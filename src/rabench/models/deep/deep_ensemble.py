"""
deep_ensemble.py

Deterministic deep ensemble of MLPs, reproducing the paper's Section 6
architecture almost verbatim:
  * E = 5 independently-seeded MLPs
  * each: input -> [256, 128, 64] hidden layers, BatchNorm + ReLU + Dropout(0.2)
    after each hidden layer -> single output neuron
  * optimizer: AdamW, lr=1e-3, weight_decay=1e-5
  * loss: MAE (L1Loss)
  * LR schedule: reduce-on-plateau, factor 0.5, patience 8
  * early stopping: patience 40
  * final prediction = mean across the 5 members; std = predictive
    uncertainty (also exposed via `predict_with_uncertainty`)

Unlike UNet1D, this model operates on the FLAT view (point-wise
regression: (frequency, geometry [+ engineered features]) -> single S11
value), exactly like the classical models -- so it can be dropped into
the same benchmarking loop with no special-casing beyond "it's a deep
model" for size/param-count reporting.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from rabench.models.base import BaseRegressor


class _MLP(nn.Module):
    def __init__(self, in_features: int, hidden=(256, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DeepEnsembleModel(BaseRegressor):
    name = "DeepEnsembleMLP"
    family = "deep"

    def __init__(
        self,
        n_members: int = 5,
        hidden=(256, 128, 64),
        dropout: float = 0.2,
        max_epochs: int = 300,
        patience: int = 40,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        val_fraction: float = 0.15,
        device: Optional[str] = None,
        base_seed: int = 0,
    ):
        self.n_members = n_members
        self.hidden = hidden
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_seed = base_seed

        self.members: list[_MLP] = []
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    def _fit_scalers(self, X, y):
        self._x_mean = X.mean(axis=0, keepdims=True)
        self._x_std = X.std(axis=0, keepdims=True) + 1e-8
        self._y_mean = y.mean()
        self._y_std = y.std() + 1e-8

    def _scale_x(self, X):
        return (X - self._x_mean) / self._x_std

    def _scale_y(self, y):
        return (y - self._y_mean) / self._y_std

    def _unscale_y(self, y):
        return y * self._y_std + self._y_mean

    def _train_one_member(self, seed: int, X_tr, y_tr, X_val, y_val) -> _MLP:
        torch.manual_seed(seed)
        model = _MLP(X_tr.shape[1], hidden=self.hidden, dropout=self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
        loss_fn = nn.L1Loss()

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=self.device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        n_train = X_tr_t.shape[0]
        batch_size = max(2, min(self.batch_size, n_train))

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                if len(idx) < 2:
                    continue  # BatchNorm needs >=2 samples
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = loss_fn(val_pred, y_val_t).item()
            scheduler.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._fit_scalers(X_train, y_train)
        Xs = self._scale_x(X_train)
        ys = self._scale_y(y_train)

        n = Xs.shape[0]
        n_val = max(2, int(n * self.val_fraction))
        rng = np.random.default_rng(self.base_seed)
        perm = rng.permutation(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_tr, y_tr = Xs[train_idx], ys[train_idx]
        X_val, y_val = Xs[val_idx], ys[val_idx]

        self.members = []
        for m in range(self.n_members):
            member = self._train_one_member(self.base_seed + m, X_tr, y_tr, X_val, y_val)
            self.members.append(member)

    def predict(self, X: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(X)
        return mean

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self._scale_x(X)
        x_t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        preds = []
        with torch.no_grad():
            for member in self.members:
                member.eval()
                preds.append(member(x_t).cpu().numpy())
        preds = np.stack(preds, axis=0)  # (E, N)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return self._unscale_y(mean), std * self._y_std

    def get_model_size_bytes(self):
        import io

        buf = io.BytesIO()
        torch.save([m.state_dict() for m in self.members], buf)
        return buf.tell()

    def get_param_count(self):
        return sum(sum(p.numel() for p in m.parameters()) for m in self.members)

    def save(self, path: str) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save([m.state_dict() for m in self.members], path)
