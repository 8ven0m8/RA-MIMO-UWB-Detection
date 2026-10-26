"""
transformer.py

Transformer-encoder regressor -- the second of the two additional popular
deep architectures added alongside UNet1D / DeepEnsembleMLP.

Same dual-mode contract as `lstm.py` (see that module's docstring):

  * SEQUENCE mode (Antenna I): a Transformer encoder attends across the
    frequency axis; a per-timestep linear head produces the 6-channel
    S11 output.
  * FLAT mode (Antenna II): each scalar feature is projected to a token
    embedding (the FT-Transformer-style "tabular-as-tokens" framing) and
    attended over; the pooled representation is regressed to a single
    S11 value.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from rabench.models.base import BaseRegressor
from rabench.utils.progress import EpochHeartbeat


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


def _make_encoder(d_model: int, nhead: int, num_layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class _SeqTransformer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        self.encoder = _make_encoder(d_model, nhead, num_layers, dropout)
        self.head = nn.Linear(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, C_in, L) -> (n, L, C_in)
        x = x.transpose(1, 2)
        h = self.in_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        out = self.head(h)  # (n, L, C_out)
        return out.transpose(1, 2)  # (n, C_out, L)


class _FlatTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        self.encoder = _make_encoder(d_model, nhead, num_layers, dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, F) -> (n, F, 1) -- one token per feature
        x = x.unsqueeze(-1)
        h = self.in_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class TransformerModel(BaseRegressor):
    """BaseRegressor wrapper around a Transformer encoder. See module
    docstring for the sequence-vs-flat dual-mode contract."""

    name = "Transformer"
    family = "deep"

    def __init__(
        self,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        seed: int = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        torch.manual_seed(seed)
        self.mode: Optional[str] = None
        self.net: Optional[nn.Module] = None
        if in_channels is not None and out_channels is not None:
            self.mode = "sequence"
            self.net = _SeqTransformer(in_channels, out_channels, d_model, nhead, num_layers, dropout).to(self.device)

        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    def _build_flat_net(self, n_features: int):
        self.mode = "flat"
        self.net = _FlatTransformer(self.d_model, self.nhead, self.num_layers, self.dropout).to(self.device)

    def _fit_scalers(self, X: np.ndarray, y: np.ndarray):
        if self.mode == "sequence":
            self._x_mean = X.mean(axis=(0, 2), keepdims=True)
            self._x_std = X.std(axis=(0, 2), keepdims=True) + 1e-8
            self._y_mean = y.mean(axis=(0, 2), keepdims=True)
            self._y_std = y.std(axis=(0, 2), keepdims=True) + 1e-8
        else:
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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.net is None:
            self._build_flat_net(X_train.shape[1])

        self._fit_scalers(X_train, y_train)
        Xs = self._scale_x(X_train)
        ys = self._scale_y(y_train)

        n = Xs.shape[0]
        use_internal_val = n >= 4
        if use_internal_val:
            n_val = max(1, n // 5)
            # Randomized split (not a fixed trailing slice) so early
            # stopping validates against a representative subset every
            # fold, matching DeepEnsembleMLP's approach.
            perm = np.random.default_rng(self.seed).permutation(n)
            val_idx = perm[:n_val].tolist()
            train_idx = perm[n_val:].tolist()
        else:
            train_idx = list(range(n))
            val_idx = []

        X_tr = torch.tensor(Xs[train_idx], dtype=torch.float32, device=self.device)
        y_tr = torch.tensor(ys[train_idx], dtype=torch.float32, device=self.device)
        if val_idx:
            X_val = torch.tensor(Xs[val_idx], dtype=torch.float32, device=self.device)
            y_val = torch.tensor(ys[val_idx], dtype=torch.float32, device=self.device)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        n_train = X_tr.shape[0]
        batch_size = max(1, min(self.batch_size, n_train))

        heartbeat = EpochHeartbeat(self.name, self.max_epochs, self.patience)
        for epoch in range(self.max_epochs):
            self.net.train()
            perm_t = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                idx = perm_t[i : i + batch_size]
                xb, yb = X_tr[idx], y_tr[idx]
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()

            if val_idx:
                self.net.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self.net(X_val), y_val).item()
            else:
                self.net.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self.net(X_tr), y_tr).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break
            heartbeat.update(epoch, val_loss, best_val_loss, epochs_no_improve)
        heartbeat.finish(epoch, best_val_loss, epochs_no_improve)

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scale_x(X)
        self.net.eval()
        with torch.no_grad():
            x_t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            pred = self.net(x_t).cpu().numpy()
        return self._unscale_y(pred)

    def get_model_size_bytes(self):
        import io

        buf = io.BytesIO()
        torch.save(self.net.state_dict(), buf)
        return buf.tell()

    def get_param_count(self):
        return sum(p.numel() for p in self.net.parameters())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
