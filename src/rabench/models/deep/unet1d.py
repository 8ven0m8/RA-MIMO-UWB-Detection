"""
unet1d.py

1D UNet CNN, reproducing the architecture described in the paper's
Section 3: "a series of down-sampling convolutional blocks with max
pooling [...] a bottleneck convolutional block [...] an upsampling path
with transposed convolutions and skip connections [...] final layer
produces a 6-channel output".

This model consumes the SEQUENCE view of Antenna1Dataset
(`to_sequences()` / `loocv_splits()`), NOT the flat view -- it needs an
actual frequency-axis sequence to convolve over, unlike the classical
models and the deep-ensemble MLP.

Because LOOCV folds here are small (n_samples - 1 training antennas,
often single digits, matching the paper's own tiny per-fold dataset),
the network is kept intentionally small to avoid grossly overfitting a
handful of curves, and internal early stopping uses one held-out
training sample as a pseudo-validation set when there are enough
samples to spare.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rabench.models.base import BaseRegressor


def _conv_block(in_ch: int, out_ch: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    )


class UNet1D(nn.Module):
    """Small 1D UNet: 2 down-sampling levels + bottleneck + 2 up-sampling
    levels with skip connections, matching the paper's described
    encoder-decoder-with-skip-connections structure at a scale
    appropriate for these tiny per-fold datasets."""

    def __init__(self, in_channels: int, out_channels: int, base_width: int = 16, dropout: float = 0.0):
        super().__init__()
        self.enc1 = _conv_block(in_channels, base_width, dropout)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = _conv_block(base_width, base_width * 2, dropout)
        self.pool2 = nn.MaxPool1d(2)

        self.bottleneck = _conv_block(base_width * 2, base_width * 4, dropout)

        self.up2 = nn.ConvTranspose1d(base_width * 4, base_width * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(base_width * 4, base_width * 2, dropout)

        self.up1 = nn.ConvTranspose1d(base_width * 2, base_width, kernel_size=2, stride=2)
        self.dec1 = _conv_block(base_width * 2, base_width, dropout)

        self.out_conv = nn.Conv1d(base_width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_len = x.shape[-1]

        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        u2 = F.interpolate(u2, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.out_conv(d1)
        out = F.interpolate(out, size=input_len, mode="linear", align_corners=False)
        return out


class UNet1DModel(BaseRegressor):
    """BaseRegressor wrapper. NOTE: `fit`/`predict` here take the SEQUENCE
    view directly (X shape (n, C_in, L), y shape (n, C_out, L)), which
    breaks the flat-array convention of the other models by necessity --
    the runner special-cases this model's task wiring. See
    `rabench.benchmarking.runner.run_antenna1_benchmark`.
    """

    name = "UNet1D"
    family = "deep"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_width: int = 16,
        max_epochs: int = 150,
        patience: int = 20,
        batch_size: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        device: Optional[str] = None,
        seed: int = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_width = base_width
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        torch.manual_seed(seed)
        self.net = UNet1D(in_channels, out_channels, base_width, dropout).to(self.device)

        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    def _fit_scalers(self, X: np.ndarray, y: np.ndarray):
        self._x_mean = X.mean(axis=(0, 2), keepdims=True)
        self._x_std = X.std(axis=(0, 2), keepdims=True) + 1e-8
        self._y_mean = y.mean(axis=(0, 2), keepdims=True)
        self._y_std = y.std(axis=(0, 2), keepdims=True) + 1e-8

    def _scale_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    def _scale_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self._y_mean) / self._y_std

    def _unscale_y(self, y: np.ndarray) -> np.ndarray:
        return y * self._y_std + self._y_mean

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._fit_scalers(X_train, y_train)
        Xs = self._scale_x(X_train)
        ys = self._scale_y(y_train)

        n = Xs.shape[0]
        use_internal_val = n >= 4
        if use_internal_val:
            n_val = max(1, n // 5)
            # Randomized (not trailing-index) split: sample order reflects
            # generation order, not anything meaningful, so always slicing
            # off the last few samples as "validation" gives early stopping
            # a fixed, arbitrary subset every fold instead of a
            # representative one -- shuffle it like DeepEnsembleMLP does.
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
        batch_size = max(2, min(self.batch_size, n_train))

        for epoch in range(self.max_epochs):
            self.net.train()
            perm_t = torch.randperm(n_train)
            epoch_loss = 0.0
            n_seen = 0
            for i in range(0, n_train, batch_size):
                idx = perm_t[i : i + batch_size]
                if len(idx) < 2:
                    continue  # BatchNorm needs >=2 samples per batch
                xb, yb = X_tr[idx], y_tr[idx]
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
                n_seen += len(idx)
            epoch_loss /= max(1, n_seen)

            if val_idx:
                self.net.eval()
                with torch.no_grad():
                    val_pred = self.net(X_val)
                    val_loss = loss_fn(val_pred, y_val).item()
            else:
                val_loss = epoch_loss  # no held-out data: use train loss
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

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
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
