"""
metrics.py

Same three metrics the paper itself reports (MAE, MSE, R^2), computed
identically for every model regardless of family, so classical-vs-deep
comparisons are apples-to-apples. For multi-channel targets (Antenna I),
both an overall (flattened) score and a per-channel breakdown are
returned, mirroring the paper's own per-channel reporting (Table 2).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def per_channel_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, channel_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """y_true / y_pred shape: (n_samples, n_channels)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_channels = y_true.shape[1]
    names = channel_names or [f"channel_{i}" for i in range(n_channels)]

    out = {}
    for i, name in enumerate(names):
        out[name] = regression_metrics(y_true[:, i], y_pred[:, i])
    return out
