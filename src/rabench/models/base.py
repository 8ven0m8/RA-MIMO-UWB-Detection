"""
base.py

Every model in this repo -- classical (RF, XGBoost, CatBoost, Decision
Tree) or deep (UNet1D, deep ensemble MLP) -- implements this same
`BaseRegressor` interface. That's what makes the benchmarking runner
(`rabench.benchmarking.runner`) able to loop over an arbitrary list of
models and produce a fair, apples-to-apples comparison table.

Design decision: `fit` / `predict` operate purely on numpy arrays. Any
model-specific preprocessing (scaling, reshaping to sequences, etc.)
happens INSIDE the model class, not in the runner. This keeps the runner
dumb and the model registry the single place new models get added.
"""

from __future__ import annotations

import io
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class BaseRegressor(ABC):
    """Common interface for all benchmarked models."""

    #: Human-readable name shown in results tables/plots.
    name: str = "base_model"

    #: "classical" or "deep" -- used purely for grouping/reporting.
    family: str = "unknown"

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    #: Set by `save()` so `get_model_size_bytes()` can reuse the on-disk
    #: size instead of pickling a second in-memory copy.
    _saved_size_bytes: Optional[int] = None

    def get_model_size_bytes(self) -> Optional[int]:
        """Default implementation: reuse the size of the file written by
        `save()` if available (cheap, no extra copy in memory). Falls back
        to pickling into an in-memory buffer only if `save()` hasn't run
        yet. Deep models override this with a torch-native state_dict size
        (usually smaller / more meaningful)."""
        if self._saved_size_bytes is not None:
            return self._saved_size_bytes
        try:
            buf = io.BytesIO()
            pickle.dump(self, buf)
            return buf.tell()
        except Exception:
            return None

    def get_param_count(self) -> Optional[int]:
        """Only meaningful for parametric models (deep nets). Classical
        tree ensembles return None -- 'number of parameters' isn't a
        well-defined comparable concept for them (num trees x depth is
        reported separately in the benchmarking report instead)."""
        return None

    def save(self, path: str) -> None:
        """Persist the trained model to disk. Default implementation
        pickles the whole object (fine for classical sklearn-style
        models); deep (torch-based) models override this to save just
        their state_dict(s), the standard/lighter torch convention."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._saved_size_bytes = Path(path).stat().st_size
