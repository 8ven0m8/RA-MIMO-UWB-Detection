"""
datasets.py

Loads the generated Antenna I / Antenna II Excel files into two
complementary representations:

  1. FLAT   -- one row per (frequency, geometry) point. This is what every
              classical ML model (Random Forest, XGBoost, CatBoost,
              Decision Tree) and the deep-ensemble MLP consume directly,
              since they are point-wise regressors.

  2. SEQUENCE -- one entry per sample/fold, shaped
              (n_samples, n_channels, n_freq_points). This is what the
              1D UNet CNN consumes, since it convolves over the frequency
              axis and needs a proper sequence, not independent rows.

Both views are derived from the SAME underlying data so that classical
and deep models are always compared on identical information -- this is
the whole point of the benchmarking repo (see PROJECT_CONTEXT.md).

Antenna I also exposes a `loocv_splits()` generator, mirroring the
leave-one-out cross-validation protocol described in the paper's
Section 3 ("the leave-one-out strategy iteratively holds out one dataset
as the validation fold while training on the remainder").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

from rabench.data_generation.antenna1_generator import PARAM_NAMES as ANTENNA1_PARAM_NAMES
from rabench.data_generation.antenna2_generator import GEOM_PARAM_NAMES as ANTENNA2_PARAM_NAMES


# --------------------------------------------------------------------------
# Antenna I
# --------------------------------------------------------------------------


@dataclass
class Antenna1Dataset:
    """Loads all `antenna1_sample_*.xlsx` files + `params.csv` from a
    directory produced by `antenna1_generator.generate_dataset`."""

    data_dir: Path
    param_names: List[str] = None
    samples: List[pd.DataFrame] = None  # one DataFrame per fold/sample
    meta: pd.DataFrame = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.param_names = ANTENNA1_PARAM_NAMES
        self.meta = pd.read_csv(self.data_dir / "params.csv")
        self.samples = []
        for _, row in self.meta.iterrows():
            df = pd.read_excel(self.data_dir / row["file"])
            self.samples.append(df)

    @property
    def channel_names(self) -> List[str]:
        return [f"s11_{p}" for p in self.param_names]

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    # -------------------- FLAT view (for classical ML + MLP ensemble) ----

    def to_flat(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Returns (X, y, feature_names, target_names).

        X columns: frequency_ghz + the 6 geometric parameter values for
        that sample (broadcast across all frequency points in the sample).
        y columns: the 6 S11 channels.
        """
        feature_names = ["frequency_ghz"] + self.param_names
        target_names = self.channel_names

        X_rows, y_rows = [], []
        for sample_idx, df in enumerate(self.samples):
            meta_row = self.meta.iloc[sample_idx]
            param_values = [meta_row[p] for p in self.param_names]
            for _, r in df.iterrows():
                X_rows.append([r["frequency_ghz"], *param_values])
                y_rows.append([r[c] for c in target_names])

        return np.array(X_rows), np.array(y_rows), feature_names, target_names

    # -------------------- SEQUENCE view (for 1D UNet) ---------------------

    def to_sequences(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Returns (X_seq, y_seq, channel_names).

        X_seq shape: (n_samples, 1 + n_params, n_freq_points)
            channel 0            -> frequency (GHz)
            channels 1..n_params -> that sample's geometric parameter
                                     values, broadcast constant across the
                                     frequency axis (a static conditioning
                                     vector, standard practice for
                                     conditioning 1D conv nets on
                                     non-sequential covariates).
        y_seq shape: (n_samples, n_params, n_freq_points)
            the 6 S11 channels, in the same order as `channel_names`.
        """
        n_samples = self.n_samples
        n_freq = len(self.samples[0])
        n_params = len(self.param_names)

        X_seq = np.zeros((n_samples, 1 + n_params, n_freq), dtype=np.float32)
        y_seq = np.zeros((n_samples, n_params, n_freq), dtype=np.float32)

        for i, df in enumerate(self.samples):
            X_seq[i, 0, :] = df["frequency_ghz"].values
            meta_row = self.meta.iloc[i]
            for j, pname in enumerate(self.param_names):
                X_seq[i, 1 + j, :] = meta_row[pname]  # broadcast constant
            for j, cname in enumerate(self.channel_names):
                y_seq[i, j, :] = df[cname].values

        return X_seq, y_seq, self.channel_names

    # -------------------- LOOCV ---------------------------------------

    def loocv_splits(self) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yields (held_out_idx, X_train_seq, y_train_seq, X_val_seq, y_val_seq)
        for each fold, matching the paper's Antenna I evaluation protocol.
        """
        X_seq, y_seq, _ = self.to_sequences()
        n = X_seq.shape[0]
        for held_out in range(n):
            train_idx = [i for i in range(n) if i != held_out]
            yield (
                held_out,
                X_seq[train_idx],
                y_seq[train_idx],
                X_seq[[held_out]],
                y_seq[[held_out]],
            )


# --------------------------------------------------------------------------
# Antenna II
# --------------------------------------------------------------------------


@dataclass
class Antenna2Dataset:
    """Loads and merges all `antenna2_sweep_*.xlsx` files from a directory
    produced by `antenna2_generator.generate_dataset`, matching the
    paper's Section 5 description: "These files are merged into a single
    structured dataset [...] augmented with second-order and interaction
    terms, including squared frequency (f^2) and the product of frequency
    with each geometric parameter."
    """

    data_dir: Path
    merged: pd.DataFrame = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        files = sorted(self.data_dir.glob("antenna2_sweep_*.xlsx"))
        if not files:
            raise FileNotFoundError(f"No antenna2 sweep files found in {self.data_dir}")

        frames = []
        for f in files:
            df = pd.read_excel(f)
            # Fill any geometric columns missing in this particular sweep
            # file (each sweep file only varies one param; the rest were
            # baseline-fixed and still written as columns by the generator,
            # so in practice every column is present -- this guards against
            # future generators that might not do that).
            for col in ANTENNA2_PARAM_NAMES:
                if col not in df.columns:
                    df[col] = np.nan
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.dropna(subset=ANTENNA2_PARAM_NAMES + ["frequency_ghz", "s11_db"])
        self.merged = merged.reset_index(drop=True)

    def to_flat(self, engineered_features: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
        """Returns (X, y, feature_names, target_name).

        If engineered_features=True, adds f^2 and f * param_i interaction
        terms exactly as described in Section 5 of the paper.
        """
        df = self.merged.copy()
        feature_names = ["frequency_ghz"] + ANTENNA2_PARAM_NAMES

        if engineered_features:
            df["frequency_ghz_sq"] = df["frequency_ghz"] ** 2
            feature_names.append("frequency_ghz_sq")
            for pname in ANTENNA2_PARAM_NAMES:
                col = f"freq_x_{pname}"
                df[col] = df["frequency_ghz"] * df[pname]
                feature_names.append(col)

        X = df[feature_names].values
        y = df["s11_db"].values
        return X, y, feature_names, "s11_db"


def load_manifest(data_dir: str | Path) -> dict:
    with open(Path(data_dir) / "manifest.json") as f:
        return json.load(f)
