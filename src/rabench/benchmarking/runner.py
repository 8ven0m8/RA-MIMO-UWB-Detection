"""
runner.py

Orchestrates model training/evaluation for both tasks and produces a
single, uniform results table (one row per model x fold) with accuracy
metrics (MAE/MSE/R2) AND the efficiency metrics your project's Objective
#2 asks for (training time, inference latency, model size / param count).

Task <-> model assignment (matches the paper's own structure, see
PROJECT_CONTEXT.md "Task definitions"):

  Antenna I  (UNet1D sequence task)   : classical models (flat) + UNet1D (sequence)
                                        evaluated via LOOCV, same protocol the
                                        paper itself uses.
  Antenna II (deep ensemble MLP task) : classical models (flat) + DeepEnsembleMLP (flat)
                                        evaluated via a single train/test split.

Both benchmarks return a pandas DataFrame with identical columns so
`report.py` can combine/plot them uniformly.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from rabench.benchmarking.metrics import per_channel_metrics, regression_metrics
from rabench.data.datasets import ANTENNA2_PARAM_NAMES, Antenna1Dataset, Antenna2Dataset
from rabench.models.base import BaseRegressor
from rabench.models.deep.unet1d import UNet1DModel
from rabench.utils.progress import fmt_duration as _fmt_duration, text_bar


def _save_model(model: BaseRegressor, models_dir: Optional[str], filename_stem: str) -> None:
    if not models_dir:
        return
    ext = "pt" if model.family == "deep" else "pkl"
    model.save(str(Path(models_dir) / f"{filename_stem}.{ext}"))


@dataclass
class BenchmarkRow:
    task: str
    model_name: str
    family: str
    fold: Optional[int]
    mae: float
    mse: float
    r2: float
    train_time_sec: float
    predict_time_sec: float
    model_size_bytes: Optional[float]
    param_count: Optional[float]


def _time_call(fn: Callable, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return result, t1 - t0


class _ProgressTracker:
    """Prints a one-line progress bar after each (fold, model) run so a
    long LOOCV benchmark isn't silent for minutes at a time -- shows how
    far along it is, this step's timing/accuracy, and a running ETA.

    The ETA is estimated per model (its mean wall time so far x its
    remaining runs), not from the overall average: the runner trains
    every classical model across all folds before any deep model, so an
    overall average would wildly underestimate during the fast classical
    phase. Models that haven't run yet can't be estimated, so the ETA is
    shown as a lower bound (">=") until every model has run once."""

    def __init__(self, runs_per_model: Dict[str, int], label: str):
        self.remaining = dict(runs_per_model)
        self.total = sum(runs_per_model.values())
        self.label = label
        self.done = 0
        self.start = time.perf_counter()
        self._last = self.start
        self._time_sum: Dict[str, float] = {}
        self._count: Dict[str, int] = {}

    def step(self, model_key: str, tag: str, train_time: float, r2: float) -> None:
        now = time.perf_counter()
        self._time_sum[model_key] = self._time_sum.get(model_key, 0.0) + (now - self._last)
        self._count[model_key] = self._count.get(model_key, 0) + 1
        self._last = now
        self.remaining[model_key] -= 1
        self.done += 1

        eta = 0.0
        lower_bound = False
        for key, left in self.remaining.items():
            if left <= 0:
                continue
            if key in self._count:
                eta += self._time_sum[key] / self._count[key] * left
            else:
                lower_bound = True
        eta_str = (">=" if lower_bound else "") + _fmt_duration(eta)
        print(
            f"[{self.label}] {text_bar(self.done, self.total)} {self.done}/{self.total} | {tag} | "
            f"train={train_time:.2f}s R2={r2:.3f} | "
            f"elapsed {_fmt_duration(now - self.start)} eta {eta_str}",
            flush=True,
        )


def _record(
    task: str,
    model: BaseRegressor,
    fold: Optional[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time: float,
    predict_time: float,
    measure_size: bool = True,
) -> BenchmarkRow:
    m = regression_metrics(y_true, y_pred)
    return BenchmarkRow(
        task=task,
        model_name=model.name,
        family=model.family,
        fold=fold,
        mae=m["MAE"],
        mse=m["MSE"],
        r2=m["R2"],
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        model_size_bytes=model.get_model_size_bytes() if measure_size else None,
        param_count=model.get_param_count(),
    )


# --------------------------------------------------------------------------
# Antenna I: LOOCV benchmark (classical flat models + UNet1D sequence model)
# --------------------------------------------------------------------------


def run_antenna1_benchmark(
    dataset: Antenna1Dataset,
    classical_model_factories: Dict[str, Callable[[], BaseRegressor]],
    unet_factory: Optional[Callable[[int, int], BaseRegressor]] = None,
    seq_deep_model_factories: Optional[Dict[str, Callable[[int, int], BaseRegressor]]] = None,
    per_channel_output_path: Optional[str] = None,
    models_dir: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[BenchmarkRow] = []
    per_channel_rows = []

    X_flat, y_flat, _, target_names = dataset.to_flat()
    n_freq = len(dataset.samples[0])
    sample_id_per_row = np.repeat(np.arange(dataset.n_samples), n_freq)

    all_seq_factories = dict(seq_deep_model_factories or {})
    if unet_factory is not None:
        all_seq_factories["unet1d"] = unet_factory

    total_iters = dataset.n_samples * (len(classical_model_factories) + len(all_seq_factories))
    print(
        f"[antenna1] starting LOOCV benchmark: {dataset.n_samples} folds x "
        f"({len(classical_model_factories)} classical + {len(all_seq_factories)} deep models) "
        f"= {total_iters} runs",
        flush=True,
    )
    progress = _ProgressTracker(
        {f"classical/{k}": dataset.n_samples for k in classical_model_factories}
        | {f"deep/{k}": dataset.n_samples for k in all_seq_factories},
        "antenna1",
    )

    # ---- classical models on the flat view, LOOCV over samples ----
    for held_out in range(dataset.n_samples):
        train_mask = sample_id_per_row != held_out
        val_mask = ~train_mask
        X_train, y_train = X_flat[train_mask], y_flat[train_mask]
        X_val, y_val = X_flat[val_mask], y_flat[val_mask]

        is_last_fold = held_out == dataset.n_samples - 1
        for name, factory in classical_model_factories.items():
            model = factory()
            _, train_time = _time_call(model.fit, X_train, y_train)
            y_pred, predict_time = _time_call(model.predict, X_val)

            # Only persist/size-measure the last fold's model per type --
            # LOOCV trains dataset.n_samples near-identical copies of each
            # model, and pickling every one of them (400 for this config)
            # is what exhausts memory. Metrics below still come from every
            # fold; only the on-disk artifact and size measurement are
            # deduped to one representative model per type.
            if is_last_fold:
                _save_model(model, models_dir, model.name)
            row = _record("antenna1", model, held_out, y_val, y_pred, train_time, predict_time, measure_size=is_last_fold)
            rows.append(row)
            progress.step(f"classical/{name}", f"fold {held_out + 1}/{dataset.n_samples} classical/{name}", train_time, row.r2)
            ch_metrics = per_channel_metrics(y_val, y_pred, target_names)
            for ch_name, m in ch_metrics.items():
                per_channel_rows.append({"task": "antenna1", "model_name": model.name, "fold": held_out, "channel": ch_name, **m})

    # ---- sequence deep models (UNet1D + any extras), LOOCV over samples ----
    if all_seq_factories:
        for held_out, X_tr, y_tr, X_val, y_val in dataset.loocv_splits():
            is_last_fold = held_out == dataset.n_samples - 1
            for name, factory in all_seq_factories.items():
                model = factory(X_tr.shape[1], y_tr.shape[1])
                _, train_time = _time_call(model.fit, X_tr, y_tr)
                y_pred, predict_time = _time_call(model.predict, X_val)

                # reshape (1, n_channels, n_freq) -> (n_freq, n_channels) so
                # metrics are computed identically to the flat classical models
                y_val_2d = y_val[0].T
                y_pred_2d = y_pred[0].T

                # Same dedup as the classical loop above: one saved model
                # per type (last fold), not dataset.n_samples of them.
                if is_last_fold:
                    _save_model(model, models_dir, model.name)
                row = _record("antenna1", model, held_out, y_val_2d, y_pred_2d, train_time, predict_time, measure_size=is_last_fold)
                rows.append(row)
                progress.step(f"deep/{name}", f"fold {held_out + 1}/{dataset.n_samples} deep/{name}", train_time, row.r2)
                ch_metrics = per_channel_metrics(y_val_2d, y_pred_2d, target_names)
                for ch_name, m in ch_metrics.items():
                    per_channel_rows.append({"task": "antenna1", "model_name": model.name, "fold": held_out, "channel": ch_name, **m})

    print(f"[antenna1] done: {total_iters} runs in {_fmt_duration(time.perf_counter() - progress.start)}", flush=True)
    df = pd.DataFrame([asdict(r) for r in rows])
    if per_channel_output_path:
        pd.DataFrame(per_channel_rows).to_csv(per_channel_output_path, index=False)
    return df


# --------------------------------------------------------------------------
# Antenna II: train/test split benchmark (classical flat models + deep ensemble MLP)
# --------------------------------------------------------------------------


def run_antenna2_benchmark(
    dataset: Antenna2Dataset,
    classical_model_factories: Dict[str, Callable[[], BaseRegressor]],
    deep_ensemble_factory: Optional[Callable[[], BaseRegressor]] = None,
    deep_model_factories: Optional[Dict[str, Callable[[], BaseRegressor]]] = None,
    test_size: float = 0.2,
    seed: int = 42,
    models_dir: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[BenchmarkRow] = []

    X, y, _, _ = dataset.to_flat(engineered_features=True)
    # Split by S11 curve (one geometry configuration), not by row: a random
    # row split puts neighbouring frequency points of the same curve in
    # both train and test, so the test set would partly measure
    # interpolation within curves the model has already seen.
    curve_ids, _ = pd.factorize(pd.MultiIndex.from_frame(dataset.merged[ANTENNA2_PARAM_NAMES]))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=curve_ids))
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    print(
        f"[antenna2] split by curve: {len(np.unique(curve_ids[train_idx]))} train / "
        f"{len(np.unique(curve_ids[test_idx]))} test curves ({len(train_idx)} / {len(test_idx)} rows)",
        flush=True,
    )

    all_factories = dict(classical_model_factories)
    all_factories.update(deep_model_factories or {})
    if deep_ensemble_factory is not None:
        all_factories["deep_ensemble"] = deep_ensemble_factory

    print(f"[antenna2] starting train/test benchmark: {len(all_factories)} models", flush=True)
    progress = _ProgressTracker({name: 1 for name in all_factories}, "antenna2")

    for name, factory in all_factories.items():
        model = factory()
        _, train_time = _time_call(model.fit, X_train, y_train)
        y_pred, predict_time = _time_call(model.predict, X_test)
        _save_model(model, models_dir, model.name)
        row = _record("antenna2", model, None, y_test, y_pred, train_time, predict_time)
        rows.append(row)
        progress.step(name, name, train_time, row.r2)

    print(f"[antenna2] done: {len(all_factories)} models in {_fmt_duration(time.perf_counter() - progress.start)}", flush=True)
    return pd.DataFrame([asdict(r) for r in rows])


def default_unet_factory(in_channels: int, out_channels: int, **kwargs) -> UNet1DModel:
    return UNet1DModel(in_channels=in_channels, out_channels=out_channels, **kwargs)
