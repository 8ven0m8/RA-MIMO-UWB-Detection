#!/usr/bin/env python
"""
run_benchmark.py

Runs the classical-vs-deep benchmark for one task, driven by a YAML config
(see configs/antenna1_benchmark.yaml, configs/antenna2_benchmark.yaml).

Usage:
    python scripts/run_benchmark.py --config configs/antenna1_benchmark.yaml
    python scripts/run_benchmark.py --config configs/antenna2_benchmark.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rabench.benchmarking.report import generate_full_report  # noqa: E402
from rabench.benchmarking.runner import (  # noqa: E402
    run_antenna1_benchmark,
    run_antenna2_benchmark,
)
from rabench.data.datasets import Antenna1Dataset, Antenna2Dataset  # noqa: E402
from rabench.models.classical.sklearn_models import CLASSICAL_MODEL_REGISTRY  # noqa: E402
from rabench.models.deep.deep_ensemble import DeepEnsembleModel  # noqa: E402
from rabench.models.deep.lstm import LSTMModel  # noqa: E402
from rabench.models.deep.transformer import TransformerModel  # noqa: E402
from rabench.models.deep.unet1d import UNet1DModel  # noqa: E402
from rabench.utils.seed import set_global_seed  # noqa: E402

#: Deep models usable on the SEQUENCE view (Antenna I), keyed by the
#: `name` field used in a config's `deep_models` list entries.
SEQ_DEEP_MODEL_REGISTRY = {
    "unet1d": UNet1DModel,
    "lstm": LSTMModel,
    "transformer": TransformerModel,
}

#: Deep models usable on the FLAT view (Antenna II).
FLAT_DEEP_MODEL_REGISTRY = {
    "deep_ensemble": DeepEnsembleModel,
    "lstm": LSTMModel,
    "transformer": TransformerModel,
}


def build_classical_factories(cfg: dict):
    factories = {}
    for model_key, params in (cfg.get("classical_models") or {}).items():
        cls = CLASSICAL_MODEL_REGISTRY[model_key]
        params = params or {}
        factories[model_key] = (lambda cls=cls, params=params: cls(**params))
    return factories


def _deep_model_configs(cfg: dict) -> list[dict]:
    """Supports both the legacy single `deep_model:` mapping and the
    newer `deep_models:` list, so a config only naming one deep model
    still works unmodified."""
    if "deep_models" in cfg:
        return list(cfg["deep_models"] or [])
    if "deep_model" in cfg and cfg["deep_model"]:
        return [cfg["deep_model"]]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_global_seed(cfg.get("seed", 42))

    task = cfg["task"]
    data_dir = REPO_ROOT / cfg["data_dir"]
    results_dir = REPO_ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    classical_factories = build_classical_factories(cfg)
    models_dir = str(results_dir / "models")

    if task == "antenna1":
        dataset = Antenna1Dataset(data_dir)

        seq_factories = {}
        for deep_cfg in _deep_model_configs(cfg):
            model_key = deep_cfg["name"]
            cls = SEQ_DEEP_MODEL_REGISTRY[model_key]
            kwargs = {k: v for k, v in deep_cfg.items() if k != "name"}
            seq_factories[model_key] = (
                lambda in_ch, out_ch, cls=cls, kwargs=kwargs: cls(in_channels=in_ch, out_channels=out_ch, **kwargs)
            )

        results_df = run_antenna1_benchmark(
            dataset,
            classical_model_factories=classical_factories,
            seq_deep_model_factories=seq_factories,
            per_channel_output_path=str(results_dir / "per_channel_metrics.csv"),
            models_dir=models_dir,
        )

    elif task == "antenna2":
        dataset = Antenna2Dataset(data_dir)

        deep_factories = {}
        for deep_cfg in _deep_model_configs(cfg):
            model_key = deep_cfg["name"]
            cls = FLAT_DEEP_MODEL_REGISTRY[model_key]
            kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in deep_cfg.items() if k != "name"}
            deep_factories[model_key] = (lambda cls=cls, kwargs=kwargs: cls(**kwargs))

        results_df = run_antenna2_benchmark(
            dataset,
            classical_model_factories=classical_factories,
            deep_model_factories=deep_factories,
            test_size=cfg.get("test_size", 0.2),
            seed=cfg.get("seed", 42),
            models_dir=models_dir,
        )

    else:
        raise ValueError(f"Unknown task: {task}")

    raw_path = results_dir / "raw_results.csv"
    results_df.to_csv(raw_path, index=False)
    print(f"Raw per-fold results -> {raw_path}")
    print(f"Trained models -> {models_dir}")

    summary = generate_full_report(results_df, str(results_dir), task=task)
    print("\n=== Summary (mean across folds) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
