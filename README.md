# RA-UWB-MIMO Benchmark

Classical ML vs. Deep Learning benchmarking framework for heterogeneous
UWB-MIMO antenna S11 sensing, built for the BCSE497J project
*"Benchmarking Deep Learning Against Classical ML for Heterogeneous
UWB-MIMO Antenna Sensing in Rheumatoid Arthritis Detection"*.

Base paper: P. Das, V. Nithin, A. Deepak, K. S. Chilakapati, *"Deep
learning enabled heterogeneous MIMO antenna sensor for detection of
rheumatoid arthritis,"* Measurement, vol. 265, 120437, 2026.

> **If you are an LLM / Claude Code reading this repo for the first
> time, read [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) first.** It's a
> denser, structured map of the codebase written specifically so a model
> can get oriented without re-deriving design decisions from scratch.

---

## What this repo actually is

The source paper designed two UWB antennas and modelled their S11
(return loss) with two different deep learning approaches:

| Antenna | Paper's model | Task shape |
|---|---|---|
| Antenna I | 1D UNet CNN | sequence: frequency → 6-channel S11 |
| Antenna II | Deterministic deep ensemble of 5 MLPs | point-wise: (frequency, geometry) → 1 S11 value |

The paper does **not** benchmark those deep models against classical ML
regressors, and doesn't report training cost / model size / inference
latency. **That comparison is this project's contribution.**

Since the paper's real dataset (from a fabricated antenna + full-wave EM
simulation + VNA measurement) isn't available to us, this repo generates
a **synthetic surrogate dataset** from the paper's own published
equivalent-circuit component values (Fig. 2e for Antenna I, Fig. 7b for
Antenna II), perturbed by the six/three geometric design parameters the
paper names. This gets you a dataset with the same *shape*, task
structure, and general S11 behaviour as the real one, so the
classical-vs-deep-learning benchmarking pipeline can be built and tested
today.

**This is explicitly a surrogate, not measured or EM-simulated data.**
See [`src/rabench/data_generation/circuit_models.py`](src/rabench/data_generation/circuit_models.py)
for the exact caveats. If real EM-simulated or measured data becomes
available later, it can be dropped into `data/raw/antenna{1,2}/` in the
same file format and everything downstream (loaders, models, benchmarks)
works unchanged — see "Swapping in real data" below.

---

## Repo structure

```
ra-uwb-mimo-benchmark/
├── configs/                        # YAML experiment configs (edit these, not the code, for most changes)
│   ├── antenna1_benchmark.yaml
│   └── antenna2_benchmark.yaml
├── data/raw/
│   ├── antenna1/                   # generated: antenna1_sample_XX.xlsx + params.csv + manifest.json
│   └── antenna2/                   # generated: antenna2_sweep_<param>.xlsx + manifest.json
├── src/rabench/
│   ├── data_generation/
│   │   ├── circuit_models.py       # shared RLC circuit math (READ THIS for the synthetic-data caveats)
│   │   ├── antenna1_generator.py   # Antenna I dataset generator
│   │   └── antenna2_generator.py   # Antenna II dataset generator
│   ├── data/
│   │   └── datasets.py             # loaders: flat view (classical/MLP) + sequence view (UNet), LOOCV splitter
│   ├── models/
│   │   ├── base.py                 # BaseRegressor interface every model implements
│   │   ├── classical/sklearn_models.py   # RandomForest, DecisionTree, XGBoost, CatBoost
│   │   └── deep/
│   │       ├── unet1d.py           # 1D UNet CNN (Antenna I)
│   │       └── deep_ensemble.py    # 5-member deep ensemble MLP (Antenna II)
│   ├── benchmarking/
│   │   ├── metrics.py              # MAE / MSE / R2, overall + per-channel
│   │   ├── runner.py               # orchestrates LOOCV (Antenna I) / train-test split (Antenna II)
│   │   └── report.py               # summary table + comparison plots
│   └── utils/seed.py
├── scripts/
│   ├── generate_dataset.py         # CLI: generate synthetic data
│   ├── run_benchmark.py            # CLI: run one task's benchmark from a YAML config
│   └── run_all.py                  # CLI: generate + benchmark both tasks end-to-end
├── results/antenna{1,2}/           # raw_results.csv, summary.csv, *.png (generated, gitignored contents)
├── tests/                          # pytest smoke tests
├── requirements.txt
└── pyproject.toml
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Generate both synthetic datasets, run both benchmarks, produce reports
python scripts/run_all.py

# ...or do it step by step / task by task:
python scripts/generate_dataset.py --task antenna1 --n-samples 8
python scripts/generate_dataset.py --task antenna2 --n-sweep-points 6
python scripts/run_benchmark.py --config configs/antenna1_benchmark.yaml
python scripts/run_benchmark.py --config configs/antenna2_benchmark.yaml
```

Outputs land in `results/antenna1/` and `results/antenna2/`:
- `raw_results.csv` — one row per (model, fold) with MAE/MSE/R², train
  time, predict time, model size, param count
- `summary.csv` — aggregated mean ± std per model
- `<task>_accuracy.png`, `<task>_efficiency.png` — comparison plots
- `per_channel_metrics.csv` (Antenna I only) — metrics broken out per
  geometric-parameter channel, matching the paper's Table 2 style

Run the test suite any time with:
```bash
9
```

## Results from the reference run in this repo

These are the numbers produced by the exact code in this repo, using the
paper's own hyperparameters (UNet1D: 150 epochs/patience 20; deep
ensemble: 300 epochs/patience 40), on the default synthetic dataset
(8 Antenna I samples, 6 sweep points per Antenna II parameter):

**Antenna I (LOOCV, 8 folds):**

| Model | R² (mean) | MAE (mean) | Train time (s) | Model size |
|---|---|---|---|---|
| CatBoost | 0.950 | 0.268 | 0.85 | 2.0 MB |
| RandomForest | 0.948 | 0.268 | 0.83 | 49.8 MB |
| XGBoost | 0.936 | 0.273 | 0.60 | 7.2 MB |
| DecisionTree | 0.908 | 0.342 | 0.003 | 57 KB |
| UNet1D | 0.214 | 1.170 | 1.03 | 173 KB |

**Antenna II (train/test split):**

| Model | R² | MAE | Train time (s) | Model size |
|---|---|---|---|---|
| XGBoost | 0.987 | 0.193 | 0.21 | 1.0 MB |
| CatBoost | 0.983 | 0.213 | 0.36 | 342 KB |
| RandomForest | 0.983 | 0.212 | 1.58 | 42.2 MB |
| DecisionTree | 0.945 | 0.359 | 0.006 | 29 KB |
| DeepEnsembleMLP | 0.939 | 0.309 | 77.0 | 941 KB |

**Reading these results honestly:** on this small, geometry-parameterized
synthetic dataset, classical ML matches or beats both deep architectures
on accuracy while training 1-2 orders of magnitude faster. This is
consistent with your project's stated hypothesis ("lighter models can
match diagnostic accuracy... more efficient, deployable path"), but
remember the *magnitude* of the gap is a property of this surrogate
dataset (small sample count for Antenna I in particular — LOOCV over 8
samples is a genuinely hard regime for a from-scratch CNN) and should be
re-validated once/if real EM-simulated or measured data is available.

---

## Adding a new model

1. Implement `BaseRegressor` (`fit`, `predict`; optionally override
   `get_model_size_bytes` / `get_param_count`) — see
   `src/rabench/models/base.py`.
2. **Classical / flat models** (point-wise regression): add it to
   `CLASSICAL_MODEL_REGISTRY` in
   `src/rabench/models/classical/sklearn_models.py`, then reference its
   registry key under `classical_models:` in a config YAML.
3. **New deep architecture**: add a module under
   `src/rabench/models/deep/`. If it's point-wise (like the MLP
   ensemble), it plugs into `run_antenna1_benchmark` /
   `run_antenna2_benchmark`'s classical-style factory pattern with no
   runner changes needed. If it's sequence-based (like UNet1D), follow
   the `unet_factory(in_channels, out_channels)` pattern in
   `scripts/run_benchmark.py` and wire it through `runner.py` the same
   way `unet_factory` is wired now.
4. Add/update a YAML config entry so the new model can be run without
   touching code again.

## Adding a new metric

Add it in `src/rabench/benchmarking/metrics.py::regression_metrics`
(and `per_channel_metrics` if it should be reported per-channel), then
surface it in `BenchmarkRow` / `_record` in `runner.py`, and in the
`.agg()` call in `report.py::summarize`.

## Swapping in real data later

If real EM-simulated or measured S11 data becomes available:
- **Antenna I**: replace the files in `data/raw/antenna1/` with real
  ones, keeping the same shape: one Excel file per sample/fold with a
  `frequency_ghz` column plus 6 `s11_<param_name>` columns, and a
  `params.csv` listing each sample's 6 geometric parameter values (see
  `antenna1_generator.py` docstring for the exact schema). Everything
  downstream (`Antenna1Dataset`, both model families, the runner) works
  unchanged.
- **Antenna II**: replace/add files in `data/raw/antenna2/` matching the
  `antenna2_sweep_*.xlsx` schema (`frequency_ghz`, the 3 geometric
  columns, `s11_db`). `Antenna2Dataset` glob-loads any file matching that
  pattern, so real sweep files can simply be added alongside or instead
  of the synthetic ones.

This is also where the project's stated Objective #4 (anomaly-detection
generalization across multiple phantom configurations) and the MIMO
diversity metrics (DG, TARC, MEG, CCL) would plug in as a new
`src/rabench/anomaly_detection/` and `src/rabench/mimo_metrics/` module
respectively — out of scope for this Option-A synthetic-data phase, see
`PROJECT_CONTEXT.md` → "Not yet implemented" for a fuller note.

## License / academic use

This repo is scaffolding for a student project benchmarking exercise. It
reimplements architectures *described* in Das et al. 2026 for comparison
purposes; it does not reproduce that paper's actual code, fabricated
hardware, or measured data. Cite the original paper when referencing the
UNet1D / deep-ensemble architectures or the RA-sensing application.
