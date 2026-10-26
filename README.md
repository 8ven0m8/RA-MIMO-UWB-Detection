# RA-UWB-MIMO Benchmark

Classical ML vs. deep learning benchmark for S11 prediction on heterogeneous
UWB-MIMO antenna sensors, built for the BCSE497J project *"Benchmarking Deep
Learning Against Classical ML for Heterogeneous UWB-MIMO Antenna Sensing in
Rheumatoid Arthritis Detection"*.

Base paper: P. Das, V. Nithin, A. Deepak, K. S. Chilakapati, *"Deep learning
enabled heterogeneous MIMO antenna sensor for detection of rheumatoid
arthritis,"* Measurement, vol. 265, 120437, 2026.

> If you are an LLM or coding agent reading this repo for the first time, read
> [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) first. It is a denser map of the
> codebase and the design decisions behind it.

## What this repo is

The base paper designed two UWB antennas and modelled their S11 (return loss)
with two deep learning approaches:

| Antenna | Paper's model | Task shape |
|---|---|---|
| Antenna I | 1D U-Net CNN | sequence: frequency → 6-channel S11 |
| Antenna II | Deterministic deep ensemble of MLPs | point-wise: (frequency, geometry) → 1 S11 value |

The paper does not compare those models with classical regressors, and it does
not report training cost, model size, or inference latency. This repo makes
that comparison: four classical models (Decision Tree, Random Forest, XGBoost,
CatBoost) against four deep models (1D U-Net, bidirectional LSTM, Transformer
encoder, deep-ensemble MLP), on identical splits and metrics.

The paper's measured and EM-simulated data are not available, so the repo
generates a **synthetic surrogate dataset** from the paper's published
equivalent-circuit element values (Fig. 2e for Antenna I, Fig. 7b for Antenna
II), perturbed by the geometric parameters the paper names, with Gaussian noise
of known size (0.15 dB for Antenna I, 0.20 dB for Antenna II). The data have the
same shape and task structure as the real data, but they are not measured or
full-wave simulated. See
[`src/rabench/data_generation/circuit_models.py`](src/rabench/data_generation/circuit_models.py)
for the caveats, and "Swapping in real data" below for how to replace them.

## Repo structure

```
RA-MIMO-UWB-Detection/
├── configs/
│   ├── antenna1_benchmark.yaml         # baseline: 50 antennas, small deep models (laptop/CPU)
│   ├── antenna2_benchmark.yaml         # baseline: 50 sweep points per parameter
│   ├── antenna1_benchmark_large.yaml   # large: 150 antennas, wider deep models (GPU)
│   └── antenna2_benchmark_large.yaml   # large: 100 sweep points, larger deep models (GPU)
├── data/                               # generated datasets (gitignored, regenerate with the scripts)
├── src/rabench/
│   ├── data_generation/                # circuit model + Antenna I/II generators
│   ├── data/datasets.py                # flat view, sequence view, LOOCV splitter
│   ├── models/
│   │   ├── base.py                     # BaseRegressor interface every model implements
│   │   ├── classical/sklearn_models.py # Random Forest, Decision Tree, XGBoost, CatBoost
│   │   └── deep/                       # unet1d.py, lstm.py, transformer.py, deep_ensemble.py
│   ├── benchmarking/                   # metrics, runner (LOOCV / split by curve), report
│   └── utils/                          # seeding, progress bar + ETA
├── scripts/
│   ├── generate_dataset.py             # generate synthetic data
│   ├── run_benchmark.py                # run one task's benchmark from a YAML config
│   └── run_all.py                      # generate + benchmark both tasks end to end
├── results/                            # CSVs and charts per run (trained models/ are gitignored)
├── tests/                              # pytest smoke tests
├── requirements.txt                    # CPU-only install (see below for GPU)
└── pyproject.toml
```

## Installation

Python 3.10 or newer. Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

**CPU only (no NVIDIA GPU).** `requirements.txt` installs the CPU build of
PyTorch and `xgboost-cpu`, so none of the multi-gigabyte CUDA packages are
downloaded:

```bash
pip install -r requirements.txt
```

**With an NVIDIA GPU.** Install PyTorch with CUDA first (pick the command for
your CUDA version at <https://pytorch.org/get-started/locally/>), then the rest
of the dependencies. Plain `xgboost` works here too:

```bash
pip install torch
pip install numpy pandas scikit-learn xgboost catboost matplotlib pyyaml openpyxl scipy pytest
```

## Running the benchmark

```bash
# Baseline: generate both datasets and run both benchmarks (CPU is fine)
python scripts/run_all.py

# Large variant on a GPU; --require-gpu stops early if CUDA is not visible
python scripts/run_all.py --variant large --require-gpu

# Or step by step
python scripts/generate_dataset.py --task antenna1 --n-samples 50
python scripts/generate_dataset.py --task antenna2 --n-sweep-points 50
python scripts/run_benchmark.py --config configs/antenna1_benchmark.yaml
python scripts/run_benchmark.py --config configs/antenna2_benchmark.yaml
```

`run_all.py` reads the sample counts and data directories from the configs.
Each config writes to its own `results_dir`:

- `raw_results.csv`: one row per (model, fold) with MAE, MSE, R², train time,
  predict time, model size, and parameter count
- `summary.csv`: mean and standard deviation per model
- `<task>_accuracy.png`, `<task>_efficiency.png`: comparison charts
- `per_channel_metrics.csv` (Antenna I only): metrics per geometric-parameter channel
- `models/`: the trained model of the last fold per type (gitignored; the
  Random Forest pickles alone are 0.35 to 1 GB)

Antenna I is evaluated with leave-one-out cross-validation over antennas.
Antenna II is split 80/20 **by curve**, so every row of a geometry lands on the
same side and the test set holds curves the models never saw.

Run the tests with:

```bash
pytest
```

Expect long runs: the large Antenna I benchmark is 150 folds × 7 models = 1050
training runs and took about 21 hours on an RTX A4000.

## Results

### Large run (150 Antenna I antennas, GPU)

Classical models trained on CPU, deep models on an NVIDIA RTX A4000.
Results in `results/antenna1_large/` and `results/antenna2_large/`.

**Antenna I** (LOOCV, 150 folds; noise floor MAE 0.120 dB):

| Model | MAE (dB) | R² | Train time per fold (s) | Size (MB) |
|---|---|---|---|---|
| XGBoost | 0.132 ± 0.004 | 0.9956 | 5.5 | 8.5 |
| Transformer | 0.134 ± 0.005 | 0.9956 | 48.3 | 14.8 |
| LSTM | 0.137 ± 0.008 | 0.9952 | 357.5 | 14.8 |
| CatBoost | 0.159 ± 0.006 | 0.9919 | 13.2 | 2.0 |
| Random Forest | 0.166 ± 0.015 | 0.9889 | 28.6 | 1059.0 |
| 1D U-Net | 0.189 ± 0.024 | 0.9883 | 53.0 | 2.6 |
| Decision Tree | 0.293 ± 0.024 | 0.9465 | 0.25 | 0.06 |

**Antenna II** (60 held-out curves, split by curve; noise floor MAE 0.160 dB):

| Model | MAE (dB) | R² | Train time (s) | Size (MB) |
|---|---|---|---|---|
| LSTM | 0.161 | 0.9903 | 109.8 | 2.1 |
| XGBoost | 0.166 | 0.9896 | 1.1 | 1.4 |
| Deep ensemble (10 MLPs) | 0.166 | 0.9895 | 443.6 | 28.3 |
| Transformer | 0.181 | 0.9877 | 142.7 | 4.2 |
| Random Forest | 0.185 | 0.9871 | 27.6 | 701.5 |
| CatBoost | 0.187 | 0.9868 | 2.3 | 0.3 |
| Decision Tree | 0.284 | 0.9633 | 0.3 | 0.04 |

`results/antenna2_large_rowsplit/` keeps an earlier Antenna II run that used a
random row split. Every model's MAE moved by at most 0.006 dB between the two
protocols.

### Baseline (50 Antenna I antennas, CPU)

`results/antenna1/`, laptop CPU, with the U-Net's dropout set to 0:

| Model | MAE (dB) | R² | Train time per fold (s) |
|---|---|---|---|
| XGBoost | 0.145 ± 0.009 | 0.9937 | 0.84 |
| Transformer | 0.159 ± 0.013 | 0.9923 | 157.2 |
| CatBoost | 0.165 ± 0.007 | 0.9908 | 1.99 |
| Random Forest | 0.196 ± 0.023 | 0.9800 | 2.02 |
| LSTM | 0.263 ± 0.051 | 0.9700 | 68.4 |
| 1D U-Net | 0.285 ± 0.089 | 0.9599 | 23.1 |
| Decision Tree | 0.305 ± 0.022 | 0.9406 | 0.02 |

The U-Net reached only R² = 0.79 in an earlier version of this config because
dropout at the end of each convolution block fed the next block's batch
normalization (a train/inference variance shift). Setting `dropout: 0.0` fixed
it in both configs.

### Reading the results

On these small, smooth, parametric datasets the best models sit within a few
hundredths of a dB of the noise floor. XGBoost matches the best deep models on
Antenna I at a fraction of their training cost. On Antenna II the top three tie
on the noisy test targets; measured against the noise-free circuit response
the LSTM has the lowest error. All of this is a property of the synthetic
surrogate and should be re-checked on measured or EM-simulated data.

## Adding a new model

1. Implement `BaseRegressor` (`fit`, `predict`; optionally override
   `get_model_size_bytes` and `get_param_count`), see
   `src/rabench/models/base.py`.
2. Classical or other point-wise models: add the class to
   `CLASSICAL_MODEL_REGISTRY` in `src/rabench/models/classical/sklearn_models.py`
   and reference its key under `classical_models:` in a config.
3. Deep models: add a module under `src/rabench/models/deep/` and register it
   in `SEQ_DEEP_MODEL_REGISTRY` (sequence view, Antenna I) or
   `FLAT_DEEP_MODEL_REGISTRY` (flat view, Antenna II) in
   `scripts/run_benchmark.py`. Then add an entry under `deep_models:` in the
   config.

## Adding a new metric

Add it to `regression_metrics` (and `per_channel_metrics` if it should be
reported per channel) in `src/rabench/benchmarking/metrics.py`, surface it in
`BenchmarkRow` and `_record` in `runner.py`, and in the `.agg()` call in
`report.py::summarize`.

## Swapping in real data

- **Antenna I:** replace the files in the config's `data_dir` with real ones in
  the same layout: one Excel file per antenna with a `frequency_ghz` column plus
  six `s11_<param_name>` columns, and a `params.csv` listing each antenna's six
  parameter values (schema in the `antenna1_generator.py` docstring).
- **Antenna II:** add or replace `antenna2_sweep_*.xlsx` files with the columns
  `frequency_ghz`, the three geometric parameters, and `s11_db`.
  `Antenna2Dataset` loads every file that matches the pattern.

The MIMO diversity metrics (DG, TARC, MEG, CCL) and the anomaly-detection part
of the base paper are not implemented; see `PROJECT_CONTEXT.md`.

## License and academic use

This is a student project. It re-implements architectures described in Das et
al. 2026 for comparison; it does not contain that paper's code, hardware, or
measured data. Cite the original paper when referring to the U-Net or
deep-ensemble architectures or to the RA-sensing application.
