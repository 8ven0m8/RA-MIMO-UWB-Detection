# PROJECT_CONTEXT.md

Machine-readable-ish context for an LLM (Claude Code or otherwise)
picking up this repo cold. Read this before editing anything. If you are
a human, `README.md` is the friendlier entry point; this file trades
friendliness for density.

## 1. What problem this repo solves

Source paper: Das et al., *"Deep learning enabled heterogeneous MIMO
antenna sensor for detection of rheumatoid arthritis,"* Measurement 265
(2026) 120437. PDF summary is not included in-repo; ask the user if you
need the full paper text.

Student project built on top of that paper (see `Review_1` deck the user
shared during scoping — not included in repo, but its objectives shaped
this architecture):

> Benchmark the paper's two deep learning models (UNet1D for Antenna I,
> deep ensemble of MLPs for Antenna II) against four classical ML
> regressors (Random Forest, XGBoost, CatBoost, Decision Tree) on
> **identical S11 prediction tasks**, using MAE/MSE/R² for accuracy AND
> training time / inference latency / model size for efficiency.

This repo is the **benchmarking harness** for that comparison. It is
built in phases; only Phase 1 (below) is implemented.

## 2. Phased scope

| Phase | Status | Content |
|---|---|---|
| **1. S11 regression benchmarking** | ✅ implemented (this repo) | Classical ML vs UNet1D (Antenna I) and vs deep ensemble MLP (Antenna II), on synthetic circuit-surrogate data |
| 2. Anomaly detection / SAR imaging benchmarking | ❌ not implemented | Paper's U-Net-based SAR anomaly detection on clay-phantom S-parameters; project's Objective #4 (multi-phantom generalization) |
| 3. MIMO diversity metrics | ❌ not implemented | Diversity Gain, TARC, MEG, CCL — project's Objective #3 |

If asked to extend this repo toward Phase 2 or 3, these do not yet exist
anywhere in the codebase — you would be adding new top-level modules
(suggested: `src/rabench/anomaly_detection/`, `src/rabench/mimo_metrics/`),
not editing existing ones. Do not assume partial implementations exist.

## 3. The central design decision: "flat" vs "sequence" data views

This is the single most important thing to understand before touching
`src/rabench/data/datasets.py` or any model.

- **Antenna I** in the paper is modelled by a UNet1D that convolves over
  the frequency axis and outputs 6 channels simultaneously (one channel
  per named geometric parameter's induced S11 curve, see Table 2 in the
  paper). That is inherently a **sequence** task.
- Classical ML models (RF/XGBoost/CatBoost/DecisionTree) can't consume
  sequences meaningfully as "one sample = one curve" — they need
  **point-wise rows**: one row per (frequency, geometry) combination.
- To make classical-vs-UNet1D a fair, identical-task comparison, this
  repo generates data once and exposes it two ways from the same
  underlying files:
  - `Antenna1Dataset.to_flat()` → `(X, y)` numpy arrays, one row per
    frequency point, `X = [frequency, *6 geometric params]`,
    `y = [6 S11 channels]`. Used by classical models.
  - `Antenna1Dataset.to_sequences()` → `(X_seq, y_seq)`,
    shape `(n_samples, n_channels, n_freq_points)`. Used by UNet1D. Row
    `i` of the flat view and sample `i`'s frequency axis in the sequence
    view represent the exact same underlying data, just reshaped.
  - `Antenna1Dataset.loocv_splits()` yields sequence-shaped LOOCV folds
    directly (paper's own evaluation protocol for Antenna I).
- **Antenna II** in the paper is point-wise by construction (a plain MLP
  ensemble, no conv/sequence structure) — so `Antenna2Dataset` only has
  a `to_flat()` view. Nothing sequence-shaped exists or is needed here.

**Implication for the benchmarking runner
(`src/rabench/benchmarking/runner.py`)**: `run_antenna1_benchmark` has to
special-case UNet1D's input (it takes a `unet_factory(in_channels,
out_channels)` callable and calls `dataset.loocv_splits()` directly,
separately from the classical-model loop which uses `to_flat()` +
manual fold masking). `run_antenna2_benchmark` has no such special case
— the deep ensemble MLP is just another flat-view factory alongside the
classical ones. **If you add a new sequence-based deep model, it needs
the same special-casing UNet1D gets; if it's point-wise, it slots into
the classical-style factory dict with zero runner changes.**

## 4. Data schemas (exact column names matter)

### Antenna I (`data/raw/antenna1/`)
- `antenna1_sample_XX.xlsx` (one per LOOCV fold/sample): columns
  `frequency_ghz`, then `s11_gndslot_width`, `s11_feedline_width`,
  `s11_patch_strip_length`, `s11_patch_strip_width`,
  `s11_ellipse_cut_radius`, `s11_radius_of_semicircle_cut` (all dB).
- `params.csv`: one row per sample — `sample_idx`, `file`, and the 6 raw
  parameter values (mm) for that sample: `gndslot_width`,
  `feedline_width`, `patch_strip_length`, `patch_strip_width`,
  `ellipse_cut_radius`, `radius_of_semicircle_cut`.
- `manifest.json`: generation metadata (seed, freq range, note).
- Canonical parameter name list lives in exactly one place:
  `rabench.data_generation.antenna1_generator.PARAM_NAMES`. Everything
  else (dataset loader's `channel_names`, per-channel metrics, plots)
  derives from it — **do not hardcode the parameter list a second time
  anywhere**; import it.

### Antenna II (`data/raw/antenna2/`)
- `antenna2_sweep_<param>.xlsx` (one per geometric parameter swept):
  columns `frequency_ghz`, `feedline_width`, `gnd_slot_width`,
  `gnd_patch_radius`, `s11_db`. `Antenna2Dataset` glob-loads
  `antenna2_sweep_*.xlsx` and concatenates all of them — you can add
  more sweep files (e.g. a 2-parameter joint sweep) without touching
  code, as long as columns match.
- Canonical parameter name list:
  `rabench.data_generation.antenna2_generator.GEOM_PARAM_NAMES`.
- `to_flat(engineered_features=True)` (default) adds `frequency_ghz_sq`
  and `freq_x_<param>` interaction columns, mirroring the paper's
  Section 5 feature engineering description. Pass `False` to get raw
  features only.

## 5. Model registry / factory pattern

- `rabench.models.classical.sklearn_models.CLASSICAL_MODEL_REGISTRY` is
  a `dict[str, Callable[..., BaseRegressor]]` — the single source of
  truth for "what classical models exist and how to construct them".
  `scripts/run_benchmark.py::build_classical_factories` reads a config's
  `classical_models:` YAML block and instantiates from this registry by
  key. **Adding a classical model = add one entry here + one registry
  key**, no runner/script changes.
- Deep models don't have a shared registry (there are only two, and they
  have fundamentally different call signatures — `UNet1DModel` needs
  `in_channels`/`out_channels` inferred from data shape at benchmark
  time, `DeepEnsembleModel` doesn't). They're wired directly in
  `scripts/run_benchmark.py`'s `main()` via small closures
  (`unet_factory`, `deep_ensemble_factory`). If a third deep model
  architecture is added, prefer this same closure pattern over trying to
  force a generic registry — the two existing models already don't fit
  one shape, a third almost certainly won't either.

## 6. Every model implements `BaseRegressor` (`src/rabench/models/base.py`)

```python
fit(X_train, y_train) -> None
predict(X) -> np.ndarray
get_model_size_bytes() -> int | None   # default: pickle-and-measure; deep models override with torch state_dict size
get_param_count() -> int | None        # default: None; deep models override with sum(p.numel())
name: str      # shown in results tables/plots
family: str    # "classical" | "deep" -- used for plot coloring in report.py
```
The benchmarking runner (`runner.py::_record`) calls all four of these
uniformly for every model, regardless of family. This is what makes
`results/*/summary.csv` directly comparable across classical and deep
rows without special-casing in the reporting layer.

## 7. The synthetic data is a surrogate — know the caveats before extending it

`src/rabench/data_generation/circuit_models.py`'s module docstring has
the full explanation, but the short version: the S11 curves come from
literally plugging the paper's own published lumped-element component
values (R/L/C) into a cascaded-RLC impedance model, then perturbing
specific components when a geometric parameter changes (via a hand-picked
`PARAM_TO_COMPONENT` sensitivity map in each generator — heuristic, not
derived from EM physics). This means:

- Absolute S11 dB values and resonance frequencies are plausible and
  UWB-like, but **not** a reconstruction of the actual fabricated
  antennas.
- The `PARAM_TO_COMPONENT` mappings (which parameter perturbs which
  L/C/R, and how strongly) are the generators' single biggest source of
  arbitrariness. If asked to make the synthetic data "more realistic",
  this is the place to focus — e.g. replacing it with an actual openEMS/
  Sonnet EM simulation sweep (Option B, discussed with the user, not
  implemented here) rather than tuning the heuristic further.
- Baseline component values ARE faithful to the paper (Antenna I:
  Fig. 2e; Antenna II: Fig. 7b) — those are not something to "fix" or
  guess differently without the user's direction.

## 8. Reproducibility

`rabench.utils.seed.set_global_seed(seed)` seeds `random`, `numpy`, and
`torch` in one call. `scripts/run_benchmark.py` calls it once at startup
using the config's `seed:` key. Each classical model also takes its own
`random_state` param (set in the YAML per-model block, currently all
default to `42`). Deep models seed internally per ensemble member
(`base_seed + member_index` in `DeepEnsembleModel`; `seed` constructor
arg in `UNet1DModel`, one model per LOOCV fold).

## 9. Known behaviour worth knowing before "fixing" it

- **UNet1D underperforms classical models badly** on the default
  8-sample synthetic Antenna I dataset (R² ≈ 0.21 vs ≈ 0.91-0.95 for
  classical models). This is expected and arguably the interesting
  result for the project's thesis, not a bug: LOOCV with a from-scratch
  CNN over only ~7 training curves per fold is a genuinely hard regime.
  Increasing `data_generation.n_samples` in
  `configs/antenna1_benchmark.yaml` (and regenerating data) will likely
  narrow this gap — that's a legitimate experiment to run, not something
  to silently "correct" in the code.
- **DeepEnsembleMLP is ~1-2 orders of magnitude slower to train** than
  any classical model on Antenna II (tens of seconds vs sub-second),
  consistent with training 5 full MLPs with early stopping vs a single
  sklearn `.fit()` call. This is real, not a measurement artifact.
- **RandomForest has by far the largest model size** of any model in
  both tasks (tens of MB, from pickling 200 full decision trees) — this
  is expected and is itself a useful data point for the efficiency
  comparison, not something to "fix" by changing hyperparameters unless
  the user asks for a size-constrained comparison.

## 10. File-by-file map (for fast navigation)

```
src/rabench/data_generation/circuit_models.py   physics core: RLCBlock, CascadedCircuit, baseline circuit builders
src/rabench/data_generation/antenna1_generator.py   Antenna I synthetic dataset (6-param, 6-channel, LOOCV-fold files)
src/rabench/data_generation/antenna2_generator.py   Antenna II synthetic dataset (3-param, 1-target, sweep files)
src/rabench/data/datasets.py                    Antenna1Dataset (flat+sequence+LOOCV), Antenna2Dataset (flat)
src/rabench/models/base.py                      BaseRegressor ABC
src/rabench/models/classical/sklearn_models.py  RF/DT/XGBoost/CatBoost wrappers + CLASSICAL_MODEL_REGISTRY
src/rabench/models/deep/unet1d.py               UNet1D nn.Module + UNet1DModel (BaseRegressor wrapper, sequence I/O)
src/rabench/models/deep/deep_ensemble.py        _MLP nn.Module + DeepEnsembleModel (BaseRegressor wrapper, flat I/O)
src/rabench/benchmarking/metrics.py             regression_metrics, per_channel_metrics
src/rabench/benchmarking/runner.py              run_antenna1_benchmark (LOOCV), run_antenna2_benchmark (train/test split)
src/rabench/benchmarking/report.py              summarize(), plot_accuracy_comparison(), plot_efficiency_comparison()
src/rabench/utils/seed.py                       set_global_seed()
scripts/generate_dataset.py                     CLI wrapper around both generators
scripts/run_benchmark.py                        CLI: config YAML -> dataset -> models -> results/*.csv + *.png
scripts/run_all.py                              generate both + benchmark both, in sequence
configs/antenna1_benchmark.yaml                 Antenna I experiment config (models, hyperparams, paths)
configs/antenna2_benchmark.yaml                 Antenna II experiment config
tests/test_circuit_models.py                    circuit math smoke tests
tests/test_generators.py                        dataset generation + loading smoke tests
```

## 11. If you're asked to "add MIMO diversity metrics" or "add anomaly detection"

Those are Phase 2/3 (section 2 above) and touch a completely different
part of the paper (Section 7-9: MIMO isolation, SAR back-projection,
U-Net anomaly segmentation on clay-phantom S-parameters) — not an
extension of the S11-regression code in this repo. Before writing code,
re-read the paper's Sections 7-9 (or ask the user for them) and propose
a new top-level module rather than bolting this onto
`rabench.data_generation` or `rabench.models`, since the input data
shape (S-parameters → SAR images → segmentation masks) is fundamentally
different from the flat/sequence S11-regression views this repo is built
around.
