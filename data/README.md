# data/

`raw/antenna1/` and `raw/antenna2/` contain **synthetic surrogate data**
generated from the paper's published equivalent-circuit values — not
real EM-simulated or measured antenna data. See
`src/rabench/data_generation/circuit_models.py` for the full explanation
and caveats, and the main `README.md` → "Swapping in real data later" for
how to replace this with real data once available.

The `.xlsx` / `.csv` / `.json` files in this folder are regenerable and
gitignored. To (re)generate them:

```bash
python scripts/generate_dataset.py --task all
```

A sample generated dataset is included in this download so you can
inspect the pipeline's output immediately without running generation
first.
