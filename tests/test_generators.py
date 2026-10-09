import tempfile
from pathlib import Path

from rabench.data_generation import antenna1_generator, antenna2_generator
from rabench.data.datasets import Antenna1Dataset, Antenna2Dataset


def test_antenna1_generation_and_loading():
    with tempfile.TemporaryDirectory() as tmp:
        meta = antenna1_generator.generate_dataset(tmp, n_samples=3, seed=1)
        assert len(meta) == 3

        ds = Antenna1Dataset(Path(tmp))
        assert ds.n_samples == 3

        X, y, feat_names, target_names = ds.to_flat()
        assert X.shape[0] == y.shape[0]
        assert len(target_names) == 6

        X_seq, y_seq, channel_names = ds.to_sequences()
        assert X_seq.shape[0] == 3
        assert y_seq.shape[1] == 6

        folds = list(ds.loocv_splits())
        assert len(folds) == 3


def test_antenna2_generation_and_loading():
    with tempfile.TemporaryDirectory() as tmp:
        files = antenna2_generator.generate_dataset(tmp, n_sweep_points=4, seed=2)
        assert len(files) == 3  # one per geometric param

        ds = Antenna2Dataset(Path(tmp))
        X, y, feat_names, target_name = ds.to_flat(engineered_features=True)
        assert X.shape[0] == y.shape[0]
        assert "frequency_ghz_sq" in feat_names
