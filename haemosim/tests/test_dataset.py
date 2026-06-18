import h5py
import numpy as np

from data.dataset import HaemodynamicsDataset


def _write_dataset_fixture(path):
    with h5py.File(path, "w") as h5:
        runs = h5.create_group("runs")
        for run_id in range(2):
            run = runs.create_group(f"{run_id:05d}")
            params = run.create_group("params")
            params.attrs["heart_rate"] = 75.0 + run_id
            params.attrs["compliance_scale"] = 1.0
            params.attrs["peripheral_resistance"] = 8.0
            run.create_dataset("t", data=np.arange(6, dtype=np.float32))
            run.create_dataset(
                "pressures",
                data=np.arange(18, dtype=np.float32).reshape(6, 3) + run_id,
            )
            run.create_dataset(
                "flows",
                data=np.arange(12, dtype=np.float32).reshape(6, 2) + run_id,
            )
            run.create_dataset("lv_volume", data=np.linspace(120.0, 100.0, 6))


def test_dataset_builds_k_step_examples_and_normalizes_inputs(tmp_path):
    path = tmp_path / "trajectories.h5"
    _write_dataset_fixture(path)

    dataset = HaemodynamicsDataset(path, k=2)
    x, y = dataset[0]

    assert len(dataset) == 8
    assert x.shape == (8,)
    assert y.shape == (5,)
    assert dataset.feature_mean.shape == (8,)
    assert dataset.feature_std.shape == (8,)
    assert np.allclose(dataset.features.mean(axis=0), 0.0, atol=1e-6)


def test_dataset_train_val_test_split_uses_default_80_10_10(tmp_path):
    path = tmp_path / "trajectories.h5"
    _write_dataset_fixture(path)
    dataset = HaemodynamicsDataset(path, k=1)

    train, val, test = dataset.train_val_test_split(seed=1)

    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1


def test_create_splits_fit_normalisation_on_train_only(tmp_path):
    path = tmp_path / "trajectories.h5"
    _write_dataset_fixture(path)

    train, val, test = HaemodynamicsDataset.create_splits(path, k=1, seed=1)

    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1
    assert train.feature_mean.shape == val.feature_mean.shape
    assert np.allclose(train.feature_mean, val.feature_mean)


def test_dataset_stride_and_max_samples_reduce_loaded_windows(tmp_path):
    path = tmp_path / "trajectories.h5"
    _write_dataset_fixture(path)

    dataset = HaemodynamicsDataset(path, k=1, stride=2, max_samples=3)

    assert len(dataset) == 3
