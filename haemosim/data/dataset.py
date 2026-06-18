import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class HaemodynamicsDataset(Dataset):
    """
    PyTorch dataset for k-step haemodynamic state prediction.

    Inputs are [pressures_t, flows_t, params_vector] normalized to zero mean
    and unit variance. Targets are the unnormalized future state
    [pressures_{t+k}, flows_{t+k}].
    """

    DEFAULT_PARAM_KEYS = (
        "heart_rate",
        "compliance_scale",
        "peripheral_resistance",
    )

    def __init__(
        self,
        hdf5_path,
        k=10,
        param_keys=None,
        indices=None,
        feature_mean=None,
        feature_std=None,
        normalize=True,
        stride=1,
        max_samples=None,
        dtype=torch.float32,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.k = int(k)
        self.param_keys = tuple(param_keys or self.DEFAULT_PARAM_KEYS)
        self.normalize = normalize
        self.stride = int(stride)
        self.max_samples = max_samples if max_samples is None else int(max_samples)
        self.dtype = dtype

        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if not self.hdf5_path.exists():
            raise FileNotFoundError(self.hdf5_path)

        features, targets, metadata = self._load_examples()

        if indices is not None:
            indices = np.asarray(indices, dtype=int)
            features = features[indices]
            targets = targets[indices]
            metadata = [metadata[i] for i in indices]

        self.raw_features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.metadata = metadata

        if feature_mean is None:
            feature_mean = self.raw_features.mean(axis=0)
        if feature_std is None:
            feature_std = self.raw_features.std(axis=0)

        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)

        if self.normalize:
            self.features = (self.raw_features - self.feature_mean) / self.feature_std
        else:
            self.features = self.raw_features

        self.input_dim = self.features.shape[1]
        self.target_dim = self.targets.shape[1]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return (
            torch.as_tensor(self.features[index], dtype=self.dtype),
            torch.as_tensor(self.targets[index], dtype=self.dtype),
        )

    def train_val_test_split(self, train=0.8, val=0.1, test=0.1, seed=42):
        """
        Return deterministic 80/10/10-style Subsets by default.

        The returned subsets share this dataset instance, so normalization
        statistics remain available as dataset.feature_mean and dataset.feature_std.
        """
        total = train + val + test
        if not np.isclose(total, 1.0):
            raise ValueError("train, val, and test fractions must sum to 1.0")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self))
        n_train = int(round(train * len(self)))
        n_val = int(round(val * len(self)))
        n_train = min(n_train, len(self))
        n_val = min(n_val, len(self) - n_train)

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        return (
            Subset(self, train_indices.tolist()),
            Subset(self, val_indices.tolist()),
            Subset(self, test_indices.tolist()),
        )

    @classmethod
    def create_splits(
        cls,
        hdf5_path,
        k=10,
        train=0.8,
        val=0.1,
        test=0.1,
        seed=42,
        param_keys=None,
        stride=1,
        max_samples=None,
    ):
        """
        Build train/val/test dataset instances with normalization fit on train.
        """
        base = cls(
            hdf5_path,
            k=k,
            param_keys=param_keys,
            normalize=False,
            stride=stride,
            max_samples=max_samples,
        )
        total = train + val + test
        if not np.isclose(total, 1.0):
            raise ValueError("train, val, and test fractions must sum to 1.0")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(base))
        n_train = int(round(train * len(base)))
        n_val = int(round(val * len(base)))
        n_train = min(n_train, len(base))
        n_val = min(n_val, len(base) - n_train)

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        mean = base.raw_features[train_indices].mean(axis=0)
        std = base.raw_features[train_indices].std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)

        kwargs = {
            "k": k,
            "param_keys": param_keys,
            "feature_mean": mean,
            "feature_std": std,
        }
        return (
            base._subset_dataset(train_indices, **kwargs),
            base._subset_dataset(val_indices, **kwargs),
            base._subset_dataset(test_indices, **kwargs),
        )

    def _subset_dataset(self, indices, feature_mean, feature_std, **_):
        subset = object.__new__(self.__class__)
        indices = np.asarray(indices, dtype=int)
        subset.hdf5_path = self.hdf5_path
        subset.k = self.k
        subset.param_keys = self.param_keys
        subset.normalize = True
        subset.stride = self.stride
        subset.max_samples = self.max_samples
        subset.dtype = self.dtype
        subset.raw_features = self.raw_features[indices].astype(np.float32)
        subset.targets = self.targets[indices].astype(np.float32)
        subset.metadata = [self.metadata[i] for i in indices]
        subset.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        subset.feature_std = np.asarray(feature_std, dtype=np.float32)
        subset.feature_std = np.where(subset.feature_std < 1e-8, 1.0, subset.feature_std)
        subset.features = (subset.raw_features - subset.feature_mean) / subset.feature_std
        subset.input_dim = subset.features.shape[1]
        subset.target_dim = subset.targets.shape[1]
        return subset

    def _load_examples(self):
        features = []
        targets = []
        metadata = []

        with h5py.File(self.hdf5_path, "r") as h5:
            runs = h5["runs"]
            for run_id in sorted(runs.keys()):
                run = runs[run_id]
                pressures = np.asarray(run["pressures"], dtype=np.float32)
                flows = np.asarray(run["flows"], dtype=np.float32)
                params = self._params_vector(run["params"].attrs)

                if pressures.shape[0] != flows.shape[0]:
                    raise ValueError(f"run {run_id} has mismatched time dimensions")
                if pressures.shape[0] <= self.k:
                    continue

                for t_index in range(0, pressures.shape[0] - self.k, self.stride):
                    current_state = np.concatenate(
                        [pressures[t_index], flows[t_index], params]
                    )
                    future_state = np.concatenate(
                        [pressures[t_index + self.k], flows[t_index + self.k]]
                    )
                    features.append(current_state)
                    targets.append(future_state)
                    metadata.append({"run_id": run_id, "t_index": t_index})
                    if self.max_samples is not None and len(features) >= self.max_samples:
                        return np.vstack(features), np.vstack(targets), metadata

        if not features:
            raise ValueError("no valid training windows found in HDF5 file")

        return np.vstack(features), np.vstack(targets), metadata

    def _params_vector(self, attrs):
        missing = [key for key in self.param_keys if key not in attrs]
        if missing:
            raise KeyError(f"missing parameter attrs: {missing}")
        return np.asarray([attrs[key] for key in self.param_keys], dtype=np.float32)


def _build_parser():
    parser = argparse.ArgumentParser(description="Inspect a HaemoSim HDF5 dataset.")
    parser.add_argument("hdf5_path", help="Path to trajectories.h5")
    parser.add_argument("--k", type=int, default=10, help="Prediction horizon.")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    dataset = HaemodynamicsDataset(args.hdf5_path, k=args.k)
    x, y = dataset[0]
    print(f"examples: {len(dataset)}")
    print(f"input shape: {tuple(x.shape)}")
    print(f"target shape: {tuple(y.shape)}")
    print(f"feature mean shape: {dataset.feature_mean.shape}")
    print(f"feature std shape: {dataset.feature_std.shape}")


if __name__ == "__main__":
    main()
