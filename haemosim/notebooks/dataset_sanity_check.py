import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

try:
    from haemosim.data.dataset import HaemodynamicsDataset
except ImportError:
    from data.dataset import HaemodynamicsDataset


def _build_parser():
    parser = argparse.ArgumentParser(description="Quick HaemoSim dataset check.")
    parser.add_argument("hdf5_path", help="Path to a trajectories HDF5 file.")
    parser.add_argument("--k", type=int, default=10, help="Prediction horizon.")
    parser.add_argument(
        "--out",
        default="notebooks/sample_trajectory.png",
        help="Path for the sample trajectory plot.",
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    dataset = HaemodynamicsDataset(args.hdf5_path, k=args.k)
    train, val, test = dataset.train_val_test_split()
    x, y = dataset[0]

    print(f"dataset examples: {len(dataset)}")
    print(f"train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"input shape: {tuple(x.shape)}")
    print(f"target shape: {tuple(y.shape)}")
    print(f"feature mean/std: {dataset.feature_mean.shape}/{dataset.feature_std.shape}")

    with h5py.File(args.hdf5_path, "r") as h5:
        run_id = sorted(h5["runs"].keys())[0]
        run = h5["runs"][run_id]
        t = run["t"][:]
        pressures = run["pressures"][:, 0]
        lv_volume = run["lv_volume"][:]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(t, pressures)
    axes[0].set_ylabel("Pressure")
    axes[0].set_title(f"Sample run {run_id}")
    axes[1].plot(t, lv_volume)
    axes[1].set_ylabel("LV volume")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
