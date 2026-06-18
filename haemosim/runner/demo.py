import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from haemosim.runner.hybrid_runner import HybridSimulator
except ImportError:
    from runner.hybrid_runner import HybridSimulator


def _build_parser():
    parser = argparse.ArgumentParser(description="Run a HaemoSim hybrid demo.")
    parser.add_argument(
        "--model",
        default="haemosim/surrogate/checkpoints/best_model.pt",
        help="Path to a trained surrogate checkpoint.",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--trust-threshold", type=float, default=5.0)
    parser.add_argument(
        "--out",
        default="haemosim/runner/hybrid_demo.png",
        help="Output plot path.",
    )
    parser.add_argument("--device", default=None)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    runner = HybridSimulator(
        trust_threshold=args.trust_threshold,
        device=args.device,
    )
    result, frame_source = runner.run(args.duration, args.dt, args.model)

    surrogate_mask = frame_source == "surrogate"
    physics_mask = frame_source == "physics"
    print(f"frames: {len(frame_source)}")
    print(f"physics frames: {int(np.sum(physics_mask))}")
    print(f"surrogate frames: {int(np.sum(surrogate_mask))}")
    print(f"surrogate fraction: {100.0 * np.mean(surrogate_mask):.1f}%")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result.t, result.aortic_pressure, color="0.35", linewidth=1.0)
    ax.scatter(
        result.t[physics_mask],
        result.aortic_pressure[physics_mask],
        s=4,
        color="tab:blue",
        label="physics",
    )
    ax.scatter(
        result.t[surrogate_mask],
        result.aortic_pressure[surrogate_mask],
        s=5,
        color="tab:orange",
        label="surrogate",
    )
    ax.set_title("Hybrid aortic pressure")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Aortic pressure")
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
