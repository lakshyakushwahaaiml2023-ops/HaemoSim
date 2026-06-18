import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc

try:
    from haemosim.core.heart import HeartModel
    from haemosim.core.simulator import CirculatorySimulator
    from haemosim.core.vessel_network import default_systemic_tree
except ImportError:
    from core.heart import HeartModel
    from core.simulator import CirculatorySimulator
    from core.vessel_network import default_systemic_tree


class SimulationDataCollector:
    """
    Generate perturbed circulatory simulations and store trajectories in HDF5.
    """

    def __init__(
        self,
        duration_seconds=10.0,
        dt=0.001,
        seed=None,
        base_heart_rate=75.0,
        base_compliance=1.0,
        base_peripheral_resistance=8.0,
    ):
        self.duration_seconds = float(duration_seconds)
        self.dt = float(dt)
        self.seed = seed
        self.base_heart_rate = float(base_heart_rate)
        self.base_compliance = float(base_compliance)
        self.base_peripheral_resistance = float(base_peripheral_resistance)

        if self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.base_heart_rate <= 0.0:
            raise ValueError("base_heart_rate must be positive")
        if self.base_compliance <= 0.0:
            raise ValueError("base_compliance must be positive")
        if self.base_peripheral_resistance <= 0.0:
            raise ValueError("base_peripheral_resistance must be positive")

    def collect(self, n_samples, output_path):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        samples = self._latin_hypercube(int(n_samples))

        with h5py.File(output_path, "w") as h5:
            runs = h5.create_group("runs")
            h5.attrs["n_samples"] = int(n_samples)
            h5.attrs["duration_seconds"] = self.duration_seconds
            h5.attrs["dt"] = self.dt

            for run_id, params in enumerate(samples):
                result = self._run_sample(params)
                group = runs.create_group(f"{run_id:05d}")

                params_group = group.create_group("params")
                for key, value in params.items():
                    params_group.attrs[key] = value

                group.create_dataset("t", data=result.t, compression="gzip")
                group.create_dataset(
                    "pressures", data=result.pressures, compression="gzip"
                )
                group.create_dataset("flows", data=result.flows, compression="gzip")
                group.create_dataset(
                    "lv_volume", data=result.lv_volume, compression="gzip"
                )
                group.create_dataset(
                    "aortic_pressure",
                    data=result.aortic_pressure,
                    compression="gzip",
                )

        return output_path

    def _latin_hypercube(self, n_samples):
        sampler = qmc.LatinHypercube(d=3, seed=self.seed)
        unit_samples = sampler.random(n_samples)

        lower = np.array([0.8, 0.7, 0.6])
        upper = np.array([1.2, 1.3, 1.4])
        scaled = qmc.scale(unit_samples, lower, upper)

        params = []
        for heart_rate_scale, compliance_scale, resistance_scale in scaled:
            params.append(
                {
                    "heart_rate": self.base_heart_rate * heart_rate_scale,
                    "heart_rate_scale": heart_rate_scale,
                    "compliance_scale": compliance_scale,
                    "peripheral_resistance": (
                        self.base_peripheral_resistance * resistance_scale
                    ),
                    "peripheral_resistance_scale": resistance_scale,
                }
            )
        return params

    def _run_sample(self, params):
        heart = HeartModel(
            heart_rate=params["heart_rate"],
            C_arterial=1.2 * params["compliance_scale"],
            R_peripheral=params["peripheral_resistance"],
        )
        network = default_systemic_tree()
        self._scale_network_compliance(network, params["compliance_scale"])

        simulator = CirculatorySimulator(
            heart=heart,
            vessel_network=network,
            terminal_resistance=params["peripheral_resistance"],
        )
        return simulator.run(self.duration_seconds, dt=self.dt)

    @staticmethod
    def _scale_network_compliance(network, compliance_scale):
        for children in network.graph.values():
            for segment in children.values():
                segment["wall_compliance"] *= compliance_scale


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Collect perturbed HaemoSim circulatory trajectories."
    )
    parser.add_argument("--n", type=int, required=True, help="Number of samples.")
    parser.add_argument(
        "--out", required=True, help="Output HDF5 path, e.g. data/trajectories.h5."
    )
    parser.add_argument("--dt", type=float, default=0.001, help="Sample timestep.")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Simulation duration in seconds.",
    )
    parser.add_argument("--seed", type=int, default=None, help="LHS random seed.")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    collector = SimulationDataCollector(
        duration_seconds=args.duration,
        dt=args.dt,
        seed=args.seed,
    )
    output_path = collector.collect(args.n, args.out)
    print(f"saved {args.n} runs to {output_path}")


if __name__ == "__main__":
    main()
