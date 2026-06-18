from dataclasses import dataclass

import h5py
import numpy as np

from data.collector import SimulationDataCollector


@dataclass
class _FakeResult:
    t: np.ndarray
    pressures: np.ndarray
    flows: np.ndarray
    lv_volume: np.ndarray
    aortic_pressure: np.ndarray


def test_latin_hypercube_samples_stay_within_requested_perturbation_ranges():
    collector = SimulationDataCollector(seed=7)

    samples = collector._latin_hypercube(16)

    for params in samples:
        assert 0.8 * collector.base_heart_rate <= params["heart_rate"]
        assert params["heart_rate"] <= 1.2 * collector.base_heart_rate
        assert 0.7 <= params["compliance_scale"] <= 1.3
        assert 0.6 * collector.base_peripheral_resistance <= params[
            "peripheral_resistance"
        ]
        assert params[
            "peripheral_resistance"
        ] <= 1.4 * collector.base_peripheral_resistance


def test_collect_writes_requested_hdf5_layout(tmp_path):
    collector = SimulationDataCollector(duration_seconds=0.01, dt=0.01, seed=3)

    def fake_run_sample(params):
        return _FakeResult(
            t=np.array([0.0, 0.01]),
            pressures=np.ones((2, 3)),
            flows=np.ones((2, 2)),
            lv_volume=np.array([120.0, 118.0]),
            aortic_pressure=np.array([90.0, 91.0]),
        )

    collector._run_sample = fake_run_sample
    output_path = tmp_path / "trajectories.h5"

    collector.collect(2, output_path)

    with h5py.File(output_path, "r") as h5:
        assert "runs" in h5
        assert set(h5["runs"].keys()) == {"00000", "00001"}

        run = h5["runs"]["00000"]
        assert "params" in run
        assert "heart_rate" in run["params"].attrs
        assert run["t"].shape == (2,)
        assert run["pressures"].shape == (2, 3)
        assert run["flows"].shape == (2, 2)
        assert run["lv_volume"].shape == (2,)
        assert run["aortic_pressure"].shape == (2,)
