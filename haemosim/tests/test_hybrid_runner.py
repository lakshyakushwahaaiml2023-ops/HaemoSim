import numpy as np
import torch

from core.vessel_network import default_systemic_tree
from runner.hybrid_runner import HybridSimulator
from surrogate.fast_model import FastStateSurrogate


def _write_fast_checkpoint(path, k=2):
    graph = default_systemic_tree().graph
    model = FastStateSurrogate(graph, hidden_dim=16, num_layers=1)
    with torch.no_grad():
        model(
            torch.zeros(1, model.num_nodes, 3),
            torch.zeros(1, model.num_edges, 1),
            torch.zeros(1, 3),
        )

    feature_dim = model.num_nodes + model.num_edges + 3
    torch.save(
        {
            "model": "fast",
            "model_state_dict": model.state_dict(),
            "hidden_dim": 16,
            "num_layers": 1,
            "k": k,
            "feature_mean": np.zeros(feature_dim, dtype=np.float32),
            "feature_std": np.ones(feature_dim, dtype=np.float32),
            "param_keys": (
                "heart_rate",
                "compliance_scale",
                "peripheral_resistance",
            ),
        },
        path,
    )


def test_hybrid_run_returns_result_and_frame_source(tmp_path):
    checkpoint_path = tmp_path / "fast_model.pt"
    _write_fast_checkpoint(checkpoint_path)
    runner = HybridSimulator(trust_threshold=1e9, device="cpu")

    result, frame_source = runner.run(0.004, 0.001, checkpoint_path)

    assert result.pressures.shape[0] == len(frame_source)
    assert result.flows.shape[0] == len(frame_source)
    assert result.lv_volume.shape == result.aortic_pressure.shape
    assert set(frame_source).issubset({"physics", "surrogate"})


def test_benchmark_returns_speedup_payload(tmp_path):
    checkpoint_path = tmp_path / "fast_model.pt"
    _write_fast_checkpoint(checkpoint_path)
    runner = HybridSimulator(trust_threshold=1e9, device="cpu")

    benchmark = runner.benchmark(0.002, 0.001, checkpoint_path)

    assert "speedup" in benchmark
    assert benchmark["frame_source"].shape[0] == benchmark["hybrid_result"].t.shape[0]
