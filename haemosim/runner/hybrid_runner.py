import time
from pathlib import Path

import numpy as np
import torch

try:
    from haemosim.core.simulator import CirculatorySimulator, SimulationResult
    from haemosim.core.vessel_network import default_systemic_tree
    from haemosim.surrogate.fast_model import FastStateSurrogate
    from haemosim.surrogate.gnn_model import GNNSurrogate
except ImportError:
    from core.simulator import CirculatorySimulator, SimulationResult
    from core.vessel_network import default_systemic_tree
    from surrogate.fast_model import FastStateSurrogate
    from surrogate.gnn_model import GNNSurrogate


class HybridSimulator:
    """
    Hybrid physics/surrogate runner for short-window surrogate acceleration.
    """

    def __init__(
        self,
        physics=None,
        trust_threshold=5.0,
        device=None,
        default_k=10,
    ):
        self.physics = physics if physics is not None else CirculatorySimulator()
        self.trust_threshold = float(trust_threshold)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.default_k = int(default_k)
        self.frame_source = []

    def run(self, duration, dt, surrogate_model_path):
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        model, checkpoint = self._load_surrogate(surrogate_model_path)
        model.eval()
        k = int(checkpoint.get("k", self.default_k))
        t_eval = np.arange(0.0, duration + 0.5 * dt, dt)
        state = self.physics._initial_state()

        states = [state.copy()]
        frame_source = ["physics"]
        current_index = 0

        while current_index < len(t_eval) - 1:
            current_t = t_eval[current_index]
            current_result = self.physics._build_result(
                np.asarray([current_t]), np.asarray([state])
            )

            prediction = self._predict_surrogate(model, checkpoint, current_result)
            can_skip = prediction is not None and self._trusted(
                prediction,
                current_result.pressures[0],
                current_result.flows[0],
            )

            if can_skip:
                steps = min(k, len(t_eval) - current_index - 1)
                next_state = self._state_from_prediction(state, prediction)
                for step in range(1, steps + 1):
                    alpha = step / steps
                    states.append((1.0 - alpha) * state + alpha * next_state)
                    frame_source.append("surrogate")
                state = states[-1].copy()
                current_index += steps
            else:
                state = self._physics_step(state, current_t, dt)
                states.append(state.copy())
                frame_source.append("physics")
                current_index += 1

        states = np.asarray(states)
        result = self.physics._build_result(t_eval[: len(states)], states)
        self.frame_source = np.asarray(frame_source, dtype=object)
        return result, self.frame_source

    def benchmark(self, duration, dt, surrogate_model_path):
        start = time.perf_counter()
        hybrid_result, frame_source = self.run(duration, dt, surrogate_model_path)
        hybrid_time = time.perf_counter() - start

        full_physics = CirculatorySimulator()
        start = time.perf_counter()
        physics_result = full_physics.run(duration, dt)
        physics_time = time.perf_counter() - start

        speedup = physics_time / max(hybrid_time, 1e-12)
        surrogate_fraction = np.mean(frame_source == "surrogate")
        print(f"full physics time: {physics_time:.3f} s")
        print(f"hybrid time: {hybrid_time:.3f} s")
        print(f"speedup: {speedup:.2f}x")
        print(f"surrogate frames: {100.0 * surrogate_fraction:.1f}%")
        return {
            "hybrid_result": hybrid_result,
            "physics_result": physics_result,
            "frame_source": frame_source,
            "hybrid_time": hybrid_time,
            "physics_time": physics_time,
            "speedup": speedup,
        }

    def _physics_step(self, state, t, dt):
        from scipy.integrate import solve_ivp

        solution = solve_ivp(
            self.physics._rhs,
            (float(t), float(t + dt)),
            state,
            method="RK45",
            t_eval=[float(t + dt)],
            max_step=float(dt),
            rtol=1e-6,
            atol=1e-8,
        )
        if not solution.success:
            raise RuntimeError(f"physics step failed: {solution.message}")
        return solution.y[:, -1]

    def _load_surrogate(self, surrogate_model_path):
        checkpoint_path = Path(surrogate_model_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        graph = default_systemic_tree().graph
        model_type = checkpoint.get("model", "gnn")
        hidden_dim = int(checkpoint.get("hidden_dim", 128))
        num_layers = int(checkpoint.get("num_layers", 3))
        if model_type == "fast":
            model = FastStateSurrogate(
                graph,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        else:
            model = GNNSurrogate(
                graph,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )

        self._initialize_lazy_modules(model)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device), checkpoint

    def _initialize_lazy_modules(self, model):
        n_nodes = model.num_nodes
        n_edges = model.num_edges
        with torch.no_grad():
            model(
                torch.zeros(1, n_nodes, 3),
                torch.zeros(1, n_edges, 1),
                torch.zeros(1, 3),
            )

    def _predict_surrogate(self, model, checkpoint, current_result):
        feature_mean = checkpoint.get("feature_mean")
        feature_std = checkpoint.get("feature_std")
        if feature_mean is None or feature_std is None:
            return None

        params = self._params_vector(checkpoint)
        raw_features = np.concatenate(
            [current_result.pressures[0], current_result.flows[0], params]
        )
        normalized = (raw_features - feature_mean) / np.where(feature_std < 1e-8, 1.0, feature_std)

        n_nodes = model.num_nodes
        n_edges = model.num_edges
        pressures = torch.as_tensor(
            normalized[:n_nodes], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        flows = torch.as_tensor(
            normalized[n_nodes : n_nodes + n_edges],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        params_tensor = torch.as_tensor(
            normalized[n_nodes + n_edges :],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        incoming = torch.zeros_like(pressures)
        outgoing = torch.zeros_like(pressures)
        edge_index = model.edge_index.to(self.device)
        for edge_id in range(n_edges):
            src = edge_index[0, edge_id]
            dst = edge_index[1, edge_id]
            outgoing[:, src] = outgoing[:, src] + flows[:, edge_id]
            incoming[:, dst] = incoming[:, dst] + flows[:, edge_id]

        x_node = torch.stack([pressures, incoming, outgoing], dim=-1)
        x_edge = flows.unsqueeze(-1)

        with torch.no_grad():
            pressure_pred, flow_pred = model(x_node, x_edge, params_tensor)

        return {
            "pressures": pressure_pred.squeeze(0).detach().cpu().numpy(),
            "flows": flow_pred.squeeze(0).detach().cpu().numpy(),
        }

    def _params_vector(self, checkpoint):
        param_keys = tuple(checkpoint.get("param_keys", ()))
        defaults = {
            "heart_rate": self.physics.heart.heart_rate,
            "compliance_scale": 1.0,
            "peripheral_resistance": self.physics.terminal_resistance,
        }
        return np.asarray([defaults.get(key, 0.0) for key in param_keys], dtype=np.float32)

    def _trusted(self, prediction, pressure, flow):
        predicted_state = np.concatenate([prediction["pressures"], prediction["flows"]])
        current_state = np.concatenate([pressure, flow])
        error = np.sqrt(np.mean((predicted_state - current_state) ** 2))
        return error < self.trust_threshold

    def _state_from_prediction(self, current_state, prediction):
        state = current_state.copy()
        pressures = prediction["pressures"]
        state[1:-1] = [
            pressures[self.physics.node_names.index(end)]
            for _, end in self.physics.segment_names
        ]
        return state
