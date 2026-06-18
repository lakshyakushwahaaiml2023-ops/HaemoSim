from dataclasses import dataclass

import numpy as np

try:
    from .heart import HeartModel
    from .vessel_network import VesselNetwork, default_systemic_tree
except ImportError:
    from heart import HeartModel
    from vessel_network import VesselNetwork, default_systemic_tree


@dataclass
class SimulationResult:
    t: np.ndarray
    pressures: np.ndarray
    flows: np.ndarray
    lv_volume: np.ndarray
    aortic_pressure: np.ndarray


class CirculatorySimulator:
    """
    Coupled left-heart and 1D vascular-tree simulator.

    State vector layout:
    [LV volume, segment compliance pressures..., venous return pressure]
    """

    def __init__(
        self,
        heart=None,
        vessel_network=None,
        venous_compliance=20.0,
        terminal_resistance=8.0,
        resistance_scale=1e-9,
        initial_lv_volume=120.0,
        initial_venous_pressure=10.0,
    ):
        self.heart = heart if heart is not None else HeartModel()
        self.network = (
            vessel_network if vessel_network is not None else default_systemic_tree()
        )
        self.venous_compliance = float(venous_compliance)
        self.terminal_resistance = float(terminal_resistance)
        self.resistance_scale = float(resistance_scale)
        self.initial_lv_volume = float(initial_lv_volume)
        self.initial_venous_pressure = float(initial_venous_pressure)

        if self.venous_compliance <= 0.0:
            raise ValueError("venous_compliance must be positive")
        if self.terminal_resistance <= 0.0:
            raise ValueError("terminal_resistance must be positive")
        if self.resistance_scale <= 0.0:
            raise ValueError("resistance_scale must be positive")

        self.node_names = self.network._topological_nodes()
        self.segment_names = self._ordered_segments()
        self._prepare_topology()

    def run(self, duration_seconds, dt=0.001, jax_backend=False):
        if duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        t_eval = np.arange(0.0, duration_seconds + 0.5 * dt, dt)
        y0 = self._initial_state()

        if jax_backend:
            states = self._run_jax(y0, t_eval)
        else:
            states = self._run_scipy(y0, t_eval)

        return self._build_result(t_eval, states)

    def _run_scipy(self, y0, t_eval):
        try:
            from scipy.integrate import solve_ivp
        except ImportError as exc:
            raise ImportError(
                "CirculatorySimulator requires scipy for the default RK45 backend."
            ) from exc

        solution = solve_ivp(
            self._rhs,
            (float(t_eval[0]), float(t_eval[-1])),
            y0,
            method="RK45",
            t_eval=t_eval,
            max_step=float(np.diff(t_eval).min()),
            rtol=1e-6,
            atol=1e-8,
        )
        if not solution.success:
            raise RuntimeError(f"circulatory integration failed: {solution.message}")
        return solution.y.T

    def _run_jax(self, y0, t_eval):
        try:
            import jax.numpy as jnp
            from jax.experimental.ode import odeint
        except ImportError as exc:
            raise ImportError(
                "jax_backend=True requires jax and jax.experimental.ode.odeint."
            ) from exc

        y0_jax = jnp.asarray(y0)
        t_jax = jnp.asarray(t_eval)

        def rhs(y, t):
            return self._rhs_jax(y, t, jnp)

        return np.asarray(odeint(rhs, y0_jax, t_jax))

    def _rhs(self, t, y):
        lv_volume = y[0]
        segment_pressures = y[1:-1]
        venous_pressure = y[-1]

        p_lv = self.heart.get_elastance(t) * (lv_volume - self.heart.V_0)
        p_aortic, q_aortic = self._aortic_coupling_np(p_lv, segment_pressures)
        q_mitral = max(
            0.0, (venous_pressure - p_lv) / max(self.heart.R_mitral, 1e-12)
        )
        q_in, q_out, terminal_outflow = self._segment_fluxes_np(
            p_aortic, segment_pressures, venous_pressure
        )

        d_segment_pressures = (q_in - q_out) / self.compliances
        d_venous_pressure = (
            np.sum(terminal_outflow) - q_mitral
        ) / self.venous_compliance
        d_lv_volume = q_mitral - q_aortic

        return np.concatenate(
            ([d_lv_volume], d_segment_pressures, [d_venous_pressure])
        )

    def _rhs_jax(self, y, t, jnp):
        lv_volume = y[0]
        segment_pressures = y[1:-1]
        venous_pressure = y[-1]

        p_lv = self._elastance_jax(t, jnp) * (lv_volume - self.heart.V_0)
        p_aortic, q_aortic = self._aortic_coupling_jax(
            p_lv, segment_pressures, jnp
        )
        q_mitral = jnp.maximum(
            0.0, (venous_pressure - p_lv) / max(self.heart.R_mitral, 1e-12)
        )
        q_in, q_out, terminal_outflow = self._segment_fluxes_jax(
            p_aortic, segment_pressures, venous_pressure, jnp
        )

        d_segment_pressures = (q_in - q_out) / jnp.asarray(self.compliances)
        d_venous_pressure = (
            jnp.sum(terminal_outflow) - q_mitral
        ) / self.venous_compliance
        d_lv_volume = q_mitral - q_aortic

        return jnp.concatenate(
            (
                jnp.asarray([d_lv_volume]),
                d_segment_pressures,
                jnp.asarray([d_venous_pressure]),
            )
        )

    def _aortic_coupling_np(self, p_lv, segment_pressures):
        if len(self.root_segment_indices) == 0:
            return p_lv, 0.0

        root_indices = np.asarray(self.root_segment_indices, dtype=int)
        conductance = 1.0 / self.resistances[root_indices]
        downstream_pressure = segment_pressures[root_indices]
        passive_root = np.sum(conductance * downstream_pressure) / np.sum(
            conductance
        )
        open_root = (
            p_lv / self.heart.Z_characteristic
            + np.sum(conductance * downstream_pressure)
        ) / (1.0 / self.heart.Z_characteristic + np.sum(conductance))

        if p_lv > passive_root:
            p_aortic = open_root
            q_aortic = max(0.0, (p_lv - p_aortic) / self.heart.Z_characteristic)
        else:
            p_aortic = passive_root
            q_aortic = 0.0

        return float(p_aortic), float(q_aortic)

    def _aortic_coupling_jax(self, p_lv, segment_pressures, jnp):
        root_indices = jnp.asarray(self.root_segment_indices, dtype=int)
        conductance = 1.0 / jnp.asarray(self.resistances)[root_indices]
        downstream_pressure = segment_pressures[root_indices]
        passive_root = jnp.sum(conductance * downstream_pressure) / jnp.sum(
            conductance
        )
        open_root = (
            p_lv / self.heart.Z_characteristic
            + jnp.sum(conductance * downstream_pressure)
        ) / (1.0 / self.heart.Z_characteristic + jnp.sum(conductance))
        valve_open = p_lv > passive_root
        p_aortic = jnp.where(valve_open, open_root, passive_root)
        q_aortic = jnp.where(
            valve_open,
            jnp.maximum(0.0, (p_lv - p_aortic) / self.heart.Z_characteristic),
            0.0,
        )
        return p_aortic, q_aortic

    def _segment_fluxes_np(self, p_aortic, segment_pressures, venous_pressure):
        q_in = np.zeros(self.n_segments)
        q_out = np.zeros(self.n_segments)
        terminal_outflow = np.zeros(self.n_segments)

        for i in range(self.n_segments):
            parent_index = self.parent_segment_indices[i]
            upstream_pressure = (
                p_aortic if parent_index < 0 else segment_pressures[parent_index]
            )
            q_in[i] = (upstream_pressure - segment_pressures[i]) / self.resistances[i]

        for i, children in enumerate(self.child_segment_indices):
            if children:
                q_out[i] = np.sum(q_in[children])
            else:
                terminal_outflow[i] = (
                    segment_pressures[i] - venous_pressure
                ) / self.terminal_resistance
                q_out[i] = terminal_outflow[i]

        return q_in, q_out, terminal_outflow

    def _segment_fluxes_jax(self, p_aortic, segment_pressures, venous_pressure, jnp):
        q_in = []
        for i in range(self.n_segments):
            parent_index = self.parent_segment_indices[i]
            upstream_pressure = (
                p_aortic if parent_index < 0 else segment_pressures[parent_index]
            )
            q_in.append((upstream_pressure - segment_pressures[i]) / self.resistances[i])
        q_in = jnp.asarray(q_in)

        q_out = []
        terminal_outflow = []
        for i, children in enumerate(self.child_segment_indices):
            if children:
                q_out_i = jnp.sum(q_in[jnp.asarray(children, dtype=int)])
                terminal_i = 0.0
            else:
                terminal_i = (
                    segment_pressures[i] - venous_pressure
                ) / self.terminal_resistance
                q_out_i = terminal_i
            q_out.append(q_out_i)
            terminal_outflow.append(terminal_i)

        return jnp.asarray(q_in), jnp.asarray(q_out), jnp.asarray(terminal_outflow)

    def _elastance_jax(self, t, jnp):
        t_cycle = jnp.mod(t, self.heart.T_period)
        y = t_cycle / self.heart.T_systole
        term1 = (y / 0.7) ** self.heart.n1
        term2 = (y / 1.17) ** self.heart.n2
        activation = self.heart.k * (term1 / (1.0 + term1)) / (1.0 + term2)
        activation = jnp.where(y == 0.0, 0.0, activation)
        return self.heart.E_min + (self.heart.E_max - self.heart.E_min) * activation

    def _build_result(self, t, states):
        pressures = np.zeros((len(t), len(self.node_names)))
        flows = np.zeros((len(t), self.n_segments))
        lv_volume = states[:, 0]
        aortic_pressure = np.zeros(len(t))

        for row, state in enumerate(states):
            segment_pressures = state[1:-1]
            venous_pressure = state[-1]
            p_lv = self.heart.get_elastance(t[row]) * (
                state[0] - self.heart.V_0
            )
            p_aortic, _ = self._aortic_coupling_np(p_lv, segment_pressures)
            q_in, _, _ = self._segment_fluxes_np(
                p_aortic, segment_pressures, venous_pressure
            )

            aortic_pressure[row] = p_aortic
            flows[row, :] = q_in
            pressures[row, :] = self._node_pressures(p_aortic, segment_pressures)

        return SimulationResult(
            t=t,
            pressures=pressures,
            flows=flows,
            lv_volume=lv_volume,
            aortic_pressure=aortic_pressure,
        )

    def _node_pressures(self, p_aortic, segment_pressures):
        values = np.zeros(len(self.node_names))
        for i, node in enumerate(self.node_names):
            if node == self.network.inlet_node:
                values[i] = p_aortic
            else:
                values[i] = segment_pressures[self.node_to_parent_segment[node]]
        return values

    def _initial_state(self):
        try:
            steady = self.network.solve()
            pressure_by_node = steady["pressures"]
        except Exception:
            pressure_by_node = {
                node: self.network.outlet_pressure for node in self.node_names
            }
            pressure_by_node[self.network.inlet_node] = self.network.inlet_pressure

        segment_pressures = np.array(
            [pressure_by_node[end] for _, end in self.segment_names], dtype=float
        )
        return np.concatenate(
            (
                [self.initial_lv_volume],
                segment_pressures,
                [self.initial_venous_pressure],
            )
        )

    def _ordered_segments(self):
        segments = []
        for start in self.network._topological_nodes():
            for end in self.network.graph.get(start, {}):
                segments.append((start, end))
        return segments

    def _prepare_topology(self):
        if self.network.inlet_node is None:
            raise ValueError("vessel_network must have an inlet pressure")

        self.n_segments = len(self.segment_names)
        self.node_to_parent_segment = {}
        self.root_segment_indices = []
        self.parent_segment_indices = np.full(self.n_segments, -1, dtype=int)
        self.child_segment_indices = [[] for _ in range(self.n_segments)]
        self.resistances = np.zeros(self.n_segments)
        self.compliances = np.zeros(self.n_segments)

        for i, (start, end) in enumerate(self.segment_names):
            self.node_to_parent_segment[end] = i
            segment = self.network.graph[start][end]
            self.resistances[i] = max(
                segment["resistance"] * self.resistance_scale, 1e-9
            )
            self.compliances[i] = max(segment["wall_compliance"], 1e-9)

        for i, (start, _) in enumerate(self.segment_names):
            if start == self.network.inlet_node:
                self.root_segment_indices.append(i)
            else:
                self.parent_segment_indices[i] = self.node_to_parent_segment[start]
                self.child_segment_indices[self.parent_segment_indices[i]].append(i)


def _print_summary(result):
    stroke_volume = float(np.max(result.lv_volume) - np.min(result.lv_volume))
    print("CirculatorySimulator 10-second default run")
    print(f"samples: {len(result.t)}")
    print(
        "aortic pressure: "
        f"min={np.min(result.aortic_pressure):.2f}, "
        f"mean={np.mean(result.aortic_pressure):.2f}, "
        f"max={np.max(result.aortic_pressure):.2f} mmHg"
    )
    print(
        "LV volume: "
        f"min={np.min(result.lv_volume):.2f}, "
        f"max={np.max(result.lv_volume):.2f}, "
        f"stroke range={stroke_volume:.2f} mL"
    )
    print(f"mean inlet segment flow: {np.mean(result.flows[:, 0]):.2f} mL/s")


if __name__ == "__main__":
    simulator = CirculatorySimulator()
    _print_summary(simulator.run(10.0, dt=0.001))
