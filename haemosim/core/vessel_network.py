import math
from collections import defaultdict, deque

import numpy as np


class VesselNetwork:
    """
    Directed 1D vascular tree with Poiseuille segment resistance and nodal
    compliance for pulsatile dynamics.
    """

    def __init__(self, blood_viscosity=0.0035, outlet_pressure=5.0):
        self.blood_viscosity = blood_viscosity
        self.outlet_pressure = outlet_pressure
        self.graph = {}
        self.inlet_node = None
        self.inlet_pressure = None
        self.pressures = {}

    def add_segment(self, start, end, length, radius, wall_compliance):
        """
        Add a directed vessel segment.

        Parameters are in SI-like units chosen consistently by the caller.
        Resistance is computed as R = 8 * mu * L / (pi * r^4).
        """
        if length <= 0.0:
            raise ValueError("length must be positive")
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if wall_compliance < 0.0:
            raise ValueError("wall_compliance must be non-negative")

        resistance = (
            8.0 * self.blood_viscosity * length / (math.pi * radius**4)
        )
        self.graph.setdefault(start, {})[end] = {
            "length": float(length),
            "radius": float(radius),
            "wall_compliance": float(wall_compliance),
            "resistance": resistance,
        }
        self.graph.setdefault(end, {})
        self.pressures.setdefault(start, self.outlet_pressure)
        self.pressures.setdefault(end, self.outlet_pressure)

    def set_inlet_pressure(self, node, pressure):
        if node not in self.nodes:
            raise ValueError(f"unknown inlet node: {node}")
        self.inlet_node = node
        self.inlet_pressure = float(pressure)
        self.pressures[node] = float(pressure)

    def solve(self, dt=None):
        """
        Solve node pressures and segment flows.

        If dt is omitted, a steady Poiseuille solution is computed. If dt is
        provided, nodal compliance contributes a backward-Euler storage term.
        """
        if self.inlet_node is None or self.inlet_pressure is None:
            raise ValueError("set_inlet_pressure() must be called before solve()")
        if dt is not None and dt <= 0.0:
            raise ValueError("dt must be positive")

        nodes = self._topological_nodes()
        outlets = self.outlet_nodes
        fixed = {self.inlet_node: self.inlet_pressure}
        fixed.update({node: self.outlet_pressure for node in outlets})
        unknowns = [node for node in nodes if node not in fixed]

        if unknowns:
            index = {node: i for i, node in enumerate(unknowns)}
            matrix = np.zeros((len(unknowns), len(unknowns)))
            rhs = np.zeros(len(unknowns))
            capacitance = self._node_compliances()

            for node in unknowns:
                row = index[node]

                if dt is not None:
                    storage = capacitance[node] / dt
                    matrix[row, row] += storage
                    rhs[row] += storage * self.pressures.get(
                        node, self.outlet_pressure
                    )

                for neighbor, resistance in self._connected_resistances(node):
                    conductance = 1.0 / resistance
                    matrix[row, row] += conductance
                    if neighbor in fixed:
                        rhs[row] += conductance * fixed[neighbor]
                    else:
                        matrix[row, index[neighbor]] -= conductance

            solved = np.linalg.solve(matrix, rhs)
            for node, value in zip(unknowns, solved):
                self.pressures[node] = float(value)

        for node, value in fixed.items():
            self.pressures[node] = float(value)

        segment_flows = self._segment_flows()
        node_flows = self._node_flows(segment_flows)

        return {
            "pressures": dict(self.pressures),
            "flows": node_flows,
            "segment_flows": segment_flows,
        }

    @property
    def nodes(self):
        nodes = set(self.graph)
        for children in self.graph.values():
            nodes.update(children)
        return nodes

    @property
    def outlet_nodes(self):
        return [node for node, children in self.graph.items() if not children]

    def _topological_nodes(self):
        indegree = {node: 0 for node in self.nodes}
        for children in self.graph.values():
            for child in children:
                indegree[child] += 1

        queue = deque(node for node, degree in indegree.items() if degree == 0)
        ordered = []
        while queue:
            node = queue.popleft()
            ordered.append(node)
            for child in self.graph.get(node, {}):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(ordered) != len(indegree):
            raise ValueError("vascular network must be acyclic")
        return ordered

    def _incoming(self):
        incoming = defaultdict(dict)
        for start, children in self.graph.items():
            for end, segment in children.items():
                incoming[end][start] = segment
        return incoming

    def _connected_resistances(self, node):
        incoming = self._incoming()
        for child, segment in self.graph[node].items():
            yield child, segment["resistance"]
        for parent, segment in incoming[node].items():
            yield parent, segment["resistance"]

    def _node_compliances(self):
        capacitance = defaultdict(float)
        for start, children in self.graph.items():
            for end, segment in children.items():
                half_compliance = 0.5 * segment["wall_compliance"]
                capacitance[start] += half_compliance
                capacitance[end] += half_compliance
        return capacitance

    def _segment_flows(self):
        flows = {}
        for start, children in self.graph.items():
            for end, segment in children.items():
                flow = (
                    self.pressures[start] - self.pressures[end]
                ) / segment["resistance"]
                flows[(start, end)] = float(flow)
        return flows

    def _node_flows(self, segment_flows):
        node_flows = {
            node: {"in": 0.0, "out": 0.0, "net": 0.0} for node in self.nodes
        }
        for (start, end), flow in segment_flows.items():
            node_flows[start]["out"] += flow
            node_flows[end]["in"] += flow

        for node, values in node_flows.items():
            values["net"] = values["in"] - values["out"]
        return node_flows


def default_systemic_tree():
    """Create a simplified aorta -> 4 major arteries -> 8 arterioles tree."""
    network = VesselNetwork()

    major_arteries = [
        "carotid",
        "subclavian",
        "renal",
        "iliac",
    ]
    for artery in major_arteries:
        network.add_segment(
            "aorta",
            artery,
            length=0.25,
            radius=0.0045,
            wall_compliance=0.45,
        )

    arteriole_pairs = {
        "carotid": ("left_cerebral_arteriole", "right_cerebral_arteriole"),
        "subclavian": ("left_arm_arteriole", "right_arm_arteriole"),
        "renal": ("left_renal_arteriole", "right_renal_arteriole"),
        "iliac": ("left_leg_arteriole", "right_leg_arteriole"),
    }
    for parent, arterioles in arteriole_pairs.items():
        for arteriole in arterioles:
            network.add_segment(
                parent,
                arteriole,
                length=0.08,
                radius=0.0012,
                wall_compliance=0.08,
            )

    network.set_inlet_pressure("aorta", 95.0)
    return network
