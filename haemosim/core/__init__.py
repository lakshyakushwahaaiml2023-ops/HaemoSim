"""Physics simulation modules."""

from .heart import HeartModel
from .vessel_network import VesselNetwork, default_systemic_tree

__all__ = [
    "HeartModel",
    "VesselNetwork",
    "default_systemic_tree",
    "CirculatorySimulator",
    "SimulationResult",
]


def __getattr__(name):
    if name in {"CirculatorySimulator", "SimulationResult"}:
        from .simulator import CirculatorySimulator, SimulationResult

        exports = {
            "CirculatorySimulator": CirculatorySimulator,
            "SimulationResult": SimulationResult,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
