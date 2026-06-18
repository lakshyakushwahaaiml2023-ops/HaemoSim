"""Data collection and storage modules."""

__all__ = ["SimulationDataCollector", "HaemodynamicsDataset"]


def __getattr__(name):
    if name == "SimulationDataCollector":
        from .collector import SimulationDataCollector

        return SimulationDataCollector
    if name == "HaemodynamicsDataset":
        from .dataset import HaemodynamicsDataset

        return HaemodynamicsDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
