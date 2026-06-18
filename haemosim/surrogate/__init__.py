"""ML surrogate model modules."""

from .fast_model import FastStateSurrogate
from .gnn_model import GNNSurrogate

__all__ = ["FastStateSurrogate", "GNNSurrogate"]
