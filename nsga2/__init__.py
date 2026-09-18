"""Educational implementation of NSGA-II."""

from .algorithm import NSGA2Result, nsga2
from .individual import Individual

__all__ = ["Individual", "NSGA2Result", "nsga2"]
