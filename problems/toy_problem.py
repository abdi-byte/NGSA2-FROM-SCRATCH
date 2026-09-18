"""A simple two-objective problem with an intuitive trade-off."""

from typing import List, Sequence


def toy_objectives(variables: Sequence[float]) -> List[float]:
    """Minimise x squared and (x - 2) squared."""

    if len(variables) != 1:
        raise ValueError("The toy problem has exactly one decision variable")
    x = variables[0]
    return [x**2, (x - 2.0) ** 2]
