"""The standard ZDT1 benchmark."""

import math
from typing import List, Sequence


def zdt1_objectives(variables: Sequence[float]) -> List[float]:
    """Return ZDT1 objectives for variables in [0, 1]."""

    if len(variables) < 2:
        raise ValueError("ZDT1 requires at least two decision variables")
    f1 = float(variables[0])
    g = 1.0 + 9.0 * sum(variables[1:]) / (len(variables) - 1)
    f2 = g * (1.0 - math.sqrt(f1 / g))
    return [f1, f2]
