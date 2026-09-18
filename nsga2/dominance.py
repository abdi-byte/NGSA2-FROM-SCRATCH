"""Pareto dominance for minimisation problems."""

from typing import Sequence


def dominates(first: Sequence[float], second: Sequence[float]) -> bool:
    """Return whether *first* Pareto-dominates *second*.

    First must be no worse in every objective and strictly better in at least
    one objective. Objectives are minimised.
    """

    if len(first) != len(second):
        raise ValueError("Individuals must have the same number of objectives")
    no_worse_everywhere = all(a <= b for a, b in zip(first, second))
    strictly_better_somewhere = any(a < b for a, b in zip(first, second))
    return no_worse_everywhere and strictly_better_somewhere
