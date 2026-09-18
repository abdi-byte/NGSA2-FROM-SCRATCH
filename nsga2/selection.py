"""NSGA-II selection operators."""

from typing import Sequence

import numpy as np

from .individual import Individual


def crowded_comparison(first: Individual, second: Individual) -> bool:
    """Return whether first is preferred by the crowded comparison rule."""

    if first.rank != second.rank:
        return first.rank < second.rank
    return first.crowding_distance > second.crowding_distance


def binary_tournament(population: Sequence[Individual], rng: np.random.Generator) -> Individual:
    """Choose one individual from two random candidates."""

    if len(population) < 2:
        raise ValueError("Binary tournament requires at least two individuals")
    first_index, second_index = rng.choice(len(population), size=2, replace=False)
    first = population[int(first_index)]
    second = population[int(second_index)]
    if crowded_comparison(first, second):
        return first
    if crowded_comparison(second, first):
        return second
    return first if rng.random() < 0.5 else second
