"""Crowding-distance assignment."""

from typing import List, Sequence

from .individual import Individual


def assign_crowding_distance(front: List[Individual]) -> None:
    """Assign summed, normalised crowding distances to one Pareto front."""

    if not front:
        return
    for individual in front:
        individual.crowding_distance = 0.0
    if len(front) <= 2:
        for individual in front:
            individual.crowding_distance = float("inf")
        return

    objective_count = len(front[0].objectives)
    for objective_index in range(objective_count):
        ordered = sorted(front, key=lambda individual: individual.objectives[objective_index])
        ordered[0].crowding_distance = float("inf")
        ordered[-1].crowding_distance = float("inf")
        minimum = ordered[0].objectives[objective_index]
        maximum = ordered[-1].objectives[objective_index]
        if maximum == minimum:
            continue
        denominator = maximum - minimum
        for index in range(1, len(ordered) - 1):
            if ordered[index].crowding_distance == float("inf"):
                continue
            contribution = (
                ordered[index + 1].objectives[objective_index]
                - ordered[index - 1].objectives[objective_index]
            ) / denominator
            ordered[index].crowding_distance += contribution


def assign_crowding_distances(fronts: Sequence[List[Individual]]) -> None:
    """Assign crowding distances to every front."""

    for front in fronts:
        assign_crowding_distance(front)
