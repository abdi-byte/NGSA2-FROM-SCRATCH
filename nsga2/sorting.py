"""Fast non-dominated sorting."""

from typing import List, Sequence

from .dominance import dominates
from .individual import Individual


def fast_non_dominated_sort(population: Sequence[Individual]) -> List[List[Individual]]:
    """Partition a population into Pareto fronts using NSGA-II's fast sort.

    For each solution p, ``dominated_solutions[p]`` stores the solutions p
    dominates, while ``domination_count[p]`` counts solutions dominating p.
    Solutions with count zero form front zero; removing that front's influence
    exposes the next front, and so on.
    """

    dominated_solutions = {id(individual): [] for individual in population}
    domination_count = {id(individual): 0 for individual in population}
    first_front: List[Individual] = []

    for index, first in enumerate(population):
        for second in population[index + 1 :]:
            if dominates(first.objectives, second.objectives):
                dominated_solutions[id(first)].append(second)
                domination_count[id(second)] += 1
            elif dominates(second.objectives, first.objectives):
                dominated_solutions[id(second)].append(first)
                domination_count[id(first)] += 1
        if domination_count[id(first)] == 0:
            first.rank = 0
            first_front.append(first)

    fronts: List[List[Individual]] = [first_front]
    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: List[Individual] = []
        for individual in fronts[front_index]:
            for dominated in dominated_solutions[id(individual)]:
                domination_count[id(dominated)] -= 1
                if domination_count[id(dominated)] == 0:
                    dominated.rank = front_index + 1
                    next_front.append(dominated)
        if next_front:
            fronts.append(next_front)
        front_index += 1

    return fronts
