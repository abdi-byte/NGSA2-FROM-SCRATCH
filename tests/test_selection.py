import numpy as np

from nsga2.individual import Individual
from nsga2.selection import binary_tournament, crowded_comparison


def test_crowded_comparison_prefers_rank_then_distance():
    better_rank = Individual([0], [0], rank=0, crowding_distance=0.0)
    worse_rank = Individual([1], [1], rank=1, crowding_distance=float("inf"))
    assert crowded_comparison(better_rank, worse_rank)
    same_rank_more_space = Individual([2], [2], rank=0, crowding_distance=2.0)
    same_rank_less_space = Individual([3], [3], rank=0, crowding_distance=1.0)
    assert crowded_comparison(same_rank_more_space, same_rank_less_space)


def test_binary_tournament_uses_rank_and_is_reproducible():
    population = [
        Individual([0], [0], rank=0, crowding_distance=0.0),
        Individual([1], [1], rank=1, crowding_distance=float("inf")),
    ]
    assert binary_tournament(population, np.random.default_rng(5)) is population[0]
    first = binary_tournament(population, np.random.default_rng(10))
    second = binary_tournament(population, np.random.default_rng(10))
    assert first is second
