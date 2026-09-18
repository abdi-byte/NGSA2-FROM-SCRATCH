from nsga2.individual import Individual
from nsga2.sorting import fast_non_dominated_sort


def test_fast_sort_returns_expected_fronts_and_ranks():
    population = [
        Individual([0], [1, 4]),
        Individual([1], [2, 3]),
        Individual([2], [3, 2]),
        Individual([3], [4, 1]),
        Individual([4], [3, 4]),
        Individual([5], [5, 5]),
    ]
    fronts = fast_non_dominated_sort(population)
    assert {tuple(individual.objectives) for individual in fronts[0]} == {(1, 4), (2, 3), (3, 2), (4, 1)}
    assert {tuple(individual.objectives) for individual in fronts[1]} == {(3, 4)}
    assert {tuple(individual.objectives) for individual in fronts[2]} == {(5, 5)}
    assert all(individual.rank == 0 for individual in fronts[0])
    assert fronts[1][0].rank == 1
    assert fronts[2][0].rank == 2
