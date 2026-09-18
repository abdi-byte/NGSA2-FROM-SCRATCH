import math

from nsga2.crowding import assign_crowding_distance
from nsga2.individual import Individual


def test_boundaries_are_infinite_and_interior_contributions_sum():
    front = [
        Individual([0], [0, 3]),
        Individual([1], [1, 2]),
        Individual([2], [2, 1]),
        Individual([3], [3, 0]),
    ]
    assign_crowding_distance(front)
    assert math.isinf(front[0].crowding_distance)
    assert math.isinf(front[-1].crowding_distance)
    assert front[1].crowding_distance == 4.0 / 3.0
    assert front[2].crowding_distance == 4.0 / 3.0


def test_constant_objective_does_not_divide_by_zero():
    front = [Individual([i], [float(i), 1.0]) for i in range(4)]
    assign_crowding_distance(front)
    assert all(individual.crowding_distance >= 0 for individual in front)
