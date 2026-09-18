import numpy as np

from nsga2.algorithm import environmental_selection, nsga2
from nsga2.individual import Individual
from nsga2.operators import polynomial_mutation, simulated_binary_crossover
from problems.toy_problem import toy_objectives


def test_operators_stay_inside_bounds():
    bounds = [(0.0, 1.0), (-2.0, 2.0)]
    rng = np.random.default_rng(3)
    for _ in range(50):
        first, second = simulated_binary_crossover([0.0, -2.0], [1.0, 2.0], bounds, rng, probability=1.0)
        for child in (first, second, polynomial_mutation(first, bounds, rng, probability=1.0)):
            assert all(lower <= value <= upper for value, (lower, upper) in zip(child, bounds))


def test_environmental_selection_returns_exact_population_size():
    population = [Individual([i], [float(i), float(10 - i)]) for i in range(10)]
    selected = environmental_selection(population, 5)
    assert len(selected) == 5
    assert all(individual.rank == 0 for individual in selected)


def test_algorithm_runs_and_seed_reproduces_result():
    first = nsga2(toy_objectives, 1, 2, [(-1.0, 3.0)], population_size=20, generations=5, seed=11)
    second = nsga2(toy_objectives, 1, 2, [(-1.0, 3.0)], population_size=20, generations=5, seed=11)
    assert len(first.population) == 20
    assert len(first.history) == 6
    assert [ind.variables for ind in first.population] == [ind.variables for ind in second.population]
    assert len(first.first_front) > 0
