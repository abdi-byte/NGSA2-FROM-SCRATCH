"""Complete NSGA-II algorithm."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .crowding import assign_crowding_distances
from .individual import Individual
from .operators import polynomial_mutation, simulated_binary_crossover
from .selection import binary_tournament
from .sorting import fast_non_dominated_sort

ObjectiveFunction = Callable[[Sequence[float]], Sequence[float]]
Bounds = Sequence[Tuple[float, float]]


@dataclass
class NSGA2Result:
    """Outputs from an NSGA-II run."""

    population: List[Individual]
    first_front: List[Individual]
    history: List[List[List[float]]]


def initialise_population(
    population_size: int,
    bounds: Bounds,
    rng: np.random.Generator,
) -> List[Individual]:
    """Sample an initial population uniformly inside the bounds."""

    if population_size < 2:
        raise ValueError("Population size must be at least two")
    variables = []
    for lower, upper in bounds:
        if lower > upper:
            raise ValueError("Each lower bound must not exceed its upper bound")
        variables.append((lower, upper))
    return [
        Individual([rng.uniform(lower, upper) for lower, upper in variables])
        for _ in range(population_size)
    ]


def evaluate_population(population: Sequence[Individual], objective_function: ObjectiveFunction) -> None:
    """Evaluate and store objectives for every individual."""

    for individual in population:
        individual.objectives = [float(value) for value in objective_function(individual.variables)]


def prepare_population(population: List[Individual]) -> List[List[Individual]]:
    """Assign rank and crowding distance to a population."""

    fronts = fast_non_dominated_sort(population)
    assign_crowding_distances(fronts)
    return fronts


def environmental_selection(
    combined_population: Sequence[Individual],
    population_size: int,
) -> List[Individual]:
    """Select the next generation from parents plus offspring."""

    if population_size <= 0 or population_size > len(combined_population):
        raise ValueError("Population size must fit within the combined population")
    fronts = prepare_population(list(combined_population))
    next_population: List[Individual] = []
    for front in fronts:
        if len(next_population) + len(front) <= population_size:
            next_population.extend(front)
            continue
        remaining = population_size - len(next_population)
        sorted_front = sorted(front, key=lambda individual: individual.crowding_distance, reverse=True)
        next_population.extend(sorted_front[:remaining])
        break
    return next_population


def create_offspring(
    population: Sequence[Individual],
    objective_function: ObjectiveFunction,
    bounds: Bounds,
    rng: np.random.Generator,
    crossover_probability: float,
    mutation_probability: Optional[float],
    crossover_eta: float,
    mutation_eta: float,
) -> List[Individual]:
    """Create and evaluate one offspring population."""

    offspring: List[Individual] = []
    while len(offspring) < len(population):
        parent_one = binary_tournament(population, rng)
        parent_two = binary_tournament(population, rng)
        child_one, child_two = simulated_binary_crossover(
            parent_one.variables,
            parent_two.variables,
            bounds,
            rng,
            probability=crossover_probability,
            eta=crossover_eta,
        )
        child_one = polynomial_mutation(child_one, bounds, rng, mutation_probability, mutation_eta)
        child_two = polynomial_mutation(child_two, bounds, rng, mutation_probability, mutation_eta)
        offspring.append(Individual(child_one))
        if len(offspring) < len(population):
            offspring.append(Individual(child_two))
    evaluate_population(offspring, objective_function)
    return offspring


def nsga2(
    objective_function: ObjectiveFunction,
    num_variables: int,
    num_objectives: int,
    bounds: Bounds,
    population_size: int = 100,
    generations: int = 100,
    crossover_probability: float = 0.9,
    mutation_probability: Optional[float] = None,
    seed: Optional[int] = None,
    crossover_eta: float = 20.0,
    mutation_eta: float = 20.0,
) -> NSGA2Result:
    """Run the complete NSGA-II pipeline for a minimisation problem."""

    if num_variables != len(bounds):
        raise ValueError("num_variables must match the number of bounds")
    if num_variables <= 0 or num_objectives <= 0:
        raise ValueError("Variable and objective counts must be positive")
    if generations < 0:
        raise ValueError("generations cannot be negative")
    rng = np.random.default_rng(seed)
    population = initialise_population(population_size, bounds, rng)
    evaluate_population(population, objective_function)
    prepare_population(population)
    history: List[List[List[float]]] = [[list(individual.objectives) for individual in population]]

    for _ in range(generations):
        offspring = create_offspring(
            population,
            objective_function,
            bounds,
            rng,
            crossover_probability,
            mutation_probability,
            crossover_eta,
            mutation_eta,
        )
        population = environmental_selection(population + offspring, population_size)
        history.append([list(individual.objectives) for individual in population])

    final_fronts = prepare_population(population)
    first_front = final_fronts[0] if final_fronts else []
    return NSGA2Result(population=population, first_front=first_front, history=history)
