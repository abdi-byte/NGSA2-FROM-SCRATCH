"""Real-valued SBX crossover and polynomial mutation."""

from typing import List, Optional, Sequence, Tuple

import numpy as np


def simulated_binary_crossover(
    first: Sequence[float],
    second: Sequence[float],
    bounds: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
    probability: float = 0.9,
    eta: float = 20.0,
) -> Tuple[List[float], List[float]]:
    """Create two children with simulated binary crossover (SBX).

    ``eta`` controls the spread of children: larger values keep children
    closer to their parents. Clipping is a final numerical guard at bounds.
    """

    if len(first) != len(second) or len(first) != len(bounds):
        raise ValueError("Parents and bounds must have matching lengths")
    child_one = list(map(float, first))
    child_two = list(map(float, second))
    if rng.random() > probability:
        return child_one, child_two

    for index, ((lower, upper), value_one, value_two) in enumerate(zip(bounds, first, second)):
        if lower > upper:
            raise ValueError("Each lower bound must not exceed its upper bound")
        if rng.random() > 0.5 or abs(value_one - value_two) < 1e-14:
            continue
        smaller, larger = sorted((float(value_one), float(value_two)))
        random_value = rng.random()
        beta = 1.0 + 2.0 * (smaller - lower) / (larger - smaller)
        alpha = 2.0 - beta ** -(eta + 1.0)
        if random_value <= 1.0 / alpha:
            beta_q = (random_value * alpha) ** (1.0 / (eta + 1.0))
        else:
            beta_q = (1.0 / (2.0 - random_value * alpha)) ** (1.0 / (eta + 1.0))
        first_child = 0.5 * ((smaller + larger) - beta_q * (larger - smaller))
        beta = 1.0 + 2.0 * (upper - larger) / (larger - smaller)
        alpha = 2.0 - beta ** -(eta + 1.0)
        if random_value <= 1.0 / alpha:
            beta_q = (random_value * alpha) ** (1.0 / (eta + 1.0))
        else:
            beta_q = (1.0 / (2.0 - random_value * alpha)) ** (1.0 / (eta + 1.0))
        second_child = 0.5 * ((smaller + larger) + beta_q * (larger - smaller))
        first_child = float(np.clip(first_child, lower, upper))
        second_child = float(np.clip(second_child, lower, upper))
        if rng.random() <= 0.5:
            child_one[index], child_two[index] = second_child, first_child
        else:
            child_one[index], child_two[index] = first_child, second_child
    return child_one, child_two


def polynomial_mutation(
    variables: Sequence[float],
    bounds: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
    probability: Optional[float] = None,
    eta: float = 20.0,
) -> List[float]:
    """Mutate real-valued variables with Deb's polynomial mutation."""

    if len(variables) != len(bounds):
        raise ValueError("Variables and bounds must have matching lengths")
    mutation_probability = probability if probability is not None else 1.0 / len(variables)
    mutated = list(map(float, variables))
    for index, ((lower, upper), value) in enumerate(zip(bounds, mutated)):
        if lower > upper:
            raise ValueError("Each lower bound must not exceed its upper bound")
        if rng.random() > mutation_probability or upper == lower:
            continue
        value = float(np.clip(value, lower, upper))
        distance_to_lower = (value - lower) / (upper - lower)
        distance_to_upper = (upper - value) / (upper - lower)
        random_value = rng.random()
        power = 1.0 / (eta + 1.0)
        if random_value <= 0.5:
            change = 2.0 * random_value + (1.0 - 2.0 * random_value) * (1.0 - distance_to_lower) ** (eta + 1.0)
            delta = change**power - 1.0
        else:
            change = 2.0 * (1.0 - random_value) + 2.0 * (random_value - 0.5) * (1.0 - distance_to_upper) ** (eta + 1.0)
            delta = 1.0 - change**power
        mutated[index] = float(np.clip(value + delta * (upper - lower), lower, upper))
    return mutated
