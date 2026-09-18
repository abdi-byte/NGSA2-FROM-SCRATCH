"""Run and plot NSGA-II on the toy problem."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsga2.algorithm import nsga2
from problems.toy_problem import toy_objectives


def main() -> None:
    result = nsga2(
        toy_objectives,
        num_variables=1,
        num_objectives=2,
        bounds=[(-1.0, 3.0)],
        population_size=80,
        generations=100,
        seed=7,
    )
    output_path = Path(__file__).resolve().parents[1] / "results" / "toy_pareto_front.png"
    objectives = [individual.objectives for individual in result.population]
    front_objectives = [individual.objectives for individual in result.first_front]
    plt.figure(figsize=(8, 5))
    plt.scatter([value[0] for value in objectives], [value[1] for value in objectives], label="Final population", alpha=0.45)
    plt.scatter([value[0] for value in front_objectives], [value[1] for value in front_objectives], label="First Pareto front", color="crimson")
    plt.xlabel("f1(x) = x^2")
    plt.ylabel("f2(x) = (x - 2)^2")
    plt.title("NSGA-II on the toy problem")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Generations: {100}")
    print(f"Population size: {len(result.population)}")
    print(f"First Pareto front size: {len(result.first_front)}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
