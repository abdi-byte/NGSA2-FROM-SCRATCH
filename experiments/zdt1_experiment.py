"""Run and plot NSGA-II on the ZDT1 benchmark."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsga2.algorithm import nsga2
from problems.zdt1 import zdt1_objectives


POPULATION_SIZE = 100
GENERATIONS = 200
NUM_VARIABLES = 30
SEED = 7


def main() -> None:
    result = nsga2(
        zdt1_objectives,
        num_variables=NUM_VARIABLES,
        num_objectives=2,
        bounds=[(0.0, 1.0)] * NUM_VARIABLES,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        seed=SEED,
    )
    output_path = Path(__file__).resolve().parents[1] / "results" / "zdt1_pareto_front.png"
    front = np.array([individual.objectives for individual in result.first_front])
    true_f1 = np.linspace(0.0, 1.0, 300)
    plt.figure(figsize=(8, 5))
    plt.scatter(front[:, 0], front[:, 1], label="NSGA-II first front", color="crimson", s=20)
    plt.plot(true_f1, 1.0 - np.sqrt(true_f1), label="True ZDT1 front", color="black", linewidth=1.5)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("NSGA-II on ZDT1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Generations: {GENERATIONS}")
    print(f"Population size: {len(result.population)}")
    print(f"First Pareto front size: {len(result.first_front)}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
