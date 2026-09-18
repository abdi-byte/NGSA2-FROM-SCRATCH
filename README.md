# NSGA-II From Scratch

This repository is a readable Python implementation of the Non-dominated Sorting Genetic Algorithm II (NSGA-II) for continuous, multi-objective minimisation problems. It is intended as a university research mini-project: the algorithm is implemented directly rather than hidden behind an optimisation library.

## What is multi-objective optimisation?

A multi-objective problem has several objectives to minimise or maximise at once. Objectives can conflict: improving one may worsen another. There is usually no single solution that is best in every objective, so the useful answer is a set of balanced trade-offs.

## Pareto dominance

For minimisation, solution A dominates solution B when A is no worse than B in every objective and strictly better in at least one objective. If each solution is better in a different objective, neither dominates the other.

## Pareto optimality and the Pareto front

A solution is Pareto-optimal when no feasible solution dominates it. The set of Pareto-optimal solutions is the Pareto set in decision space. Their objective values form the Pareto front in objective space.

## Why NSGA-II?

NSGA-II combines two ideas:

1. **Pareto rank**: non-dominated sorting gives rank 0 to the current best trade-offs and higher ranks to progressively dominated solutions. This encourages convergence toward good objective values.
2. **Crowding distance**: solutions in the same front are rewarded for being far from their neighbours. This preserves diversity across the trade-off curve.

## Algorithm flow

```text
Initial population
        |
Evaluate objectives
        |
Fast non-dominated sorting
        |
Crowding distance
        |
Binary tournament selection
        |
SBX crossover + polynomial mutation
        |
Evaluate offspring
        |
Parents + offspring
        |
Fast non-dominated sorting
        |
Environmental selection
        |
Next generation
```

### Fast non-dominated sorting

For every solution `p`, the algorithm stores the solutions that `p` dominates and a domination count: how many solutions dominate `p`. Every solution with count zero belongs to the first front. Once that front is known, its influence is removed from the dominated solutions, revealing the next front. This repeats until every solution has a rank.

### Crowding distance

For each objective, a front is sorted by objective value. Its two boundary solutions receive infinite distance so they are retained as edge points. Each interior solution receives the normalised distance between its neighbours; contributions are summed over objectives. Larger distance is preferred when ranks are equal.

### Genetic operators

- **SBX (simulated binary crossover)** creates children around two real-valued parents, with `eta` controlling spread. Larger `eta` concentrates children closer to the parents.
- **Polynomial mutation** perturbs each variable with a configurable probability, also using `eta` to control how local the perturbation is. Every result is clipped to its variable bounds.

A single NumPy random generator is created for each run. Passing the same seed therefore reproduces the same initial population and evolutionary decisions.

## Experiments

### Toy problem

The one-variable problem minimises:

```text
f1(x) = x^2
f2(x) = (x - 2)^2
```

with `x` in `[-1, 3]`. The two objectives pull the solution toward different values of `x`, making the trade-off easy to inspect.

### ZDT1

ZDT1 uses 30 variables in `[0, 1]` by default:

```text
f1(x) = x1
g(x) = 1 + 9 * sum(x2...xn) / (n - 1)
f2(x) = g * (1 - sqrt(f1 / g))
```

Its analytical Pareto front is `f2 = 1 - sqrt(f1)` for `f1` in `[0, 1]`.

## Running on Windows PowerShell

From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
python experiments/toy_experiment.py
python experiments/zdt1_experiment.py
```

The plots are saved as `results/toy_pareto_front.png` and `results/zdt1_pareto_front.png`. Population size, generation count, and seed are constants near the top of the ZDT1 experiment and arguments in the toy experiment.

## Suggested study order

1. Read `nsga2/dominance.py` and `nsga2/sorting.py` to understand ranks.
2. Trace `nsga2/crowding.py` and `nsga2/selection.py` to see how diversity affects selection.
3. Read `nsga2/algorithm.py`, especially `environmental_selection` and `create_offspring`.
4. Compare `problems/toy_problem.py` and `problems/zdt1.py` with the plots.
5. Use the tests as small executable examples of each rule.

## Project structure

```text
README.md
requirements.txt
main.py
nsga2/             # algorithm components
problems/          # objective functions
experiments/       # runnable plotting scripts
tests/             # focused correctness tests
results/           # generated plots
```
