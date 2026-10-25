# Research Extension: NSGA-II for study allocation

**Status:** Educational simulation; not a validated student grade predictor.

This extension adds (1) repeated seeded NSGA-II runs on ZDT1, (2) Inverted Generational Distance (IGD) and exact 2D hypervolume metrics, (3) an optional `pymoo` baseline, (4) convergence plots, (5) a synthetic five-module student study allocation problem, and (6) new tests.

## Install
Copy the `problems/`, `experiments/`, `tests/` files into the corresponding locations in the existing repository, without removing anything. All paths are relative to the repo root.

```
python -m pip install -r requirements.txt
python -m pip install pymoo  # optional baseline
python -m pytest
python experiments/research_evaluation.py
```

Output generated in `results/research/`. This script runs 10 seeds by default. Increase `SEEDS` to `range(30)` after a trial run; 200-generation ZDT1 runs may be computationally expensive.

## Student simulation
Five illustrative modules are assigned baseline grades, maximum possible study-related gains and learning-rate coefficients. For each module, the assumed grade response is `base + gain * (1-exp(-rate*weekly_hours))`. NSGA-II optimises (1) negative predicted mean grade, (2) hours, and (3) sum of shortfalls under 70. Candidates are repaired to obey a 30-hour weekly limit and each module has a 15-hour upper bound.

**Important:** These grade-response assumptions are synthetic. No causal claim, wellbeing inference, or advice to sacrifice sleep is justified. Real schedules need lecture/lab constraints, time-of-day availability and realistic breaks. Students should understand the model as an optimisation illustration, not a promise of improvement.

## Research limitations and next steps
- Add repeated-seed sensitivity analyses for population size, crossover/mutation rates and 2D/3D hypervolume (using pymoo HV for 3D).
- Improve convergence measurement: record each generation's first nondominated front rather than all population objectives.
- Account for duplicate objective points, numerical robustness, and finite-valued parameters.
- Compare baselines using equal objective-evaluation budgets and report confidence intervals, not just single-best runs.
- Collect ethically sourced, consent-based observations before making any real-world predictive claims.

## Parameter sensitivity
Run `python experiments/sensitivity.py` to evaluate changes in population size, crossover probability and mutation distribution index across five random seeds. This introductory analysis holds generations constant, **not total function evaluations**. Compare results cautiously; use matched evaluation budgets for a controlled follow-up study.
