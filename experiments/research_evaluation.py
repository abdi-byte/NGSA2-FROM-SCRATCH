"""Multi-run evaluation, metrics, baseline comparison and optional pymoo comparison.
Run: python experiments/research_evaluation.py
"""
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nsga2.algorithm import nsga2
from problems.zdt1 import zdt1_objectives
from problems.student_study import (MODULES, predict, repair_study_hours,
                                    objectives_under_budget, WEEKLY_BUDGET)

OUT = Path(__file__).resolve().parents[1] / 'results' / 'research'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(10))  # expand to 30 for publication-quality statistical summaries

def igd(points, reference):
    """Inverted generational distance: lower is better."""
    p = np.asarray(points, dtype=float); r = np.asarray(reference, dtype=float)
    if p.ndim != 2 or r.ndim != 2 or not len(p) or not len(r) or p.shape[1] != r.shape[1]:
        raise ValueError('Nonempty equally-dimensional point arrays required')
    return float(np.mean(np.min(np.linalg.norm(r[:, None, :] - p[None, :, :], axis=2), axis=1)))

def hypervolume_2d(points, reference=(1.1, 1.1)):
    """Exact dominated 2D hypervolume for minimisation; higher is better."""
    p = np.asarray(points, dtype=float); ref = np.asarray(reference, dtype=float)
    if p.size == 0: return 0.0
    if p.ndim != 2 or p.shape[1] != 2: raise ValueError('Expected 2D objectives')
    p = p[np.all(p < ref, axis=1)]
    if not len(p): return 0.0
    p = p[np.argsort(p[:, 0])]
    area, best_y = 0., ref[1]
    for x, y in p:
        if y < best_y:
            area += (ref[0] - x) * (best_y - y)
            best_y = y
    return float(area)

def zdt1_runs():
    true_f1 = np.linspace(0, 1, 501)
    true_front = np.column_stack((true_f1, 1-np.sqrt(true_f1)))
    rows=[]
    plt.figure(figsize=(8,5))
    for seed in SEEDS:
        r = nsga2(zdt1_objectives, 30, 2, [(0.,1.)]*30,
                  population_size=100, generations=200, seed=seed)
        front = np.asarray([v.objectives for v in r.first_front])
        rows.append({'algorithm':'scratch','seed':seed,'igd':igd(front,true_front),
                     'hv':hypervolume_2d(front)})
        if seed == 0:
            plt.scatter(front[:,0],front[:,1],s=10,label='Scratch NSGA-II (seed 0)')
            # Compute approximate IGD by generation for a diagnostic convergence curve
            intervals = list(range(0,len(r.history),10))
            history_igd=[igd(np.asarray(r.history[t]),true_front) for t in intervals]
            fig2,ax2=plt.subplots(figsize=(8,5))
            ax2.plot(intervals,history_igd);ax2.set(xlabel='Generation',ylabel='IGD (all population points)',
                    title='ZDT1 convergence (seed 0)');fig2.tight_layout()
            fig2.savefig(OUT/'convergence.png',dpi=160);plt.close(fig2)
    plt.plot(true_f1,true_front[:,1],label='Analytic ZDT1 Pareto front')
    plt.legend();plt.xlabel('f1');plt.ylabel('f2');plt.title('Benchmark against analytic front')
    plt.tight_layout();plt.savefig(OUT/'zdt1_scatter.png',dpi=160);plt.close()
    # Optional library validation on identical budgets; direct differences are expected.
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.problems import get_problem
        problem=get_problem('zdt1')
        for seed in SEEDS:
            result=minimize(problem,NSGA2(pop_size=100),('n_gen',201),seed=seed,verbose=False)
            front=np.asarray(result.F)
            rows.append({'algorithm':'pymoo','seed':seed,'igd':igd(front,true_front),
                         'hv':hypervolume_2d(front)})
    except ImportError:
        print('pymoo missing; install with: python -m pip install pymoo')
    with (OUT/'zdt1_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['algorithm','seed','igd','hv']);writer.writeheader();writer.writerows(rows)
    for algo in sorted(set(row['algorithm'] for row in rows)):
        subset=[row for row in rows if row['algorithm']==algo]
        print(algo, 'IGD mean/std',np.mean([x['igd'] for x in subset]),np.std([x['igd'] for x in subset],ddof=1))
    return rows

def student_runs():
    # Feasible parameterisation: the decision variables are allocations in [0,15],
    # mapped to a <=30-hour schedule before the grade simulator is evaluated.
    results=[]
    for seed in SEEDS:
        r=nsga2(objectives_under_budget,5,3,[(0.,15.)]*5,
                population_size=100,generations=150,seed=seed)
        for idx,ind in enumerate(r.first_front):
            hours=np.array(repair_study_hours(ind.variables))
            p=predict(hours)
            results.append({'seed':seed,'solution':idx,'mean_grade':round(p.weighted_average,3),
                            'hours':round(p.study_hours,3),'shortfall':round(p.shortfall,3),
                            **{f'{name}_hours':round(float(hour),3) for name,hour in zip(MODULES,hours)}})
    fields=['seed','solution','mean_grade','hours','shortfall']+[f'{m}_hours' for m in MODULES]
    with (OUT/'student_pareto.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
    # Plot only seed 0 for interpretability, coloring by shortage below grade floor.
    one=[x for x in results if x['seed']==0]
    plt.figure(figsize=(8,5)); sc=plt.scatter([x['hours'] for x in one],[x['mean_grade'] for x in one],
                        c=[x['shortfall'] for x in one],cmap='viridis',s=20)
    plt.colorbar(sc,label='Total shortfall below 70');plt.xlabel('Study hours per week')
    plt.ylabel('Synthetic predicted weighted grade (%)')
    plt.title('Student strategies: NSGA-II Pareto trade-offs (seed 0)')
    plt.tight_layout();plt.savefig(OUT/'student_pareto.png',dpi=160);plt.close()
    print('Student feasible Pareto solutions:',len(results),'weekly budget:',WEEKLY_BUDGET)

if __name__=='__main__':
    zdt1_runs();student_runs()
