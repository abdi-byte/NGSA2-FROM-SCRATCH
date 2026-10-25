"""Parameter sensitivity on ZDT1, equal number of generations (not evaluations).
Run after merging extension: python experiments/sensitivity.py
"""
import csv
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nsga2.algorithm import nsga2
from problems.zdt1 import zdt1_objectives
from experiments.research_evaluation import igd, hypervolume_2d, OUT

REF_X=np.linspace(0.,1.,501)
REF=np.column_stack([REF_X,1.-np.sqrt(REF_X)])
CONFIGS=[('baseline',100,0.9,20.),('smaller_population',50,0.9,20.),
         ('larger_population',150,0.9,20.),('lower_crossover',100,0.6,20.),
         ('higher_mutation_spread',100,0.9,5.)]

def main():
    records=[]
    for name,pop,cross,mutation_eta in CONFIGS:
        for seed in range(5):
            result=nsga2(zdt1_objectives,30,2,[(0,1)]*30,population_size=pop,
                         generations=100,seed=seed,crossover_probability=cross,
                         mutation_eta=mutation_eta)
            front=np.asarray([i.objectives for i in result.first_front])
            records.append(dict(config=name,seed=seed,population_size=pop,
                                function_evaluations=pop*101,
                                igd=igd(front,REF),hv=hypervolume_2d(front)))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'sensitivity.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for name,*_ in CONFIGS:
        scores=[row['igd'] for row in records if row['config']==name]
        print(f'{name}: mean IGD={np.mean(scores):.4f}, sd={np.std(scores,ddof=1):.4f}')
    print('WARNING: different population sizes have unequal evaluation budgets; do not claim controlled performance superiority.')

if __name__=='__main__':main()
