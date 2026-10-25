import numpy as np
import pytest
from problems.student_study import predict, repair_study_hours, objectives_under_budget
from experiments.research_evaluation import igd, hypervolume_2d

def test_more_study_yields_better_synthetic_grades():
    assert (predict([5]*5).grades > predict([0]*5).grades).all()

def test_budget_repair():
    repaired=repair_study_hours([15]*5)
    assert sum(repaired)==pytest.approx(30)
    assert min(repaired)>=0 and max(repaired)<=15

def test_grade_metrics_three_objectives():
    assert len(objectives_under_budget([1]*5))==3

def test_invalid_allocation_is_rejected():
    with pytest.raises(ValueError): predict([30,0,0,0,0])

def test_igd_exact_match_zero():
    p=np.array([[0.,1.],[1.,0.]])
    assert igd(p,p)==pytest.approx(0.)

def test_hypervolume_two_disjoint_steps():
    assert hypervolume_2d([[0.2,0.8],[0.8,0.2]],(1.,1.))==pytest.approx(0.28)
