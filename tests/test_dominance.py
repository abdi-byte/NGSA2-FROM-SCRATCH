from nsga2.dominance import dominates


def test_dominance_for_minimisation():
    assert dominates([1, 2], [2, 3])
    assert not dominates([2, 3], [1, 2])


def test_non_dominated_tradeoff():
    assert not dominates([1, 4], [2, 3])
    assert not dominates([2, 3], [1, 4])


def test_equal_objectives_do_not_dominate():
    assert not dominates([1, 2], [1, 2])
