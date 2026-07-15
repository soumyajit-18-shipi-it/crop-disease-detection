from src.evaluation.metrics import compute_accuracy


def test_compute_accuracy():
    assert compute_accuracy([0, 1, 1], [0, 0, 1]) == 2 / 3
