from src.training.benchmark import _select_best


def _metrics(f1, ece, speed, size):
    return {
        "validation": {"macro": {"f1": f1}, "calibration": {"expected_calibration_error": ece}},
        "onnx_parity": {"passed": True, "onnx_cpu_inference": {"images_per_second": speed}},
        "onnx_size_bytes": size,
    }


def test_selection_balances_quality_calibration_speed_and_size():
    selected, scores = _select_best({
        "quality": _metrics(0.95, 0.10, 50, 100),
        "balanced": _metrics(0.93, 0.03, 90, 60),
        "small": _metrics(0.85, 0.05, 100, 40),
    })
    assert selected == "balanced"
    assert set(scores) == {"quality", "balanced", "small"}


def test_failed_onnx_parity_is_not_eligible():
    bad = _metrics(0.99, 0.01, 100, 10)
    bad["onnx_parity"]["passed"] = False
    selected, _ = _select_best({"bad": bad, "good": _metrics(0.8, 0.1, 10, 100)})
    assert selected == "good"
