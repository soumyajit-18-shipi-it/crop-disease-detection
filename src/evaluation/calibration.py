from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import log_loss


def softmax_probabilities(logits, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    values -= values.max(axis=1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(axis=1, keepdims=True)


def calibration_bins(y_true, probabilities, bins: int = 15) -> list[dict]:
    targets = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    confidence = scores.max(axis=1)
    predicted = scores.argmax(axis=1)
    correct = predicted == targets
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (confidence >= lower if index == 0 else confidence > lower) & (confidence <= upper)
        count = int(mask.sum())
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "accuracy": float(correct[mask].mean()) if count else None,
                "confidence": float(confidence[mask].mean()) if count else None,
                "gap": (
                    float(abs(correct[mask].mean() - confidence[mask].mean()))
                    if count
                    else None
                ),
            }
        )
    return rows


def expected_calibration_error(y_true, probabilities, bins: int = 15) -> float:
    total = max(len(y_true), 1)
    return float(
        sum((row["count"] / total) * row["gap"] for row in calibration_bins(y_true, probabilities, bins) if row["count"])
    )


def calibration_metrics(y_true, probabilities, bins: int = 15) -> dict:
    targets = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    class_count = scores.shape[1]
    one_hot = np.eye(class_count, dtype=np.float64)[targets]
    return {
        "expected_calibration_error": expected_calibration_error(targets, scores, bins),
        "negative_log_likelihood": float(log_loss(targets, scores, labels=list(range(class_count)))),
        "brier_score": float(np.mean(np.sum((scores - one_hot) ** 2, axis=1))),
        "bins": calibration_bins(targets, scores, bins),
    }


def fit_temperature(logits, y_true, max_iterations: int = 75) -> dict:
    """Fit one positive temperature on validation logits by minimizing NLL."""
    values = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    targets = torch.as_tensor(y_true, dtype=torch.long, device="cpu")
    if values.ndim != 2 or values.shape[0] != targets.numel() or not targets.numel():
        raise ValueError("Temperature scaling requires non-empty logits shaped (N, C)")
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    criterion = torch.nn.CrossEntropyLoss()
    before_nll = float(criterion(values, targets).item())
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=int(max_iterations),
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = criterion(values / temperature, targets)
        loss.backward()
        return loss

    status = "optimized"
    try:
        optimizer.step(closure)
        temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
        after_nll = float(criterion(values / temperature, targets).item())
        if not np.isfinite(after_nll) or after_nll > before_nll + 1e-9:
            temperature = 1.0
            after_nll = before_nll
            status = "fallback_identity"
    except (RuntimeError, ValueError):
        temperature = 1.0
        after_nll = before_nll
        status = "fallback_identity"
    return {
        "method": "temperature_scaling",
        "temperature": temperature,
        "optimization_status": status,
        "objective": "cross_entropy_nll",
        "nll_before": before_nll,
        "nll_after": after_nll,
        "fitted_samples": int(targets.numel()),
        "max_iterations": int(max_iterations),
    }


def plot_reliability_diagram(
    y_true,
    logits,
    temperature: float,
    output_path: str | Path,
    bins: int = 15,
) -> Path:
    raw = calibration_bins(y_true, softmax_probabilities(logits), bins)
    calibrated = calibration_bins(y_true, softmax_probabilities(logits, temperature), bins)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", color="#64748b", label="Perfect calibration")
    for rows, label, color, marker in (
        (raw, "Before scaling", "#d97706", "o"),
        (calibrated, "After scaling", "#15803d", "s"),
    ):
        points = [(row["confidence"], row["accuracy"]) for row in rows if row["count"]]
        if points:
            x_values, y_values = zip(*points)
            ax.plot(x_values, y_values, marker=marker, linewidth=2, color=color, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy")
    ax.set_title("Reliability Diagram")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
