"""Platt calibration on a frozen model, ECE, reliability diagram."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator


def calibrate_model(base_model, X_cal, y_cal, method: str = "sigmoid"):
    """Fit calibrator on cal_A only; base model stays frozen (no cv='prefit')."""
    cal = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
    return cal.fit(X_cal, y_cal)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-width bins; weighted mean of |observed rate - mean predicted|."""
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    idx = np.clip((y_prob * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def plot_reliability_diagram(y_true, p_before, p_after, out_path, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for p, name in [(p_before, "before"), (p_after, "after")]:
        frac, mean_p = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", label=name)
    ax.set_xlabel("mean predicted P(default)")
    ax.set_ylabel("observed default rate")
    ax.legend()
    ax.set_title("Reliability diagram")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)