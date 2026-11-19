"""Profit-based threshold sweep (evaluation profit = total_pymnt - loan_amnt)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_profit_per_threshold(y_true, y_prob, loan_amnt, total_pymnt,
                                 sweep_range, sweep_step, min_roi: float = 0.0) -> pd.DataFrame:
    """Approve if p_default < t.
    expected_profit = sum(total_pymnt - loan_amnt - min_roi * loan_amnt) over approved loans.
    raw_profit = same without the hurdle. n_tp = approved non-defaults; n_fp = approved defaults."""
    y = np.asarray(y_true); p = np.asarray(y_prob)
    loan = np.asarray(loan_amnt, dtype=float)
    raw = np.asarray(total_pymnt, dtype=float) - loan
    adj = raw - min_roi * loan
    lo, hi = sweep_range
    thresholds = np.round(np.arange(lo, hi + sweep_step / 2, sweep_step), 10)
    rows = []
    for t in thresholds:
        m = p < t
        rows.append({
            "threshold": float(t),
            "expected_profit": float(adj[m].sum()),
            "raw_profit": float(raw[m].sum()),
            "n_approved": int(m.sum()),
            "n_tp": int((y[m] == 0).sum()),
            "n_fp": int((y[m] == 1).sum()),
        })
    return pd.DataFrame(rows)

def select_optimal_threshold(df: pd.DataFrame) -> float:
    """Threshold with maximum total profit (first one on ties)."""
    return float(df.loc[df["expected_profit"].idxmax(), "threshold"])


def plot_profit_curve(df: pd.DataFrame, optimal_t: float, out_path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(df["threshold"], df["expected_profit"] / 1e6)
    ax.axvline(optimal_t, color="r", ls="--", label=f"optimal t = {optimal_t:.2f}")
    ax.set_xlabel("approve if P(default) < threshold")
    ax.set_ylabel("total profit ($M)")
    ax.set_title("Profit vs threshold")
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)