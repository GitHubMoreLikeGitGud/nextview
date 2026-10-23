"""Statistics for the pre-registered analysis (spec 11.5): paired tests, bootstrap CIs, Holm,
censored frames-to-target curves, coverage and reliability.

All functions are deterministic for a given input and seed.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "wilcoxon_paired",
    "bootstrap_median_diff",
    "holm",
    "paired_endpoint",
    "km_curve",
    "km_median",
    "reliability",
    "quantiles101",
]


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test of a - b (``scipy.stats.wilcoxon``, zero differences dropped)."""
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    ok = np.isfinite(d)
    d = d[ok]
    nz = d[d != 0]
    if len(nz) == 0:
        return {"n": int(len(d)), "n_nonzero": 0, "statistic": None, "p": 1.0}
    res = wilcoxon(nz, alternative="two-sided")
    return {"n": int(len(d)), "n_nonzero": int(len(nz)), "statistic": float(res.statistic), "p": float(res.pvalue)}


def bootstrap_median_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 0,
                          level: float = 0.95) -> dict:
    """Median of the paired differences a - b and its percentile bootstrap CI (resampling pairs)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"median_diff": None, "ci": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {"median_diff": float(np.median(d)), "ci": [float(lo), float(hi)], "n": int(len(d)), "n_boot": n_boot,
            "level": level, "seed": seed}


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_endpoint(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Wilcoxon + bootstrap CI of the median paired difference, plus win/tie/loss counts (a < b = win)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    w = wilcoxon_paired(a[ok], b[ok])
    bs = bootstrap_median_diff(a[ok], b[ok], n_boot, seed)
    d = a[ok] - b[ok]
    return {**bs, "wilcoxon_p": w["p"], "wilcoxon_stat": w["statistic"], "n_nonzero": w["n_nonzero"],
            "median_a": float(np.median(a[ok])) if ok.any() else None, "median_b": float(np.median(b[ok])) if ok.any() else None,
            "wins": int(np.sum(d < 0)), "ties": int(np.sum(d == 0)), "losses": int(np.sum(d > 0))}


def km_curve(times: Sequence[float], events: Sequence[bool], t_max: int = 30) -> list[dict]:
    """Kaplan-Meier survival S(t) = P(not yet below target after t frames), right-censored at ``t_max``."""
    t = np.asarray(times, float)
    e = np.asarray(events, bool)
    out = []
    s = 1.0
    for k in range(0, t_max + 1):
        at_risk = int(np.sum(t >= k))
        d = int(np.sum((t == k) & e))
        if at_risk > 0:
            s *= 1.0 - d / at_risk
        out.append({"t": k, "S": round(s, 6), "at_risk": at_risk, "events": d})
    return out


def km_median(times: Sequence[float], events: Sequence[bool], t_max: int = 30) -> float | None:
    """First t with S(t) <= 0.5, or None when the curve stays above 0.5 (median not reached)."""
    for p in km_curve(times, events, t_max):
        if p["S"] <= 0.5:
            return float(p["t"])
    return None


def reliability(fractions: Sequence[Sequence[float]], levels: Sequence[float]) -> list[dict]:
    """Mean empirical coverage per nominal level from per-row coverage fractions (rows x levels)."""
    F = np.asarray(fractions, float)
    out = []
    for j, q in enumerate(levels):
        col = F[:, j]
        col = col[np.isfinite(col)]
        out.append({"nominal": q, "empirical_mean": float(col.mean()) if len(col) else None,
                    "empirical_median": float(np.median(col)) if len(col) else None, "rows": int(len(col))})
    return out


def quantiles101(x: Sequence[float]) -> list[float]:
    """101 empirical quantiles at levels 0.00, 0.01, ..., 1.00 (numpy 'linear' rule)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite values")
    q = np.quantile(x, np.linspace(0.0, 1.0, 101))
    return [float(v) for v in np.maximum.accumulate(q)]


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False
