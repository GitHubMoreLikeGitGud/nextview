"""Reading benchmark outputs and the per-session derived quantities used by every analysis.

* ``load_runs``: ``runs.csv`` -> typed rows; ``sessions_of``: rows grouped per job, sorted by n.
* ``value_at``: a metric after ``n`` accepted frames, carrying the last value forward when the
  run ended earlier (an abstention or the attempt cap) and NaN before the first calibration.
* ``frames_to``: frames until the true field p95 first drops below a target (censored).
* ``replay_stop``: the R11 stop rule (``agent.policy.decide``) replayed on a recorded trajectory
  for any stop calibration, tau and P_exceed threshold. Trajectories do not depend on the stop
  rule (the bench continues after a stop), so this is exact; ``run_synthetic`` records the
  in-loop stop too and ``summarize`` checks that the two agree.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import ROW_FIELDS

__all__ = ["load_runs", "load_sessions", "sessions_of", "value_at", "frames_to", "replay_stop", "p_exceed"]

_STR = {"split", "family", "actuator", "policy", "model"}
_INT = {"camera", "seed", "n", "attempts", "views", "stable", "open_flags"}


def load_runs(path: Path | str) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {}
            for k in ROW_FIELDS:
                v = r.get(k, "")
                if k in _STR:
                    row[k] = v or None
                elif v == "":
                    row[k] = None if k in _INT else math.nan
                elif k in _INT:
                    row[k] = int(float(v))
                else:
                    row[k] = float(v)
            out.append(row)
    return out


def load_sessions(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sessions_of(rows: list[dict], policies=None) -> dict[tuple, list[dict]]:
    """{(camera, family, seed, actuator, policy): rows sorted by n}."""
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if policies is not None and r["policy"] not in policies:
            continue
        g[(r["camera"], r["family"], r["seed"], r["actuator"], r["policy"])].append(r)
    for v in g.values():
        v.sort(key=lambda r: r["n"])
    return dict(g)


def value_at(srows: list[dict], n: int, field: str = "true_p95") -> float:
    """Metric after ``n`` accepted frames (last value carried forward if the run ended before n)."""
    best = None
    for r in srows:
        if r["n"] <= n:
            best = r
        else:
            break
    if best is None:
        return math.nan
    v = best.get(field)
    return math.nan if v is None else float(v)


def frames_to(srows: list[dict], target: float = 1.0, t_max: int = 30) -> tuple[int, bool]:
    """(first n with true field p95 < target, True) or (t_max, False) when never reached (censored)."""
    for r in srows:
        v = r.get("true_p95")
        if v is not None and math.isfinite(v) and v < target and r["n"] <= t_max:
            return int(r["n"]), True
    return t_max, False


def p_exceed(u95: float, quantiles, tau: float) -> float:
    """P_exceed = 1 - F_r(tau / U95), F_r interpolated from 101 quantiles (as ``config.StopCalibration``)."""
    if not math.isfinite(u95):
        return 1.0
    if u95 <= 0:
        return 0.0
    q = np.asarray(quantiles, float)
    return 1.0 - float(np.interp(tau / u95, q, np.linspace(0.0, 1.0, len(q)), left=0.0, right=1.0))


def replay_stop(srows: list[dict], quantiles, tau: float, threshold: float, min_views: int = 6) -> dict | None:
    """First row where R11 would fire: views >= min_views, P_exceed < threshold, prequential
    stability and no open flags. Returns that row (plus its P_exceed), or None (no stop)."""
    for r in srows:
        if r.get("views") is None or r["views"] < min_views or not r.get("stable") or (r.get("open_flags") or 0) > 0:
            continue
        u = r.get("U95")
        if u is None or not math.isfinite(u):
            continue
        pe = p_exceed(u, quantiles, tau)
        if pe < threshold:
            return {**r, "p_exceed_replay": pe}
    return None
