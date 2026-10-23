"""Fit the stop calibration on the DEV split (spec 2.6) and apply the tau rule.

    python -m nextview.bench.fit_stop --runs results/synth_dev/runs.csv --out config/stop_calibration.json

r = true field p95 / U95 is collected at every accepted-frame step of policy A with >= 6 views
(``min_views_stop``), separately per actuator and pooled over the four families; its 101
empirical quantiles go into ``config/stop_calibration.json``. Only A's steps are used: the
probability is consulted only on A's trajectories. B0's r distribution is reported for
comparison, not fitted.

tau rule (spec 2.6): with the fitted calibration, replay R11 on the DEV A trajectories of the
normal family under the robot actuator; if the median frames-to-stop exceeds 25 (no stop counts
as 31), tau = 0.75 px, else tau stays 0.5 px.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .analysis import load_runs, replay_stop, sessions_of
from .stats import quantiles101

__all__ = ["collect_r", "fit", "tau_rule", "main"]

TAU_DEFAULT = 0.5
TAU_FALLBACK = 0.75
TAU_RULE_FRAMES = 25
NO_STOP = 31


def collect_r(rows, policy: str = "A", min_views: int = 6) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {"robot": [], "human": []}
    for r in rows:
        if r["policy"] != policy or r["views"] is None or r["views"] < min_views:
            continue
        u, t = r["U95"], r["true_p95"]
        if math.isfinite(u) and u > 0 and math.isfinite(t):
            out[r["actuator"]].append(t / u)
    return {k: np.asarray(v) for k, v in out.items()}


def fit(rows, min_views: int = 6) -> dict:
    r = collect_r(rows, "A", min_views)
    return {act: quantiles101(v) for act, v in r.items()}


def stops(rows, quantiles: dict, tau: float, threshold: float, family=None, actuator=None, policy="A",
          min_views: int = 6) -> list[int]:
    out = []
    for (cam, fam, seed, act, pol), srows in sessions_of(rows, {policy}).items():
        if (family and fam != family) or (actuator and act != actuator):
            continue
        s = replay_stop(srows, quantiles[act], tau, threshold, min_views)
        out.append(NO_STOP if s is None else int(s["n"]))
    return out


def tau_rule(rows, quantiles: dict, threshold: float = 0.10, min_views: int = 6) -> dict:
    s = stops(rows, quantiles, TAU_DEFAULT, threshold, family="normal", actuator="robot", min_views=min_views)
    med = float(np.median(s))
    tau = TAU_FALLBACK if med > TAU_RULE_FRAMES else TAU_DEFAULT
    return {"tau_px": tau, "median_frames_to_stop_normal_robot_at_0p5": med, "stops": s,
            "reason": (f"DEV: median frames-to-stop for normal/A/robot at tau 0.5 px = {med:g} > {TAU_RULE_FRAMES}; "
                       f"tau set to {TAU_FALLBACK} (spec 2.6)") if tau == TAU_FALLBACK else
            (f"DEV: median frames-to-stop for normal/A/robot at tau 0.5 px = {med:g} <= {TAU_RULE_FRAMES}; "
             "tau kept at 0.5 px (spec 2.6)")}


def _describe(v: np.ndarray) -> dict:
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "p95": float(np.percentile(v, 95)), "frac_gt1": float(np.mean(v > 1.0))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True, help="stop calibration JSON to write")
    ap.add_argument("--report", default=None, help="DEV fit report JSON (default <runs dir>/stop_fit.json)")
    ap.add_argument("--threshold", type=float, default=0.10)
    a = ap.parse_args(argv)
    from .run_synthetic import file_sha256, portable_path

    runs = Path(a.runs)
    rows = load_runs(runs)
    if any(r["split"] != "dev" for r in rows):
        raise SystemExit("fit_stop uses the DEV split only")
    q = fit(rows)
    rule = tau_rule(rows, q, a.threshold)
    tau = rule["tau_px"]
    meta_path = runs.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    r_a = collect_r(rows, "A")
    r_b = collect_r(rows, "B0")
    by_family = {}
    for fam in ("normal", "wide", "tele", "webcam"):
        for act in ("robot", "human"):
            sub = [x for x in rows if x["family"] == fam]
            by_family[f"{fam}/{act}"] = _describe(collect_r(sub, "A")[act])
    stopcal = {
        "calibrated": True,
        "note": "Fitted on the DEV split only (cameras 0-39, policy seed 0, policy A) by nextview.bench.fit_stop.",
        "ratio": "true_p95 / U95, collected at every accepted-frame step of policy A with >= 6 views",
        "levels": 101,
        "fit_split": "dev",
        "fit_commit": (meta.get("code", {}).get("git", {}) or {}).get("head"),
        "fit_source_tree_sha256": meta.get("code", {}).get("source_tree_sha256"),
        "fit_runs_sha256": file_sha256(runs),
        "fit_samples": {k: int(len(v)) for k, v in r_a.items()},
        "fitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tau_px": tau,
        "quantiles": q,
    }
    out = Path(a.out)
    out.write_text(json.dumps(stopcal, indent=1), encoding="utf-8")
    # DEV in-sample stop behaviour at the frozen tau (descriptive)
    dev_stop = {}
    for act in ("robot", "human"):
        for fam in ("normal", "wide", "tele", "webcam", None):
            s = stops(rows, q, tau, a.threshold, family=fam, actuator=act)
            dev_stop[f"{fam or 'all'}/{act}"] = {"sessions": len(s), "stopped": int(sum(x < NO_STOP for x in s)),
                                                 "median_frames_to_stop": float(np.median(s))}
    report = {
        "runs": portable_path(runs), "runs_sha256": stopcal["fit_runs_sha256"], "stop_calibration": portable_path(out),
        "stop_calibration_sha256": file_sha256(out), "tau_rule": {k: v for k, v in rule.items() if k != "stops"},
        "tau_rule_stops": rule["stops"], "threshold": a.threshold,
        "r_A": {k: _describe(v) for k, v in r_a.items()}, "r_B0": {k: _describe(v) for k, v in r_b.items()},
        "r_A_by_family": by_family, "dev_stop_at_frozen_tau": dev_stop,
    }
    rep = Path(a.report) if a.report else runs.parent / "stop_fit.json"
    rep.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("tau_rule", "r_A", "r_B0", "dev_stop_at_frozen_tau")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
