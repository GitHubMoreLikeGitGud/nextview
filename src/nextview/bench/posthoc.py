"""Post-hoc descriptive numbers for the TEST split -> ``posthoc.json`` (no hypothesis tests).

    python -m nextview.bench.posthoc --runs results/tmp/full/synth_test --out results/synth_test/posthoc.json

Written after the TEST run, in response to review; nothing here changes a pre-registered endpoint,
and ``summary.json`` stays the pre-registered ``summarize.py`` output. Contents:

* ``r7``: per policy and actuator, corrupted views accepted / dropped by R7 / cleared by an
  approved R6 restart, clean views dropped, recall and precision of R7's drops; R7's accuracy
  effect as A minus A-outlier at n = 10, run by run (same camera, seed and actuator; a
  difference within 0.01 px counts as equal, so the full and the committed 4-digit tables agree).
* ``r6_restarts``: approved R6 drift restarts per policy and actuator (TEST has no drift nuisance).
* ``no_estimate_at_n10``: runs without a finite estimate at n = 10 (dropped from their camera's
  seed median) and the A-vs-baseline human endpoints with A's dropped runs scored at their last
  estimate before n = 10 instead.
* ``human_vs_robot``: the paired median differences A minus B0 / B1 / B2 per actuator (the stated
  hypothesis "gains shrink under the human actuator").
* ``mm_wide_tail``: per-run p95 length error at n = 10 and Z = 1 m on the wide family.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .analysis import sessions_of, value_at
from .files import load_runs, load_sessions, runs_file
from .summarize import _paired

__all__ = ["posthoc", "main"]

ACTS = ("robot", "human")
A_POLICIES = ("A", "A-gate", "A-outlier", "A-model", "A-nbv")
EQUAL_PX = 0.01


def _r7(S, sessions) -> dict:
    out = {}
    for pol in A_POLICIES:
        for act in ACTS:
            ss = [s for s in sessions if s.get("policy") == pol and s.get("actuator") == act]
            ca = sum(s.get("corrupted_accepted", 0) for s in ss)
            cd = sum(s.get("corrupted_dropped", 0) for s in ss)
            cr = sum(s.get("corrupted_cleared_by_restart", 0) for s in ss)
            fd = sum(s.get("clean_dropped", 0) for s in ss)
            out[f"{pol}/{act}"] = {"corrupted_accepted": ca, "corrupted_dropped_by_R7": cd, "corrupted_cleared_by_R6_restart": cr,
                                   "corrupted_left_in_set": ca - cd - cr, "clean_dropped_by_R7": fd,
                                   "recall": cd / ca if ca else None, "precision": cd / (cd + fd) if cd + fd else None}
    eff = {}
    for act in ACTS:
        d = []
        for k, s in S.items():
            if k[4] != "A" or k[3] != act:
                continue
            a, b = value_at(s, 10), value_at(S.get(k[:4] + ("A-outlier",), []), 10)
            if math.isfinite(a) and math.isfinite(b):
                d.append(a - b)
        x = np.asarray(d, float)
        pr = _paired(S, "A", "A-outlier", act, "p95@10", n_boot=2000)
        eff[act] = {"runs": int(len(x)), "A_better": int(np.sum(x < -EQUAL_PX)), "A_worse": int(np.sum(x > EQUAL_PX)),
                    "equal_within_0.01px": int(np.sum(np.abs(x) <= EQUAL_PX)), "mean_diff_px": float(x.mean()),
                    "camera_median_diff_px": pr["median_diff"], "camera_median_ci": pr["ci"], "cameras": pr["cameras"],
                    "pooled_run_median_A": float(np.nanmedian([value_at(s, 10) for k, s in S.items() if k[4] == "A" and k[3] == act])),
                    "pooled_run_median_A-outlier": float(np.nanmedian([value_at(s, 10) for k, s in S.items()
                                                                       if k[4] == "A-outlier" and k[3] == act]))}
    return {"counts": out, "effect_A_minus_A-outlier_p95@10": eff}


def _no_estimate(S, sessions) -> dict:
    sess = {(s["camera"], s["seed"], s["actuator"], s["policy"]): s for s in sessions}
    runs = defaultdict(list)
    for k, s in S.items():
        if math.isfinite(value_at(s, 10)):
            continue
        rec = sess.get((k[0], k[2], k[3], k[4]), {})
        runs[f"{k[4]}/{k[3]}"].append({"camera": k[0], "seed": k[2], "end": (rec.get("end") or {}).get("rule"),
                                       "n_final": rec.get("n_final"),
                                       "r6_restart_at_n": [a["n"] for a in rec.get("approvals") or []
                                                           if a["rule"] == "R6_DRIFT" and a["decision"] == "approve"]})
    out = {"runs": dict(sorted(runs.items())), "sensitivity": {}}
    filled = dict(S)
    for act in ACTS:
        for e in runs.get(f"A/{act}", []):
            k = next(k for k in S if k[4] == "A" and k[3] == act and k[0] == e["camera"] and k[2] == e["seed"])
            rr = [dict(r) for r in S[k]]
            last = None
            for r in rr:
                if r["n"] <= 10 and math.isfinite(r["true_p95"]):
                    last = r["true_p95"]
            for r in rr:
                if r["n"] == 10:
                    r["true_p95"] = math.nan if last is None else last
            filled[k] = rr
            e["last_estimate_before_n10"] = last
    for base in ("B0", "B2", "B1"):
        for act in ACTS:
            a = _paired(S, "A", base, act, "p95@10")
            b = _paired(filled, "A", base, act, "p95@10")
            out["sensitivity"][f"A-{base}/{act}"] = {
                "as_reported": {"median_diff": a["median_diff"], "ci": a["ci"], "wilcoxon_p": a["wilcoxon_p"]},
                "A_dropped_runs_at_last_estimate": {"median_diff": b["median_diff"], "ci": b["ci"], "wilcoxon_p": b["wilcoxon_p"]}}
    return out


def posthoc(runs_dir: Path, test_dir: Path) -> dict:
    rows = load_runs(runs_dir)
    sessions = load_sessions(runs_dir)
    S = sessions_of(rows)
    r6 = Counter()
    for s in sessions:
        for a in s.get("approvals") or []:
            if a["rule"] == "R6_DRIFT":
                r6[f"{s['policy']}/{s['actuator']}:{a['decision']}"] += 1
    out = {"runs_table": runs_file(runs_dir).name, "r7": _r7(S, sessions), "r6_restarts": dict(sorted(r6.items())),
           "no_estimate_at_n10": _no_estimate(S, sessions)}
    sm = json.loads((test_dir / "summary.json").read_text(encoding="utf-8"))
    hv = {}
    for e in sm["primary_endpoints"]:
        if e["metric"] == "p95@10":
            hv.setdefault(e["comparison"], {})[e["actuator"]] = {"median_diff": e["median_diff"], "ci": e["ci"]}
    out["human_vs_robot"] = hv
    cp = test_dir / "consequence.json"
    if cp.exists():
        t = json.loads(cp.read_text(encoding="utf-8"))["table"]
        out["mm_wide_tail"] = {f"{pol}/{act}": {"median_of_run_p95_mm": t.get(f"{pol}/{act}/n10/wide", {}).get("Z1.0", {}).get("median_of_run_p95_mm"),
                                                "median_of_run_median_mm": t.get(f"{pol}/{act}/n10/wide", {}).get("Z1.0", {}).get("median_of_run_median_mm")}
                               for pol in ("A", "B0", "B1", "B2", "B3", "B4") for act in ACTS}
        out["mm_wide_tail"]["_"] = "n = 10, plane Z = 1.0 m, 100 mm segments in the outer 30 % of the image, wide family"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/synth_test", help="directory with the runs table and sessions")
    ap.add_argument("--test", default="results/synth_test", help="directory with summary.json and consequence.json")
    ap.add_argument("--out", default="results/synth_test/posthoc.json")
    a = ap.parse_args(argv)
    res = posthoc(Path(a.runs), Path(a.test))
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
