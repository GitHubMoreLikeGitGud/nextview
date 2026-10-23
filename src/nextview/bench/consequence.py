"""Measurement consequence in mm (spec 11.5 item 10; graft G3).

    python -m nextview.bench.consequence --runs results/synth_test --out results/synth_test/consequence.json

For each run, the estimate (K, dist from ``runs.csv``) is used to measure 200 segments of 100 mm
on a fronto-parallel plane at Z = 0.5 m and 1.0 m, both endpoints in the outer 30 % of the image
(``sim.truth.planar_mm_error``: endpoints projected with the GT camera, back-projected with the
estimate onto the known plane). Per run: median and p95 of |L_hat - 100| mm. Reported per policy
and actuator at n = 10, at n = 30, and for the A family at its stop (R11 replayed at the frozen
tau and P_exceed < 0.10; a run without a stop is scored at n = 30 and counted).

Post-hoc diagnostic (descriptive, added after review), from the committed ``estimates.csv.gz``:

    python -m nextview.bench.consequence --runs results/synth_test --inversion-only

checks every back-projected segment endpoint by projecting it forward again with the same estimate.
An endpoint whose reprojection misses the original pixel by more than 0.1 px was not inverted
(``undistortPoints`` left the region where the fitted polynomial is invertible). Per policy,
actuator, n and family it reports the share of such endpoints and the length error over the pairs
whose two endpoints invert; written to ``<runs>/consequence_inversion.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analysis import replay_stop, sessions_of
from .files import load_runs
from .common import A_FAMILY, DIST_COLS

__all__ = ["run_errors", "table", "inversion_check", "main"]

ZS = (0.5, 1.0)
INVERT_TOL_PX = 0.1


def _row_at(srows, n):
    best = None
    for r in srows:
        if r["n"] <= n:
            best = r
    return best


def _kd(r):
    K = np.array([[r["fx"], 0.0, r["cx"]], [0.0, r["fy"], r["cy"]], [0.0, 0.0, 1.0]])
    nd = 8 if r.get("model") == "rational" else 5
    d = np.array([r[c] for c in DIST_COLS[:nd]], float)
    return K, d


def _errors_for(task):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import cv2

    cv2.setNumThreads(1)
    from ..sim.lens import make_camera
    from ..sim.truth import planar_mm_error

    family, camera, items = task
    cam = make_camera(family, camera)
    out = []
    for key, label, r in items:
        rec = {"key": key, "label": label, "n": r["n"]}
        if r is None or not math.isfinite(r.get("fx", math.nan)):
            out.append(rec)
            continue
        K, d = _kd(r)
        for Z in ZS:
            try:
                e = planar_mm_error(cam, K, d, Z)
                rec[f"median_mm@{Z}"] = e.median
                rec[f"p95_mm@{Z}"] = e.p95
            except (RuntimeError, ValueError, cv2.error):
                rec[f"median_mm@{Z}"] = math.nan
                rec[f"p95_mm@{Z}"] = math.nan
        out.append(rec)
    return out


def run_errors(rows, quantiles, tau, workers: int = 1) -> list[dict]:
    S = sessions_of(rows)
    by_cam = defaultdict(list)
    for key, srows in S.items():
        cam, fam, seed, act, pol = key
        for label, n in (("n10", 10), ("n30", 30)):
            r = _row_at(srows, n)
            if r is not None:
                by_cam[(fam, cam)].append((key, label, r))
        if pol in A_FAMILY:
            st = replay_stop(srows, quantiles[act], tau, 0.10)
            r = st if st is not None else _row_at(srows, 30)
            if r is not None:
                by_cam[(fam, cam)].append((key, "stop" if st is not None else "stop_none_n30", r))
    tasks = [(fam, cam, items) for (fam, cam), items in sorted(by_cam.items())]
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_errors_for, tasks)
    else:
        parts = [_errors_for(t) for t in tasks]
    return [x for p in parts for x in p]


def table(errs) -> dict:
    g = defaultdict(list)
    for e in errs:
        cam, fam, seed, act, pol = e["key"]
        label = "stop" if e["label"].startswith("stop") else e["label"]
        g[(pol, act, label)].append(e)
        g[(pol, act, label, fam)].append(e)
    out = {}
    for k, v in sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        ent = {"runs": len(v)}
        if k[2] == "stop":
            ent["runs_without_stop_scored_at_30"] = sum(1 for e in v if e["label"] == "stop_none_n30")
        for Z in ZS:
            med = np.array([e.get(f"median_mm@{Z}", math.nan) for e in v], float)
            p95 = np.array([e.get(f"p95_mm@{Z}", math.nan) for e in v], float)
            ent[f"Z{Z}"] = {"median_of_run_median_mm": float(np.nanmedian(med)) if np.isfinite(med).any() else None,
                            "p90_of_run_median_mm": float(np.nanpercentile(med, 90)) if np.isfinite(med).any() else None,
                            "median_of_run_p95_mm": float(np.nanmedian(p95)) if np.isfinite(p95).any() else None}
        out["/".join(str(x) for x in k)] = ent
    return out


def inversion_check(est_path, Z_m: float = 1.0, ns=(10, 30), policies=("A", "B0", "B1", "B2", "B3", "B4")) -> dict:
    """Share of segment endpoints that the estimate does not invert, and the length error without them."""
    import csv
    import gzip

    import cv2

    from ..sim.lens import UNDISTORT_CRITERIA, make_camera
    from ..sim.truth import planar_pairs

    with gzip.open(est_path, "rt", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["policy"] in policies and int(r["n"]) in ns and r["fx"] != ""]
    cams, pairs, g = {}, {}, defaultdict(list)
    for r in rows:
        ck = (r["family"], int(r["camera"]))
        if ck not in cams:
            cams[ck] = make_camera(*ck)
            pairs[ck] = planar_pairs(cams[ck], Z_m, 200, 100.0, 0.30, 0)
        K, d = _kd({k: (v if k == "model" else float(v) if v != "" else math.nan) for k, v in r.items()
                    if k in ("model", "fx", "fy", "cx", "cy") or k in DIST_COLS})
        ends, miss = [], []
        for u in pairs[ck]:
            nrm = cv2.undistortPoints(u.reshape(-1, 1, 2), K, d, criteria=UNDISTORT_CRITERIA).reshape(-1, 2)
            back, _ = cv2.projectPoints(np.column_stack([nrm, np.ones(len(nrm))]).reshape(-1, 1, 3),
                                        np.zeros(3), np.zeros(3), K, d)
            ends.append(Z_m * nrm)
            miss.append(np.linalg.norm(back.reshape(-1, 2) - u, axis=1))
        e = np.abs(np.linalg.norm(ends[1] - ends[0], axis=1) * 1000.0 - 100.0)
        bad = ~(np.isfinite(e) & (miss[0] <= INVERT_TOL_PX) & (miss[1] <= INVERT_TOL_PX))
        ok = e[~bad]
        rec = {"fail_share": float(np.mean(np.concatenate([miss[0], miss[1]]) > INVERT_TOL_PX)),
               "pair_fail_share": float(bad.mean()), "p95": float(np.percentile(e, 95)),
               "p95_invertible": float(np.percentile(ok, 95)) if ok.size >= 20 else math.nan,
               "median_invertible": float(np.median(ok)) if ok.size >= 20 else math.nan}
        for key in ((r["policy"], r["actuator"], f"n{r['n']}"), (r["policy"], r["actuator"], f"n{r['n']}", r["family"])):
            g[key].append(rec)
    out = {}
    for k, v in sorted(g.items()):
        col = lambda name: np.array([x[name] for x in v], float)  # noqa: E731
        out["/".join(k)] = {
            "runs": len(v), "runs_with_failed_endpoints": int(np.sum(col("fail_share") > 0)),
            "median_endpoint_fail_share": float(np.median(col("fail_share"))),
            "median_pair_fail_share": float(np.median(col("pair_fail_share"))),
            "median_of_run_p95_mm": float(np.median(col("p95"))),
            "median_of_run_p95_mm_invertible_pairs": float(np.nanmedian(col("p95_invertible"))),
            "median_of_run_median_mm_invertible_pairs": float(np.nanmedian(col("median_invertible"))),
        }
    return {"Z_m": Z_m, "tolerance_px": INVERT_TOL_PX, "segment_mm": 100, "pairs": 200, "outer_band": 0.30,
            "note": "post-hoc diagnostic (descriptive); an endpoint fails when its back-projection, projected again "
                    "with the same estimate, misses the pixel by more than the tolerance",
            "table": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--stopcal", default="config/stop_calibration.json")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--inversion-only", action="store_true",
                    help="only the post-hoc back-projection check, from <runs>/estimates.csv.gz")
    a = ap.parse_args(argv)
    d = Path(a.runs)
    if a.inversion_only:
        res = inversion_check(d / "estimates.csv.gz")
        out = Path(a.out) if a.out else d / "consequence_inversion.json"
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        for k, v in res["table"].items():
            if k.count("/") == 3 and k.endswith("/wide") and "/n10/" in k:
                print(f"{k:24s} endpoint fail share {v['median_endpoint_fail_share']:.3f}  run p95 "
                      f"{v['median_of_run_p95_mm']:.2f} mm, invertible pairs {v['median_of_run_p95_mm_invertible_pairs']:.2f} mm")
        print(f"wrote {out}")
        return 0
    rows = load_runs(d, need_estimates=True)
    sc = json.loads(Path(a.stopcal).read_text(encoding="utf-8"))
    if a.tau is None:
        from ..config import load_thresholds

        tau = float(load_thresholds().tau_px)
    else:
        tau = a.tau
    errs = run_errors(rows, sc["quantiles"], tau, a.workers)
    res = {"tau_px": tau, "segment_mm": 100, "pairs": 200, "outer_band": 0.30, "Z_m": list(ZS), "table": table(errs)}
    out = Path(a.out) if a.out else d / "consequence.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res["table"].items():
        if k.count("/") == 2:
            z1 = v["Z1.0"]
            print(f"{k:28s} runs {v['runs']:4d}  Z=1 m: median |dL| {z1['median_of_run_median_mm']:.3f} mm, "
                  f"run p95 {z1['median_of_run_p95_mm']:.3f} mm")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
