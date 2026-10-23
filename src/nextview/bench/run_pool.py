"""Real-pool evaluation on METRIC (spec 11.4): frame-selection policies over a fixed image pool.

    python -m nextview.bench.run_pool --out results/pool

Per camera (D455 real cameras 1-4, KinectV2 synthetic cameras 1-4), seed 0:

* Held-out frames: 1/3 of the full-board frames (at most 50), drawn with seed 0. The spec's 50
  held-out frames do not exist on this data: only 18-104 frames per camera show the whole board
  (``data_ext/metric/detection_report.json``). The pool is every other frame with a usable
  detection (>= 4 corners over 2 rows and 2 columns), partial boards included.
* Policies, each giving one frame sequence (budget K = its first K frames):
    P-rand  100 independent random draws of K pool frames per budget
    P-first the first K pool frames by filename
    P-tmpl  per B1 template pose in turn, the nearest unused pool frame (board centre, size,
            tilt vector), the pose read under the reference intrinsics, i.e. the physical pose
            an operator following the template would see
    P-cov   coverage greedy: most new 16 x 9 cells covered by the frame's detected corners
    P-agent the agent's gate (R0: whole board, R2: ESF edge rise <= 3 px), then NBV: the frame
            whose actual corners, at its solvePnP pose under the current estimate, give the
            largest predicted U95 drop (Schur-complemented information, spec 2.4-2.5); R7 drops
            outlier views (they still count against the budget). The first 3 frames are the
            first 3 gate-passing frames by filename.
  R1 (board area >= 4 %) and R3 (clipped pixels in the board) are not applied: the METRIC board
  covers 0.4-4.5 % of every frame by design of the dataset, and the KinectV2 renders clip the
  white squares at 255, so R3 fires on nearly every synthetic frame without any glare. Their
  counts are reported per camera.
* Every policy uses the same k5 estimator (``fit_k5``: OpenCV's start and a staged start,
  lowest RMS kept).
* Metrics per budget {6, 8, 10, 15, 20, 30} (budgets beyond what a policy can select are
  skipped): held-out reprojection RMS (each held-out pose refit with IPPE + LM under the
  estimate), fx/cx/cy against the reference, and the field difference (32 x 18 grid p95 / mean)
  against the reference: factory intrinsics for D455 (a REFERENCE, not ground truth), GT for
  KinectV2 (principal point converted to OpenCV's convention, ``data.metric.load_reference``).
  P-agent's percentile within the P-rand distribution is reported for held-out RMS and field p95.
* R8 check at the largest budget: 3-fold held-out RMS of rational vs k5 on P-agent's frames; a
  correct agent should not switch on these near-zero-distortion lenses.

Detections and object points are in square units of the 5 x 4 METRIC board (``Frame.board``);
the square size does not affect the intrinsics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

__all__ = ["fit_k5", "heldout_rms", "evaluate_camera", "main"]

BUDGETS = (6, 8, 10, 15, 20, 30)
N_RAND = 100
CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-10)
CAMERAS = {"d455": (1, 2, 3, 4), "kinect_synth": (1, 2, 3, 4)}
MAX_EDGE_RISE = 3.0
OUTLIER_K, OUTLIER_MIN = 3.0, 0.5


# ---- estimator -------------------------------------------------------------------------------------

def fit_k5(frames, size, model: str = "k5"):
    """k5 (or rational) calibration of pool frames: OpenCV's own start and a staged start
    (k1, k2 only, then all coefficients from it); the lowest-RMS solution is returned as
    ``(rms, K, dist, rvecs, tvecs, per_view)`` or None."""
    objs = [np.asarray(f.det.obj, np.float32) for f in frames]
    imgs = [np.asarray(f.det.corners, np.float32).reshape(-1, 1, 2) for f in frames]
    base = cv2.CALIB_RATIONAL_MODEL if model == "rational" else 0
    best = None
    starts = [(base, None, None)]
    try:
        r = cv2.calibrateCameraExtended(objs, imgs, size, None, None,
                                        flags=base | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
                                        | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6, criteria=CRITERIA)
        starts.append((base | cv2.CALIB_USE_INTRINSIC_GUESS, r[1], r[2]))
    except cv2.error:
        pass
    for flags, K0, d0 in starts:
        try:
            out = cv2.calibrateCameraExtended(objs, imgs, size, None if K0 is None else K0.copy(),
                                              None if d0 is None else d0.copy(), flags=flags, criteria=CRITERIA)
        except cv2.error:
            continue
        rms, K, d, rv, tv, _si, _se, pv = out
        if not (np.all(np.isfinite(K)) and np.all(np.isfinite(d)) and math.isfinite(rms)):
            continue
        if best is None or rms < best[0]:
            best = (float(rms), np.asarray(K, float), np.asarray(d, float).ravel(), [np.asarray(x, float).ravel() for x in rv],
                    [np.asarray(x, float).ravel() for x in tv], np.asarray(pv, float).ravel())
    return best


def heldout_rms(frames, K, dist) -> float:
    from nextview.data.pool_tools import estimate_pose

    sse, n = 0.0, 0
    for f in frames:
        try:
            p = estimate_pose(f, K, dist)
        except (ValueError, cv2.error):
            return math.nan
        sse += p.reproj_px ** 2 * f.n_corners
        n += f.n_corners
    return math.sqrt(sse / n) if n else math.nan


# ---- policies -----------------------------------------------------------------------------------

def _cells(frame, size, grid=(16, 9)) -> np.ndarray:
    W, H = size
    cols, rows = grid
    c = np.asarray(frame.det.corners, float)
    ix = np.clip((c[:, 0] / W * cols).astype(int), 0, cols - 1)
    iy = np.clip((c[:, 1] / H * rows).astype(int), 0, rows - 1)
    return np.unique(iy * cols + ix)


def p_cov(pool, size, k_max):
    counts = np.zeros(16 * 9, int)
    chosen, left = [], list(range(len(pool)))
    cells = [_cells(f, size) for f in pool]
    while left and len(chosen) < k_max:
        best = max(left, key=lambda i: (int(np.sum(counts[cells[i]] == 0)), pool[i].n_corners, -i))
        chosen.append(best)
        left.remove(best)
        np.add.at(counts, cells[best], 1)
    return [pool[i] for i in chosen]


def _pose_features(frame, K, dist, size):
    from nextview.data.pool_tools import estimate_pose

    W, H = size
    c = np.asarray(frame.det.corners, float)
    try:
        p = estimate_pose(frame, K, dist)
    except (ValueError, cv2.error):
        return None
    R, _ = cv2.Rodrigues(np.asarray(p.rvec, float).reshape(3, 1))
    n = R[:, 2] if R[2, 2] >= 0 else -R[:, 2]  # board normal in camera coordinates, facing away
    tilt = np.degrees(np.arccos(min(1.0, abs(n[2]))))
    tvec2 = np.array([n[0], n[1]]) / max(np.hypot(n[0], n[1]), 1e-9) * tilt
    xs = c[:, 0]
    width_frac = (xs.max() - xs.min()) / W * frame.board.squares[0] / max(frame.board.n_cols - 1, 1)
    return np.array([c[:, 0].mean() / W, c[:, 1].mean() / H, width_frac / 0.3, tvec2[0] / 30.0, tvec2[1] / 30.0])


def p_tmpl(pool, ref, size, k_max):
    from nextview.bench.policies import template_b1

    tmpl = []
    for (x, y, s, tilt, az, roll) in template_b1():
        a = math.radians(az)  # tilting about axis (sin az, -cos az, 0) moves the normal along -(cos az, sin az)
        tmpl.append(np.array([x, y, s / 0.3, -tilt * math.cos(a) / 30.0, -tilt * math.sin(a) / 30.0]))
    feats = [_pose_features(f, ref.K, ref.dist, size) for f in pool]
    idx = [i for i, v in enumerate(feats) if v is not None]
    chosen = []
    k = 0
    while len(chosen) < min(k_max, len(idx)):
        t = tmpl[k % len(tmpl)]
        left = [i for i in idx if i not in chosen]
        chosen.append(min(left, key=lambda i: (float(np.linalg.norm(feats[i] - t)), i)))
        k += 1
    return [pool[i] for i in chosen]


def _u95(K, dist, cov, size, grid=(32, 18)) -> float:
    from nextview.tools.uncertainty import grid_jacobian, u95_from_cov

    return u95_from_cov(grid_jacobian(K, dist, size, grid), cov)


def p_agent(pool, gate_ok, size, k_max, log=None):
    """Gate, then NBV over the gate-passing pool frames (robot mode, effort weight 0)."""
    from nextview.data.pool_tools import estimate_pose
    from nextview.tools.uncertainty import covariance, information_matrix, noise_variance, view_information

    cand = [i for i in range(len(pool)) if gate_ok[i]]
    chosen: list[int] = []
    used: list[int] = []
    dropped: list[int] = []
    steps = []
    for i in cand[:3]:
        chosen.append(i)
        used.append(i)
    while len(chosen) < k_max:
        fit = fit_k5([pool[i] for i in used], size)
        left = [i for i in cand if i not in chosen]
        if fit is None or not left:
            break
        rms, K, d, rv, tv, pv = fit
        objs = [pool[i].det.obj for i in used]
        F = information_matrix(K, d, rv, tv, objs)
        npts = int(sum(len(o) for o in objs))
        s2 = noise_variance(rms, npts, 9, len(used))
        if not math.isfinite(s2):
            s2 = max(rms * rms / 2.0, 1e-6)
        G_u = None
        from nextview.tools.uncertainty import grid_jacobian

        G_u = grid_jacobian(K, d, size, (32, 18))

        def u95(cov):
            var = np.einsum("mrp,pq,mrq->m", G_u, cov, G_u, optimize=True)
            return float(np.percentile(np.sqrt(np.maximum(var, 0.0)), 95))

        now = u95(covariance(F, s2))
        best, best_gain = None, -math.inf
        for i in left:
            try:
                p = estimate_pose(pool[i], K, d)
            except (ValueError, cv2.error):
                continue
            S = view_information(K, d, p.rvec, p.tvec, pool[i].det.obj)
            g = now - u95(covariance(F + S, s2))
            if g > best_gain:
                best, best_gain = i, g
        if best is None:
            break
        chosen.append(best)
        used.append(best)
        steps.append({"k": len(chosen), "frame": pool[best].name, "U95_before": round(now, 5), "pred_gain": round(best_gain, 5)})
        fit2 = fit_k5([pool[i] for i in used], size)
        if fit2 is not None and len(used) >= 4:
            pv = fit2[5]
            med = float(np.median(pv))
            mad = float(np.median(np.abs(pv - med)))
            out = [used[j] for j in range(len(used)) if pv[j] > med + OUTLIER_K * 1.4826 * mad and pv[j] > OUTLIER_MIN]
            if out and len(used) - len(out) >= 3:
                dropped += [(i, len(chosen)) for i in out]
                used = [i for i in used if i not in out]
    return [pool[i] for i in chosen], [pool[i] for i in used], dropped, steps


# ---- evaluation ------------------------------------------------------------------------------

def _metrics(fit, held, ref_cam, ref) -> dict:
    from nextview.sim.truth import field_error

    if fit is None:
        return {"ok": False}
    rms, K, d, *_ = fit
    fe = field_error(ref_cam, K, d)
    return {"ok": True, "rms": rms, "heldout_rms": heldout_rms(held, K, d), "field_p95": fe.p95, "field_mean": fe.mean,
            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "fx_rel_pct": float((K[0, 0] - ref.K[0, 0]) / ref.K[0, 0] * 100), "cx_err_px": float(K[0, 2] - ref.K[0, 2]),
            "cy_err_px": float(K[1, 2] - ref.K[1, 2]), "dist": [float(x) for x in d]}


def evaluate_camera(subset: str, camera: int, seed: int = 0, n_rand: int = N_RAND, log=print) -> dict:
    from nextview.data import pool_tools
    from nextview.data.metric import load_pool, load_reference
    from nextview.sim.lens import GTCamera

    t0 = time.perf_counter()
    frames = load_pool(camera, subset, detect=False)
    ref = load_reference(subset, camera)
    size = tuple(ref.image_size)
    ref_cam = GTCamera.pinhole(ref.K, ref.dist, size=size, family=f"metric_{subset}")
    det = [f for f in frames if f.det.ok]
    full = [f for f in det if f.n_corners >= f.board.n_corners]
    rng = np.random.default_rng([seed, camera, len(subset)])
    n_held = min(50, len(full) // 3)
    held_idx = set(rng.choice(len(full), size=n_held, replace=False).tolist())
    held = [f for i, f in enumerate(full) if i in held_idx]
    held_names = {f.name for f in held}
    pool = [f for f in det if f.name not in held_names]
    # gate measurements on every pool frame
    gate = []
    diag = {"R0_not_full_board": 0, "R2_blur": 0, "R2_unmeasurable": 0, "R1_would_reject": 0, "R3_would_reject": 0}
    for f in pool:
        if f.n_corners < f.board.n_corners:
            gate.append(False)
            diag["R0_not_full_board"] += 1
            continue
        q = pool_tools.assess_frame(f.image(cv2.IMREAD_GRAYSCALE), f)
        diag["R1_would_reject"] += int(not q.board_area_frac >= 0.04)
        diag["R3_would_reject"] += int(q.saturation_frac > 0.02)
        if not math.isfinite(q.edge_rise_px):
            gate.append(False)
            diag["R2_unmeasurable"] += 1
        elif q.edge_rise_px > MAX_EDGE_RISE:
            gate.append(False)
            diag["R2_blur"] += 1
        else:
            gate.append(True)
    k_max = min(max(BUDGETS), len(pool))
    seqs = {"P-first": pool[:k_max], "P-cov": p_cov(pool, size, k_max), "P-tmpl": p_tmpl(pool, ref, size, k_max)}
    chosen, used_final, dropped, steps = p_agent(pool, gate, size, k_max)
    seqs["P-agent"] = chosen
    res = {"subset": subset, "camera": camera, "image_size": list(size), "reference": ref.kind, "frames": len(frames),
           "detected": len(det), "full_board": len(full), "heldout": len(held), "pool": len(pool),
           "pool_gate_pass": int(sum(gate)), "gate_diagnostics": diag, "agent_dropped_R7": len(dropped),
           "agent_steps": steps, "budgets": {}}
    for K_b in BUDGETS:
        if K_b > len(pool):
            continue
        e = {}
        for name, seq in seqs.items():
            if len(seq) < K_b:
                continue
            sel = seq[:K_b]
            if name == "P-agent":  # the agent's calibration excludes views R7 dropped by then
                drop = {pool[i].name for i, k in dropped if k <= K_b}
                sel = [f for f in sel if f.name not in drop]
            e[name] = _metrics(fit_k5(sel, size), held, ref_cam, ref)
            e[name]["frames"] = len(sel)
        rr = np.random.default_rng([seed, camera, K_b, 0x5A])
        rand = []
        for _ in range(n_rand):
            idx = rr.choice(len(pool), size=K_b, replace=False)
            m = _metrics(fit_k5([pool[i] for i in sorted(idx)], size), held, ref_cam, ref)
            if m["ok"]:
                rand.append(m)
        if rand:
            hr = np.array([m["heldout_rms"] for m in rand], float)
            fp = np.array([m["field_p95"] for m in rand], float)
            e["P-rand"] = {"draws": len(rand), "heldout_rms": _band(hr), "field_p95": _band(fp),
                           "fx_rel_pct": _band(np.array([m["fx_rel_pct"] for m in rand]))}
            a = e.get("P-agent")
            if a and a.get("ok"):
                e["P-agent"]["heldout_percentile_in_rand"] = float(np.mean(hr[np.isfinite(hr)] <= a["heldout_rms"]) * 100)
                e["P-agent"]["field_p95_percentile_in_rand"] = float(np.mean(fp[np.isfinite(fp)] <= a["field_p95"]) * 100)
        res["budgets"][str(K_b)] = e
    # R8 check on the agent's frames at its largest budget
    kb = max(int(b) for b in res["budgets"]) if res["budgets"] else None
    if kb and len(chosen) >= 6:
        res["r8_check"] = _cv_rational_vs_k5(chosen[:kb], size)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    log(f"{subset}/camera{camera}: pool {len(pool)} (gate pass {sum(gate)}), held-out {len(held)}, "
        f"budgets {list(res['budgets'])}, {res['seconds']} s", flush=True)
    return res


def _band(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    return {"median": float(np.median(x)), "p05": float(np.percentile(x, 5)), "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)), "p95": float(np.percentile(x, 95))}


def _cv_rational_vs_k5(frames, size, folds: int = 3) -> dict:
    out = {}
    for model in ("k5", "rational"):
        sse, n = 0.0, 0
        for k in range(folds):
            train = [f for i, f in enumerate(frames) if i % folds != k]
            test = [f for i, f in enumerate(frames) if i % folds == k]
            fit = fit_k5(train, size, model)
            if fit is None:
                sse = math.nan
                break
            for f in test:
                h = heldout_rms([f], fit[1], fit[2])
                sse += h * h * f.n_corners
                n += f.n_corners
        out[f"cv_{model}"] = math.sqrt(sse / n) if n and math.isfinite(sse) else math.nan
    k5, rat = out["cv_k5"], out["cv_rational"]
    out["frames"] = len(frames)
    out["rational_gain"] = 1 - rat / k5 if math.isfinite(k5) and math.isfinite(rat) and k5 > 0 else math.nan
    out["would_switch_cv_criterion"] = bool(math.isfinite(out["rational_gain"]) and out["rational_gain"] >= 0.10)
    return out


def _ranges(d: dict) -> dict:
    """Min and max of each metric over P-agent, P-first, P-tmpl and P-cov, every camera and budget
    (with where they occur), and the share of gate-measured pool frames whose board covers < 4 %."""
    out = {}
    for m in ("field_p95", "heldout_rms", "fx_rel_pct"):
        vals = []
        for c in d["cameras"]:
            for b, e in c["budgets"].items():
                for pol in ("P-agent", "P-first", "P-tmpl", "P-cov"):
                    v = e.get(pol) or {}
                    if v.get("ok") and math.isfinite(v.get(m, math.nan)):
                        vals.append((float(v[m]), pol, c["camera"], int(b)))
        if vals:
            lo, hi = min(vals), max(vals)
            out[m] = {"min": lo[0], "min_at": list(lo[1:]), "max": hi[0], "max_at": list(hi[1:]), "values": len(vals)}
            if m == "fx_rel_pct":
                a = max(vals, key=lambda t: abs(t[0]))
                out[m].update(abs_max=abs(a[0]), abs_max_at=list(a[1:]))
    measured = sum(c["pool"] - c["gate_diagnostics"]["R0_not_full_board"] for c in d["cameras"])
    small = sum(c["gate_diagnostics"]["R1_would_reject"] for c in d["cameras"])
    out["board_area_lt_4pct"] = {"frames": small, "of_full_board_pool_frames": measured}
    return out


def pool_summary(out: Path) -> dict:
    """Per subset and budget: each policy's per-camera held-out RMS and field p95, and P-agent's
    percentile within P-rand (0 = better than every random draw)."""
    summ = {}
    for f in sorted(Path(out).glob("metric_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sub = d["subset"]
        e = {}
        budgets = sorted({int(b) for c in d["cameras"] for b in c["budgets"]})
        for b in budgets:
            row = {}
            for pol in ("P-agent", "P-first", "P-tmpl", "P-cov"):
                vals = {c["camera"]: c["budgets"][str(b)][pol] for c in d["cameras"]
                        if str(b) in c["budgets"] and c["budgets"][str(b)].get(pol, {}).get("ok")}
                row[pol] = {"cameras": sorted(vals), "heldout_rms": [round(v["heldout_rms"], 5) for v in vals.values()],
                            "field_p95": [round(v["field_p95"], 3) for v in vals.values()],
                            "fx_rel_pct": [round(v["fx_rel_pct"], 3) for v in vals.values()]}
            rnd = {c["camera"]: c["budgets"][str(b)]["P-rand"] for c in d["cameras"]
                   if str(b) in c["budgets"] and c["budgets"][str(b)].get("P-rand")}
            row["P-rand"] = {"cameras": sorted(rnd), "heldout_rms_median": [round(v["heldout_rms"]["median"], 5) for v in rnd.values()],
                             "field_p95_median": [round(v["field_p95"]["median"], 3) for v in rnd.values()]}
            pa = {c["camera"]: c["budgets"][str(b)].get("P-agent", {}) for c in d["cameras"] if str(b) in c["budgets"]}
            row["P-agent_percentile_in_P-rand"] = {
                "heldout_rms": {k: v.get("heldout_percentile_in_rand") for k, v in pa.items() if v.get("ok")},
                "field_p95": {k: v.get("field_p95_percentile_in_rand") for k, v in pa.items() if v.get("ok")}}
            e[str(b)] = row
        summ[sub] = {"reference": d["cameras"][0]["reference"] if d["cameras"] else None, "budgets": e,
                     "r8_check": {c["camera"]: c.get("r8_check") for c in d["cameras"]},
                     "ranges_non_random": _ranges(d)}
    (Path(out) / "summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    return summ


def _job(args):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)
    sub, cam, n_rand = args
    return evaluate_camera(sub, cam, n_rand=n_rand)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/pool")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-rand", type=int, default=N_RAND)
    ap.add_argument("--subsets", nargs="*", default=list(CAMERAS))
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args(argv)
    if a.summarize_only:
        pool_summary(Path(a.out))
        print(f"wrote {Path(a.out) / 'summary.json'}")
        return 0
    jobs = [(s, c, a.n_rand) for s in a.subsets for c in CAMERAS[s]]
    if a.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
            results = pool.map(_job, jobs)
    else:
        results = [_job(j) for j in jobs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    by = {}
    for r in results:
        by.setdefault(r["subset"], []).append(r)
    for sub, rs in by.items():
        (out / f"metric_{sub}.json").write_text(json.dumps({"subset": sub, "cameras": rs}, indent=1, default=float),
                                                encoding="utf-8")
    pool_summary(out)
    print(f"wrote {sorted(by)} and summary.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
