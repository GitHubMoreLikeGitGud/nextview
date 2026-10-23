"""DEV-split blur curve: corner error against the measured ESF edge rise (spec 2.9 ``max_edge_rise_px``).

    python -m nextview.bench.blur_curve --out results/dev/blur_curve.json

For every DEV camera (seeds 0-39, the four in-scope families) 6 fixed poses are rendered exactly
(no actuator error), blurred with a motion PSF of length L in {0, 2, 4, 6, 8, 10} px (random
direction) or a defocus disk of radius r in {1, 2, 3, 4} px, given the family's sensor noise and
JPEG quality, then detected and assessed with the agent's own tools. Corner error is the median
distance between detected corners and the GT projection. Frames are binned by measured rise
(0.5 px bins); the freeze rule (spec 2.9) is: the threshold is the lower edge of the first bin,
above the minimum-error bin, whose median corner error is at least 2x the error of the
unblurred frames; bins with fewer than 20 frames are not used. The rule's outcome is recorded
whatever it is.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

__all__ = ["measure", "freeze_rule", "main"]

MOTION_L = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
DEFOCUS_R = (1.0, 2.0, 3.0, 4.0)
POSES_PER_CAMERA = 6
BIN_PX = 0.5
MIN_BIN = 20


def _camera_rows(cam_seed: int) -> list[dict]:
    import cv2

    cv2.setNumThreads(1)
    from nextview.bench.common import family_for
    from nextview.board import object_points
    from nextview.sim.nuisance import add_noise, apply_psf, defocus_kernel, jpeg_roundtrip, motion_kernel, to_uint8
    from nextview.sim.poses import random_pose
    from nextview.sim.scenarios import make_simulator
    from nextview.tools.detect import detect_board
    from nextview.tools.quality import assess_frame

    fam = family_for("dev", cam_seed)
    sim = make_simulator(fam, "robot", cam_seed, 0)
    cam = sim.camera
    rng = np.random.default_rng([cam_seed, 0xB1A5])
    obj = object_points()
    rows = []
    for k in range(POSES_PER_CAMERA):
        rv, tv = random_pose(cam, rng, size_range=(0.35, 0.7), max_tilt_deg=45.0)
        clean = sim.renderer(cam).render(rv, tv).astype(np.float32)
        gt = cam.project(obj, rv, tv)
        levels = [("motion", L) for L in MOTION_L] + [("defocus", r) for r in DEFOCUS_R]
        for kind, v in levels:
            ang = float(rng.uniform(0.0, 180.0))
            ker = motion_kernel(v, ang) if kind == "motion" else defocus_kernel(v)
            img = apply_psf(clean, ker)
            img = add_noise(img, float(cam.noise_sigma), np.random.default_rng([cam_seed, k, int(10 * v), 3]))
            img = jpeg_roundtrip(to_uint8(img), cam.jpeg_quality)
            det = detect_board(img)
            row = {"camera": cam_seed, "family": fam, "pose": k, "kind": kind, "level": v, "detected": bool(det.ok),
                   "n_corners": int(len(det.ids)), "rise": None, "corner_err": None}
            if det.ok and len(det.ids) >= 12:
                q = assess_frame(img, det)
                e = np.linalg.norm(det.corners - gt[det.ids], axis=1)
                row["rise"] = None if not math.isfinite(q.edge_rise_px) else round(float(q.edge_rise_px), 4)
                row["corner_err"] = round(float(np.median(e)), 5)
            rows.append(row)
    return rows


def measure(cameras=range(40), workers: int = 1) -> list[dict]:
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_camera_rows, list(cameras))
    else:
        parts = [_camera_rows(c) for c in cameras]
    return [r for p in parts for r in p]


def freeze_rule(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["rise"] is not None and r["corner_err"] is not None]
    sharp = [r["corner_err"] for r in ok if r["kind"] == "motion" and r["level"] == 0.0]
    sharp_err = float(np.median(sharp))
    rises = np.array([r["rise"] for r in ok])
    errs = np.array([r["corner_err"] for r in ok])
    edges = np.arange(0.0, float(np.ceil(rises.max() / BIN_PX) * BIN_PX) + BIN_PX, BIN_PX)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (rises >= lo) & (rises < hi)
        if sel.sum() == 0:
            continue
        bins.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": int(sel.sum()),
                     "median_err": round(float(np.median(errs[sel])), 5),
                     "p90_err": round(float(np.percentile(errs[sel], 90)), 5),
                     "median_rise": round(float(np.median(rises[sel])), 4)})
    use = [b for b in bins if b["n"] >= MIN_BIN]
    threshold = None
    if use:
        kmin = int(np.argmin([b["median_err"] for b in use]))
        for b in use[kmin:]:
            if b["median_err"] >= 2.0 * sharp_err:
                threshold = b["lo"]
                break
    det_rate = {}
    for kind, levels in (("motion", MOTION_L), ("defocus", DEFOCUS_R)):
        for v in levels:
            sel = [r for r in rows if r["kind"] == kind and r["level"] == v]
            det_rate[f"{kind}_{v:g}"] = {
                "frames": len(sel), "detected_ge12": sum(1 for r in sel if r["corner_err"] is not None),
                "median_rise": None if not any(r["rise"] for r in sel) else
                round(float(np.median([r["rise"] for r in sel if r["rise"] is not None])), 3),
                "median_err": None if not any(r["corner_err"] for r in sel) else
                round(float(np.median([r["corner_err"] for r in sel if r["corner_err"] is not None])), 5)}
    return {"sharp_median_err_px": round(sharp_err, 5), "rule": "first 0.5 px rise bin (n >= 20) above the minimum-error bin "
            "with median corner error >= 2x the unblurred median", "threshold_px": threshold, "bins": bins,
            "by_level": det_rate, "frames": len(rows), "measured_frames": len(ok)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args(argv)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    rows = measure(range(40), a.workers)
    res = freeze_rule(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("bins",)}, indent=1))
    for b in res["bins"]:
        print(f"rise [{b['lo']:4.1f},{b['hi']:4.1f}) n={b['n']:4d} median err {b['median_err']:.4f} px p90 {b['p90_err']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
