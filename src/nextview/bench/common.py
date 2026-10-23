"""Shared benchmark definitions: splits, the per-frame metric row and its CSV schema (spec 11.2, 11.5).

A run of one policy on one (camera, seed, actuator) produces one row per accepted frame ``n``
(1..budget) with the estimate at that point and its error against the GT camera, plus one
session record. Every policy is scored by the same function (``metric_row``), so the numbers
are comparable across policies.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..sim.lens import GTCamera, make_camera
from ..sim.poses import random_pose
from ..sim.truth import field_error_vectors, intrinsics_error
from ..tools.uncertainty import grid_covariances, grid_jacobian

__all__ = [
    "FAMILIES",
    "POLICIES",
    "A_FAMILY",
    "COV_LEVELS",
    "ROW_FIELDS",
    "Job",
    "split_jobs",
    "family_for",
    "heldout_views",
    "heldout_rms",
    "metric_row",
]

FAMILIES = ("normal", "wide", "tele", "webcam")
POLICIES = ("A", "B0", "B1", "B2", "B3", "B4", "A-gate", "A-outlier", "A-model", "A-nbv", "B2-k5")
A_FAMILY = ("A", "A-gate", "A-outlier", "A-model", "A-nbv")  # run through the agent loop
COV_LEVELS = (0.5, 0.8, 0.9, 0.95)
CHI2_2 = tuple(-2.0 * math.log(1.0 - q) for q in COV_LEVELS)  # chi-square(2) quantiles
HELDOUT_VIEWS = 20
HELDOUT_SALT = 0x4E1D

# synthetic-split nuisances: each family's own noise / JPEG / near-focus defocus, occluders and
# corruption (spec 6.3); glare on for the glossy webcam family only (spec 6.3 "glossy scenarios")
SPLIT_NUISANCES = {"normal": (), "wide": (), "tele": (), "webcam": ("glare",)}

DIST_COLS = tuple(f"d{i}" for i in range(8))
ROW_FIELDS = (
    "split", "camera", "family", "seed", "actuator", "policy", "n", "attempts", "views", "model",
    "fx", "fy", "cx", "cy", *DIST_COLS, "rms", "U95", "p_exceed_loop", "stable", "open_flags",
    "true_p95", "true_mean", "fx_rel_pct", "fy_rel_pct", "cx_err_px", "cy_err_px", "heldout_rms",
    "cov50", "cov80", "cov90", "cov95",
)


@dataclass(frozen=True)
class Job:
    split: str
    camera: int  # camera seed
    family: str
    seed: int  # policy seed (simulator streams, opening poses, NBV sampling)
    actuator: str
    policy: str
    budget: int = 30
    nuisances: tuple = ()
    variant: dict = field(default_factory=dict)  # failure-set extras: sag_m, approvals

    @property
    def key(self) -> tuple:
        return (self.split, self.camera, self.family, self.seed, self.actuator, self.policy)


def family_for(split: str, camera: int) -> str:
    if split == "dev":
        return FAMILIES[camera // 10]
    if split == "test":
        return FAMILIES[(camera - 1000) // 25]
    raise ValueError(f"no family rule for split {split!r}")


def split_jobs(split: str, policies=None, actuators=("robot", "human"), budget: int = 30) -> list[Job]:
    """All jobs of a pre-registered split (spec 11.2).

    dev: cameras 0-39 (10 per family, in family order), policy seed 0, A and B0.
    test: cameras 1000-1099 (25 per family), policy seeds 0-2, every policy.
    smoke: 2 cameras (normal 0, wide 10), seed 0, B0 and A, budget 8 (the test-suite smoke bench).
    failure: fisheye130 (robot), autofocus drift (webcam, human), warped board (normal, robot,
    RO denied), 20 cameras each, A only.
    """
    jobs: list[Job] = []
    if split == "dev":
        pol = policies or ("A", "B0")
        for cam in range(40):
            fam = family_for("dev", cam)
            for act in actuators:
                for p in pol:
                    jobs.append(Job("dev", cam, fam, 0, act, p, budget, SPLIT_NUISANCES[fam]))
    elif split == "test":
        pol = policies or POLICIES
        for cam in range(1000, 1100):
            fam = family_for("test", cam)
            for seed in (0, 1, 2):
                for act in actuators:
                    for p in pol:
                        jobs.append(Job("test", cam, fam, seed, act, p, budget, SPLIT_NUISANCES[fam]))
    elif split == "smoke":
        pol = policies or ("B0", "A")
        for cam, fam in ((0, "normal"), (10, "wide")):
            for p in pol:
                jobs.append(Job("smoke", cam, fam, 0, "robot", p, min(budget, 8), SPLIT_NUISANCES[fam]))
    elif split == "failure":
        pol = policies or ("A",)
        rng = np.random.default_rng(0xFA11)
        sags = rng.uniform(0.002, 0.004, 20)
        for k in range(20):
            for p in pol:
                jobs.append(Job("failure", 2000 + k, "fisheye130", 0, "robot", p, budget, (),
                                {"case": "fisheye130", "approvals": {"switch_model": "approve"}}))
                jobs.append(Job("failure", 2100 + k, "webcam", 0, "human", p, budget, ("autofocus_drift",),
                                {"case": "autofocus_drift", "approvals": {"switch_model": "approve"}}))
                jobs.append(Job("failure", 2200 + k, "normal", 0, "robot", p, budget, (),
                                {"case": "warped_board", "sag_m": float(sags[k]),
                                 "approvals": {"switch_model": "approve", "use_ro": "deny"}}))
    else:
        raise ValueError(f"unknown split {split!r}")
    return jobs


# ---- held-out GT views (metric 8) -------------------------------------------------------------

def heldout_views(cam: GTCamera, n: int = HELDOUT_VIEWS) -> list[tuple[np.ndarray, np.ndarray]]:
    """``n`` fixed full-board views per camera: (object points, noise-free GT corners).

    Poses come from the simulator sampler with an RNG seeded by the camera alone, so every
    policy and seed is scored on the same views.
    """
    from ..board import object_points

    rng = np.random.default_rng([int(cam.seed or 0), HELDOUT_SALT, zlib.crc32(cam.family.encode()) & 0xFFFF])
    obj = object_points()
    out = []
    for _ in range(n):
        rv, tv = random_pose(cam, rng)
        p, ok = cam.project_checked(obj, rv, tv)
        out.append((obj[ok], p[ok]))
    return out


def heldout_rms(views, K, dist) -> float:
    """RMS over all held-out corners, each view's pose refit (IPPE + LM) under (K, dist)."""
    K = np.asarray(K, np.float64)
    d = np.asarray(dist, np.float64).ravel()
    sse, npts = 0.0, 0
    for obj, img in views:
        try:
            n, rvs, tvs, errs = cv2.solvePnPGeneric(obj, img, K, d, flags=cv2.SOLVEPNP_IPPE)
            if n < 1:
                return math.nan
            j = int(np.argmin(np.asarray(errs).ravel()[:n]))
            rv, tv = cv2.solvePnPRefineLM(obj, img, K, d, rvs[j], tvs[j])
            p, _ = cv2.projectPoints(obj, rv, tv, K, d)
        except cv2.error:
            return math.nan
        e = p.reshape(-1, 2) - img
        sse += float(np.sum(e * e))
        npts += len(img)
    return math.sqrt(sse / npts) if npts else math.nan


# ---- the per-frame metric row -------------------------------------------------------------------

def _f(x, nd=6):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def metric_row(job: Job, n: int, attempts: int, n_views: int, cam: GTCamera, calib, held, *,
               grid=(32, 18), p_exceed_loop=None, stable=None, open_flags=None) -> dict:
    """One row: the estimate after accepted frame ``n`` and its error against ``cam`` (GT).

    ``calib`` is a ``CalibResult`` (with ``cov``) or None (fewer than 3 views: metrics empty).
    U95 is recomputed from ``calib.cov`` on the same grid for every policy. The coverage columns
    are the fractions of grid cells whose Mahalanobis error e^T Cov^-1 e is within the
    chi-square(2) quantile at q = 0.5, 0.8, 0.9, 0.95 (spec 11.5 item 5).
    """
    row = {k: None for k in ROW_FIELDS}
    row.update(split=job.split, camera=job.camera, family=job.family, seed=job.seed, actuator=job.actuator,
               policy=job.policy, n=int(n), attempts=int(attempts), views=int(n_views))
    if calib is None:
        return row
    K = np.asarray(calib.K, np.float64)
    d = np.asarray(calib.dist, np.float64).ravel()
    row.update(model=calib.model, fx=_f(K[0, 0], 4), fy=_f(K[1, 1], 4), cx=_f(K[0, 2], 4), cy=_f(K[1, 2], 4),
               rms=_f(calib.rms, 5))
    for i, v in enumerate(d[:8]):
        row[f"d{i}"] = _f(v, 8)
    ev = field_error_vectors(cam, K, d, grid).reshape(-1, 2)
    e = np.linalg.norm(ev, axis=1)
    fin = np.isfinite(e)
    if fin.any():
        row["true_p95"] = _f(np.percentile(e[fin], 95))
        row["true_mean"] = _f(e[fin].mean())
    ie = intrinsics_error(cam, K)
    row.update(fx_rel_pct=_f(ie["fx_rel_pct"]), fy_rel_pct=_f(ie["fy_rel_pct"]), cx_err_px=_f(ie["cx_err_px"]),
               cy_err_px=_f(ie["cy_err_px"]))
    row["heldout_rms"] = _f(heldout_rms(held, K, d))
    cov = getattr(calib, "cov", None)
    if cov is not None and np.all(np.isfinite(cov)):
        G = grid_jacobian(K, d, cam.size, grid)
        C = grid_covariances(G, np.asarray(cov, np.float64))
        s = np.sqrt(np.maximum(C[:, 0, 0] + C[:, 1, 1], 0.0))
        row["U95"] = _f(np.percentile(s, 95))
        ok = fin & (np.linalg.det(C) > 0)
        if ok.any():
            m = np.einsum("mi,mij,mj->m", ev[ok], np.linalg.inv(C[ok]), ev[ok])
            for name, c2 in zip(("cov50", "cov80", "cov90", "cov95"), CHI2_2):
                row[name] = _f(np.mean(m <= c2), 5)
    row["p_exceed_loop"] = _f(p_exceed_loop)
    row["stable"] = None if stable is None else int(bool(stable))
    row["open_flags"] = open_flags
    return row
