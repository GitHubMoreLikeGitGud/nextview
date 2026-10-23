"""Board pose from one detection (IPPE with an ambiguity ratio, then LM refinement)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..geometry import distance_of, tilt_of
from ..types import Detection, PoseResult

__all__ = ["estimate_pose", "solve_pose", "reprojection_rms"]


def _as_dist(dist) -> np.ndarray:
    return np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).ravel()


def reprojection_rms(obj: np.ndarray, img: np.ndarray, rvec, tvec, K, dist) -> float:
    """sqrt(mean ||e||^2) over the points, in pixels."""
    p, _ = cv2.projectPoints(
        np.asarray(obj, np.float64).reshape(-1, 1, 3),
        np.asarray(rvec, np.float64).reshape(3, 1),
        np.asarray(tvec, np.float64).reshape(3, 1),
        np.asarray(K, np.float64),
        _as_dist(dist),
    )
    e = p.reshape(-1, 2) - np.asarray(img, np.float64).reshape(-1, 2)
    return float(math.sqrt(np.mean(np.sum(e * e, axis=1))))


def solve_pose(obj: np.ndarray, img: np.ndarray, K, dist) -> tuple[np.ndarray, np.ndarray, float, float]:
    """IPPE on a planar target, best solution refined with LM.

    Returns ``(rvec, tvec, rms_px, ippe_ratio)``; ``ippe_ratio`` = err_best / err_second of the
    two IPPE solutions (1 = indistinguishable, 0 = only one solution). Raises ``ValueError``
    for fewer than 4 points or collinear board points (OpenCV would return a meaningless pose).
    """
    obj = np.asarray(obj, np.float64).reshape(-1, 3)
    img = np.asarray(img, np.float64).reshape(-1, 2)
    if len(obj) < 4:
        raise ValueError(f"pose needs at least 4 corners, got {len(obj)}")
    sv = np.linalg.svd(obj[:, :2] - obj[:, :2].mean(axis=0), compute_uv=False)
    if sv[1] <= 1e-6 * sv[0]:
        raise ValueError("pose needs corners that are not collinear on the board")
    K = np.asarray(K, np.float64)
    d = _as_dist(dist)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, d, flags=cv2.SOLVEPNP_IPPE)
    if n < 1:
        raise ValueError("solvePnPGeneric returned no solution")
    errs = np.asarray(errs, np.float64).ravel()
    order = np.argsort(errs[:n])
    best = int(order[0])
    ratio = 0.0
    if n >= 2:
        e_best, e_second = errs[order[0]], errs[order[1]]
        ratio = float(e_best / e_second) if e_second > 0 else 1.0
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, d, rvecs[best].copy(), tvecs[best].copy())
    rvec, tvec = rvec.ravel(), tvec.ravel()
    return rvec, tvec, reprojection_rms(obj, img, rvec, tvec, K, d), ratio


def estimate_pose(det: Detection, K, dist) -> PoseResult:
    """Pose of a detection. Apply the R0 gate first: corners must span at least 2 rows and 2
    columns (``solve_pose`` raises ``ValueError`` on collinear corners)."""
    rvec, tvec, rms, ratio = solve_pose(det.obj, det.corners, K, dist)
    return PoseResult(
        rvec=rvec,
        tvec=tvec,
        reproj_px=rms,
        ippe_ratio=ratio,
        tilt_deg=tilt_of(rvec),
        distance=distance_of(rvec, tvec),
    )
