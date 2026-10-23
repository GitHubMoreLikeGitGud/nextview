"""Calibration wrappers (k5, rational, release-object RO) and held-out error measures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from ..board import N_CORNERS, SQUARES, object_points
from ..types import CalibResult
from ._views import board_points, full_board_views, view_arrays, views_used
from .pose import reprojection_rms, solve_pose
from .uncertainty import covariance, information_matrix, noise_variance

__all__ = [
    "MODELS",
    "ROCalibResult",
    "N_DIST",
    "RO_FIXED_POINT",
    "view_arrays",
    "full_board_views",
    "views_used",
    "calibrate",
    "heldout_view_error",
    "prequential_rms",
    "cv_heldout_rms",
    "ro_cv_gain",
]

MODELS = ("k5", "rational", "ro")
N_DIST = {"k5": 5, "rational": 8, "ro": 5}
RO_FIXED_POINT = SQUARES[0] - 2  # index of the top-right inner corner
# Final fits: wide-angle views that reach the image edge need a few hundred LM iterations from a
# poor start. Fits that only provide a starting point stop earlier.
CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-10)
INIT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-6)


@dataclass(eq=False)
class ROCalibResult(CalibResult):
    """``CalibResult`` of the release-object model, with the refined board corners."""

    new_obj: np.ndarray | None = None  # (40, 3) metres, board frame


def _flags(model: str) -> int:
    if model == "rational":
        return cv2.CALIB_RATIONAL_MODEL
    if model in ("k5", "ro"):
        return 0
    raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")


def _sorted_full(view) -> np.ndarray:
    ids, c = view_arrays(view)
    order = np.argsort(ids)
    return c[order]


_K1_ONLY = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST


def _run(objs, imgs, size, flags: int, K0=None, d0=None, criteria=CRITERIA):
    if K0 is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        K0 = np.array(K0, np.float64)
        d0 = np.array(d0, np.float64).reshape(1, -1)
    try:
        out = cv2.calibrateCameraExtended(objs, imgs, size, K0, d0, flags=flags, criteria=criteria)
    except cv2.error:
        return None
    return out if np.isfinite(out[0]) else None


def _run_ro(objs, imgs, size, K0=None, d0=None):
    flags = 0
    if K0 is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        K0 = np.array(K0, np.float64)
        d0 = np.array(d0, np.float64).reshape(1, -1)
    try:
        out = cv2.calibrateCameraROExtended(objs, imgs, size, RO_FIXED_POINT, K0, d0, flags=flags, criteria=CRITERIA)
    except cv2.error:
        return None
    return out if np.isfinite(out[0]) else None


def _staged_init(objs, imgs, size):
    """(K, k5) from a k1-only fit refined to k5: a start that survives strong wide-angle distortion."""
    s1 = _run(objs, imgs, size, _K1_ONLY, criteria=INIT_CRITERIA)
    if s1 is None:
        return None
    s2 = _run(objs, imgs, size, 0, s1[1], s1[2].ravel()[:5], criteria=INIT_CRITERIA)
    s = s2 if s2 is not None else s1
    return s[1], s[2].ravel()[:5]


def _central_init(objs, imgs, size):
    """(K, k5) fitted on the more central half of the views, where distortion is milder."""
    if len(objs) < 6:
        return None
    c = np.array([(size[0] - 1) / 2.0, (size[1] - 1) / 2.0])
    rad = [float(np.mean(np.linalg.norm(np.asarray(i, np.float64).reshape(-1, 2) - c, axis=1))) for i in imgs]
    sub = np.argsort(rad)[: max(3, (len(objs) + 1) // 2)]
    s = _run([objs[i] for i in sub], [imgs[i] for i in sub], size, 0, criteria=INIT_CRITERIA)
    return None if s is None else (s[1], s[2].ravel()[:5])


def _pad(dist, n: int) -> np.ndarray:
    d = np.zeros(n)
    src = np.asarray(dist, np.float64).ravel()[:n]
    d[: src.size] = src
    return d


def _init_guess(init, n: int):
    if init is None:
        return None
    if isinstance(init, CalibResult):
        K0, d0 = init.K, init.dist
    else:
        K0, d0 = init
    K0 = np.asarray(K0, np.float64)
    if K0.shape != (3, 3) or not np.all(np.isfinite(K0)):
        return None
    return K0, _pad(d0, n)


def _init_rms(init) -> float | None:
    if isinstance(init, CalibResult) and np.isfinite(init.rms):
        return float(init.rms)
    return None


# A warm start is accepted without the multi-start search when its RMS is within this much of
# the RMS of the calibration it started from (one view more or fewer moves the RMS little; a
# jump means the start fell into a different basin).
WARM_ACCEPT_RATIO = 1.2
WARM_ACCEPT_PX = 0.05


def _raw_starts(objs, imgs, size) -> list[tuple[np.ndarray, np.ndarray]]:
    """(K, k5) starting points: staged (k1 -> k5) and central-half fits."""
    return [st for st in (_staged_init(objs, imgs, size), _central_init(objs, imgs, size)) if st is not None]


def _k5_converged(objs, imgs, size, raw):
    """(K, k5) of the best full k5 fit from the raw starts and OpenCV's own start, or None."""
    cands = [_run(objs, imgs, size, 0, K0, _pad(d0, 5)) for K0, d0 in raw]
    cands.append(_run(objs, imgs, size, 0))
    good = [c for c in cands if c is not None]
    if not good:
        return None
    best = min(good, key=lambda c: c[0])
    return best[1], best[2].ravel()[:5]


def _best(cands, what: str):
    cands = [c for c in cands if c is not None]
    if not cands:
        raise ValueError(f"{what} failed from every start")
    return min(cands, key=lambda c: c[0])


def _search(run, objs, imgs, size, nd: int, init, own_start_first: bool, from_k5: bool):
    """Warm start (``init`` plus, if ``own_start_first``, OpenCV's own start) when ``init`` is a
    ``CalibResult``; the multi-start search is added only if that RMS is poor. The search starts
    from the staged and central starts and, when ``from_k5``, from the converged k5 fit
    (k1 -> k5 -> rational / RO; a rational fit from a raw k5 start can stall far from the optimum)."""
    cands = []
    guess = _init_guess(init, nd)
    if guess is not None:
        cands.append(run(guess[0], guess[1]))
    own_done = False
    ref = _init_rms(init)
    if ref is not None:
        if own_start_first:
            cands.append(run(None, None))
            own_done = True
        good = [c for c in cands if c is not None]
        if good and min(c[0] for c in good) <= WARM_ACCEPT_RATIO * ref + WARM_ACCEPT_PX:
            return cands, own_done
    raw = _raw_starts(objs, imgs, size)
    starts = list(raw)
    if from_k5:
        k5 = _k5_converged(objs, imgs, size, raw)
        if k5 is not None:
            starts.insert(0, k5)
    cands += [run(K0, _pad(d0, nd)) for K0, d0 in starts]
    return cands, own_done


def _fit_planar(objs, imgs, size, model: str, init=None):
    """Lowest-RMS fit over a staged start (k1 -> k5), a start from the central half of the views,
    ``init``, for rational the converged k5 fit, and for k5 (or when every other start fails)
    OpenCV's own initialisation."""
    flags = _flags(model)
    nd = N_DIST[model]

    def run(K0, d0):
        return _run(objs, imgs, size, flags, K0, d0)

    cands, own_done = _search(run, objs, imgs, size, nd, init, own_start_first=model == "k5", from_k5=model != "k5")
    if not own_done and (model == "k5" or not any(c is not None for c in cands)):
        cands.append(run(None, None))
    return _best(cands, "calibrateCameraExtended")


def _fit_ro(objs, imgs, size, init=None):
    def run(K0, d0):
        return _run_ro(objs, imgs, size, K0, d0)

    cands, own_done = _search(run, objs, imgs, size, 5, init, own_start_first=True, from_k5=True)
    if not own_done:
        cands.append(run(None, None))
    return _best(cands, "calibrateCameraROExtended")


def calibrate(views: Sequence, model: str, size: tuple[int, int], init=None) -> CalibResult:
    """Calibrate from accepted views and attach the intrinsic information and covariance.

    ``k5``: ``calibrateCameraExtended`` with (k1, k2, p1, p2, k3). ``rational``: adds
    ``CALIB_RATIONAL_MODEL`` (8 coefficients). ``ro``: ``calibrateCameraROExtended`` on the
    full-board views only, with the top-right inner corner as the fixed point; it returns an
    ``ROCalibResult`` whose ``new_obj`` holds the refined board. For RO the covariance treats
    the refined board as known. ``size`` is the (W, H) of the frames the views came from.

    Without ``init``, each model is fitted from a staged start (k1 only, then k5), from a k5 fit
    on the central half of the views, for rational and RO also from the converged k5 fit, and
    for k5 and RO from OpenCV's own start; the lowest-RMS solution is kept. ``init`` may be ``(K, dist)``, which is added as one more start, or a
    ``CalibResult`` such as the previous step's calibration: then the fit starts from it (plus
    OpenCV's own start for k5 and RO) and runs the full multi-start search only when the RMS is
    worse than 1.2 x ``init.rms`` + 0.05 px.
    """
    _flags(model)
    obj_all = object_points()
    size = (int(size[0]), int(size[1]))
    new_obj = None
    if model == "ro":
        used = full_board_views(views)
        if len(used) < 3:
            raise ValueError(f"RO model needs at least 3 full-board views, got {len(used)}")
        obj32 = obj_all.astype(np.float32)
        objs = [obj32 for _ in used]
        imgs = [_sorted_full(v).astype(np.float32) for v in used]
        rms, K, dist, rvecs, tvecs, new_obj_raw, std_intr, _, _, per_view = _fit_ro(objs, imgs, size, init)
        new_obj = np.asarray(new_obj_raw, np.float64).reshape(-1, 3)
        obj_used = [new_obj for _ in used]
        extra = 3 * N_CORNERS - 9
    else:
        used = list(views)
        if len(used) < 2:
            raise ValueError(f"calibration needs at least 2 views, got {len(used)}")
        objs, imgs, obj_used = [], [], []
        for v in used:
            ids, c = view_arrays(v)
            o = obj_all[ids]
            objs.append(o.astype(np.float32))
            imgs.append(c.astype(np.float32))
            obj_used.append(o)
        rms, K, dist, rvecs, tvecs, std_intr, _, per_view = _fit_planar(objs, imgs, size, model, init)
        extra = 0
    nd = N_DIST[model]
    dist = np.asarray(dist, np.float64).ravel()[:nd]
    K = np.asarray(K, np.float64)
    rvecs = [np.asarray(r, np.float64).ravel() for r in rvecs]
    tvecs = [np.asarray(t, np.float64).ravel() for t in tvecs]
    n_pts = int(sum(len(o) for o in obj_used))
    p = 4 + nd
    sigma2 = noise_variance(float(rms), n_pts, p, len(used), extra)
    F = information_matrix(K, dist, rvecs, tvecs, obj_used)
    cov = covariance(F, sigma2) if np.isfinite(sigma2) else np.full((p, p), np.nan)
    kwargs = dict(
        model=model,
        K=K,
        dist=dist,
        rms=float(rms),
        per_view=np.asarray(per_view, np.float64).ravel(),
        rvecs=rvecs,
        tvecs=tvecs,
        std_intr=np.asarray(std_intr, np.float64).ravel(),
        sigma2=sigma2,
        F=F,
        cov=cov,
    )
    if model == "ro":
        return ROCalibResult(**kwargs, new_obj=new_obj)
    return CalibResult(**kwargs)


def heldout_view_error(view, K, dist, board_pts: np.ndarray | None = None) -> tuple[float, int]:
    """Sum of squared residuals and point count of a view under a fixed model, pose re-fitted."""
    ids, c = view_arrays(view)
    base = object_points() if board_pts is None else np.asarray(board_pts, np.float64).reshape(-1, 3)
    pts = base[ids]
    if np.all(np.abs(pts[:, 2]) < 1e-12):
        _, _, rms, _ = solve_pose(pts, c, K, dist)
    else:
        # A refined (RO) board is not exactly planar: start from IPPE on the flattened board.
        planar = pts.copy()
        planar[:, 2] = 0.0
        rvec, tvec, _, _ = solve_pose(planar, c, K, dist)
        rvec, tvec = cv2.solvePnPRefineLM(pts, c, np.asarray(K, np.float64), np.asarray(dist, np.float64).ravel(), rvec.reshape(3, 1), tvec.reshape(3, 1))
        rms = reprojection_rms(pts, c, rvec, tvec, K, dist)
    return rms * rms * len(ids), len(ids)


def prequential_rms(view, calib: CalibResult) -> float:
    """RMS of a new view against a calibration that did not include it (pose re-fitted)."""
    sse, n = heldout_view_error(view, calib.K, calib.dist, board_points(calib))
    return math.sqrt(sse / n)


def cv_heldout_rms(views: Sequence, model: str, size: tuple[int, int], folds: int = 3, init=None) -> float:
    """k-fold held-out reprojection RMS (views assigned to folds by index modulo ``folds``).

    Each held-out view keeps its own fitted pose (6 dof) and is scored with the intrinsics
    calibrated on the other folds. Returns NaN when a fold cannot be calibrated.

    The score covers the views ``model`` is fitted on: for ``ro`` only the full-board views,
    for ``k5`` and ``rational`` all views. Compare RO against k5 on the same views with
    ``ro_cv_gain``. Each fold is warm-started from ``init`` (a ``CalibResult``, e.g. the
    session's current calibration); without it, from a calibration of ``model`` on all the
    scored views. A fold whose warm start lands at a poor RMS falls back to the multi-start fit.
    """
    views = views_used(model, views)
    if len(views) < 2 * folds:
        return math.nan
    if not isinstance(init, CalibResult):
        try:
            init = calibrate(views, model, size)
        except ValueError:
            init = None
    sse, n = 0.0, 0
    for f in range(folds):
        train = [v for i, v in enumerate(views) if i % folds != f]
        test = [v for i, v in enumerate(views) if i % folds == f]
        try:
            calib = calibrate(train, model, size, init=init)
        except ValueError:
            return math.nan
        for v in test:
            try:
                s, m = heldout_view_error(v, calib.K, calib.dist, board_points(calib))
            except (ValueError, cv2.error):
                return math.nan
            sse += s
            n += m
    return math.sqrt(sse / n) if n else math.nan


def ro_cv_gain(views: Sequence, size: tuple[int, int], folds: int = 3, init=None) -> tuple[float, float, float]:
    """R9 evidence: ``(gain, cv_ro, cv_k5)`` with gain = 1 - cv_ro / cv_k5.

    Both held-out RMS values are computed on the full-board views only (the views RO can use),
    with the same fold assignment, so the gain compares the two models on the same data.
    """
    full = full_board_views(views)
    cv_ro = cv_heldout_rms(full, "ro", size, folds, init)
    cv_k5 = cv_heldout_rms(full, "k5", size, folds, init)
    gain = 1.0 - cv_ro / cv_k5 if (cv_k5 > 0 and math.isfinite(cv_ro)) else math.nan
    return gain, cv_ro, cv_k5
