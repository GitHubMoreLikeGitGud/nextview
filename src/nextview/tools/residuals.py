"""Residual diagnostics for rules R6 to R10: radial trend, outlier views, board warp, drift.

Statistics use numpy only (the Lambda image has no scipy).
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from ..types import CalibResult, Diagnostics
from ._views import board_points, full_board_views, view_arrays, views_used

__all__ = [
    "view_residuals",
    "chi2_sf_even",
    "radial_trend_p",
    "outlier_views",
    "warp_score",
    "drift_flag",
    "residual_field",
    "estimate_hfov_deg",
    "out_of_scope",
    "residual_analysis",
]

RADIAL_BINS = 8  # even, so the chi-square tail has a closed form
WARP_DZ_M = 1e-4


def _project(obj, rvec, tvec, K, dist) -> np.ndarray:
    p, _ = cv2.projectPoints(
        np.asarray(obj, np.float64).reshape(-1, 1, 3),
        np.asarray(rvec, np.float64).reshape(3, 1),
        np.asarray(tvec, np.float64).reshape(3, 1),
        np.asarray(K, np.float64),
        np.asarray(dist, np.float64).ravel(),
    )
    return p.reshape(-1, 2)


def view_residuals(calib: CalibResult, views: Sequence) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per used view: (ids, observed (N, 2), residual = observed - projected (N, 2))."""
    used = views_used(calib.model, views)
    base = board_points(calib)
    out = []
    for v, r, t in zip(used, calib.rvecs, calib.tvecs):
        ids, c = view_arrays(v)
        out.append((ids, c, c - _project(base[ids], r, t, calib.K, calib.dist)))
    return out


def chi2_sf_even(x: float, k: int) -> float:
    """Upper tail P(X > x) of a chi-square with even ``k`` degrees of freedom."""
    if k <= 0 or k % 2:
        raise ValueError("k must be a positive even integer")
    if x <= 0:
        return 1.0
    h = 0.5 * x
    term, total = 1.0, 1.0
    for i in range(1, k // 2):
        term *= h / i
        total += term
    return float(min(1.0, math.exp(-h) * total))


def radial_trend_p(calib: CalibResult, views: Sequence, n_bins: int = RADIAL_BINS) -> float:
    """p-value for a systematic radial residual pattern (lens-model under-fit).

    Radial residual components are binned by normalised radius into ``n_bins`` equal-count
    bins; under a correct model each bin mean is zero. Statistic: sum n_b m_b^2 / s_w^2 with
    the pooled within-bin variance s_w^2, compared with chi-square(n_bins).
    """
    res = view_residuals(calib, views)
    if not res:
        return math.nan
    obs = np.concatenate([c for _, c, _ in res])
    e = np.concatenate([r for _, _, r in res])
    c0 = np.array([calib.K[0, 2], calib.K[1, 2]])
    f = 0.5 * (calib.K[0, 0] + calib.K[1, 1])
    v = obs - c0
    rad = np.hypot(v[:, 0], v[:, 1])
    keep = rad > 0.02 * f
    if np.count_nonzero(keep) < 4 * n_bins:
        return math.nan
    rho = rad[keep] / f
    er = np.sum(e[keep] * (v[keep] / rad[keep, None]), axis=1)
    edges = np.quantile(rho, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, rho, side="right") - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    if np.any(cnt < 2):
        return math.nan
    means = np.bincount(idx, weights=er, minlength=n_bins) / cnt
    within = float(np.sum((er - means[idx]) ** 2)) / (er.size - n_bins)
    if within <= 0:
        return math.nan
    stat = float(np.sum(cnt * means**2) / within)
    return chi2_sf_even(stat, n_bins)


def outlier_views(calib: CalibResult, views: Sequence, mad_k: float = 3.0, min_abs_px: float = 0.5) -> tuple[int, ...]:
    """Ids of views with per-view RMS > median + mad_k * 1.4826 * MAD and > ``min_abs_px``."""
    used = views_used(calib.model, views)
    pv = np.asarray(calib.per_view, np.float64).ravel()
    if pv.size < 3 or pv.size != len(used):
        return ()
    med = float(np.median(pv))
    mad = float(np.median(np.abs(pv - med)))
    limit = med + mad_k * 1.4826 * mad
    return tuple(int(getattr(v, "id", i)) for i, (v, e) in enumerate(zip(used, pv)) if e > limit and e > min_abs_px)


def warp_score(calib: CalibResult, views: Sequence) -> tuple[float, int]:
    """Fraction of full-board residual energy explained by a common out-of-plane offset per corner.

    For each corner k one dz_k (board normal) is shared by all full-board views; its image
    effect in view v is j_vk * dz_k with j_vk = d(projection)/dz. A sagging or warped board
    gives a large fraction; pure noise gives about 1 / (2 V). Returns (score, n_full_views).
    """
    used = views_used(calib.model, views)
    full_ids = {id(v) for v in full_board_views(used)}
    base = board_points(calib)
    num = np.zeros(len(base))
    den = np.zeros(len(base))
    total = 0.0
    n_full = 0
    lifted = base + np.array([0.0, 0.0, WARP_DZ_M])
    for v, r, t in zip(used, calib.rvecs, calib.tvecs):
        if id(v) not in full_ids:
            continue
        ids, c = view_arrays(v)
        p0 = _project(base[ids], r, t, calib.K, calib.dist)
        p1 = _project(lifted[ids], r, t, calib.K, calib.dist)
        j = (p1 - p0) / WARP_DZ_M
        e = c - p0
        num[ids] += np.sum(j * e, axis=1)
        den[ids] += np.sum(j * j, axis=1)
        total += float(np.sum(e * e))
        n_full += 1
    if n_full == 0 or total <= 0:
        return math.nan, n_full
    ok = den > 0
    explained = float(np.sum(num[ok] ** 2 / den[ok]))
    return explained / total, n_full


def drift_flag(views: Sequence, factor: float = 3.0, consecutive: int = 3) -> bool:
    """True when the last ``consecutive`` prequential RMS values all exceed ``factor`` x the median
    of the earlier ones (at least 3 earlier values are needed)."""
    vals = [getattr(v, "prequential_rms", None) for v in views]
    vals = [float(x) for x in vals if x is not None and np.isfinite(x)]
    if len(vals) < consecutive + 3:
        return False
    ref = float(np.median(vals[:-consecutive]))
    return bool(ref > 0 and all(x > factor * ref for x in vals[-consecutive:]))


def residual_field(calib: CalibResult, views: Sequence, size: tuple[int, int], grid: tuple[int, int] = (16, 9)) -> np.ndarray:
    """(rows, cols, 2) mean residual vector per image cell; NaN where no corner fell."""
    cols, rows = grid
    W, H = size
    sums = np.zeros((rows * cols, 2))
    cnt = np.zeros(rows * cols)
    for _, c, e in view_residuals(calib, views):
        ix = np.clip((c[:, 0] / W * cols).astype(int), 0, cols - 1)
        iy = np.clip((c[:, 1] / H * rows).astype(int), 0, rows - 1)
        k = iy * cols + ix
        np.add.at(sums, k, e)
        np.add.at(cnt, k, 1.0)
    out = np.full((rows * cols, 2), np.nan)
    m = cnt > 0
    out[m] = sums[m] / cnt[m, None]
    return out.reshape(rows, cols, 2)


def estimate_hfov_deg(K, dist, size: tuple[int, int]) -> float:
    """Horizontal field of view through the principal row, from the fitted model."""
    K = np.asarray(K, np.float64)
    W = size[0]
    pts = np.array([[[0.0, K[1, 2]]], [[W - 1.0, K[1, 2]]]])
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
    n = cv2.undistortPoints(pts, K, np.asarray(dist, np.float64).ravel(), criteria=crit).reshape(-1, 2)
    return math.degrees(math.atan(abs(n[0, 0])) + math.atan(abs(n[1, 0])))


def _th(th, name: str, default):
    return default if th is None else getattr(th, name, default)


def out_of_scope(calib: CalibResult, size: tuple[int, int], th=None) -> tuple[bool, float]:
    """R10 evidence: ``(fires, hfov_deg)``.

    Fires when the model is already ``rational`` and the HFOV estimated from that fit exceeds
    ``out_of_scope_hfov_deg``. The model only becomes rational after R8 found a radial trend
    under k5, so the k5 under-fit is already established. A radial trend that persists after
    the switch is not required: inside a 1280 x 720 frame the rational model fits a 130 degree
    fisheye to about 0.1 px with no trend left, while its HFOV estimate stays accurate.
    """
    hfov = estimate_hfov_deg(calib.K, calib.dist, size)
    limit = float(_th(th, "out_of_scope_hfov_deg", 110.0))
    return bool(calib.model == "rational" and math.isfinite(hfov) and hfov > limit), hfov


def residual_analysis(calib: CalibResult, views: Sequence, th=None, size: tuple[int, int] | None = None) -> Diagnostics:
    """Diagnostics for the planner. ``th`` is a ``config.Thresholds`` (defaults used when None).

    ``residual_field`` (16 x 9 by default) and ``hfov_deg`` are filled only when the image
    ``size`` is given. ``outliers`` holds ``View.id`` values. ``warp_score`` is NaN without
    full-board views.
    """
    radial_p = radial_trend_p(calib, views)
    outliers = outlier_views(calib, views, _th(th, "outlier_mad_k", 3.0), _th(th, "outlier_min_abs_px", 0.5))
    warp, _ = warp_score(calib, views)
    drift = drift_flag(views, _th(th, "drift_factor", 3.0), int(_th(th, "drift_consecutive", 3)))
    field = None
    hfov = math.nan
    if size is not None:
        grid = tuple(_th(th, "grid_coverage", (16, 9)))
        field = residual_field(calib, views, size, grid)
        hfov = estimate_hfov_deg(calib.K, calib.dist, size)
    return Diagnostics(radial_p=radial_p, outliers=outliers, warp_score=warp, drift=drift, residual_field=field, hfov_deg=hfov)
