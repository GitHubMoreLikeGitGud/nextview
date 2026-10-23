"""Intrinsic covariance and the per-pixel projection-uncertainty map (spec section 2.5).

theta = (fx, fy, cx, cy, d1..dn), p = 4 + n. For each view v the ``projectPoints`` Jacobian
is split into pose columns J_phi (2N_v x 6) and intrinsic columns J_theta (2N_v x p), and the
pose is eliminated with a Schur complement:

    S_v = A_v - B_v C_v^-1 B_v^T,  A = J_theta^T J_theta,  B = J_theta^T J_phi,  C = J_phi^T J_phi.

Each pose appears in its own view only, so F = sum_v S_v is exactly the Gauss-Newton marginal
information for theta. Sigma_theta = sigma^2 F^-1 with sigma^2 = rms^2 N / (2N - p - 6V).
OpenCV's own ``stdDeviationsIntrinsics`` agree for k5; for the rational model they come from an
unscaled eigen-inverse of an ill-conditioned matrix and can be NaN or too small.

The map projects Sigma_theta through the intrinsic Jacobian G of a fixed ray per grid cell:
Cov_ij = G Sigma G^T, sigma_ij = sqrt(trace Cov_ij), U95 = 95th percentile of sigma over the grid.
This is first order and assumes the lens model is correct.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from ..geometry import grid_centres
from ..types import CalibResult, UncMap
from ._views import board_points, view_arrays, views_used

__all__ = [
    "UNDISTORT_CRITERIA",
    "n_dist",
    "projection_jacobian",
    "view_information",
    "information_matrix",
    "noise_variance",
    "covariance",
    "grid_rays",
    "grid_jacobian",
    "sigma_from_cov",
    "u95_from_cov",
    "grid_covariances",
    "uncertainty_map",
]

UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
POSE_JITTER = 1e-9


def n_dist(dist: np.ndarray) -> int:
    return int(np.asarray(dist).size)


def projection_jacobian(K, dist, rvec, tvec, obj) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``projectPoints`` at a pose: returns (points (N, 2), J_phi (2N, 6), J_theta (2N, p))."""
    d = np.asarray(dist, np.float64).ravel()
    p, J = cv2.projectPoints(
        np.asarray(obj, np.float64).reshape(-1, 1, 3),
        np.asarray(rvec, np.float64).reshape(3, 1),
        np.asarray(tvec, np.float64).reshape(3, 1),
        np.asarray(K, np.float64),
        d,
    )
    npar = 4 + d.size
    return p.reshape(-1, 2), J[:, :6], J[:, 6 : 6 + npar]


def view_information(K, dist, rvec, tvec, obj) -> np.ndarray:
    """Schur-complemented (pose-eliminated) information S_v of one view, (p, p)."""
    _, Jp, Jt = projection_jacobian(K, dist, rvec, tvec, obj)
    A = Jt.T @ Jt
    B = Jt.T @ Jp
    C = Jp.T @ Jp + POSE_JITTER * np.eye(6)
    S = A - B @ np.linalg.solve(C, B.T)
    return 0.5 * (S + S.T)


def information_matrix(K, dist, rvecs: Sequence, tvecs: Sequence, objs: Sequence) -> np.ndarray:
    """F = sum_v S_v over the views."""
    p = 4 + n_dist(dist)
    F = np.zeros((p, p))
    for r, t, o in zip(rvecs, tvecs, objs):
        F += view_information(K, dist, r, t, o)
    return F


def noise_variance(rms: float, n_points: int, p: int, n_views: int, extra_params: int = 0) -> float:
    """Per-coordinate noise variance: rms^2 N / (2N - p - 6V - extra).

    OpenCV's rms is sqrt(sum ||e||^2 / N), so rms^2 N is the residual sum of squares.
    """
    dof = 2 * n_points - p - 6 * n_views - extra_params
    if dof <= 0:
        return math.nan
    return float(rms * rms * n_points / dof)


def covariance(F: np.ndarray, sigma2: float) -> np.ndarray:
    """sigma^2 F^-1.

    F is Jacobi-scaled first (unit diagonal) and a 1e-12 trace jitter is added to the scaled
    matrix. The intrinsics have very different units (focal length in px, distortion
    coefficients unitless), so a jitter on the unscaled trace would swamp the focal-length
    terms of a rational model.
    """
    F = np.asarray(F, np.float64)
    p = F.shape[0]
    d = np.sqrt(np.clip(np.diag(F), 1e-300, None))
    Fs = F / np.outer(d, d)
    jitter = 1e-12 * float(np.trace(Fs))
    inv_s = np.linalg.solve(Fs + jitter * np.eye(p), np.eye(p))
    cov = sigma2 * inv_s / np.outer(d, d)
    return 0.5 * (cov + cov.T)


def grid_rays(K, dist, size: tuple[int, int], grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Grid cell centres (M, 2) and their normalised rays (M, 3) under the model (K, dist)."""
    u = grid_centres(size, grid)
    n = cv2.undistortPoints(
        u.reshape(-1, 1, 2), np.asarray(K, np.float64), np.asarray(dist, np.float64).ravel(), criteria=UNDISTORT_CRITERIA
    ).reshape(-1, 2)
    rays = np.column_stack([n, np.ones(len(n))])
    return u, rays


def grid_jacobian(K, dist, size: tuple[int, int], grid: tuple[int, int] = (32, 18)) -> np.ndarray:
    """Intrinsic Jacobian of each grid cell's fixed ray, (M, 2, p); row-major over the grid."""
    _, rays = grid_rays(K, dist, size, grid)
    zero = np.zeros(3)
    _, _, G = projection_jacobian(K, dist, zero, zero, rays)
    p = G.shape[1]
    return G.reshape(-1, 2, p)


def grid_covariances(G: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """(M, 2, 2) image-point covariance per grid cell."""
    return np.einsum("mrp,pq,msq->mrs", G, cov, G, optimize=True)


def sigma_from_cov(G: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """(M,) sqrt(trace(G cov G^T)) per grid cell."""
    var = np.einsum("mrp,pq,mrq->m", G, cov, G, optimize=True)
    return np.sqrt(np.maximum(var, 0.0))


def u95_from_cov(G: np.ndarray, cov: np.ndarray) -> float:
    return float(np.percentile(sigma_from_cov(G, cov), 95))


def _view_objs(calib: CalibResult, views) -> list[np.ndarray]:
    base = board_points(calib)
    return [base[view_arrays(v)[0]] for v in views_used(calib.model, views)]


def uncertainty_map(
    calib: CalibResult,
    views,
    size: tuple[int, int],
    stopcal=None,
    *,
    actuator: str | None = None,
    tau_px: float = 0.5,
    grid: tuple[int, int] = (32, 18),
) -> UncMap:
    """Per-cell predicted reprojection std (rows, cols), its 95th percentile and P_exceed.

    ``stopcal`` is a ``config.StopCalibration`` (or None, giving ``p_exceed = nan``); with a
    ``stopcal`` the ``actuator`` ("robot" or "human") must be given, because the stop quantiles
    differ per actuator. ``calib.cov`` is used when present; otherwise it is rebuilt from ``views``.
    """
    if stopcal is not None and actuator is None:
        raise ValueError("uncertainty_map needs actuator= ('robot' or 'human') when stopcal is given")
    cov = calib.cov
    if cov is None or not np.all(np.isfinite(cov)):
        F = information_matrix(calib.K, calib.dist, calib.rvecs, calib.tvecs, _view_objs(calib, views))
        cov = covariance(F, calib.sigma2)
    G = grid_jacobian(calib.K, calib.dist, size, grid)
    s = sigma_from_cov(G, cov)
    u95 = float(np.percentile(s, 95))
    cols, rows = grid
    p_ex = math.nan if stopcal is None else float(stopcal.p_exceed(u95, actuator, tau_px))
    return UncMap(grid=(cols, rows), sigma_px=s.reshape(rows, cols), U95=u95, p_exceed=p_ex)
