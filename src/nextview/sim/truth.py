"""Ground-truth error metrics for an estimated calibration (spec 2.5 step 6 and 11.5).

field error: for each cell centre u of a (cols x rows) grid, take the GT ray through u and
project it with the estimate; the error is that projection minus u.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .lens import UNDISTORT_CRITERIA, GTCamera

__all__ = [
    "grid_centres",
    "FieldError",
    "field_error_vectors",
    "field_error",
    "fx_rel_err",
    "intrinsics_error",
    "planar_pairs",
    "PlanarError",
    "planar_mm_error",
]


def grid_centres(size=(1280, 720), grid=(32, 18)) -> np.ndarray:
    """(rows, cols, 2) pixel coordinates of cell centres; 40 px cells give 19.5, 59.5, ..."""
    W, H = size
    cols, rows = grid
    xs = (np.arange(cols) + 0.5) * (W / cols) - 0.5
    ys = (np.arange(rows) + 0.5) * (H / rows) - 0.5
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X, Y], -1)


def _project_rays(rays_xy: np.ndarray, K, dist) -> np.ndarray:
    X = np.column_stack([rays_xy, np.ones(len(rays_xy))]).reshape(-1, 1, 3)
    d = np.asarray(dist, dtype=np.float64).ravel()
    p, _ = cv2.projectPoints(np.ascontiguousarray(X), np.zeros(3), np.zeros(3),
                             np.asarray(K, dtype=np.float64), d)
    return p.reshape(-1, 2)


def field_error_vectors(cam: GTCamera, K, dist, grid=(32, 18)) -> np.ndarray:
    """(rows, cols, 2) error vectors pi(r_gt; theta_hat) - u, in pixels."""
    u = grid_centres(cam.size, grid)
    flat = u.reshape(-1, 2)
    rays = cam.pixel_to_normalized(flat)
    out = np.full_like(flat, np.nan)
    ok = np.all(np.isfinite(rays), axis=1)
    if ok.any():
        out[ok] = _project_rays(rays[ok], K, dist) - flat[ok]
    return out.reshape(u.shape)


@dataclass(frozen=True, eq=False)
class FieldError:
    mean: float
    p95: float
    max: float
    err: np.ndarray  # (rows, cols) error norms, px
    vec: np.ndarray  # (rows, cols, 2) error vectors, px

    def to_dict(self) -> dict:
        return {"mean": round(self.mean, 5), "p95": round(self.p95, 5), "max": round(self.max, 5)}


def field_error(cam: GTCamera, K, dist, grid=(32, 18)) -> FieldError:
    v = field_error_vectors(cam, K, dist, grid)
    e = np.linalg.norm(v, axis=-1)
    finite = e[np.isfinite(e)]
    if finite.size == 0:
        return FieldError(float("nan"), float("nan"), float("nan"), e, v)
    return FieldError(float(finite.mean()), float(np.percentile(finite, 95)), float(finite.max()), e, v)


def fx_rel_err(cam: GTCamera, K) -> float:
    """|fx_hat - fx| / fx, in percent."""
    return float(abs(float(np.asarray(K)[0, 0]) - cam.fx) / cam.fx * 100.0)


def intrinsics_error(cam: GTCamera, K) -> dict:
    """fx/fy relative errors (percent) and cx/cy absolute errors (px)."""
    K = np.asarray(K, dtype=np.float64)
    return {
        "fx_rel_pct": fx_rel_err(cam, K),
        "fy_rel_pct": float(abs(K[1, 1] - cam.fy) / cam.fy * 100.0),
        "cx_err_px": float(abs(K[0, 2] - cam.cx)),
        "cy_err_px": float(abs(K[1, 2] - cam.cy)),
    }


def _in_outer_band(uv: np.ndarray, size, outer: float) -> np.ndarray:
    W, H = size
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    cheb = np.maximum(np.abs(uv[:, 0] - cx) / (W / 2.0), np.abs(uv[:, 1] - cy) / (H / 2.0))
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
    return inside & (cheb >= 1.0 - outer)


def planar_pairs(cam: GTCamera, Z_m: float, n_pairs: int = 200, length_mm: float = 100.0,
                 outer: float = 0.30, seed: int = 0, max_batches: int = 50):
    """Image point pairs of 3D segments ``length_mm`` long on the fronto-parallel plane z = Z_m.

    Both endpoints lie in the outer band of the image: max(|du|/(W/2), |dv|/(H/2)) >= 1 - outer.
    The pairs depend only on (GT camera, Z, seed), so every policy is scored on the same pairs.
    Returns (u1, u2), each (n_pairs, 2) px.
    """
    rng = np.random.default_rng([int(seed), 7919])
    W, H = cam.size
    L = length_mm / 1000.0
    got1, got2 = [], []
    n = 0
    for _ in range(max_batches):
        b = rng.uniform(0.0, 1.0, (4096, 3))
        u1 = np.column_stack([b[:, 0] * (W - 1), b[:, 1] * (H - 1)])
        keep = _in_outer_band(u1, cam.size, outer)
        u1 = u1[keep]
        phi = b[keep, 2] * 2.0 * np.pi
        r1 = cam.pixel_to_normalized(u1)
        P1 = Z_m * np.column_stack([r1, np.ones(len(r1))])
        P2 = P1 + L * np.column_stack([np.cos(phi), np.sin(phi), np.zeros(len(phi))])
        u2, seen = cam.project_checked(P2)
        ok = np.all(np.isfinite(r1), axis=1) & seen & _in_outer_band(u2, cam.size, outer)
        got1.append(u1[ok])
        got2.append(u2[ok])
        n += int(ok.sum())
        if n >= n_pairs:
            break
    u1 = np.concatenate(got1)[:n_pairs]
    u2 = np.concatenate(got2)[:n_pairs]
    if len(u1) < n_pairs:
        raise RuntimeError(f"only {len(u1)} valid pairs at Z = {Z_m} m")
    return u1, u2


@dataclass(frozen=True, eq=False)
class PlanarError:
    abs_err_mm: np.ndarray  # (n_pairs,) |L_hat - L|
    median: float
    p95: float
    Z_m: float

    def to_dict(self) -> dict:
        return {"Z_m": self.Z_m, "median_mm": round(self.median, 4), "p95_mm": round(self.p95, 4)}


def planar_mm_error(cam: GTCamera, K, dist, Z_m: float, n_pairs: int = 200, length_mm: float = 100.0,
                    outer: float = 0.30, seed: int = 0) -> PlanarError:
    """Length error of 100 mm segments measured with the estimate on a known plane (spec 11.5 item 10).

    Endpoints are projected with the GT camera, then back-projected with (K, dist) onto z = Z_m.
    """
    u1, u2 = planar_pairs(cam, Z_m, n_pairs, length_mm, outer, seed)
    K = np.asarray(K, dtype=np.float64)
    d = np.asarray(dist, dtype=np.float64).ravel()

    def back(u):
        n = cv2.undistortPoints(u.reshape(-1, 1, 2), K, d, criteria=UNDISTORT_CRITERIA).reshape(-1, 2)
        return Z_m * np.column_stack([n, np.ones(len(n))])

    L_hat = np.linalg.norm(back(u2) - back(u1), axis=1) * 1000.0
    e = np.abs(L_hat - length_mm)
    return PlanarError(e, float(np.median(e)), float(np.percentile(e, 95)), float(Z_m))
