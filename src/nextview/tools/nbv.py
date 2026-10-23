"""Next-best-view planning (spec section 2.4).

Candidate lattice: board centre on a 5 x 3 grid of image cells, size fraction s in
{0.35, 0.5, 0.7} of the image width, orientation tilt 0 or tilt {20, 35, 50} deg x 8 azimuths,
in-plane roll uniform in +-30 deg. 15 x 3 x 25 = 1125 points; ``nbv_candidates`` (300) are
sampled with the session RNG.

Each candidate pose comes from the current estimate (K, dist): depth Z = fx * board_width /
(s * W), rotation from (tilt, azimuth, roll), translation so the board centre projects to the
cell centre. A candidate is feasible when >= 50 % of the corners project inside a 5 px margin
(rays the fitted model folds back into the frame do not count), tilt < 60 deg, the predicted
board area is >= 4 % of the image, and tilt >= 25 deg while the AMBIGUOUS flag is set. All
corners and the whole board outline (so every marker) must be inside the margin instead of half
the corners when only full-board views count: the model is RO (``calibrate`` fits RO on
full-board views only), or the session is collecting full-board views for the R9 warp check
(``state.seek_full_board``).

Gain = U95(sigma^2 F^-1) - U95(sigma^2 (F + S_c)^-1), with S_c the pose-eliminated (Schur)
information of the candidate's visible corners (``tools.uncertainty``). All candidates are
projected in one ``projectPoints`` call on camera-frame points with a zero pose: the Jacobian's
pose columns then describe a rigid motion about the camera centre instead of the board pose,
which spans the same 6-dof tangent space, and the Schur complement does not depend on how that
space is parametrised.

Score = gain - lambda * effort * max(gain), lambda = 0.3 for a human operator and 0 for a robot;
effort = |dtilt| / 30 deg + |dcentre| / (0.5 W) + |ds| / 0.3 from the last realised pose.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from ..board import BOARD_SIZE_M, N_CORNERS, SQUARES
from ..geometry import BOARD_CENTRE_M, BOARD_OUTLINE_M, as_rotation, ghost_quad, project, tilt_of
from ..types import CalibResult, Candidate
from ._views import board_points
from .uncertainty import UNDISTORT_CRITERIA, grid_jacobian

__all__ = [
    "LATTICE_CELLS",
    "SIZE_FRACTIONS",
    "ORIENTATIONS",
    "CandidateList",
    "lattice",
    "k_initial",
    "OpeningSpec",
    "opening_specs",
    "opening_target",
    "focal_from_detection",
    "rotations",
    "batch_covariance",
    "grid_gram",
    "u95_batch",
    "radial_limit",
    "propose_next_views",
]

LATTICE_CELLS = (5, 3)  # (cols, rows)
SIZE_FRACTIONS = (0.35, 0.5, 0.7)
TILTS = (20.0, 35.0, 50.0)
AZIMUTHS = tuple(float(a) for a in range(0, 360, 45))
ORIENTATIONS = ((0.0, 0.0),) + tuple((t, a) for t in TILTS for a in AZIMUTHS)  # 25 (tilt, azimuth)
ROLL_MAX_DEG = 30.0
CHUNK = 60
MIN_DEPTH_M = 0.02
POSE_JITTER = 1e-9
INNER_AREA_FRAC = (SQUARES[0] - 2) / SQUARES[0] * (SQUARES[1] - 2) / SQUARES[1]
OPENING_SALT = 0x5EED


class CandidateList(list):
    """``list[Candidate]`` plus planner statistics (how many lattice points were sampled,
    found feasible and evaluated, the current U95 and the time spent)."""

    n_sampled: int = 0
    n_feasible: int = 0
    n_evaluated: int = 0
    u95_now: float = math.nan
    ms: float = 0.0
    stopped_by_deadline: bool = False
    full_board_only: bool = False

    def stats(self) -> dict:
        return {"n_candidates": self.n_evaluated, "n_sampled": self.n_sampled, "n_feasible": self.n_feasible,
                "u95_now": None if not math.isfinite(self.u95_now) else round(self.u95_now, 4),
                "deadline_hit": self.stopped_by_deadline, "full_board_only": self.full_board_only}


def lattice(size: tuple[int, int]) -> np.ndarray:
    """(1125, 5) lattice points: centre x px, centre y px, size fraction, tilt deg, azimuth deg."""
    W, H = size
    cols, rows = LATTICE_CELLS
    out = []
    for j in range(rows):
        for i in range(cols):
            cx = (i + 0.5) * W / cols - 0.5
            cy = (j + 0.5) * H / rows - 0.5
            for s in SIZE_FRACTIONS:
                for tilt, az in ORIENTATIONS:
                    out.append((cx, cy, s, tilt, az))
    return np.asarray(out, np.float64)


def k_initial(size: tuple[int, int], focal_px: float | None = None) -> np.ndarray:
    """K0 = [[0.9 W, 0, W/2], [0, 0.9 W, H/2], [0, 0, 1]] (spec 2.4), or with a focal estimate."""
    W, H = size
    f = 0.9 * W if focal_px is None else float(focal_px)
    return np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])


def rotations(tilt_deg: np.ndarray, az_deg: np.ndarray, roll_deg: np.ndarray) -> np.ndarray:
    """(n, 3, 3) board rotations, vectorised ``geometry.rotation_from_tilt``."""
    tilt = np.radians(np.asarray(tilt_deg, float))
    az = np.radians(np.asarray(az_deg, float))
    roll = np.radians(np.asarray(roll_deg, float))
    n = tilt.size
    k = np.stack([np.sin(az), -np.cos(az), np.zeros(n)], axis=1)  # unit tilt axis
    c, s = np.cos(tilt), np.sin(tilt)
    Kx = np.zeros((n, 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    I = np.eye(3)[None]
    R_tilt = I + s[:, None, None] * Kx + (1 - c)[:, None, None] * (Kx @ Kx)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.zeros((n, 3, 3))
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = cr, -sr, sr, cr, 1.0
    return R_tilt @ Rz


def _undistort(px: np.ndarray, K, dist) -> np.ndarray:
    px = np.asarray(px, np.float64).reshape(-1, 1, 2)
    d = None if dist is None else np.asarray(dist, np.float64).ravel()
    if d is None or not np.any(d != 0):
        Kf = np.asarray(K, np.float64)
        p = px.reshape(-1, 2)
        return np.column_stack([(p[:, 0] - Kf[0, 2]) / Kf[0, 0], (p[:, 1] - Kf[1, 2]) / Kf[1, 1]])
    return cv2.undistortPoints(px, np.asarray(K, np.float64), d, criteria=UNDISTORT_CRITERIA).reshape(-1, 2)


def _poses_from_points(pts: np.ndarray, rolls: np.ndarray, K, dist, size) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``geometry.pose_from_lattice``: returns (R (n, 3, 3), t (n, 3))."""
    K = np.asarray(K, np.float64)
    W = float(size[0])
    xn = _undistort(pts[:, :2], K, dist)
    Z = K[0, 0] * BOARD_SIZE_M[0] / (pts[:, 2] * W)
    R = rotations(pts[:, 3], pts[:, 4], rolls)
    centre = Z[:, None] * np.column_stack([xn, np.ones(len(xn))])
    t = centre - R @ BOARD_CENTRE_M
    return R, t


# ---- opening poses ------------------------------------------------------------------------

@dataclass(frozen=True)
class OpeningSpec:
    """An opening pose in image terms: board centre (fractions of W, H), size fraction, tilt, azimuth, roll."""

    cx: float
    cy: float
    s: float
    tilt: float
    az: float
    roll: float


def opening_specs(camera_seed: int, seed: int, n: int = 3) -> list[OpeningSpec]:
    """The seeded opening poses shared by every policy for the same (camera, seed) (spec 2.4).

    Centres in the central 30 % of the image, the first at s = 0.45 (large enough to detect on a
    105 deg lens, small enough to stay mostly in frame on a 20 deg lens under K0), the next two at
    s in [0.45, 0.6]; tilts 15-30 deg then 25-40 deg, azimuths about 120 deg apart; roll +-20 deg.
    """
    rng = np.random.default_rng([int(camera_seed), int(seed), OPENING_SALT])
    a0 = float(rng.uniform(0.0, 360.0))
    out = []
    for k in range(n):
        u = rng.uniform(0.0, 1.0, 6)
        s = 0.45 if k == 0 else 0.45 + 0.15 * float(u[2])
        tilt = (15.0 if k == 0 else 25.0) + 15.0 * float(u[3])
        out.append(OpeningSpec(0.35 + 0.3 * float(u[0]), 0.35 + 0.3 * float(u[1]), s, tilt,
                               (a0 + 120.0 * k + 40.0 * (float(u[4]) - 0.5)) % 360.0, 40.0 * (float(u[5]) - 0.5)))
    return out


def opening_target(k: int, camera_seed: int, seed: int, size: tuple[int, int],
                   focal_px: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(rvec, tvec) of opening pose ``k`` (taken modulo 3) under K0, or K0 with ``focal_px``."""
    spec = opening_specs(camera_seed, seed, 3)[int(k) % 3]
    W, H = size
    K = k_initial(size, focal_px)
    pts = np.array([[spec.cx * (W - 1), spec.cy * (H - 1), spec.s, spec.tilt, spec.az]])
    R, t = _poses_from_points(pts, np.array([spec.roll]), K, None, size)
    rvec, _ = cv2.Rodrigues(R[0])
    return rvec.ravel(), t[0]


def focal_from_detection(ids, corners, target_rvec, target_tvec, focal_used: float,
                         size: tuple[int, int]) -> float | None:
    """Focal-length estimate from one frame captured at a commanded target pose.

    For the same metric pose the image of the board scales with f, so f ~ f_used * (observed
    corner spread / spread predicted at the target with f_used). Used only before the first
    calibration (fewer than 3 views) to place the opening poses; needs >= 4 corners.
    """
    from ..board import object_points

    ids = np.asarray(ids).ravel().astype(int)
    if ids.size < 4:
        return None
    obs = np.asarray(corners, np.float64).reshape(-1, 2)
    pred = project(object_points()[ids], target_rvec, target_tvec, k_initial(size, focal_used))
    sp = float(np.sqrt(np.mean(np.sum((pred - pred.mean(0)) ** 2, axis=1))))
    so = float(np.sqrt(np.mean(np.sum((obs - obs.mean(0)) ** 2, axis=1))))
    if not (sp > 5.0 and so > 5.0 and math.isfinite(sp) and math.isfinite(so)):
        return None
    f = float(focal_used) * so / sp
    W = float(size[0])
    return float(np.clip(f, 0.15 * W, 6.0 * W))


# ---- batched information and U95 ------------------------------------------------------------

def radial_limit(dist, r_cap: float = 12.0) -> float:
    """Largest undistorted radius up to which the fitted radial map r -> r * radial(r) rises.

    Beyond it a polynomial model folds rays back into the frame, so predicted corners there are
    not real observations.
    """
    d = np.asarray(dist, np.float64).ravel()
    k1, k2, k3 = d[0], d[1], (d[4] if d.size >= 5 else 0.0)
    k4, k5, k6 = (d[5], d[6], d[7]) if d.size >= 8 else (0.0, 0.0, 0.0)
    r = np.linspace(0.0, r_cap, 6001)
    r2 = r * r
    rd = r * (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    inc = np.diff(rd) > 0
    if inc.all():
        return r_cap
    return float(r[max(int(np.argmin(inc)), 1)])


def batch_covariance(Fb: np.ndarray, sigma2: float) -> np.ndarray:
    """sigma^2 F^-1 for a stack (n, p, p), with the Jacobi scaling and jitter of
    ``uncertainty.covariance`` (so a single matrix gives the identical result)."""
    Fb = np.asarray(Fb, np.float64)
    n, p, _ = Fb.shape
    d = np.sqrt(np.clip(np.diagonal(Fb, axis1=1, axis2=2), 1e-300, None))
    dd = d[:, :, None] * d[:, None, :]
    Fs = Fb / dd
    jit = 1e-12 * np.trace(Fs, axis1=1, axis2=2)
    inv = np.linalg.inv(Fs + jit[:, None, None] * np.eye(p)[None])
    cov = sigma2 * inv / dd
    return 0.5 * (cov + np.transpose(cov, (0, 2, 1)))


def grid_gram(K, dist, size, grid) -> np.ndarray:
    """(M, p*p) per grid cell, sum_r G_r^T G_r flattened: var_m = GG_m . vec(cov)."""
    G = grid_jacobian(K, dist, size, grid)
    p = G.shape[2]
    return np.einsum("mrp,mrq->mpq", G, G).reshape(len(G), p * p)


def u95_batch(GG: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """(n,) 95th percentile over the grid of sqrt(trace(G cov G^T)) for each covariance."""
    n, p, _ = covs.shape
    var = GG @ covs.reshape(n, p * p).T  # (M, n)
    return np.percentile(np.sqrt(np.maximum(var, 0.0)), 95, axis=0)


# ---- planner ----------------------------------------------------------------------------------

def _context(state):
    """(actuator, ambiguous, last_pose, sim, seek_full_board) from a SessionState, a mapping or None."""
    if state is None:
        return "robot", False, None, False, False
    if isinstance(state, Mapping):
        get = state.get
        last = get("last_pose")
        return (get("actuator", "robot"), "AMBIGUOUS" in (get("flags") or ()), last, get("mode") == "sim",
                bool(get("seek_full_board")))
    last = state.last_pose() if callable(getattr(state, "last_pose", None)) else getattr(state, "last_pose", None)
    return (getattr(state, "actuator", "robot"), "AMBIGUOUS" in (getattr(state, "flags", None) or ()), last,
            getattr(state, "mode", None) == "sim", bool(getattr(state, "seek_full_board", False)))


def _th(th, name, default):
    return default if th is None else getattr(th, name, default)


def _area_fracs(px: np.ndarray, ok: np.ndarray, vis: np.ndarray, size) -> np.ndarray:
    """Predicted visible board area / image area per candidate."""
    W, H = size
    rect = np.array([[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H - 0.5], [-0.5, H - 0.5]], np.float32)
    out = np.zeros(len(px))
    for i in range(len(px)):
        if ok[i, N_CORNERS:].all():
            q = px[i, N_CORNERS:].astype(np.float32)
            if cv2.contourArea(q, oriented=True) < 0:
                q = q[::-1].copy()
            if cv2.isContourConvex(q):
                a, _ = cv2.intersectConvexConvex(q, rect)
                out[i] = max(float(a), 0.0) / (W * H)
                continue
        v = px[i, :N_CORNERS][vis[i]]
        if len(v) >= 3:
            out[i] = float(cv2.contourArea(cv2.convexHull(v.astype(np.float32)))) / INNER_AREA_FRAC / (W * H)
    return out


def _evaluate(R, t, obj, K, dist, size, margin):
    """Project candidates: returns (px (n, 44, 2), ok (n, 44), J (n, 44, 2, cols), vis (n, 40))."""
    n = len(R)
    P = np.concatenate([obj, BOARD_OUTLINE_M], axis=0)  # (44, 3)
    X = np.einsum("nij,kj->nki", R, P) + t[:, None, :]
    z = X[..., 2]
    front = z > MIN_DEPTH_M
    zs = np.where(front, z, 1.0)
    xn = X[..., 0] / zs
    yn = X[..., 1] / zs
    r_lim = radial_limit(dist)
    ok = front & (np.hypot(xn, yn) < 0.98 * r_lim)
    safe = np.where(ok[..., None], X, np.array([0.0, 0.0, 1.0]))  # camera-frame points: depth matters for the pose columns
    d = np.asarray(dist, np.float64).ravel()
    p, J = cv2.projectPoints(np.ascontiguousarray(safe.reshape(-1, 1, 3)), np.zeros(3), np.zeros(3),
                             np.asarray(K, np.float64), d)
    px = p.reshape(n, len(P), 2)
    J = J.reshape(n, len(P), 2, -1)
    W, H = size
    inside = (px[..., 0] >= margin) & (px[..., 0] <= W - 1 - margin) & (px[..., 1] >= margin) & (px[..., 1] <= H - 1 - margin)
    vis = (ok & inside)[:, :N_CORNERS]
    return px, ok, J, vis


def _schur(J: np.ndarray, vis: np.ndarray, p: int) -> np.ndarray:
    """(n, p, p) Schur-complemented information of each candidate's visible corners."""
    n = len(J)
    Jm = J[:, :N_CORNERS] * vis[:, :, None, None]
    Jm = Jm.reshape(n, 2 * N_CORNERS, -1)
    Jp, Jt = Jm[:, :, :6], Jm[:, :, 6 : 6 + p]
    A = np.einsum("nkp,nkq->npq", Jt, Jt)
    B = np.einsum("nkp,nkq->npq", Jt, Jp)
    C = np.einsum("nkp,nkq->npq", Jp, Jp) + POSE_JITTER * np.eye(6)[None]
    S = A - B @ np.linalg.solve(C, np.transpose(B, (0, 2, 1)))
    return 0.5 * (S + np.transpose(S, (0, 2, 1)))


def _efforts(R, pts, K, dist, size, last_pose) -> np.ndarray:
    if last_pose is None:
        return np.zeros(len(pts))
    from ..geometry import board_centre_px

    rv, tv = last_pose
    W = float(size[0])
    tilt_last = tilt_of(rv)
    c_last = board_centre_px(rv, tv, K, dist)
    Rl = as_rotation(rv)
    Z = float((Rl @ BOARD_CENTRE_M + np.asarray(tv, float).ravel())[2])
    s_last = float(np.asarray(K)[0, 0]) * BOARD_SIZE_M[0] / (Z * W) if Z > 0 else 0.0
    if not np.all(np.isfinite(c_last)):
        c_last = np.array([W / 2.0, size[1] / 2.0])
    dc = np.hypot(pts[:, 0] - c_last[0], pts[:, 1] - c_last[1])
    return np.abs(pts[:, 3] - tilt_last) / 30.0 + dc / (0.5 * W) + np.abs(pts[:, 2] - s_last) / 0.3


def propose_next_views(state, calib: CalibResult, size, th=None, rng=None, k: int = 5, deadline: float | None = None,
                       *, n_candidates: int | None = None, mode_lambda: float | None = None) -> CandidateList:
    """Ranked candidate poses with predicted U95 gain, effort, ghost quad and instruction.

    ``state``: a ``SessionState`` (or a mapping with ``actuator``, ``flags``, ``last_pose``, ``mode``,
    ``seek_full_board``, or None) supplies the actuator (effort weight), the AMBIGUOUS flag (tilt
    >= 25 deg), the full-board requirement and the last realised pose (effort). ``rng``: the session RNG (lattice sample and rolls).
    ``deadline``: a ``time.perf_counter()`` value after which no further chunk of 60 candidates is
    evaluated; at least ``nbv_min_candidates`` are always evaluated. Returns a ``CandidateList``.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(0) if rng is None else rng
    size = (int(size[0]), int(size[1]))
    actuator, ambiguous, last_pose, sim, seek_full = _context(state)
    lam = (float(_th(th, "effort_lambda_human", 0.3)) if actuator == "human" else 0.0) if mode_lambda is None else mode_lambda
    n_cand = int(_th(th, "nbv_candidates", 300) if n_candidates is None else n_candidates)
    min_eval = int(_th(th, "nbv_min_candidates", 60))
    margin = float(_th(th, "nbv_margin_px", 5.0))
    min_vis = float(_th(th, "nbv_min_visible_frac", 0.5))
    full_only = bool(seek_full or calib.model == "ro")
    if full_only:
        min_vis = 1.0
    min_area = float(_th(th, "min_board_area_frac", 0.04))
    max_tilt = float(_th(th, "max_tilt_deg", 60.0))
    amb_tilt = float(_th(th, "ambiguous_min_tilt_deg", 25.0))
    grid = tuple(_th(th, "grid_uncertainty", (32, 18)))

    K = np.asarray(calib.K, np.float64)
    dist = np.asarray(calib.dist, np.float64).ravel()
    p = 4 + dist.size
    obj = board_points(calib)
    F = np.asarray(calib.F, np.float64)
    sigma2 = float(calib.sigma2)
    if not math.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = max(float(calib.rms) ** 2 / 2.0, 1e-6)

    GG = grid_gram(K, dist, size, grid)
    u95_now = float(u95_batch(GG, batch_covariance(F[None], sigma2))[0])

    lat = lattice(size)
    out = CandidateList()
    out.u95_now = u95_now
    out.full_board_only = full_only

    def feasible_set(idx):
        pts = lat[idx]
        rolls = rng.uniform(-ROLL_MAX_DEG, ROLL_MAX_DEG, len(idx))
        R, t = _poses_from_points(pts, rolls, K, dist, size)
        px, ok, J, vis = _evaluate(R, t, obj, K, dist, size, margin)
        frac = vis.mean(axis=1)
        good = (frac >= min_vis) & (pts[:, 3] < max_tilt)
        if full_only:  # every marker must be in frame too, or the edge corners are not detected
            o = px[:, N_CORNERS:]
            good &= ok[:, N_CORNERS:].all(axis=1) & ((o[..., 0] >= margin) & (o[..., 0] <= size[0] - 1 - margin)
                                                     & (o[..., 1] >= margin) & (o[..., 1] <= size[1] - 1 - margin)).all(axis=1)
        if ambiguous:
            good &= pts[:, 3] >= amb_tilt
        area = np.zeros(len(idx))
        if good.any():
            area[good] = _area_fracs(px[good], ok[good], vis[good], size)
        good &= area >= min_area
        return pts, rolls, R, t, J, vis, good

    idx = rng.choice(len(lat), size=min(n_cand, len(lat)), replace=False)
    pts, rolls, R, t, J, vis, good = feasible_set(idx)
    out.n_sampled = len(idx)
    if good.sum() < k:  # too few feasible in the sample: fall back to the whole lattice
        idx = np.arange(len(lat))
        pts, rolls, R, t, J, vis, good = feasible_set(idx)
        out.n_sampled = len(idx)
    fi = np.nonzero(good)[0]
    out.n_feasible = len(fi)
    if len(fi) == 0:
        out.ms = 1000 * (time.perf_counter() - t0)
        return out

    gains = np.full(len(fi), np.nan)
    after = np.full(len(fi), np.nan)
    done = 0
    while done < len(fi):
        if deadline is not None and done >= min_eval and time.perf_counter() > deadline:
            out.stopped_by_deadline = True
            break
        sl = fi[done : done + CHUNK]
        S = _schur(J[sl], vis[sl], p)
        u_after = u95_batch(GG, batch_covariance(F[None] + S, sigma2))
        after[done : done + len(sl)] = u_after
        gains[done : done + len(sl)] = u95_now - u_after
        done += len(sl)
    out.n_evaluated = done
    ev = fi[:done]
    gains, after = gains[:done], after[:done]
    effort = _efforts(R[ev], pts[ev], K, dist, size, last_pose)
    gmax = float(np.max(gains)) if done else 0.0
    score = gains - lam * effort * gmax if gmax > 0 else gains.copy()
    order = np.argsort(-score, kind="stable")[:k]

    from ..agent.instructions import describe_target

    for rank, j in enumerate(order, start=1):
        c = ev[j]
        rvec, _ = cv2.Rodrigues(R[c])
        rvec = rvec.ravel()
        tvec = t[c].copy()
        out.append(Candidate(
            rank=rank, rvec=rvec, tvec=tvec, gain=float(gains[j]), pred_U95_after=float(after[j]),
            effort=float(effort[j]), score=float(score[j]), ghost_quad=ghost_quad(rvec, tvec, K, dist),
            instruction=describe_target(rvec, tvec, K, dist, size, sim=sim),
        ))
    out.ms = 1000 * (time.perf_counter() - t0)
    return out
