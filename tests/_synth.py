"""Synthetic ChArUco frames rendered directly with cv2 for the perception and calibration tests.

The board texture is ``board_png("screen")`` (160 px per square, 80 px margin). A frame is
``warpPerspective`` of the texture by ``K [r1 r2 t] S`` followed by a lens-distortion ``remap``.
``ss`` > 1 renders at ``ss`` times the resolution and area-averages down, which models the pixel
aperture of a real sensor. Blur is applied at the supersampled resolution, before averaging.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from nextview import board
from nextview.geometry import BOARD_CENTRE_M, rotation_from_tilt

TEX = cv2.imdecode(np.frombuffer(board.board_png("screen"), np.uint8), cv2.IMREAD_GRAYSCALE)
PPM = board.SCREEN_SQUARE_PX / board.SQUARE_M
MARGIN = board.SCREEN_MARGIN_PX
# texture pixel centre (u, v) -> board metres; square boundaries fall on texture pixel edges
S_TEX = np.array([[1 / PPM, 0, (-MARGIN + 0.5) / PPM], [0, 1 / PPM, (-MARGIN + 0.5) / PPM], [0, 0, 1]])
OBJ = board.object_points()


def make_K(f: float, size=(1280, 720), aspect: float = 1.0, pp_offset=(0.0, 0.0)) -> np.ndarray:
    W, H = size
    return np.array([[f, 0, (W - 1) / 2 + pp_offset[0]], [0, f * aspect, (H - 1) / 2 + pp_offset[1]], [0, 0, 1.0]])


def _scaled_K(K: np.ndarray, ss: int) -> np.ndarray:
    Ks = K.astype(np.float64).copy()
    Ks[0, 0] *= ss
    Ks[1, 1] *= ss
    Ks[0, 2] = ss * K[0, 2] + (ss - 1) / 2.0
    Ks[1, 2] = ss * K[1, 2] + (ss - 1) / 2.0
    return Ks


@lru_cache(maxsize=8)
def _distortion_maps(K_bytes: bytes, d_bytes: bytes, size: tuple[int, int], ss: int):
    K = np.frombuffer(K_bytes, np.float64).reshape(3, 3)
    dist = np.frombuffer(d_bytes, np.float64)
    W, H = size[0] * ss, size[1] * ss
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    xl = (xs - (ss - 1) / 2.0) / ss
    yl = (ys - (ss - 1) / 2.0) / ss
    pts = np.stack([xl.ravel(), yl.ravel()], 1).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
    n = cv2.undistortPoints(pts, K, dist, criteria=crit).reshape(H, W, 2)
    Ks = _scaled_K(K, ss)
    mx = (Ks[0, 0] * n[..., 0] + Ks[0, 2]).astype(np.float32)
    my = (Ks[1, 1] * n[..., 1] + Ks[1, 2]).astype(np.float32)
    return mx, my


def motion_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    """Normalised line PSF of the given length and direction (bilinear splat of a fine line)."""
    half = int(math.ceil(length_px / 2.0)) + 1
    k = np.zeros((2 * half + 1, 2 * half + 1), np.float64)
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    for t in np.linspace(-length_px / 2.0, length_px / 2.0, int(length_px * 20) + 1):
        x, y = half + t * c, half + t * s
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        k[y0, x0] += (1 - fx) * (1 - fy)
        k[y0, x0 + 1] += fx * (1 - fy)
        k[y0 + 1, x0] += (1 - fx) * fy
        k[y0 + 1, x0 + 1] += fx * fy
    return (k / k.sum()).astype(np.float32)


def render(
    K: np.ndarray,
    dist,
    rvec,
    tvec,
    size=(1280, 720),
    ss: int = 1,
    blur_sigma: float = 0.0,
    motion: tuple[float, float] | None = None,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
    dark: float = 20.0,
    light: float = 230.0,
    background: float = 90.0,
) -> np.ndarray:
    """Grey uint8 frame of the board at pose (rvec, tvec) seen by camera (K, dist)."""
    K = np.asarray(K, dtype=np.float64)
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).ravel()
    W, H = size
    Ks = _scaled_K(K, ss)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).ravel()
    Hm = Ks @ np.column_stack([R[:, 0], R[:, 1], t]) @ S_TEX
    tex = (dark + (light - dark) * (TEX.astype(np.float32) / 255.0)).astype(np.float32)
    img = cv2.warpPerspective(tex, Hm, (W * ss, H * ss), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    if np.any(dist != 0):
        mx, my = _distortion_maps(K.tobytes(), dist.tobytes(), (W, H), ss)
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma * ss)
    if motion is not None and motion[0] > 0:
        img = cv2.filter2D(img, -1, motion_kernel(motion[0] * ss, motion[1]))
    if ss > 1:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0.0, noise, img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def pose_at(
    K: np.ndarray,
    centre_px,
    distance: float,
    tilt_deg: float = 0.0,
    azimuth_deg: float = 0.0,
    roll_deg: float = 0.0,
):
    """Pose with the board centre at depth ``distance`` on the ray through ``centre_px`` (no distortion)."""
    R = rotation_from_tilt(tilt_deg, azimuth_deg, roll_deg)
    ray = np.array([(centre_px[0] - K[0, 2]) / K[0, 0], (centre_px[1] - K[1, 2]) / K[1, 1], 1.0])
    tvec = distance * ray - R @ BOARD_CENTRE_M
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel(), tvec


def project(K, dist, rvec, tvec, obj=OBJ) -> np.ndarray:
    d = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
    p, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), np.asarray(rvec, float).reshape(3, 1), np.asarray(tvec, float).reshape(3, 1), K, d)
    return p.reshape(-1, 2)


def random_pose(K, size, rng, dist=None, z_range=(0.35, 0.8), max_tilt=50.0, margin=5.0, full=True, tries=200):
    """Random pose whose board corners all project inside the image (``full``) or at least 50 %."""
    W, H = size
    for _ in range(tries):
        tilt = rng.uniform(0, max_tilt)
        az = rng.uniform(0, 360)
        roll = rng.uniform(-30, 30)
        z = rng.uniform(*z_range)
        c = (rng.uniform(0.2 * W, 0.8 * W), rng.uniform(0.2 * H, 0.8 * H))
        rv, tv = pose_at(K, c, z, tilt, az, roll)
        p = project(K, dist, rv, tv)
        inside = (p[:, 0] > margin) & (p[:, 0] < W - 1 - margin) & (p[:, 1] > margin) & (p[:, 1] < H - 1 - margin)
        if (full and inside.all()) or (not full and inside.mean() >= 0.5):
            return rv, tv
    raise RuntimeError("no feasible pose found")


# --- corner-level observations of simulator cameras -------------------------------------------


def camera_views(cam, rng, n, noise=0.25, min_visible=0.5, min_corners=12, margin=5.0):
    """Corner-level views of a simulator ``GTCamera`` (``nextview.sim.lens``) as ``View`` objects.

    Poses come from the simulator's own ``random_pose`` (board centre anywhere in the image,
    partial boards allowed); a corner is kept when the camera really sees it
    (``project_checked``) inside a ``margin`` px border. Observations get Gaussian noise.
    """
    from nextview.sim.poses import random_pose
    from nextview.types import View

    W, H = cam.size
    out = []
    for _ in range(50 * n):
        if len(out) == n:
            break
        rv, tv = random_pose(cam, rng, min_visible=min_visible, margin_px=margin)
        p, ok = cam.project_checked(OBJ, rv, tv)
        ok &= (p[:, 0] >= margin) & (p[:, 0] <= W - 1 - margin) & (p[:, 1] >= margin) & (p[:, 1] <= H - 1 - margin)
        ids = np.nonzero(ok)[0]
        rows, cols = np.unique(ids // (board.SQUARES[0] - 1)).size, np.unique(ids % (board.SQUARES[0] - 1)).size
        if len(ids) < min_corners or rows < 2 or cols < 2:
            continue
        obs = p[ids] + rng.normal(0.0, noise, (len(ids), 2))
        out.append(View(len(out), ids.astype(np.int32), obs, None, np.asarray(rv, float).ravel(), np.asarray(tv, float).ravel()))
    if len(out) < n:
        raise RuntimeError("could not generate enough views")
    return out
