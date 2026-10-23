"""Image nuisances for simulated captures (spec 6.3).

Every ``sample_*`` function draws a fixed number of random numbers whether or not the nuisance
fires, so two policies that use the same (camera, seed, attempt) stream see the same nuisances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..board import BOARD_SIZE_M, N_CORNERS
from .render import PPM, TEX_MARGIN_PX, board_to_texture

__all__ = [
    "LAMP_CAM_M",
    "NuisanceConfig",
    "NUISANCE_NAMES",
    "motion_kernel",
    "defocus_kernel",
    "defocus_radius",
    "combine_kernels",
    "apply_psf",
    "add_noise",
    "to_uint8",
    "jpeg_roundtrip",
    "OccluderSpec",
    "sample_occluder",
    "draw_occluder",
    "GlareSpec",
    "sample_glare",
    "glare_hotspot",
    "add_glare",
    "CorruptionSpec",
    "sample_corruption",
    "apply_corruption",
]

LAMP_CAM_M = np.array([0.0, -0.4, 0.1])  # lamp position in camera coordinates
NUISANCE_NAMES = ("glare", "autofocus_drift")


@dataclass(frozen=True)
class NuisanceConfig:
    noise_sigma: float = 1.0
    jpeg_quality: int | None = 95
    z_near: float = 0.0  # m; defocus when the board centre is nearer than this
    occluder_p: float = 0.15
    corruption_p: float = 0.05
    glare: bool = False
    autofocus_drift: bool = False
    drift_after_accepted: int = 10
    drift_focal_scale: float = 1.015
    drift_dk1: float = 0.01
    names: tuple[str, ...] = field(default=())

    @classmethod
    def for_camera(cls, cam, names=(), **overrides) -> "NuisanceConfig":
        """Family noise/JPEG/near-focus from ``cam`` plus the named preset nuisances."""
        names = tuple(names)
        unknown = [n for n in names if n not in NUISANCE_NAMES]
        if unknown:
            raise ValueError(f"unknown nuisances {unknown}; known: {list(NUISANCE_NAMES)}")
        kw = dict(
            noise_sigma=float(cam.noise_sigma),
            jpeg_quality=cam.jpeg_quality,
            z_near=float(cam.z_near),
            glare="glare" in names,
            autofocus_drift="autofocus_drift" in names,
            names=names,
        )
        kw.update(overrides)
        return cls(**kw)

    @classmethod
    def clean(cls) -> "NuisanceConfig":
        """No noise, no JPEG, no occluders, no corruption: geometry only."""
        return cls(noise_sigma=0.0, jpeg_quality=None, z_near=0.0, occluder_p=0.0, corruption_p=0.0)


# ---- point-spread functions -------------------------------------------------------------

def motion_kernel(length_px: float, angle_deg: float) -> np.ndarray | None:
    """Normalised line PSF of ``length_px`` centred on the anchor (no image shift). None if < 0.05 px."""
    L = float(length_px)
    if L < 0.05:
        return None
    half = int(math.ceil(L / 2.0)) + 1
    size = 2 * half + 1
    k = np.zeros((size, size), np.float64)
    a = math.radians(angle_deg)
    n = max(int(math.ceil(L * 8)), 2) + 1
    s = np.linspace(-L / 2.0, L / 2.0, n)
    xs = half + s * math.cos(a)
    ys = half + s * math.sin(a)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(k, (y0 + dy, x0 + dx), w)
    return (k / k.sum()).astype(np.float32)


def defocus_radius(Z: float, z_near: float) -> float:
    """r = min(4, 8 (z_near - Z) / z_near) px when Z < z_near, else 0."""
    if z_near <= 0.0 or Z >= z_near:
        return 0.0
    return float(min(4.0, 8.0 * (z_near - Z) / z_near))


def defocus_kernel(radius_px: float, oversample: int = 8) -> np.ndarray | None:
    """Normalised disk PSF with anti-aliased rim. None if the radius is below 0.25 px."""
    r = float(radius_px)
    if r < 0.25:
        return None
    half = int(math.ceil(r)) + 1
    size = 2 * half + 1
    o = oversample
    g = (np.arange(size * o) + 0.5) / o - 0.5 - half
    X, Y = np.meshgrid(g, g)
    inside = (X * X + Y * Y <= r * r).astype(np.float64)
    k = inside.reshape(size, o, size, o).mean(axis=(1, 3))
    return (k / k.sum()).astype(np.float32)


def combine_kernels(*kernels: np.ndarray | None) -> np.ndarray | None:
    ks = [k for k in kernels if k is not None]
    if not ks:
        return None
    out = ks[0].astype(np.float64)
    for k in ks[1:]:
        ph, pw = k.shape[0] // 2, k.shape[1] // 2
        padded = np.pad(out, ((ph, ph), (pw, pw)))
        flipped = np.ascontiguousarray(k.astype(np.float64)[::-1, ::-1])
        out = cv2.filter2D(padded, -1, flipped, borderType=cv2.BORDER_CONSTANT)
    return (out / out.sum()).astype(np.float32)


def apply_psf(img: np.ndarray, kernel: np.ndarray | None) -> np.ndarray:
    if kernel is None:
        return img
    # filter2D correlates; flip so this is a true convolution (the kernels here are symmetric anyway).
    return cv2.filter2D(img, -1, np.ascontiguousarray(kernel[::-1, ::-1]), borderType=cv2.BORDER_REPLICATE)


# ---- sensor ------------------------------------------------------------------------------

def add_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise; always draws H*W normals so the stream stays aligned."""
    n = rng.standard_normal(img.shape, dtype=np.float32)
    if sigma <= 0.0:
        return img
    return img + np.float32(sigma) * n


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def jpeg_roundtrip(img: np.ndarray, quality: int | None) -> np.ndarray:
    if quality is None:
        return img
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if out is None:
        raise RuntimeError("JPEG decoding failed")
    return out


# ---- occluder ----------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class OccluderSpec:
    active: bool
    polygon_m: np.ndarray  # (n, 2) convex polygon in board metres
    grey: float
    area_frac: float  # polygon area / board area (before clipping at the board edge)

    def to_dict(self) -> dict:
        return {"active": self.active, "polygon_m": self.polygon_m.round(5).tolist(),
                "grey": round(self.grey, 2), "area_frac": round(self.area_frac, 4)}


def sample_occluder(rng: np.random.Generator, p: float = 0.15) -> OccluderSpec:
    """A random convex 4-6 vertex polygon, grey 60-120, covering 5-20 % of the board; fires with prob. p.

    Vertices sit in angular order on a rotated ellipse, which keeps the polygon convex.
    """
    u = rng.uniform(0.0, 1.0, 14)
    active = bool(u[0] < p)
    n_vert = 4 + min(int(u[1] * 3), 2)
    frac = 0.05 + 0.15 * u[2]
    grey = 60.0 + 60.0 * u[3]
    bw, bh = BOARD_SIZE_M
    cx = bw * (0.15 + 0.7 * u[4])
    cy = bh * (0.15 + 0.7 * u[5])
    k = np.arange(n_vert)
    ang = (k + 0.7 * (u[6 : 6 + n_vert] - 0.5)) * (2.0 * math.pi / n_vert)
    aspect = 0.5 + 0.5 * u[12]
    rot = math.pi * u[13]
    pts = np.column_stack([np.cos(ang), aspect * np.sin(ang)])
    c, s_ = math.cos(rot), math.sin(rot)
    pts = pts @ np.array([[c, s_], [-s_, c]])
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    scale = math.sqrt(frac * bw * bh / area)
    poly = pts * scale + [cx, cy]
    return OccluderSpec(active, poly, grey, frac)


def draw_occluder(texture: np.ndarray, spec: OccluderSpec) -> np.ndarray:
    """Paint the occluder into a texture copy (board-plane coordinates)."""
    if not spec.active:
        return texture
    out = texture.copy()
    shift = 4
    pts = np.rint(board_to_texture(spec.polygon_m) * (1 << shift)).astype(np.int32)
    cv2.fillConvexPoly(out, pts, float(spec.grey), lineType=cv2.LINE_8, shift=shift)
    return out


# ---- glare -------------------------------------------------------------------------------

@dataclass(frozen=True)
class GlareSpec:
    amplitude: float  # 120-255 grey levels
    sigma_frac: float  # 0.05-0.12 board widths


def sample_glare(rng: np.random.Generator) -> GlareSpec:
    u = rng.uniform(0.0, 1.0, 2)
    return GlareSpec(amplitude=120.0 + 135.0 * float(u[0]), sigma_frac=0.05 + 0.07 * float(u[1]))


def glare_hotspot(rvec, tvec, lamp=LAMP_CAM_M, margin_m: float = TEX_MARGIN_PX / PPM) -> np.ndarray | None:
    """Specular hotspot of ``lamp`` on the board plane, in board metres, or None if it misses the board.

    ``margin_m`` defaults to the white margin around the squares, i.e. the whole glossy surface.

    The hotspot is where the line from the lamp to the mirror image of the camera centre crosses
    the board plane; the lamp and the camera must be on the same side of the board.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    n = R[:, 2]
    L = np.asarray(lamp, dtype=np.float64)
    side_cam = float((np.zeros(3) - t) @ n)
    side_lamp = float((L - t) @ n)
    if side_cam * side_lamp <= 0.0:
        return None
    C_mirror = 2.0 * float(t @ n) * n  # origin reflected about the plane
    d = C_mirror - L
    denom = float(d @ n)
    if abs(denom) < 1e-12:
        return None
    lam = float((t - L) @ n) / denom
    P = L + lam * d
    b = R.T @ (P - t)
    bw, bh = BOARD_SIZE_M
    if -margin_m <= b[0] <= bw + margin_m and -margin_m <= b[1] <= bh + margin_m:
        return b[:2]
    return None


def add_glare(texture: np.ndarray, centre_m: np.ndarray, spec: GlareSpec) -> np.ndarray:
    """Add a Gaussian hotspot to a texture copy; it saturates once the image is clipped to 8 bits."""
    th, tw = texture.shape[:2]
    cu, cv_ = board_to_texture(np.asarray(centre_m).reshape(1, 2))[0]
    sig = spec.sigma_frac * BOARD_SIZE_M[0] * PPM
    xs = np.arange(tw, dtype=np.float32) - np.float32(cu)
    ys = np.arange(th, dtype=np.float32) - np.float32(cv_)
    gx = np.exp(-0.5 * (xs / sig) ** 2)
    gy = np.exp(-0.5 * (ys / sig) ** 2)
    return texture + np.float32(spec.amplitude) * np.outer(gy, gx).astype(np.float32)


# ---- corruption (applied after detection) ------------------------------------------------

@dataclass(frozen=True, eq=False)
class CorruptionSpec:
    """Shift three detected corners by 2-5 px. ``keys`` rank the 40 corner ids; the three
    detected ids with the smallest keys are the ones moved."""

    active: bool
    keys: np.ndarray  # (40,)
    offsets: np.ndarray  # (3, 2) px

    def to_dict(self) -> dict:
        return {"active": self.active, "offsets": self.offsets.round(3).tolist()}


def sample_corruption(rng: np.random.Generator, p: float = 0.05) -> CorruptionSpec:
    u = rng.uniform(0.0, 1.0, 1 + N_CORNERS + 6)
    active = bool(u[0] < p)
    keys = u[1 : 1 + N_CORNERS].copy()
    mag = 2.0 + 3.0 * u[1 + N_CORNERS : 4 + N_CORNERS]
    ang = 2.0 * math.pi * u[4 + N_CORNERS : 7 + N_CORNERS]
    offsets = np.column_stack([mag * np.cos(ang), mag * np.sin(ang)])
    return CorruptionSpec(active, keys, offsets)


def apply_corruption(ids: np.ndarray, corners: np.ndarray, spec: CorruptionSpec | None):
    """Return (corners, moved_ids). Corners are (N, 2) px; unchanged when the spec is inactive."""
    ids = np.asarray(ids).ravel().astype(int)
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if spec is None or not spec.active or len(ids) == 0:
        return corners.copy(), np.array([], dtype=int)
    order = np.argsort(spec.keys[ids], kind="stable")[:3]
    out = corners.copy()
    out[order] += spec.offsets[: len(order)]
    return out, ids[order]
