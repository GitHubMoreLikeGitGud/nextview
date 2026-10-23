"""Ground-truth lens families and exact camera models for the simulator (spec section 6.1).

Every camera renders at 1280x720. Pinhole families use OpenCV's distortion model
(``k5``: k1 k2 p1 p2 k3; ``rational``: k1 k2 p1 p2 k3 k4 k5 k6); ``fisheye130`` uses the
``cv2.fisheye`` model. Pixel coordinates follow OpenCV: integer values are pixel centres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = [
    "IMAGE_SIZE",
    "UNDISTORT_CRITERIA",
    "FISHEYE_MAX_THETA_DEG",
    "FamilySpec",
    "LENS_FAMILIES",
    "GTCamera",
    "sample_camera",
    "make_camera",
]

IMAGE_SIZE = (1280, 720)  # (W, H)
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9)
FISHEYE_MAX_THETA_DEG = 75.0  # rays beyond this are not rendered (spec 6.2)

_FAMILY_SALT = {"normal": 11, "wide": 12, "tele": 13, "webcam": 14, "fisheye130": 15}


@dataclass(frozen=True)
class FamilySpec:
    name: str
    model: str  # GT model: "k5" | "rational" | "fisheye"
    hfov_deg: tuple[float, float]
    noise_sigma: float
    jpeg_quality: int | None
    pp_jitter_px: tuple[float, float]  # principal point offset bound (x, y)
    z_near: float = 0.0  # m; defocus below this board distance (0 = never)
    failure_only: bool = False
    note: str = ""


LENS_FAMILIES: dict[str, FamilySpec] = {
    "normal": FamilySpec("normal", "k5", (55.0, 65.0), 1.0, 95, (30.0, 20.0), note="baseline case"),
    "wide": FamilySpec("wide", "rational", (95.0, 110.0), 2.0, 95, (30.0, 20.0), note="model-selection case"),
    "tele": FamilySpec("tele", "k5", (20.0, 28.0), 1.0, 95, (40.0, 40.0), note="principal point hard to pin down"),
    "webcam": FamilySpec("webcam", "k5", (65.0, 75.0), 4.0, 70, (30.0, 20.0), z_near=0.25, note="gate case"),
    "fisheye130": FamilySpec(
        "fisheye130", "fisheye", (130.0, 130.0), 2.0, 95, (20.0, 20.0), failure_only=True,
        note="out-of-scope abstention",
    ),
}


def _radial_pinhole(r2: np.ndarray, d: np.ndarray) -> np.ndarray:
    k1, k2, _, _, k3 = d[:5]
    k4, k5, k6 = (d[5], d[6], d[7]) if len(d) >= 8 else (0.0, 0.0, 0.0)
    num = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    den = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
    return num / den


def _theta_d(theta: np.ndarray, d: np.ndarray) -> np.ndarray:
    t2 = theta * theta
    return theta * (1.0 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3]))))


def _theta_d_prime(theta: np.ndarray, d: np.ndarray) -> np.ndarray:
    t2 = theta * theta
    return 1.0 + t2 * (3 * d[0] + t2 * (5 * d[1] + t2 * (7 * d[2] + t2 * 9 * d[3])))


@dataclass(frozen=True, eq=False)
class GTCamera:
    """A ground-truth camera. ``K`` is 3x3 float64; ``dist`` has 5 (k5), 8 (rational) or 4 (fisheye) entries."""

    family: str
    model: str
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int] = IMAGE_SIZE
    noise_sigma: float = 1.0
    jpeg_quality: int | None = 95
    z_near: float = 0.0
    seed: int | None = None
    meta: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        d = np.asarray(self.dist, dtype=np.float64).ravel()
        expected = {"k5": 5, "rational": 8, "fisheye": 4}
        if self.model not in expected:
            raise ValueError(f"unknown model {self.model!r}")
        if len(d) != expected[self.model]:
            raise ValueError(f"model {self.model} needs {expected[self.model]} coefficients, got {len(d)}")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "dist", d)
        object.__setattr__(self, "size", (int(self.size[0]), int(self.size[1])))

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def pinhole(cls, K, dist=None, size=IMAGE_SIZE, **kw) -> "GTCamera":
        """A camera from plain intrinsics (e.g. an estimate or K0 with zero distortion)."""
        d = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).ravel()
        if len(d) < 5:
            d = np.concatenate([d, np.zeros(5 - len(d))])
        model = "k5" if len(d) == 5 else "rational"
        if len(d) not in (5, 8):
            raise ValueError("pinhole() takes 4, 5 or 8 distortion coefficients")
        return cls(family=kw.pop("family", "custom"), model=model, K=K, dist=d, size=size, **kw)

    def drifted(self, focal_scale: float = 1.015, dk1: float = 0.01) -> "GTCamera":
        """The same camera after a focus/zoom change: fx, fy scaled and k1 shifted."""
        K = self.K.copy()
        K[0, 0] *= focal_scale
        K[1, 1] *= focal_scale
        d = self.dist.copy()
        d[0] += dk1
        meta = dict(self.meta, drifted=True)
        return GTCamera(self.family, self.model, K, d, self.size, self.noise_sigma, self.jpeg_quality,
                        self.z_near, self.seed, meta)

    # ---- properties ---------------------------------------------------------------------
    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def key(self) -> tuple:
        return (self.model, self.K.tobytes(), self.dist.tobytes(), self.size)

    @property
    def hfov_deg(self) -> float:
        """True horizontal field of view between the outer pixel edges on the principal row."""
        W = self.size[0]
        uv = np.array([[-0.5, self.cy], [W - 0.5, self.cy]])
        r = self.pixel_to_ray(uv)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        return float(np.degrees(np.arccos(np.clip(r[0] @ r[1], -1.0, 1.0))))

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "model": self.model,
            "K": self.K.round(6).tolist(),
            "dist": self.dist.round(8).tolist(),
            "size": list(self.size),
            "noise_sigma": self.noise_sigma,
            "jpeg_quality": self.jpeg_quality,
            "z_near": self.z_near,
            "seed": self.seed,
            "hfov_deg": round(self.hfov_deg, 3),
        }

    # ---- projection ---------------------------------------------------------------------
    def project(self, obj: np.ndarray, rvec, tvec) -> np.ndarray:
        """Project (N, 3) points from a frame with pose (rvec, tvec) to (N, 2) pixels."""
        obj = np.ascontiguousarray(np.asarray(obj, dtype=np.float64).reshape(-1, 1, 3))
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if self.model == "fisheye":
            p, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, self.K, self.dist)
        else:
            p, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        return p.reshape(-1, 2)

    def project_camera_points(self, X: np.ndarray) -> np.ndarray:
        """Project (N, 3) camera-frame points to (N, 2) pixels."""
        return self.project(X, np.zeros(3), np.zeros(3))

    def project_checked(self, obj: np.ndarray, rvec=None, tvec=None) -> tuple[np.ndarray, np.ndarray]:
        """Project and report which points the camera really sees: returns ((N, 2) px, (N,) bool).

        A point counts only if it is in front of the camera and its pixel maps back to its own ray.
        Strong pinhole distortion polynomials fold rays from outside the field of view back into
        the frame; those points come back False. Fisheye rays beyond the rendered 75 degrees are
        False too. The pixel may still lie outside the image; callers check bounds.
        """
        X = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
        if rvec is not None:
            R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
            X = X @ R.T + np.asarray(tvec, dtype=np.float64).reshape(3)
        uv = self.project_camera_points(X)
        ok = X[:, 2] > 1e-9
        n = np.zeros((len(X), 2))
        n[ok] = X[ok, :2] / X[ok, 2:3]
        if self.model == "fisheye":
            theta = np.degrees(np.arctan2(np.hypot(X[:, 0], X[:, 1]), X[:, 2]))
            ok &= theta <= FISHEYE_MAX_THETA_DEG
        elif ok.any():
            back = np.full((len(X), 2), np.inf)
            back[ok] = self.pixel_to_normalized(uv[ok])
            ok &= np.all(np.abs(back - n) <= 1e-6 * (1.0 + np.abs(n)), axis=1)
        ok &= np.all(np.isfinite(uv), axis=1)
        return uv, ok

    def pixel_to_normalized(self, uv: np.ndarray) -> np.ndarray:
        """Exact inverse of the projection: (N, 2) pixels -> (N, 2) undistorted (x/z, y/z).

        Fisheye pixels whose ray is at or beyond 90 degrees off-axis come back as NaN.
        """
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if self.model == "fisheye":
            return self._fisheye_unproject(uv)
        out = cv2.undistortPoints(uv.reshape(-1, 1, 2), self.K, self.dist, criteria=UNDISTORT_CRITERIA)
        return out.reshape(-1, 2)

    def pixel_to_ray(self, uv: np.ndarray) -> np.ndarray:
        n = self.pixel_to_normalized(uv)
        return np.column_stack([n, np.ones(len(n))])

    def ray_angle_deg(self, uv: np.ndarray) -> np.ndarray:
        """Angle of each pixel's ray from the optical axis, in degrees."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if self.model == "fisheye":
            xd = (uv[:, 0] - self.cx) / self.fx
            yd = (uv[:, 1] - self.cy) / self.fy
            return np.degrees(self._solve_theta(np.hypot(xd, yd)))
        n = self.pixel_to_normalized(uv)
        return np.degrees(np.arctan(np.hypot(n[:, 0], n[:, 1])))

    def _solve_theta(self, theta_d: np.ndarray) -> np.ndarray:
        d = self.dist
        theta = theta_d.astype(np.float64).copy()
        for _ in range(60):
            step = (_theta_d(theta, d) - theta_d) / _theta_d_prime(theta, d)
            theta -= step
            if np.nanmax(np.abs(step), initial=0.0) < 1e-15:
                break
        return theta

    def _fisheye_unproject(self, uv: np.ndarray) -> np.ndarray:
        xd = (uv[:, 0] - self.cx) / self.fx
        yd = (uv[:, 1] - self.cy) / self.fy
        td = np.hypot(xd, yd)
        theta = self._solve_theta(td)
        out = np.full((len(uv), 2), np.nan)
        ok = theta < math.radians(89.9)
        small = td < 1e-12
        scale = np.where(small, 1.0, np.tan(np.where(ok, theta, 0.0)) / np.where(small, 1.0, td))
        out[ok, 0] = xd[ok] * scale[ok]
        out[ok, 1] = yd[ok] * scale[ok]
        return out


# ---- sampling ---------------------------------------------------------------------------

def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _sample_dist(family: str, rng: np.random.Generator) -> np.ndarray:
    if family == "normal":
        return np.array([_u(rng, -0.15, 0.05), _u(rng, -0.05, 0.10), _u(rng, -1e-3, 1e-3),
                         _u(rng, -1e-3, 1e-3), _u(rng, -0.03, 0.02)])
    if family == "wide":
        # Pade-like k1/k4 pair: r (1 + k1 r^2) / (1 + k4 r^2) with k4 - k1 in [0.40, 0.55]. Its series
        # in r^2 does not converge over the image (k4 r^2 > 1 near the edges), so a k5 polynomial
        # cannot follow it. k3 >= 0 and a small k6 keep the radial map rising past the image corners.
        k1 = _u(rng, 0.10, 0.30)
        return np.array([k1, _u(rng, 0.0, 0.02), _u(rng, -5e-4, 5e-4), _u(rng, -5e-4, 5e-4),
                         _u(rng, 0.0, 0.005), k1 + _u(rng, 0.40, 0.55), _u(rng, 0.0, 0.03),
                         _u(rng, 0.0, 0.002)])
    if family == "tele":
        return np.array([_u(rng, -0.05, 0.05), 0.0, 0.0, 0.0, 0.0])
    if family == "webcam":
        return np.array([_u(rng, -0.10, 0.05), _u(rng, -0.05, 0.05), _u(rng, -5e-4, 5e-4),
                         _u(rng, -5e-4, 5e-4), _u(rng, -0.01, 0.01)])
    if family == "fisheye130":
        return np.array([_u(rng, -0.05, 0.05), _u(rng, -0.02, 0.02), _u(rng, -0.005, 0.005),
                         _u(rng, -0.002, 0.002)])
    raise ValueError(f"unknown lens family {family!r}; known: {list(LENS_FAMILIES)}")


def _corner_radius(K: np.ndarray, size: tuple[int, int]) -> float:
    W, H = size
    c = np.array([[-0.5, -0.5], [W - 0.5, -0.5], [-0.5, H - 0.5], [W - 0.5, H - 0.5]])
    x = (c[:, 0] - K[0, 2]) / K[0, 0]
    y = (c[:, 1] - K[1, 2]) / K[1, 1]
    return float(np.max(np.hypot(x, y)))


def _pinhole_radial_ok(d: np.ndarray, rd_needed: float, ru_edge: float, margin: float = 1.03) -> bool:
    """The radial map r -> r * radial(r) must rise monotonically past the image corner,
    and the nominal edge ray ``ru_edge`` must lie on that rising branch."""
    r = np.linspace(0.0, 12.0, 24001)
    rd = r * _radial_pinhole(r * r, d)
    inc = np.diff(rd) > 0
    if not inc[0]:
        return False
    stop = int(np.argmin(inc)) if not inc.all() else len(inc)
    if ru_edge >= r[stop]:
        return False
    return float(rd[: stop + 1].max()) >= margin * rd_needed


def _fisheye_ok(d: np.ndarray, td_needed: float) -> bool:
    th = np.linspace(0.0, math.radians(89.0), 8901)
    td = _theta_d(th, d)
    if not np.all(np.diff(td) > 0):
        return False
    return float(td[-1]) >= td_needed


def _roundtrip_ok(cam: GTCamera, tol_px: float = 1e-6) -> bool:
    W, H = cam.size
    xs = np.linspace(0, W - 1, 33)
    ys = np.linspace(0, H - 1, 19)
    g = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    n = cam.pixel_to_normalized(g)
    ok = np.all(np.isfinite(n), axis=1)
    if cam.model != "fisheye" and not ok.all():
        return False
    X = np.column_stack([n[ok], np.ones(int(ok.sum()))])
    back = cam.project_camera_points(X)
    return bool(np.max(np.linalg.norm(back - g[ok], axis=1)) < tol_px)


def sample_camera(family: str, rng: np.random.Generator, size: tuple[int, int] = IMAGE_SIZE,
                  seed: int | None = None, max_tries: int = 500) -> GTCamera:
    """Draw a GT camera from a lens family.

    fx is set so that the ray through the image's horizontal edge (at W/2 from the principal
    point) is at half the sampled HFOV, distortion included. Samples whose radial map is not
    monotonic across the whole image, or that fail an exact projection round trip, are redrawn.
    """
    if family not in LENS_FAMILIES:
        raise ValueError(f"unknown lens family {family!r}; known: {list(LENS_FAMILIES)}")
    spec = LENS_FAMILIES[family]
    W, H = size
    for attempt in range(max_tries):
        hfov = _u(rng, *spec.hfov_deg) if spec.hfov_deg[0] != spec.hfov_deg[1] else spec.hfov_deg[0]
        d = _sample_dist(family, rng)
        aspect = _u(rng, 0.995, 1.005)
        ppx = _u(rng, -spec.pp_jitter_px[0], spec.pp_jitter_px[0])
        ppy = _u(rng, -spec.pp_jitter_px[1], spec.pp_jitter_px[1])
        half = math.radians(hfov / 2.0)
        if spec.model == "fisheye":
            rd_edge = float(_theta_d(np.array(half), d))
        else:
            ru = math.tan(half)
            rd_edge = ru * float(_radial_pinhole(np.array(ru * ru), d))
        if not rd_edge > 0:
            continue
        fx = (W / 2.0) / rd_edge
        K = np.array([[fx, 0.0, W / 2.0 + ppx], [0.0, fx * aspect, H / 2.0 + ppy], [0.0, 0.0, 1.0]])
        rc = _corner_radius(K, size)
        if spec.model == "fisheye":
            if not _fisheye_ok(d, rc):
                continue
        elif not _pinhole_radial_ok(d, rc, math.tan(half)):
            continue
        cam = GTCamera(family, spec.model, K, d, size, spec.noise_sigma, spec.jpeg_quality, spec.z_near, seed,
                       {"hfov_nominal_deg": hfov, "sample_attempts": attempt + 1})
        if not _roundtrip_ok(cam):
            continue
        return cam
    raise RuntimeError(f"could not sample a valid {family} camera in {max_tries} tries")


def make_camera(family: str, camera_seed: int, size: tuple[int, int] = IMAGE_SIZE) -> GTCamera:
    """The camera for (family, camera_seed); the same pair always gives the same camera."""
    if family not in _FAMILY_SALT:
        raise ValueError(f"unknown lens family {family!r}; known: {list(LENS_FAMILIES)}")
    rng = np.random.default_rng([int(camera_seed), _FAMILY_SALT[family]])
    return sample_camera(family, rng, size=size, seed=int(camera_seed))
