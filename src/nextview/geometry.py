"""Board pose geometry shared by the tools, the planner and the simulator.

Conventions
-----------
* Camera frame: x right, y down, z forward (OpenCV).
* Board frame: origin at the top-left outer corner of the board, x along the squares row,
  y down the board, z into the board. ``rvec = 0`` is a fronto-parallel board facing the camera.
* Tilt is the angle between the board normal and the optical axis. The tilt azimuth is the
  image-plane direction (degrees from +x towards +y) of the board edge that moves away from
  the camera: 0 = right edge away, 90 = bottom edge away, 180 = left edge away, 270 = top edge away.
* Pixel coordinates use the OpenCV pixel-centre convention (pixel (0, 0) is centred on (0, 0)).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .board import BOARD_SIZE_M

__all__ = [
    "BOARD_CENTRE_M",
    "BOARD_OUTLINE_M",
    "as_rotation",
    "rotation_from_tilt",
    "rvec_from_tilt",
    "tilt_of",
    "tilt_azimuth_of",
    "roll_of",
    "distance_of",
    "pose_from_lattice",
    "project",
    "ghost_quad",
    "board_centre_px",
    "grid_centres",
    "polygon_area",
]

BOARD_CENTRE_M = np.array([BOARD_SIZE_M[0] / 2.0, BOARD_SIZE_M[1] / 2.0, 0.0])
# Outer board outline, clockwise in the image for rvec = 0: TL, TR, BR, BL.
BOARD_OUTLINE_M = np.array(
    [
        [0.0, 0.0, 0.0],
        [BOARD_SIZE_M[0], 0.0, 0.0],
        [BOARD_SIZE_M[0], BOARD_SIZE_M[1], 0.0],
        [0.0, BOARD_SIZE_M[1], 0.0],
    ]
)


def as_rotation(r: np.ndarray) -> np.ndarray:
    """Rotation matrix from a Rodrigues vector, or the matrix itself if already 3 x 3."""
    r = np.asarray(r, dtype=np.float64)
    if r.shape == (3, 3):
        return r
    R, _ = cv2.Rodrigues(r.reshape(3, 1))
    return R


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_from_tilt(tilt_deg: float, azimuth_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """Board rotation: in-plane roll first, then a tilt that sends the ``azimuth`` edge away."""
    az = math.radians(azimuth_deg)
    axis = np.array([math.sin(az), -math.cos(az), 0.0]) * math.radians(tilt_deg)
    R_tilt, _ = cv2.Rodrigues(axis.reshape(3, 1))
    return R_tilt @ _rot_z(roll_deg)


def rvec_from_tilt(tilt_deg: float, azimuth_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(rotation_from_tilt(tilt_deg, azimuth_deg, roll_deg))
    return rvec.ravel()


def tilt_of(rvec: np.ndarray) -> float:
    """Angle in degrees between the board normal and the camera optical axis."""
    R = as_rotation(rvec)
    return math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))


def tilt_azimuth_of(rvec: np.ndarray) -> float:
    """Image direction (degrees, [0, 360)) of the board edge farthest from the camera."""
    R = as_rotation(rvec)
    return math.degrees(math.atan2(-R[1, 2], -R[0, 2])) % 360.0


def roll_of(rvec: np.ndarray) -> float:
    """In-plane roll in degrees, (-180, 180], after undoing the tilt (inverse of ``rotation_from_tilt``)."""
    R = as_rotation(rvec)
    tilt = tilt_of(R)
    az = tilt_azimuth_of(R) if tilt > 1e-9 else 0.0
    R_roll = rotation_from_tilt(tilt, az, 0.0).T @ R
    return math.degrees(math.atan2(R_roll[1, 0], R_roll[0, 0]))


def distance_of(rvec: np.ndarray, tvec: np.ndarray) -> float:
    """Distance in metres from the camera centre to the board centre."""
    R = as_rotation(rvec)
    return float(np.linalg.norm(R @ BOARD_CENTRE_M + np.asarray(tvec, dtype=np.float64).ravel()))


def pose_from_lattice(
    K: np.ndarray,
    centre_px: tuple[float, float],
    size_frac: float,
    tilt_deg: float,
    azimuth_deg: float,
    roll_deg: float,
    image_size: tuple[int, int],
    dist: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pose whose board centre projects to ``centre_px`` at width ``size_frac`` of the image.

    Depth of the board centre: Z = fx * board_width / (size_frac * W).
    Returns ``(rvec (3,), tvec (3,))``.
    """
    K = np.asarray(K, dtype=np.float64)
    W = float(image_size[0])
    Z = K[0, 0] * BOARD_SIZE_M[0] / (size_frac * W)
    pt = np.array([[centre_px]], dtype=np.float64)
    if dist is not None and np.any(np.asarray(dist) != 0):
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
        xn = cv2.undistortPoints(pt, K, np.asarray(dist, dtype=np.float64), criteria=crit).reshape(2)
    else:
        xn = np.array([(centre_px[0] - K[0, 2]) / K[0, 0], (centre_px[1] - K[1, 2]) / K[1, 1]])
    R = rotation_from_tilt(tilt_deg, azimuth_deg, roll_deg)
    centre_cam = Z * np.array([xn[0], xn[1], 1.0])
    tvec = centre_cam - R @ BOARD_CENTRE_M
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel(), tvec


def project(points_m: np.ndarray, rvec, tvec, K, dist=None) -> np.ndarray:
    """(N, 3) board-frame points -> (N, 2) pixels."""
    d = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).ravel()
    p, _ = cv2.projectPoints(
        np.asarray(points_m, dtype=np.float64).reshape(-1, 1, 3),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64),
        d,
    )
    return p.reshape(-1, 2)


def ghost_quad(rvec, tvec, K, dist=None) -> np.ndarray:
    """(4, 2) image outline of the board at the given pose: TL, TR, BR, BL."""
    return project(BOARD_OUTLINE_M, rvec, tvec, K, dist)


def board_centre_px(rvec, tvec, K, dist=None) -> np.ndarray:
    return project(BOARD_CENTRE_M[None, :], rvec, tvec, K, dist)[0]


def grid_centres(size: tuple[int, int], grid: tuple[int, int]) -> np.ndarray:
    """Cell centres of a (cols, rows) grid over a (W, H) image, row-major, shape (rows * cols, 2).

    Pixel-centre convention: for 1280 x 720 and (32, 18) the centres are 19.5, 59.5, ..., 1259.5
    by 19.5, 59.5, ..., 699.5 (the same grid as ``sim.truth.grid_centres``).
    """
    W, H = size
    cols, rows = grid
    xs = (np.arange(cols) + 0.5) * (W / cols) - 0.5
    ys = (np.arange(rows) + 0.5) * (H / rows) - 0.5
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def polygon_area(poly: np.ndarray) -> float:
    """Unsigned area of a simple polygon (N, 2)."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
