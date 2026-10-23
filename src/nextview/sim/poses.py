"""Board poses for the simulator: (centre pixel, size fraction, tilt, azimuth, roll) -> (rvec, tvec).

A pose maps board-frame points to camera coordinates: X_cam = R X_board + t. The identity
rotation is a fronto-parallel board that reads the right way round in the image. Tilt, azimuth
and roll follow the conventions of ``nextview.geometry`` (the planner uses the same ones).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..board import BOARD_SIZE_M, object_points
from ..geometry import BOARD_CENTRE_M, pose_from_lattice
from .lens import GTCamera

__all__ = [
    "board_centre_cam",
    "pose_for_view",
    "visible_fraction",
    "random_pose",
]

_OBJ = object_points()


def board_centre_cam(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R @ BOARD_CENTRE_M + np.asarray(tvec, dtype=np.float64).reshape(3)


def pose_for_view(cam: GTCamera, centre_px, size_frac: float, tilt_deg: float = 0.0,
                  azimuth_deg: float = 0.0, roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Pose whose board centre projects to ``centre_px`` at depth Z = fx * board_width / (s * W).

    ``geometry.pose_from_lattice`` with the GT camera's exact inverse projection (which also
    covers the fisheye model): the centre pixel is first moved to its undistorted position.
    """
    n = cam.pixel_to_normalized(np.asarray(centre_px, dtype=np.float64).reshape(1, 2))[0]
    if not np.all(np.isfinite(n)):
        raise ValueError(f"pixel {centre_px} has no forward ray")
    K = cam.K
    undist_px = (K[0, 0] * n[0] + K[0, 2], K[1, 1] * n[1] + K[1, 2])
    rvec, tvec = pose_from_lattice(K, undist_px, size_frac, tilt_deg, azimuth_deg, roll_deg, cam.size)
    return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)


def visible_fraction(cam: GTCamera, rvec, tvec, margin_px: float = 5.0, min_depth_m: float = 0.05) -> float:
    """Fraction of the 40 corners that project inside the image with ``margin_px`` to spare.

    Returns 0 when any part of the board is closer than ``min_depth_m`` (or behind the camera).
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    outline = np.array([[0, 0, 0], [BOARD_SIZE_M[0], 0, 0], [BOARD_SIZE_M[0], BOARD_SIZE_M[1], 0],
                        [0, BOARD_SIZE_M[1], 0]], dtype=np.float64)
    if np.min((outline @ R.T + t)[:, 2]) < min_depth_m:
        return 0.0
    p, ok_ray = cam.project_checked(_OBJ, rvec, tvec)
    W, H = cam.size
    m = margin_px
    inside = ok_ray & (p[:, 0] >= m) & (p[:, 0] <= W - 1 - m) & (p[:, 1] >= m) & (p[:, 1] <= H - 1 - m)
    return float(inside.mean())


def random_pose(cam: GTCamera, rng: np.random.Generator, size_range=(0.3, 0.8), max_tilt_deg: float = 50.0,
                max_roll_deg: float = 30.0, min_visible: float = 1.0, margin_px: float = 5.0,
                max_tries: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """A random feasible pose: centre uniform over the image, size fraction, tilt, azimuth and roll uniform.

    Draws exactly six numbers per try, so a fixed rng state always yields the same pose.
    """
    W, H = cam.size
    for _ in range(max_tries):
        u = rng.uniform(0.0, 1.0, 6)
        centre = (u[0] * (W - 1), u[1] * (H - 1))
        s = size_range[0] + u[2] * (size_range[1] - size_range[0])
        tilt = u[3] * max_tilt_deg
        az = u[4] * 360.0
        roll = (2.0 * u[5] - 1.0) * max_roll_deg
        try:
            rvec, tvec = pose_for_view(cam, centre, s, tilt, az, roll)
        except ValueError:
            continue
        if visible_fraction(cam, rvec, tvec, margin_px) >= min_visible:
            return rvec, tvec
    raise RuntimeError("no feasible random pose found")
