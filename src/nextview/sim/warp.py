"""Warped-board observations at corner level (spec 6.3), for the RO-model failure set.

The board sags as z = a ((x - x_c) / (W_b / 2))^2 with a = 2-4 mm; observations are the
projected sagged corners plus N(0, 0.15 px). Calibration is given the flat object points.
Nothing is rendered: these feed the calibration directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..board import BOARD_SIZE_M, object_points
from .lens import GTCamera

__all__ = ["SAG_RANGE_M", "sample_sag", "sag_object_points", "WarpedObservation", "warped_observation"]

SAG_RANGE_M = (0.002, 0.004)
_OBJ = object_points()


def sample_sag(rng: np.random.Generator) -> float:
    return float(rng.uniform(*SAG_RANGE_M))


def sag_object_points(amplitude_m: float, obj: np.ndarray | None = None) -> np.ndarray:
    """(N, 3) true corner positions of the sagged board, in the board frame."""
    p = (_OBJ if obj is None else np.asarray(obj, dtype=np.float64)).reshape(-1, 3).copy()
    half_w = BOARD_SIZE_M[0] / 2.0
    p[:, 2] = amplitude_m * ((p[:, 0] - half_w) / half_w) ** 2
    return p


@dataclass(frozen=True, eq=False)
class WarpedObservation:
    ids: np.ndarray  # (N,) int32 corner ids inside the image
    corners: np.ndarray  # (N, 2) px, noisy
    obj: np.ndarray  # (N, 3) flat board points (what calibration assumes)
    obj_true: np.ndarray  # (N, 3) sagged points
    amplitude_m: float


def warped_observation(cam: GTCamera, rvec, tvec, amplitude_m: float, rng: np.random.Generator,
                       noise_px: float = 0.15, margin_px: float = 0.0) -> WarpedObservation:
    """Corners of a sagged board seen at (rvec, tvec); always draws 40 x 2 normals."""
    true = sag_object_points(amplitude_m)
    p, seen = cam.project_checked(true, rvec, tvec)
    p = p + noise_px * rng.standard_normal(p.shape)
    W, H = cam.size
    m = margin_px
    keep = seen & (p[:, 0] >= m) & (p[:, 0] <= W - 1 - m) & (p[:, 1] >= m) & (p[:, 1] <= H - 1 - m)
    ids = np.nonzero(keep)[0].astype(np.int32)
    return WarpedObservation(ids, p[keep], _OBJ[keep].copy(), true[keep], float(amplitude_m))
