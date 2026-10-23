"""Shared data contracts passed between the tools, the agent, the simulator and the benchmark.

Array-bearing records use ``eq=False``: element-wise ``==`` on numpy arrays is ambiguous,
so these compare and hash by identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Detection",
    "Quality",
    "PoseResult",
    "View",
    "CalibResult",
    "UncMap",
    "Diagnostics",
    "Candidate",
]


@dataclass(frozen=True, eq=False)
class Detection:
    """ChArUco detection for one frame.

    ids: (N,) int32 ChArUco corner ids; corners: (N, 2) float64 image px;
    obj: (N, 3) float64 board-frame metres (``CharucoBoard.matchImagePoints``). External pool
    frames (``nextview.data.Frame``) can hold other boards in square units; see ``Frame.board``.
    """

    ids: np.ndarray
    corners: np.ndarray
    obj: np.ndarray
    n_markers: int
    ok: bool


@dataclass(frozen=True)
class Quality:
    """Frame quality measured on the board region (slanted-edge ESF, saturation, coverage)."""

    edge_rise_px: float
    rise_by_dir: tuple[float, float, float, float]  # edge-normal bins 0/45/90/135 deg
    anisotropy: float
    blur_dir_deg: float | None
    saturation_frac: float
    board_area_frac: float
    glare_centroid: tuple[float, float] | None
    n_edges: int


@dataclass(frozen=True, eq=False)
class PoseResult:
    rvec: np.ndarray  # (3,)
    tvec: np.ndarray  # (3,) metres
    reproj_px: float
    ippe_ratio: float  # err_best / err_second of the two IPPE solutions
    tilt_deg: float
    distance: float  # metres, camera centre to board centre


@dataclass(frozen=True, eq=False)
class View:
    """An accepted view in the session state (spec section 2.1)."""

    id: int
    corner_ids: np.ndarray  # (N,) int32
    corners: np.ndarray  # (N, 2) float64 px, rounded to 0.01 in the state token
    quality: Quality | None
    rvec: np.ndarray  # (3,) pose at acceptance
    tvec: np.ndarray  # (3,)
    prequential_rms: float | None = None  # error against the calibration before this view was added


@dataclass(eq=False)
class CalibResult:
    model: str  # "k5" | "rational" | "ro"
    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # (n,) n = 5 for k5/ro, 8 for rational
    rms: float
    per_view: np.ndarray  # (V,) per-view RMS px
    rvecs: list
    tvecs: list
    std_intr: np.ndarray  # stdDeviationsIntrinsics, flattened
    sigma2: float  # noise variance estimate (section 2.5 step 3)
    F: np.ndarray  # (p, p) Schur-complemented information for theta
    cov: np.ndarray  # (p, p) sigma2 * F^-1


@dataclass(frozen=True, eq=False)
class UncMap:
    grid: tuple[int, int]  # (cols, rows), default (32, 18)
    sigma_px: np.ndarray  # (rows, cols) predicted reprojection std per cell
    U95: float
    p_exceed: float


@dataclass(frozen=True, eq=False)
class Diagnostics:
    """Output of ``residual_analysis`` (rules R6 to R10)."""

    radial_p: float  # radial residual-trend p-value
    outliers: tuple[int, ...]  # view ids flagged by R7
    warp_score: float
    drift: bool
    residual_field: np.ndarray | None = None  # (rows, cols, 2) mean residual vectors on the coverage grid
    hfov_deg: float = float("nan")  # horizontal field of view estimated from the fitted model (needs the image size)


@dataclass(frozen=True, eq=False)
class Candidate:
    rank: int
    rvec: np.ndarray
    tvec: np.ndarray
    gain: float  # predicted U95 reduction, px
    pred_U95_after: float
    effort: float
    score: float
    ghost_quad: np.ndarray  # (4, 2) px, board outline at the target pose
    instruction: str
