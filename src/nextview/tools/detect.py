"""ChArUco board detection."""

from __future__ import annotations

import cv2
import numpy as np

from ..board import BOARD, SQUARES
from ..types import Detection

__all__ = ["detect_board", "to_gray", "corner_span", "empty_detection"]

_N_COLS = SQUARES[0] - 1  # inner corners per row
_detector: cv2.aruco.CharucoDetector | None = None


def _get_detector() -> cv2.aruco.CharucoDetector:
    global _detector
    if _detector is None:
        _detector = cv2.aruco.CharucoDetector(BOARD)
    return _detector


def to_gray(img: np.ndarray) -> np.ndarray:
    """8-bit single-channel view of a grey, BGR or BGRA image."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img[:, :, 0]


def empty_detection(n_markers: int = 0) -> Detection:
    return Detection(
        ids=np.zeros(0, np.int32),
        corners=np.zeros((0, 2), np.float64),
        obj=np.zeros((0, 3), np.float64),
        n_markers=int(n_markers),
        ok=False,
    )


def corner_span(ids: np.ndarray) -> tuple[int, int]:
    """Number of distinct inner-corner rows and columns covered by ``ids``."""
    ids = np.asarray(ids).ravel()
    if ids.size == 0:
        return 0, 0
    return int(np.unique(ids // _N_COLS).size), int(np.unique(ids % _N_COLS).size)


def detect_board(img: np.ndarray) -> Detection:
    """Detect ChArUco corners.

    Corners are sorted by id. ``ok`` means the detection is usable for a homography or PnP:
    at least 4 corners spanning at least 2 rows and 2 columns. The stricter acceptance gate
    (``min_corners``) is applied by the planner.
    """
    gray = to_gray(np.asarray(img))
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)
    ch_corners, ch_ids, _, marker_ids = _get_detector().detectBoard(gray)
    n_markers = 0 if marker_ids is None else len(marker_ids)
    if ch_ids is None or len(ch_ids) == 0:
        return empty_detection(n_markers)
    obj, imgp = BOARD.matchImagePoints(ch_corners, ch_ids)
    ids = ch_ids.reshape(-1).astype(np.int32)
    corners = np.asarray(imgp, dtype=np.float64).reshape(-1, 2)
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    order = np.argsort(ids, kind="stable")
    ids, corners, obj = ids[order], corners[order], obj[order]
    rows, cols = corner_span(ids)
    ok = bool(len(ids) >= 4 and rows >= 2 and cols >= 2)
    return Detection(ids=ids, corners=corners, obj=obj, n_markers=n_markers, ok=ok)
