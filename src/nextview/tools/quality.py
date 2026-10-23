"""Frame quality on the board region: edge blur (ESF), clipping/glare and board size."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..geometry import BOARD_OUTLINE_M, polygon_area
from ..types import Detection, Quality
from .detect import to_gray
from .esf import measure_edges

__all__ = ["assess_frame", "board_outline_px", "SATURATION_LEVEL"]

SATURATION_LEVEL = 250  # 8-bit level treated as clipped (leaves room for JPEG ringing around 255)


def board_outline_px(det: Detection) -> np.ndarray | None:
    """(4, 2) image outline of the whole board from a homography fitted to the detected corners."""
    if not det.ok or len(det.ids) < 4:
        return None
    H, _ = cv2.findHomography(det.obj[:, :2].astype(np.float64), det.corners.astype(np.float64), 0)
    if H is None:
        return None
    return cv2.perspectiveTransform(BOARD_OUTLINE_M[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2)


def _clip_to_image(poly: np.ndarray, w: int, h: int) -> np.ndarray | None:
    rect = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]], np.float32)
    p = poly.astype(np.float32)
    if cv2.contourArea(p, oriented=True) < 0:
        p = p[::-1].copy()
    area, inter = cv2.intersectConvexConvex(p, rect)
    if area <= 0 or inter is None:
        return None
    return inter.reshape(-1, 2)


def assess_frame(
    img: np.ndarray,
    det: Detection,
    *,
    anisotropy_threshold: float = 1.5,
    saturation_level: int = SATURATION_LEVEL,
) -> Quality:
    """Blur, clipping and coverage measurements for one frame (spec section 2.2).

    ``saturation_frac`` is the clipped fraction of the board area (any channel at or above
    ``saturation_level``); ``glare_centroid`` is the centroid of those pixels.
    ``board_area_frac`` is the visible board area over the image area.
    """
    img = np.asarray(img)
    gray = to_gray(img)
    h, w = gray.shape[:2]
    nan4 = (math.nan, math.nan, math.nan, math.nan)
    if not det.ok:
        return Quality(math.nan, nan4, math.nan, None, 0.0, 0.0, None, 0)

    esf = measure_edges(gray, det, anisotropy_threshold)

    outline = board_outline_px(det)
    if outline is None or not np.all(np.isfinite(outline)):
        outline = cv2.convexHull(det.corners.astype(np.float32)).reshape(-1, 2)
    region = _clip_to_image(outline, w, h)
    if region is None:
        return Quality(esf.edge_rise_px, esf.rise_by_dir, esf.anisotropy, esf.blur_dir_deg, 0.0, 0.0, None, esf.n_edges)
    area_frac = polygon_area(region) / float(w * h)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(region * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    peak = img[..., :3].max(axis=2) if img.ndim == 3 else img
    sat = (peak >= saturation_level) & (mask > 0)
    n_in = int(np.count_nonzero(mask))
    n_sat = int(np.count_nonzero(sat))
    sat_frac = n_sat / n_in if n_in else 0.0
    centroid = None
    if n_sat:
        ys, xs = np.nonzero(sat)
        centroid = (float(xs.mean()), float(ys.mean()))
    return Quality(
        edge_rise_px=esf.edge_rise_px,
        rise_by_dir=esf.rise_by_dir,
        anisotropy=esf.anisotropy,
        blur_dir_deg=esf.blur_dir_deg,
        saturation_frac=float(sat_frac),
        board_area_frac=float(area_frac),
        glare_centroid=centroid,
        n_edges=esf.n_edges,
    )
