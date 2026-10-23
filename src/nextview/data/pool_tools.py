"""Frame quality and board pose for pool frames, parameterised by ``Frame.board``.

Same measurements and return types as ``tools.quality.assess_frame``, ``tools.esf.measure_edges``
and ``tools.pose.estimate_pose``, but the board outline, the inner-edge list, the square size and
the marker gap come from the frame's own board instead of the NextView board. On a NextView-board
frame the results match the ``tools`` versions.

Pose translations and distances are in ``frame.board.units`` (square units for METRIC and the
OpenCV samples); multiply by ``frame.board.metres_per_unit`` when that is known.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from nextview.geometry import polygon_area, tilt_of
from nextview.tools import esf
from nextview.tools.detect import to_gray
from nextview.tools.pose import solve_pose
from nextview.types import Detection, PoseResult, Quality

from .frame import Frame, PlanarBoard

__all__ = [
    "SATURATION_LEVEL",
    "board_outline_px",
    "board_area_frac",
    "inner_edges",
    "measure_edges",
    "assess_frame",
    "estimate_pose",
]

SATURATION_LEVEL = 250
_NEAR_SQUARES = 1.6  # local homography neighbourhood around an edge, in squares


def _homography(obj_xy: np.ndarray, img: np.ndarray) -> np.ndarray | None:
    if len(obj_xy) < 4:
        return None
    H, _ = cv2.findHomography(np.asarray(obj_xy, np.float64), np.asarray(img, np.float64), 0)
    return H


def board_outline_px(det: Detection, board: PlanarBoard) -> np.ndarray | None:
    """(4, 2) image outline of the whole board from a homography fitted to the detected corners."""
    if not board.usable(det.ids):
        return None
    H = _homography(det.obj[:, :2], det.corners)
    if H is None:
        return None
    return cv2.perspectiveTransform(board.outline()[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2)


def _clip_to_image(poly: np.ndarray, w: int, h: int) -> np.ndarray | None:
    rect = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]], np.float32)
    p = np.asarray(poly, np.float32)
    if cv2.contourArea(p, oriented=True) < 0:
        p = p[::-1].copy()
    area, inter = cv2.intersectConvexConvex(p, rect)
    if area <= 0 or inter is None:
        return None
    return inter.reshape(-1, 2)


def _board_region(frame: Frame) -> np.ndarray | None:
    outline = board_outline_px(frame.det, frame.board)
    if outline is None or not np.all(np.isfinite(outline)):
        if len(frame.det.ids) < 3:
            return None
        outline = cv2.convexHull(frame.det.corners.astype(np.float32)).reshape(-1, 2)
    w, h = frame.image_size
    return _clip_to_image(outline, w, h)


def board_area_frac(frame: Frame) -> float:
    """Visible whole-board area (homography-projected outline, clipped to the image) over the image area."""
    if not frame.det.ok:
        return 0.0
    region = _board_region(frame)
    if region is None:
        return 0.0
    w, h = frame.image_size
    return polygon_area(region) / float(w * h)


def inner_edges(board: PlanarBoard) -> list[tuple[int, int]]:
    """Corner-id pairs bounding each inner square edge of ``board``."""
    nc, nr = board.n_cols, board.n_rows
    pairs = [(r * nc + c, r * nc + c + 1) for r in range(nr) for c in range(nc - 1)]
    pairs += [(r * nc + c, (r + 1) * nc + c) for r in range(nr - 1) for c in range(nc)]
    return pairs


def _gap_frac(board: PlanarBoard) -> float | None:
    # distance from a square edge to the marker border on the white side, in squares;
    # a plain chessboard has uniform white squares
    if board.kind != "charuco" or board.marker_len is None:
        return None
    return (board.square_len - board.marker_len) / 2.0 / board.square_len


def _across_scale(H: np.ndarray, mid: np.ndarray, normal_obj: np.ndarray, n_img: np.ndarray, square: float) -> float:
    pts = np.array([mid - 0.5 * square * normal_obj, mid + 0.5 * square * normal_obj], dtype=np.float64)
    q = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    return abs(float(np.dot(q[1] - q[0], n_img)))


def measure_edges(gray: np.ndarray, frame: Frame, anisotropy_threshold: float = 1.5) -> esf.EsfResult:
    """Board-level slanted-edge blur summary (the method of ``tools.esf``) on ``frame.board``."""
    det, board = frame.det, frame.board
    nan4 = (math.nan, math.nan, math.nan, math.nan)
    if not det.ok or len(det.ids) < 4:
        return esf.EsfResult(math.nan, nan4, math.nan, None, 0, ())
    gray = to_gray(np.asarray(gray))
    sq = board.square_len
    gap = _gap_frac(board)
    pos = {int(i): k for k, i in enumerate(det.ids)}
    obj_xy = det.obj[:, :2]
    H_global = _homography(obj_xy, det.corners)
    fits: list[esf.EdgeFit] = []
    for a, b in inner_edges(board):
        ka, kb = pos.get(a), pos.get(b)
        if ka is None or kb is None:
            continue
        p0, p1 = det.corners[ka], det.corners[kb]
        v = p1 - p0
        L = float(np.hypot(v[0], v[1]))
        if L < esf.MIN_EDGE_PX:
            continue
        n_img = np.array([-v[1], v[0]]) / L
        ma, mb = obj_xy[ka], obj_xy[kb]
        mid = 0.5 * (ma + mb)
        e = (mb - ma) / np.linalg.norm(mb - ma)
        normal_obj = np.array([-e[1], e[0]])
        near = np.max(np.abs(obj_xy - mid), axis=1) <= _NEAR_SQUARES * sq
        H = None
        if np.count_nonzero(near) >= 6:
            rows, cols = board.corner_span(det.ids[near])
            if rows >= 2 and cols >= 2:
                H = _homography(obj_xy[near], det.corners[near])
        if H is None:
            H = H_global
        if H is None:
            continue
        s_across = _across_scale(H, mid, normal_obj, n_img, sq)
        fits.append(esf.measure_edge(gray, p0, p1, s_across, gap_frac=gap))
    good = [f for f in fits if f.ok]
    if len(good) < esf.MIN_EDGES:
        return esf.EsfResult(math.nan, nan4, math.nan, None, len(good), tuple(fits))
    by_bin: list[list[float]] = [[], [], [], []]
    for f in good:
        by_bin[int(round(f.normal_deg / 45.0)) % 4].append(f.rise)
    rise_by_dir = tuple(float(np.median(r)) if r else math.nan for r in by_bin)
    qualified = [k for k in range(4) if len(by_bin[k]) >= esf.MIN_EDGES_PER_DIR]
    if qualified:
        meds = np.array([rise_by_dir[k] for k in qualified])
        edge_rise = float(meds.max())
        anisotropy = float(meds.max() / meds.min()) if len(qualified) >= 2 else 1.0
        kmax = qualified[int(np.argmax(meds))]
    else:
        edge_rise = float(np.median([f.rise for f in good]))
        anisotropy = 1.0
        kmax = None
    blur_dir = esf.DIR_BINS_DEG[kmax] if (kmax is not None and anisotropy >= anisotropy_threshold) else None
    return esf.EsfResult(edge_rise, rise_by_dir, anisotropy, blur_dir, len(good), tuple(fits))


def assess_frame(
    img: np.ndarray,
    frame: Frame,
    *,
    anisotropy_threshold: float = 1.5,
    saturation_level: int = SATURATION_LEVEL,
) -> Quality:
    """Blur, clipping and board-area measurements for one pool frame, on its own board."""
    img = np.asarray(img)
    gray = to_gray(img)
    h, w = gray.shape[:2]
    nan4 = (math.nan, math.nan, math.nan, math.nan)
    if not frame.det.ok:
        return Quality(math.nan, nan4, math.nan, None, 0.0, 0.0, None, 0)
    if (w, h) != tuple(frame.image_size):
        raise ValueError(f"image is {w}x{h}, frame says {frame.image_size}")
    res = measure_edges(gray, frame, anisotropy_threshold)
    region = _board_region(frame)
    if region is None:
        return Quality(res.edge_rise_px, res.rise_by_dir, res.anisotropy, res.blur_dir_deg, 0.0, 0.0, None, res.n_edges)
    area_frac = polygon_area(region) / float(w * h)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(region * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    peak = img[..., :3].max(axis=2) if img.ndim == 3 else img
    sat = (peak >= saturation_level) & (mask > 0)
    n_in = int(np.count_nonzero(mask))
    n_sat = int(np.count_nonzero(sat))
    centroid = None
    if n_sat:
        ys, xs = np.nonzero(sat)
        centroid = (float(xs.mean()), float(ys.mean()))
    return Quality(
        edge_rise_px=res.edge_rise_px,
        rise_by_dir=res.rise_by_dir,
        anisotropy=res.anisotropy,
        blur_dir_deg=res.blur_dir_deg,
        saturation_frac=float(n_sat / n_in) if n_in else 0.0,
        board_area_frac=float(area_frac),
        glare_centroid=centroid,
        n_edges=res.n_edges,
    )


def estimate_pose(frame: Frame, K, dist) -> PoseResult:
    """IPPE + LM pose of ``frame.board``; ``tvec`` and ``distance`` (to the board centre) in board units."""
    if not frame.board.usable(frame.det.ids):
        raise ValueError(f"{frame.name}: detection is not usable for a pose ({len(frame.det.ids)} corners)")
    rvec, tvec, rms, ratio = solve_pose(frame.det.obj, frame.det.corners, K, dist)
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    distance = float(np.linalg.norm(R @ frame.board.centre() + np.asarray(tvec, np.float64).ravel()))
    return PoseResult(rvec=rvec, tvec=tvec, reproj_px=rms, ippe_ratio=ratio, tilt_deg=tilt_of(rvec), distance=distance)
