import math

import cv2
import numpy as np
import pytest

import _synth as S
from nextview.data import pool_tools
from nextview.data.board_probe import BoardSpec, detect_charuco
from nextview.data.frame import NEXTVIEW_BOARD, Frame, PlanarBoard
from nextview.data.samples import chessboard_board, detect_chessboard
from nextview.geometry import polygon_area, rotation_from_tilt
from nextview.tools import pose as tool_pose
from nextview.tools import quality
from nextview.tools.detect import detect_board

METRIC_BOARD = BoardSpec("DICT_6X6_50", (5, 4), 1.0, 0.79, True)
SIZE = (1920, 1080)
K = np.array([[1400.0, 0, 959.5], [0, 1400.0, 539.5], [0, 0, 1]])


def _pose(board: PlanarBoard, distance: float, tilt=25.0, azimuth=40.0, roll=10.0):
    R = rotation_from_tilt(tilt, azimuth, roll)
    tvec = np.array([0.0, 0.0, distance]) - R @ board.centre()
    return cv2.Rodrigues(R)[0].ravel(), tvec


def _render(tex: np.ndarray, tex_to_board: np.ndarray, rvec, tvec, blur: float, bg: int = 110):
    """Warp a board texture into a SIZE frame; returns the image and the true board outline (px)."""
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], tvec]) @ tex_to_board
    W, H = SIZE
    th, tw = tex.shape
    img = cv2.warpPerspective(tex, Hm, SIZE, flags=cv2.INTER_LINEAR, borderValue=bg)
    mask = cv2.warpPerspective(np.full_like(tex, 255), Hm, SIZE, flags=cv2.INTER_NEAREST) > 0
    out = np.full((H, W), bg, np.uint8)
    out[mask] = img[mask]
    edges = np.float64([[-0.5, -0.5], [tw - 0.5, -0.5], [tw - 0.5, th - 0.5], [-0.5, th - 0.5]])
    outline = cv2.perspectiveTransform(edges.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    return out, outline


def metric_frame(blur=1.0, distance=30.0):
    board = METRIC_BOARD.planar()
    P = 60  # texture px per square
    sx, sy = METRIC_BOARD.squares
    tex = METRIC_BOARD.make().generateImage((sx * P, sy * P), marginSize=0, borderBits=1)
    tex_to_board = np.array([[1 / P, 0, 0.5 / P], [0, 1 / P, 0.5 / P], [0, 0, 1]])
    rvec, tvec = _pose(board, distance)
    img, outline = _render(tex, tex_to_board, rvec, tvec, blur)
    det = detect_charuco(img, METRIC_BOARD.make(), METRIC_BOARD)
    frame = Frame("synthetic", path=None, source="test", image_size=SIZE, det=det, board=board)
    return frame, img, outline, rvec, tvec


def test_frames_carry_their_board():
    frame, *_ = metric_frame()
    assert frame.board.kind == "charuco" and frame.board.units == "square" and not frame.board.is_nextview
    assert frame.board.metres_per_unit is None
    assert NEXTVIEW_BOARD.is_nextview and NEXTVIEW_BOARD.metres_per_unit == 1.0
    np.testing.assert_allclose(frame.board.outline()[2], [5, 4, 0])


def test_metric_board_area_pose_and_blur():
    frame, img, outline, rvec, tvec = metric_frame(blur=1.0)
    assert frame.det.ok and frame.n_corners == 12
    true_area = polygon_area(outline) / (SIZE[0] * SIZE[1])
    assert pool_tools.board_area_frac(frame) == pytest.approx(true_area, rel=0.02)

    pose = pool_tools.estimate_pose(frame, K, np.zeros(5))
    assert pose.distance == pytest.approx(30.0, rel=0.005)  # square units, to the board centre
    assert pose.tilt_deg == pytest.approx(25.0, abs=0.5)

    q1 = pool_tools.assess_frame(img, frame)
    assert q1.n_edges >= 8 and 1.8 < q1.edge_rise_px < 4.0
    assert q1.board_area_frac == pytest.approx(true_area, rel=0.02)
    _, img2, *_ = metric_frame(blur=2.0)
    q2 = pool_tools.assess_frame(img2, frame)
    assert q2.edge_rise_px / q1.edge_rise_px > 1.5


def test_chessboard_area_and_edges():
    pattern = (9, 6)
    board = chessboard_board(pattern)
    P = 50
    sx, sy = board.squares
    tex = np.zeros((sy * P, sx * P), np.uint8)
    for r in range(sy):
        for c in range(sx):
            if (r + c) % 2 == 0:
                tex[r * P : (r + 1) * P, c * P : (c + 1) * P] = 230
    # texture pixel centre -> object units (first inner corner at the origin)
    tex_to_board = np.array([[1 / P, 0, 0.5 / P - 1], [0, 1 / P, 0.5 / P - 1], [0, 0, 1]])
    rvec, tvec = _pose(board, 25.0, tilt=20.0, azimuth=120.0, roll=0.0)
    img, outline = _render(tex, tex_to_board, rvec, tvec, blur=1.0)
    det, _ = detect_chessboard(img, pattern)
    frame = Frame("cb", path=None, source="test", image_size=SIZE, det=det, board=board)
    assert det.ok
    true_area = polygon_area(outline) / (SIZE[0] * SIZE[1])
    assert pool_tools.board_area_frac(frame) == pytest.approx(true_area, rel=0.02)
    q = pool_tools.assess_frame(img, frame)
    assert q.n_edges >= 8 and math.isfinite(q.edge_rise_px)


def test_matches_tools_on_the_nextview_board():
    Kn = S.make_K(900.0)
    rvec, tvec = S.pose_at(Kn, (640, 360), 0.45, tilt_deg=25.0, azimuth_deg=30.0)
    img = S.render(Kn, None, rvec, tvec, ss=2, blur_sigma=1.0)
    det = detect_board(img)
    frame = Frame("nv", path=None, source="test", image_size=(1280, 720), det=det, board=NEXTVIEW_BOARD)
    a, b = pool_tools.assess_frame(img, frame), quality.assess_frame(img, det)
    assert a.n_edges == b.n_edges and a.edge_rise_px == pytest.approx(b.edge_rise_px, rel=1e-9)
    np.testing.assert_allclose(a.rise_by_dir, b.rise_by_dir, rtol=1e-9)
    assert a.board_area_frac == pytest.approx(b.board_area_frac, rel=1e-9)
    assert a.saturation_frac == b.saturation_frac
    pa, pb = pool_tools.estimate_pose(frame, Kn, None), tool_pose.estimate_pose(det, Kn, None)
    assert pa.distance == pytest.approx(pb.distance, rel=1e-9) and pa.tilt_deg == pytest.approx(pb.tilt_deg, abs=1e-9)


def test_unusable_detection_is_refused():
    frame, *_ = metric_frame()
    from nextview.types import Detection

    det = frame.det
    row = Detection(ids=det.ids[:4], corners=det.corners[:4], obj=det.obj[:4], n_markers=det.n_markers, ok=False)
    flat = Frame("row", path=None, source="test", image_size=SIZE, det=row, board=frame.board)
    assert pool_tools.board_outline_px(row, frame.board) is None and pool_tools.board_area_frac(flat) == 0.0
    with pytest.raises(ValueError):
        pool_tools.estimate_pose(flat, K, None)
