import math

import numpy as np
import pytest

import _synth as S
from nextview import geometry as g
from nextview.board import BOARD_SIZE_M
from nextview.tools.detect import corner_span, detect_board
from nextview.tools.pose import estimate_pose, solve_pose

SIZE = (1280, 720)
K = S.make_K(900.0, SIZE)
DIST = np.array([-0.12, 0.04, 0.0005, -0.0003, 0.0])


@pytest.mark.parametrize(
    "centre,distance,tilt,az,roll",
    [
        ((640, 360), 0.45, 0.0, 0.0, 5.0),
        ((520, 300), 0.55, 20.0, 45.0, -10.0),
        ((760, 420), 0.60, 40.0, 180.0, 20.0),
        ((640, 360), 0.70, 50.0, 270.0, 0.0),
    ],
)
def test_pose_tilt_and_distance(centre, distance, tilt, az, roll):
    rv, tv = S.pose_at(K, centre, distance, tilt, az, roll)
    img = S.render(K, DIST, rv, tv, SIZE, ss=2, noise=1.0, rng=np.random.default_rng(3))
    det = detect_board(img)
    assert det.ok and len(det.ids) >= 36
    p = estimate_pose(det, K, DIST)
    assert abs(p.tilt_deg - tilt) < 1.0
    assert p.distance == pytest.approx(g.distance_of(rv, tv), rel=0.01)
    assert p.reproj_px < 0.5
    if tilt >= 20:
        assert p.ippe_ratio < 0.5


def test_ippe_ratio_small_frontal_vs_tilted():
    rng = np.random.default_rng(0)
    rv, tv = S.pose_at(K, (640, 360), 1.4, 0.0, 0.0, 0.0)  # board about 170 px wide
    p = S.project(K, None, rv, tv)
    frontal = np.array([solve_pose(S.OBJ, p + rng.normal(0, 0.3, p.shape), K, None)[3] for _ in range(20)])
    rv, tv = S.pose_at(K, (640, 360), 0.6, 35.0, 180.0, 0.0)
    p = S.project(K, None, rv, tv)
    tilted = [solve_pose(S.OBJ, p + rng.normal(0, 0.3, p.shape), K, None)[3] for _ in range(20)]
    # Per frame, not only on average: about 95 % of single frames exceed 0.85 at this size and noise.
    assert np.mean(frontal > 0.85) >= 0.8
    assert max(tilted) < 0.5


def test_pose_needs_four_non_collinear_corners():
    with pytest.raises(ValueError):
        solve_pose(S.OBJ[:3], np.zeros((3, 2)), K, None)
    rv, tv = S.pose_at(K, (640, 360), 0.5, 20.0, 0.0, 0.0)
    row = S.OBJ[:5]  # five corners of one row
    with pytest.raises(ValueError):
        solve_pose(row, S.project(K, None, rv, tv, obj=row), K, None)
    two_rows = S.OBJ[[0, 1, 2, 8, 9]]
    solve_pose(two_rows, S.project(K, None, rv, tv, obj=two_rows), K, None)


def test_detection_contract():
    rv, tv = S.pose_at(K, (640, 360), 0.5, 10.0, 30.0, 0.0)
    img = S.render(K, None, rv, tv, SIZE, ss=2)
    det = detect_board(img)
    assert det.ids.dtype == np.int32 and det.corners.shape == (40, 2) and det.obj.shape == (40, 3)
    assert np.all(np.diff(det.ids) > 0)
    np.testing.assert_allclose(det.obj, S.OBJ[det.ids], atol=1e-6)
    err = np.linalg.norm(det.corners - S.project(K, None, rv, tv)[det.ids], axis=1)
    assert np.median(err) < 0.2
    assert det.n_markers == 27
    assert corner_span(det.ids) == (5, 8)
    bgr = np.dstack([img, img, img])
    assert len(detect_board(bgr).ids) == 40


# --- geometry -----------------------------------------------------------------------------------


@pytest.mark.parametrize("tilt,az,roll", [(0, 0, 0), (20, 0, 10), (35, 90, -25), (50, 180, 5), (30, 270, 0), (45, 315, -30)])
def test_tilt_round_trip(tilt, az, roll):
    rv = g.rvec_from_tilt(tilt, az, roll)
    assert g.tilt_of(rv) == pytest.approx(tilt, abs=1e-9)
    if tilt > 0:
        assert (g.tilt_azimuth_of(rv) - az + 180) % 360 - 180 == pytest.approx(0, abs=1e-9)
        assert g.roll_of(rv) == pytest.approx(roll, abs=1e-9)


def test_azimuth_names_the_far_edge():
    # azimuth 180 = left edge away from the camera
    rv, tv = S.pose_at(K, (640, 360), 0.5, 30.0, 180.0, 0.0)
    R = g.as_rotation(rv)
    left = R @ np.array([0.0, BOARD_SIZE_M[1] / 2, 0]) + tv
    right = R @ np.array([BOARD_SIZE_M[0], BOARD_SIZE_M[1] / 2, 0]) + tv
    assert left[2] > right[2]


def test_pose_from_lattice_places_board():
    rv, tv = g.pose_from_lattice(K, (300.0, 200.0), 0.5, 35.0, 90.0, 15.0, SIZE, DIST)
    c = g.board_centre_px(rv, tv, K, DIST)
    np.testing.assert_allclose(c, [300.0, 200.0], atol=1e-6)
    assert g.tilt_of(rv) == pytest.approx(35.0)
    Z = (g.as_rotation(rv) @ g.BOARD_CENTRE_M + tv)[2]
    assert Z == pytest.approx(K[0, 0] * BOARD_SIZE_M[0] / (0.5 * SIZE[0]))
    # fronto-parallel at the centre: the board spans the requested fraction of the width
    rv, tv = g.pose_from_lattice(K, (639.5, 359.5), 0.5, 0.0, 0.0, 0.0, SIZE)
    quad = g.ghost_quad(rv, tv, K)
    assert (quad[1, 0] - quad[0, 0]) / SIZE[0] == pytest.approx(0.5, rel=1e-9)
    assert g.polygon_area(quad) == pytest.approx(640 * 640 * BOARD_SIZE_M[1] / BOARD_SIZE_M[0], rel=1e-9)


def test_grid_centres():
    c = g.grid_centres((1280, 720), (32, 18))
    assert c.shape == (576, 2)
    np.testing.assert_allclose(c[0], [19.5, 19.5])
    np.testing.assert_allclose(c[31], [1259.5, 19.5])
    np.testing.assert_allclose(c[-1], [1259.5, 699.5])
    assert math.isclose(g.distance_of(np.zeros(3), np.array([0.0, 0.0, 1.0]) - g.BOARD_CENTRE_M), 1.0)
