import cv2
import numpy as np
import pytest

from nextview.board import BOARD_SIZE_M, object_points
from nextview.sim.lens import GTCamera, make_camera
from nextview.sim.poses import pose_for_view, random_pose
from nextview.sim.truth import (
    field_error,
    field_error_vectors,
    fx_rel_err,
    grid_centres,
    intrinsics_error,
    planar_mm_error,
    planar_pairs,
)
from nextview.sim.warp import sag_object_points, warped_observation


def test_grid_centres():
    g = grid_centres((1280, 720), (32, 18))
    assert g.shape == (18, 32, 2)
    np.testing.assert_allclose(g[0, 0], [19.5, 19.5])
    np.testing.assert_allclose(g[-1, -1], [1259.5, 699.5])


@pytest.mark.parametrize("family", ["normal", "wide", "tele", "webcam"])
def test_field_error_zero_for_gt(family):
    cam = make_camera(family, 2)
    fe = field_error(cam, cam.K, cam.dist)
    assert fe.err.shape == (18, 32) and fe.vec.shape == (18, 32, 2)
    assert fe.max < 1e-6


def test_field_error_responds_to_principal_point_shift():
    cam = make_camera("normal", 0)
    K = cam.K.copy()
    K[0, 2] += 1.5
    v = field_error_vectors(cam, K, cam.dist)
    # a pure cx shift moves every projection by +1.5 px in x
    np.testing.assert_allclose(v[..., 0], 1.5, atol=1e-6)
    np.testing.assert_allclose(v[..., 1], 0.0, atol=1e-6)
    fe = field_error(cam, K, cam.dist)
    assert fe.mean == pytest.approx(1.5) and fe.p95 == pytest.approx(1.5)


def _exact_views(cam, rng, n=30, min_visible=0.4):
    """``n`` noise-free corner views whose boards reach the image corners (partial boards allowed)."""
    obj = object_points()
    W, H = cam.size
    objs, imgs = [], []
    while len(objs) < n:
        rv, tv = random_pose(cam, rng, min_visible=min_visible)
        p, ok = cam.project_checked(obj, rv, tv)
        m = ok & (p[:, 0] >= 0) & (p[:, 0] <= W - 1) & (p[:, 1] >= 0) & (p[:, 1] <= H - 1)
        if m.sum() >= 12:
            objs.append(obj[m].astype(np.float32))
            imgs.append(p[m].astype(np.float32))
    return objs, imgs


def _best_fit(cam, objs, imgs, model):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_RATIONAL_MODEL if model == "rational" else 0)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    _, K, d, _, _ = cv2.calibrateCamera(objs, imgs, cam.size, cam.K.copy(), np.zeros(8 if model == "rational" else 5),
                                        flags=flags, criteria=crit)
    return K, d.ravel()


def test_best_fit_k5_cannot_follow_wide():
    """The wide family is the model-selection case: with exact corners over the whole frame, the
    least-squares k5 fit still misses the GT field by more than 1 px (p95) on at least 8 of 10
    cameras, while a rational fit on the same views recovers it."""
    k5_p95, rat_p95 = [], []
    for cs in range(10):
        cam = make_camera("wide", cs)
        objs, imgs = _exact_views(cam, np.random.default_rng(100 + cs))
        k5_p95.append(field_error(cam, *_best_fit(cam, objs, imgs, "k5")).p95)
        rat_p95.append(field_error(cam, *_best_fit(cam, objs, imgs, "rational")).p95)
    print("k5 p95", np.round(k5_p95, 2).tolist(), "rational p95", np.round(rat_p95, 4).tolist())
    assert sum(p > 1.0 for p in k5_p95) >= 8
    assert sum(p < 0.05 for p in rat_p95) >= 9


def test_best_fit_k5_follows_normal():
    """Control for the test above: the same procedure on the k5 normal family recovers the field."""
    for cs in range(3):
        cam = make_camera("normal", cs)
        objs, imgs = _exact_views(cam, np.random.default_rng(200 + cs))
        assert field_error(cam, *_best_fit(cam, objs, imgs, "k5")).p95 < 0.05


def test_fisheye_truth_is_finite():
    cam = make_camera("fisheye130", 0)
    fe = field_error(cam, cam.K, np.zeros(5))
    assert np.isfinite(fe.p95) and fe.p95 > 10


def test_intrinsic_errors():
    cam = make_camera("normal", 1)
    K = cam.K.copy()
    K[0, 0] *= 1.02
    K[1, 1] *= 0.99
    K[1, 2] -= 3.0
    assert fx_rel_err(cam, K) == pytest.approx(2.0)
    e = intrinsics_error(cam, K)
    assert e["fy_rel_pct"] == pytest.approx(1.0) and e["cx_err_px"] == 0.0 and e["cy_err_px"] == pytest.approx(3.0)


@pytest.mark.parametrize("Z", [0.5, 1.0])
def test_planar_pairs_geometry(Z):
    cam = make_camera("normal", 0)
    u1, u2 = planar_pairs(cam, Z)
    assert u1.shape == (200, 2) and u2.shape == (200, 2)
    for u in (u1, u2):
        cheb = np.maximum(np.abs(u[:, 0] - 639.5) / 640, np.abs(u[:, 1] - 359.5) / 360)
        assert np.all(cheb >= 0.7)
        assert np.all((u[:, 0] >= 0) & (u[:, 0] <= 1279) & (u[:, 1] >= 0) & (u[:, 1] <= 719))
    P1 = Z * np.column_stack([cam.pixel_to_normalized(u1), np.ones(200)])
    P2 = Z * np.column_stack([cam.pixel_to_normalized(u2), np.ones(200)])
    np.testing.assert_allclose(np.linalg.norm(P2 - P1, axis=1), 0.1, atol=1e-9)
    u1b, _ = planar_pairs(cam, Z)
    np.testing.assert_array_equal(u1, u1b)  # same pairs for every policy


def test_planar_mm_error():
    cam = make_camera("normal", 0)
    assert planar_mm_error(cam, cam.K, cam.dist, 0.5).p95 < 1e-6
    # distortion-free camera: a 1 % longer focal length shrinks every length by exactly 1/1.01
    pin = GTCamera.pinhole(cam.K)
    K = cam.K.copy()
    K[0, 0] *= 1.01
    K[1, 1] *= 1.01
    pe = planar_mm_error(pin, K, np.zeros(5), 1.0)
    np.testing.assert_allclose(pe.abs_err_mm, 100 * (1 - 1 / 1.01), rtol=1e-6)
    assert pe.abs_err_mm.shape == (200,)
    # with the real distortion the same focal error is still about 1 mm, but not uniform
    pe2 = planar_mm_error(cam, K, cam.dist, 1.0)
    assert 0.5 < pe2.median < 2.0


def test_warp_sag_and_observation():
    cam = make_camera("normal", 0)
    p = sag_object_points(0.003)
    flat = object_points()
    np.testing.assert_allclose(p[:, :2], flat[:, :2])
    half = BOARD_SIZE_M[0] / 2
    np.testing.assert_allclose(p[:, 2], 0.003 * ((flat[:, 0] - half) / half) ** 2)
    assert sag_object_points(0.003, np.array([[0.0, 0.0, 0.0]]))[0, 2] == pytest.approx(0.003)
    rv, tv = pose_for_view(cam, (640, 360), 0.6, 20, 30)
    obs = warped_observation(cam, rv, tv, 0.003, np.random.default_rng(0), noise_px=0.0)
    assert len(obs.ids) == 40 and np.all(obs.obj[:, 2] == 0)
    np.testing.assert_allclose(obs.corners, cam.project(obs.obj_true, rv, tv), atol=1e-9)
    # the sag is visible: flat projection differs by more than the 0.15 px noise somewhere
    assert np.abs(cam.project(obs.obj, rv, tv) - obs.corners).max() > 0.3
    noisy = warped_observation(cam, rv, tv, 0.003, np.random.default_rng(0))
    assert np.std(noisy.corners - obs.corners) == pytest.approx(0.15, rel=0.25)
