import math
import time

import cv2
import numpy as np
import pytest

import _synth as S
from nextview.board import BOARD_SIZE_M, object_points
from nextview.config import load_thresholds
from nextview.geometry import pose_from_lattice, rotation_from_tilt, tilt_of
from nextview.sim.lens import make_camera
from nextview.tools import nbv
from nextview.tools.calibrate import calibrate
from nextview.tools.uncertainty import covariance, grid_jacobian, u95_from_cov, uncertainty_map, view_information
from nextview.types import View

TH = load_thresholds()
SIZE = (1280, 720)
OBJ = object_points()


def _calib(family="normal", seed=1, n=8):
    cam = make_camera(family, seed)
    views = S.camera_views(cam, np.random.default_rng(3), n, noise=0.2)
    return cam, views, calibrate(views, "k5", cam.size)


def _visible(rvec, tvec, calib, size, margin=5.0):
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    Xc = OBJ @ R.T + tvec
    p = cv2.projectPoints(OBJ, rvec, tvec, calib.K, calib.dist)[0].reshape(-1, 2)
    W, H = size
    rl = nbv.radial_limit(calib.dist)
    return ((Xc[:, 2] > 0.02) & (np.hypot(Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]) < 0.98 * rl) & (p[:, 0] >= margin)
            & (p[:, 0] <= W - 1 - margin) & (p[:, 1] >= margin) & (p[:, 1] <= H - 1 - margin))


def test_vectorised_pose_matches_geometry():
    rng = np.random.default_rng(0)
    K = np.array([[800.0, 0, 650], [0, 801, 350], [0, 0, 1]])
    d = np.array([-0.2, 0.05, 0.001, -0.001, 0.01])
    for _ in range(10):
        t, a, r = rng.uniform(0, 60), rng.uniform(0, 360), rng.uniform(-30, 30)
        np.testing.assert_allclose(nbv.rotations(np.array([t]), np.array([a]), np.array([r]))[0], rotation_from_tilt(t, a, r), atol=1e-12)
        pt = np.array([[rng.uniform(0, 1279), rng.uniform(0, 719), rng.choice([0.35, 0.5, 0.7]), t, a]])
        R, tv = nbv._poses_from_points(pt, np.array([r]), K, d, SIZE)
        rv_ref, tv_ref = pose_from_lattice(K, tuple(pt[0, :2]), pt[0, 2], t, a, r, SIZE, d)
        np.testing.assert_allclose(cv2.Rodrigues(R[0])[0].ravel(), rv_ref, atol=1e-9)
        np.testing.assert_allclose(tv[0], tv_ref, atol=1e-9)


def test_lattice_size_and_orientations():
    lat = nbv.lattice(SIZE)
    assert lat.shape == (1125, 5)
    assert len(nbv.ORIENTATIONS) == 25 and set(np.unique(lat[:, 2])) == {0.35, 0.5, 0.7}
    assert len(np.unique(lat[:, :2], axis=0)) == 15


@pytest.mark.parametrize("family", ["normal", "wide", "fisheye130"])
def test_gain_matches_direct_schur_computation(family):
    cam, views, c = _calib(family)
    cands = nbv.propose_next_views({"actuator": "robot"}, c, cam.size, TH, np.random.default_rng(1))
    assert cands.u95_now == pytest.approx(uncertainty_map(c, views, cam.size).U95, rel=1e-9)
    top = cands[0]
    vis = _visible(top.rvec, top.tvec, c, cam.size)
    S_v = view_information(c.K, c.dist, top.rvec, top.tvec, OBJ[vis])
    G = grid_jacobian(c.K, c.dist, cam.size, (32, 18))
    after = u95_from_cov(G, covariance(c.F + S_v, c.sigma2))
    assert top.pred_U95_after == pytest.approx(after, rel=1e-9)
    assert top.gain == pytest.approx(cands.u95_now - after, rel=1e-9)


def test_all_candidates_feasible():
    cam, views, c = _calib("wide")
    for state in ({"actuator": "robot"}, {"actuator": "human", "flags": ["AMBIGUOUS"]}):
        cands = nbv.propose_next_views(state, c, cam.size, TH, np.random.default_rng(5))
        assert len(cands) == 5 and cands.n_evaluated >= TH.nbv_min_candidates
        for cd in cands:
            vis = _visible(cd.rvec, cd.tvec, c, cam.size)
            assert vis.mean() >= TH.nbv_min_visible_frac
            assert tilt_of(cd.rvec) < TH.max_tilt_deg
            if "flags" in state:
                assert tilt_of(cd.rvec) >= TH.ambiguous_min_tilt_deg - 1e-6
            assert cd.ghost_quad.shape == (4, 2) and cd.instruction.startswith("Board")
        assert [cd.rank for cd in cands] == [1, 2, 3, 4, 5]
        assert all(a.score >= b.score for a, b in zip(cands, cands[1:]))


def test_centre_frontal_views_send_the_board_to_a_tilt_or_a_corner():
    K = S.make_K(900.0, SIZE)
    dist = np.array([-0.12, 0.04, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(2)
    views = []
    for i in range(4):
        rv, tv = S.pose_at(K, (640 + rng.normal(0, 5), 360 + rng.normal(0, 5)), 0.45 + 0.02 * i, tilt_deg=rng.uniform(0, 3),
                           azimuth_deg=rng.uniform(0, 360))
        p = S.project(K, dist, rv, tv)
        views.append(View(i, np.arange(40, dtype=np.int32), p + rng.normal(0, 0.1, p.shape), None, rv, tv))
    c = calibrate(views, "k5", SIZE)
    top = nbv.propose_next_views({"actuator": "robot"}, c, SIZE, TH, np.random.default_rng(0))[0]
    centre = cv2.projectPoints(np.array([[BOARD_SIZE_M[0] / 2, BOARD_SIZE_M[1] / 2, 0.0]]), top.rvec, top.tvec, c.K, c.dist)[0].ravel()
    corner_cell = (centre[0] < 1280 / 5 or centre[0] > 4 * 1280 / 5) and (centre[1] < 720 / 3 or centre[1] > 2 * 720 / 3)
    assert tilt_of(top.rvec) >= 20 - 1e-6 or corner_cell


def test_deterministic_for_a_seed_and_fast():
    cam, views, c = _calib("normal", n=10)
    a = nbv.propose_next_views({"actuator": "human"}, c, cam.size, TH, np.random.default_rng(11))
    t = time.perf_counter()
    b = nbv.propose_next_views({"actuator": "human"}, c, cam.size, TH, np.random.default_rng(11))
    dt = time.perf_counter() - t
    assert [x.score for x in a] == [x.score for x in b]
    np.testing.assert_array_equal(a[0].rvec, b[0].rvec)
    assert b.n_sampled == 300 and b.n_evaluated == b.n_feasible
    assert dt < 1.0
    other = nbv.propose_next_views({"actuator": "human"}, c, cam.size, TH, np.random.default_rng(12))
    assert not np.array_equal(other[0].rvec, a[0].rvec) or other[0].score != a[0].score


def test_effort_term_only_in_human_mode():
    cam, views, c = _calib("normal", n=8)
    last = (views[-1].rvec, views[-1].tvec)
    robot = nbv.propose_next_views({"actuator": "robot", "last_pose": last}, c, cam.size, TH, np.random.default_rng(4))
    human = nbv.propose_next_views({"actuator": "human", "last_pose": last}, c, cam.size, TH, np.random.default_rng(4))
    assert all(x.score == pytest.approx(x.gain) for x in robot)
    gmax = max(x.gain for x in robot)  # same sample and rolls: the same max gain over the lattice sample
    for x in human:
        assert x.score == pytest.approx(x.gain - TH.effort_lambda_human * x.effort * gmax, abs=1e-12)
        assert x.effort > 0
    no_last = nbv.propose_next_views({"actuator": "human"}, c, cam.size, TH, np.random.default_rng(4))
    assert all(x.effort == 0 for x in no_last)


def test_deadline_still_evaluates_the_minimum():
    cam, views, c = _calib("normal")
    cands = nbv.propose_next_views(None, c, cam.size, TH, np.random.default_rng(0), deadline=time.perf_counter() - 1)
    assert cands.stopped_by_deadline and cands.n_evaluated == TH.nbv_min_candidates and len(cands) == 5


def test_opening_poses_and_focal_estimate():
    a = nbv.opening_specs(3, 0)
    assert a == nbv.opening_specs(3, 0) and a != nbv.opening_specs(3, 1) and len(a) == 3
    assert a[0].s == 0.45 and all(15 <= s.tilt <= 40 for s in a)
    rv, tv = nbv.opening_target(0, 3, 0, SIZE)
    K0 = nbv.k_initial(SIZE)
    assert K0[0, 0] == pytest.approx(0.9 * 1280)
    c = cv2.projectPoints(np.array([[BOARD_SIZE_M[0] / 2, BOARD_SIZE_M[1] / 2, 0.0]]), rv, tv, K0, None)[0].ravel()
    assert c[0] == pytest.approx(a[0].cx * 1279, abs=1e-6) and c[1] == pytest.approx(a[0].cy * 719, abs=1e-6)
    # a tele lens seen at a K0 opening pose: the corner spread gives f back
    for family in ("tele", "wide", "normal"):
        cam = make_camera(family, 0)
        p, ok = cam.project_checked(OBJ, rv, tv)
        inside = ok & (p[:, 0] > 0) & (p[:, 0] < 1279) & (p[:, 1] > 0) & (p[:, 1] < 719)
        ids = np.nonzero(inside)[0]
        f = nbv.focal_from_detection(ids, p[ids], rv, tv, K0[0, 0], SIZE)
        tol = 0.15 if family == "wide" else 0.03  # barrel distortion shrinks the spread of a wide lens
        assert f == pytest.approx(cam.fx, rel=tol)
    assert nbv.focal_from_detection([0, 1, 2], np.zeros((3, 2)), rv, tv, 1000.0, SIZE) is None


def test_radial_limit():
    assert nbv.radial_limit(np.zeros(5)) == 12.0
    lim = nbv.radial_limit(np.array([-0.4, 0.0, 0, 0, 0]))  # r (1 - 0.4 r^2) peaks at r = sqrt(1/1.2)
    assert lim == pytest.approx(math.sqrt(1 / 1.2), abs=0.01)


def _outline_inside(rvec, tvec, calib, size, margin=5.0):
    from nextview.geometry import BOARD_OUTLINE_M

    p = cv2.projectPoints(BOARD_OUTLINE_M, rvec, tvec, calib.K, calib.dist)[0].reshape(-1, 2)
    W, H = size
    return bool(np.all((p[:, 0] >= margin) & (p[:, 0] <= W - 1 - margin) & (p[:, 1] >= margin) & (p[:, 1] <= H - 1 - margin)))


@pytest.mark.parametrize("how", ["seek_full_board", "ro_model"])
def test_full_board_mode_plans_only_whole_board_views(how):
    cam = make_camera("normal", 2)
    views = S.camera_views(cam, np.random.default_rng(3), 10, noise=0.2, min_visible=0.9)
    k5 = calibrate(views, "k5", cam.size)
    calib = calibrate(views, "ro", cam.size) if how == "ro_model" else k5
    state = {"actuator": "robot", "seek_full_board": how == "seek_full_board"}
    cands = nbv.propose_next_views(state, calib, cam.size, TH, np.random.default_rng(4), k=5)
    assert calib.model == ("ro" if how == "ro_model" else "k5")
    assert len(cands) == 5 and cands.stats()["full_board_only"] is True
    for c in cands:
        assert _visible(c.rvec, c.tvec, calib, cam.size).all() and _outline_inside(c.rvec, c.tvec, calib, cam.size)
    part = nbv.propose_next_views({"actuator": "robot"}, k5, cam.size, TH, np.random.default_rng(4), k=5)
    assert part.stats()["full_board_only"] is False and part.n_feasible > cands.n_feasible
