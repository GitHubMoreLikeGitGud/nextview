import math

import cv2
import numpy as np
import pytest

from nextview.sim.actuator import HumanActuator, RobotActuator, make_actuator
from nextview.sim.lens import make_camera
from nextview.sim.poses import BOARD_CENTRE_M, board_centre_cam, pose_for_view


def _errors(act, cam, rv, tv, n=3000, prev=(None, None)):
    rng = np.random.default_rng(0)
    R_t, _ = cv2.Rodrigues(rv.reshape(3, 1))
    c_t = board_centre_cam(rv, tv)
    rot, trans, outs = [], [], []
    for _ in range(n):
        a = act.execute(cam, rv, tv, rng, *prev)
        R, _ = cv2.Rodrigues(a.rvec.reshape(3, 1))
        d, _ = cv2.Rodrigues(R @ R_t.T)
        rot.append(np.degrees(d.ravel()))
        trans.append(board_centre_cam(a.rvec, a.tvec) - c_t)
        outs.append(a)
    return np.array(rot), np.array(trans), outs


def test_robot_noise_levels():
    cam = make_camera("normal", 0)
    rv, tv = pose_for_view(cam, (640, 360), 0.5, 20, 30)
    rot, trans, outs = _errors(RobotActuator(), cam, rv, tv)
    np.testing.assert_allclose(rot.std(0), [1.0, 1.0, 1.0], rtol=0.08)
    np.testing.assert_allclose(trans.std(0), [0.005] * 3, rtol=0.08)
    assert np.allclose(rot.mean(0), 0, atol=0.08) and np.allclose(trans.mean(0), 0, atol=4e-4)
    assert all(o.blur_len_px == 0.0 for o in outs)


@pytest.mark.parametrize("size_frac", [0.7, 0.35])
def test_human_noise_scales_with_distance(size_frac):
    cam = make_camera("normal", 0)
    rv, tv = pose_for_view(cam, (640, 360), size_frac, 0, 0)
    Z = board_centre_cam(rv, tv)[2]
    rot, trans, _ = _errors(HumanActuator(), cam, rv, tv)
    np.testing.assert_allclose(rot.std(0), [6.0] * 3, rtol=0.08)
    np.testing.assert_allclose(trans.std(0), [0.03 * Z / 0.5] * 3, rtol=0.08)


def test_human_motion_blur_rule():
    cam = make_camera("normal", 0)
    prev = pose_for_view(cam, (200, 200), 0.5)
    rv, tv = pose_for_view(cam, (1000, 500), 0.5)
    _, _, outs = _errors(HumanActuator(), cam, rv, tv, n=2000, prev=prev)
    L = np.array([o.blur_len_px for o in outs])
    long = L > 1.0 + 1e-9
    # p = 0.5 of frames are blurred along the move; the move is ~850 px so L hits the 10 px cap
    assert long.mean() == pytest.approx(0.5, abs=0.04)
    assert np.allclose(L[long], 10.0)
    assert L[~long].min() >= 0.0 and L[~long].max() <= 1.0
    c_prev = cam.project(BOARD_CENTRE_M.reshape(1, 3), *prev)[0]
    for o in np.array(outs, dtype=object)[long]:
        d = cam.project(BOARD_CENTRE_M.reshape(1, 3), o.rvec, o.tvec)[0] - c_prev
        assert o.blur_angle_deg == pytest.approx(math.degrees(math.atan2(d[1], d[0])) % 360.0, abs=1e-9)
        # roughly the commanded move direction, give or take the hand's translation jitter
        assert abs(((o.blur_angle_deg - math.degrees(math.atan2(300, 800)) + 180) % 360) - 180) < 25.0


def test_human_blur_length_formula_small_move():
    cam = make_camera("normal", 0)
    prev = pose_for_view(cam, (600, 360), 0.5)
    rv, tv = pose_for_view(cam, (640, 360), 0.5)
    rng = np.random.default_rng(4)
    act = HumanActuator()
    for _ in range(200):
        a = act.execute(cam, rv, tv, rng, *prev)
        if a.blur_len_px > 1.0:
            c_now = cam.project(BOARD_CENTRE_M.reshape(1, 3), a.rvec, a.tvec)[0]
            c_prev = cam.project(BOARD_CENTRE_M.reshape(1, 3), *prev)[0]
            assert a.blur_len_px == pytest.approx(min(10.0, 1.0 + 0.04 * np.linalg.norm(c_now - c_prev)))


def test_common_random_numbers():
    cam = make_camera("wide", 0)
    rv, tv = pose_for_view(cam, (640, 360), 0.5, 10, 10)
    for kind in ("robot", "human"):
        a = make_actuator(kind).execute(cam, rv, tv, np.random.default_rng([3, 1, 7]))
        b = make_actuator(kind).execute(cam, rv, tv, np.random.default_rng([3, 1, 7]))
        np.testing.assert_array_equal(a.rvec, b.rvec)
        np.testing.assert_array_equal(a.tvec, b.tvec)
        assert a.blur_len_px == b.blur_len_px
    with pytest.raises(ValueError):
        make_actuator("drone")


def test_pose_for_view_uses_geometry_conventions():
    import nextview.sim as sim_pkg
    from nextview import geometry as g
    from nextview.board import object_points
    from nextview.sim.lens import GTCamera

    pin = GTCamera.pinhole(make_camera("normal", 0).K)
    for tilt, az, roll in [(30.0, 0.0, 0.0), (30.0, 90.0, 10.0), (45.0, 200.0, -20.0)]:
        rv, tv = pose_for_view(pin, (500.0, 300.0), 0.5, tilt, az, roll)
        rv2, tv2 = g.pose_from_lattice(pin.K, (500.0, 300.0), 0.5, tilt, az, roll, pin.size)
        np.testing.assert_allclose(rv, rv2, atol=1e-12)
        np.testing.assert_allclose(tv, tv2, atol=1e-12)
        assert g.tilt_of(rv) == pytest.approx(tilt)
        assert g.tilt_azimuth_of(rv) == pytest.approx(az)
        assert g.roll_of(rv) == pytest.approx(roll)
    # azimuth 0 sends the right edge of the board away from the camera, 90 the bottom edge
    obj = object_points()
    for az, axis in [(0.0, 0), (90.0, 1)]:
        rv, tv = pose_for_view(pin, (640.0, 360.0), 0.5, 30.0, az, 0.0)
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
        z = (obj @ R.T + tv)[:, 2]
        far = obj[:, axis] > obj[:, axis].mean()
        assert z[far].mean() > z[~far].mean()
    assert "rotation_from_tilt" not in sim_pkg.__all__ and "tilt_of" not in sim_pkg.__all__
