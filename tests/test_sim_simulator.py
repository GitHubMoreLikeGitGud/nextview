import cv2
import numpy as np
import pytest

from nextview.board import BOARD, object_points
from nextview.config import load_presets
from nextview.sim import Simulator, make_camera, make_simulator, scenario_from_preset, scenario_table
from nextview.sim.nuisance import NuisanceConfig
from nextview.sim.poses import pose_for_view, random_pose, visible_fraction
from nextview.sim.render import BACKGROUND

OBJ = object_points()


def _detect(img):
    corners, ids, _, _ = cv2.aruco.CharucoDetector(BOARD).detectBoard(img)
    return (np.zeros(0, int), np.zeros((0, 2))) if ids is None else (ids.ravel(), corners.reshape(-1, 2))


def test_capture_is_deterministic_and_stateless():
    sim = make_simulator("normal", "human", camera_seed=3, seed=1)
    rv, tv = pose_for_view(sim.camera, (640, 360), 0.5, 20, 40)
    a = sim.capture(rv, tv, attempt=5)
    for k in range(3):  # unrelated captures in between must not change attempt 5
        sim.capture(rv, tv, attempt=k)
    b = make_simulator("normal", "human", camera_seed=3, seed=1).capture(rv, tv, attempt=5)
    c = sim.capture(rv, tv, attempt=5)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.image, c.image)
    np.testing.assert_array_equal(a.truth.rvec, b.truth.rvec)
    d = sim.capture(rv, tv, attempt=6)
    assert not np.array_equal(a.image, d.image)
    e = make_simulator("normal", "human", camera_seed=3, seed=2).capture(rv, tv, attempt=5)
    assert not np.array_equal(a.truth.rvec, e.truth.rvec)


def test_common_random_numbers_across_targets():
    """Two policies asking for different targets at the same attempt get the same actuator error."""
    sim = make_simulator("normal", "robot", camera_seed=0, seed=0)
    t1 = pose_for_view(sim.camera, (500, 300), 0.5, 10, 0)
    t2 = pose_for_view(sim.camera, (800, 400), 0.6, 30, 90)
    a = sim.capture(*t1, attempt=2).truth.actuation
    b = sim.capture(*t2, attempt=2).truth.actuation
    assert a.rot_err_deg == pytest.approx(b.rot_err_deg) and a.centre_err_m == pytest.approx(b.centre_err_m)


def test_frame_format_and_truth_dict():
    sc = scenario_from_preset("phone_crack_metrology", camera_seed=0)
    rv, tv = random_pose(sc.camera, np.random.default_rng(0))
    fr = sc.simulator.capture(rv, tv, attempt=0)
    assert fr.image.dtype == np.uint8 and fr.image.shape == (720, 1280)
    d = fr.truth.to_dict()
    import json

    json.dumps(d)
    assert set(d) >= {"rvec", "tvec", "K", "dist", "corrupted", "occluder", "blur_len_px", "drifted"}


def test_robot_capture_matches_projection():
    sim = make_simulator("webcam", "robot", camera_seed=1, seed=0, occluder_p=0.0, corruption_p=0.0)
    rv, tv = pose_for_view(sim.camera, (640, 360), 0.6, 25, 60)
    fr = sim.capture(rv, tv, attempt=0)
    ids, corners = _detect(fr.image)
    assert len(ids) >= 36
    gt = fr.truth.camera.project(OBJ[ids], fr.truth.rvec, fr.truth.tvec)
    assert np.sqrt(((corners - gt) ** 2).sum(1).mean()) < 0.2


def test_autofocus_drift_after_ten_accepted():
    sc = scenario_from_preset("autofocus_drift", camera_seed=0)
    sim = sc.simulator
    rv, tv = pose_for_view(sc.camera, (640, 360), 0.5)
    f9 = sim.capture(rv, tv, attempt=0, n_accepted=9)
    f10 = sim.capture(rv, tv, attempt=0, n_accepted=10)
    assert not f9.truth.drifted and f10.truth.drifted
    assert f10.truth.camera.fx == pytest.approx(sc.camera.fx * 1.015)
    assert f10.truth.camera.dist[0] == pytest.approx(sc.camera.dist[0] + 0.01)
    # the other presets never drift
    other = scenario_from_preset("inspection_gauge_cam", camera_seed=0).simulator
    assert not other.capture(rv, tv, attempt=0, n_accepted=25).truth.drifted


def test_nuisance_rates_over_attempts():
    sim = make_simulator("normal", "robot", camera_seed=0, seed=0)
    rv, tv = pose_for_view(sim.camera, (640, 360), 0.5)
    occ = corr = 0
    n = 300
    for a in range(n):
        t = sim.capture(rv, tv, attempt=a).truth
        occ += t.occluder.active
        corr += t.corrupted
    assert occ / n == pytest.approx(0.15, abs=0.05)
    assert corr / n == pytest.approx(0.05, abs=0.035)


def test_occluder_hides_corners():
    sim = make_simulator("normal", "robot", camera_seed=0, seed=0, occluder_p=1.0, corruption_p=0.0)
    clean = make_simulator("normal", "robot", camera_seed=0, seed=0, occluder_p=0.0, corruption_p=0.0)
    rv, tv = pose_for_view(sim.camera, (640, 360), 0.6)
    lost = 0
    for a in range(6):
        n_occ = len(_detect(sim.capture(rv, tv, attempt=a).image)[0])
        n_clean = len(_detect(clean.capture(rv, tv, attempt=a).image)[0])
        lost += n_clean - n_occ
    assert lost > 0


def test_glare_saturates_board_only_in_glossy_preset():
    glossy = scenario_from_preset("phone_crack_metrology", camera_seed=0).simulator
    matte = make_simulator("webcam", "human", camera_seed=0, seed=0)
    rng = np.random.default_rng(11)
    hits = 0
    for a in range(200):
        rv, tv = random_pose(glossy.camera, rng, max_tilt_deg=30)
        f = glossy.capture(rv, tv, attempt=a)
        if f.truth.glare is None or f.truth.glare_px is None:
            continue
        u, v = np.rint(f.truth.glare_px).astype(int)
        if not (50 <= u < 1230 and 50 <= v < 670):
            continue
        hits += 1
        # the hotspot adds 120-255 levels: white paper (224) clips at 255, black ink (20) may not
        patch = f.image[v - 50 : v + 51, u - 50 : u + 51]
        assert patch.max() == 255
        g = matte.capture(rv, tv, attempt=a)
        assert g.truth.glare is None
        if hits >= 5:
            break
    assert hits >= 3


def test_board_behind_camera_gives_background_frame():
    sim = make_simulator("normal", "robot", camera_seed=0, seed=0, clean=True)
    rv = np.zeros(3)
    tv = np.array([0.0, 0.0, -1.0])
    fr = sim.capture(rv, tv, attempt=0)
    assert not fr.truth.rendered and fr.truth.visible_frac == 0.0
    assert np.all(fr.image == int(BACKGROUND))


def test_corruption_applies_after_detection():
    sim = make_simulator("normal", "robot", camera_seed=0, seed=0, occluder_p=0.0, corruption_p=1.0)
    rv, tv = pose_for_view(sim.camera, (640, 360), 0.6)
    fr = sim.capture(rv, tv, attempt=0)
    assert fr.truth.corrupted
    ids, corners = _detect(fr.image)
    out, moved = fr.corrupt(ids, corners)
    assert len(moved) == 3 and (np.linalg.norm(out - corners, axis=1) > 1.9).sum() == 3


def test_all_presets_build_and_capture():
    table = scenario_table()
    assert {r["id"] for r in table} == set(load_presets().ids)
    assert sum(r["default"] for r in table) == 1
    for row in table:
        sc = scenario_from_preset(row["id"], camera_seed=0)
        assert sc.family == row["family"] and sc.simulator.actuator_kind == row["actuator"]
        assert sc.simulator.nuisances.glare == ("glare" in row["nuisances"])
        rv, tv = random_pose(sc.camera, np.random.default_rng(1))
        fr = sc.simulator.capture(rv, tv, attempt=0)
        assert fr.image.shape == (720, 1280)
        assert sc.to_dict()["camera"]["family"] == row["family"]


def test_human_preset_blurs_more_than_robot():
    cam_seed = 2
    lengths = {}
    for kind in ("human", "robot"):
        sim = make_simulator("normal", kind, camera_seed=cam_seed, seed=0, occluder_p=0.0, corruption_p=0.0)
        rng = np.random.default_rng(5)
        prev = (None, None)
        lengths[kind] = []
        for a in range(30):
            rv, tv = random_pose(sim.camera, rng)
            fr = sim.capture(rv, tv, attempt=a, prev_rvec=prev[0], prev_tvec=prev[1])
            prev = (fr.truth.rvec, fr.truth.tvec)
            lengths[kind].append(fr.truth.actuation.blur_len_px)
    assert max(lengths["human"]) == pytest.approx(10.0) and min(lengths["human"]) <= 1.0
    assert all(L == 0.0 for L in lengths["robot"])


def test_observe_applies_corruption_to_detection():
    from nextview.tools.detect import detect_board

    bad = make_simulator("normal", "robot", camera_seed=0, seed=0, occluder_p=0.0, corruption_p=1.0)
    good = make_simulator("normal", "robot", camera_seed=0, seed=0, occluder_p=0.0, corruption_p=0.0)
    rv, tv = pose_for_view(bad.camera, (640, 360), 0.6)
    fr = bad.capture(rv, tv, attempt=0)
    plain = detect_board(fr.image)
    det, moved = fr.observe()
    assert fr.truth.corrupted and len(moved) == 3
    np.testing.assert_array_equal(det.ids, plain.ids)
    shift = np.linalg.norm(det.corners - plain.corners, axis=1)
    assert np.all(shift[np.isin(det.ids, moved)] > 1.9) and np.all(shift[~np.isin(det.ids, moved)] == 0)
    fr_ok = good.capture(rv, tv, attempt=0)
    det_ok, moved_ok = fr_ok.observe()
    assert not fr_ok.truth.corrupted and len(moved_ok) == 0
    np.testing.assert_array_equal(det_ok.corners, detect_board(fr_ok.image).corners)


def test_scenario_actuator_override():
    sc = scenario_from_preset("wide_robot_handheld", camera_seed=0)
    assert sc.actuator == "human" and sc.simulator.actuator_kind == "human"
    robot = scenario_from_preset("wide_robot_handheld", camera_seed=0, actuator="robot")
    assert robot.actuator == "robot" and robot.simulator.actuator_kind == "robot"
    assert robot.to_dict()["actuator"] == "robot" and robot.family == sc.family
    np.testing.assert_array_equal(robot.camera.K, sc.camera.K)
    with pytest.raises(ValueError):
        scenario_from_preset("wide_robot_handheld", actuator="drone")


def test_simulator_custom_nuisances():
    cam = make_camera("tele", 0)
    sim = Simulator(cam, "robot", NuisanceConfig.clean(), seed=4)
    rv, tv = pose_for_view(cam, (640, 360), 0.5)
    fr = sim.capture(rv, tv, attempt=0)
    exact = sim.render_exact(fr.truth.rvec, fr.truth.tvec)
    assert np.array_equal(fr.image, exact)  # clean config: capture == render at the realised pose
    assert visible_fraction(cam, fr.truth.rvec, fr.truth.tvec) == 1.0
