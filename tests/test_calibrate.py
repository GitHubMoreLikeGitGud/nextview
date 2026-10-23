import math

import numpy as np
import pytest

import _synth as S
from nextview import board
from nextview.config import load_thresholds
from nextview.sim.lens import make_camera
from nextview.tools.calibrate import calibrate, cv_heldout_rms, prequential_rms, ro_cv_gain
from nextview.tools.detect import detect_board
from nextview.tools.residuals import (
    chi2_sf_even,
    drift_flag,
    estimate_hfov_deg,
    out_of_scope,
    radial_trend_p,
    residual_analysis,
    warp_score,
)
from nextview.types import View

SIZE = (1280, 720)
TH = load_thresholds()


def _clean_views(K, dist, n, seed, noise=0.0):
    rng = np.random.default_rng(seed)
    views = []
    for i in range(n):
        rv, tv = S.random_pose(K, SIZE, rng, dist)
        p = S.project(K, dist, rv, tv)
        views.append(View(i, np.arange(40, dtype=np.int32), p + rng.normal(0, noise, p.shape), None, rv, tv))
    return views


def test_noise_free_views_recover_K_within_0p1_percent():
    K = S.make_K(950.0, SIZE, aspect=1.002, pp_offset=(12.0, -7.0))
    dist = np.array([-0.21, 0.07, 0.0006, -0.0004, -0.01])
    c = calibrate(_clean_views(K, dist, 15, seed=1), "k5", SIZE)
    for (i, j) in [(0, 0), (1, 1), (0, 2), (1, 2)]:
        assert c.K[i, j] == pytest.approx(K[i, j], rel=1e-3)
    np.testing.assert_allclose(c.dist, dist, atol=1e-4)
    assert c.rms < 1e-3
    assert c.dist.shape == (5,) and c.F.shape == (9, 9) and c.cov.shape == (9, 9)
    assert len(c.rvecs) == len(c.tvecs) == len(c.per_view) == 15


def test_rendered_views_recover_fx_within_1_percent():
    K = S.make_K(900.0, SIZE, pp_offset=(8.0, -5.0))
    dist = np.array([-0.25, 0.08, 0.0005, -0.0003, -0.01])
    rng = np.random.default_rng(3)
    views = []
    for i in range(15):
        rv, tv = S.random_pose(K, SIZE, rng, dist)
        det = detect_board(S.render(K, dist, rv, tv, SIZE, ss=2, noise=1.0, rng=rng))
        assert det.ok and len(det.ids) >= 12
        views.append(View(i, det.ids, det.corners, None, rv, tv))
    c = calibrate(views, "k5", SIZE)
    assert c.K[0, 0] == pytest.approx(K[0, 0], rel=0.01)
    assert c.K[1, 1] == pytest.approx(K[1, 1], rel=0.01)
    assert c.rms < 0.3


def test_rational_and_ro_shapes():
    K = S.make_K(900.0, SIZE)
    dist = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])
    views = _clean_views(K, dist, 8, seed=4, noise=0.1)
    r = calibrate(views, "rational", SIZE)
    assert r.dist.shape == (8,) and r.cov.shape == (12, 12)
    ro = calibrate(views, "ro", SIZE)
    assert ro.dist.shape == (5,) and ro.new_obj.shape == (40, 3)
    with pytest.raises(ValueError):
        calibrate(views, "fisheye", SIZE)
    with pytest.raises(ValueError):
        calibrate(views[:1], "k5", SIZE)


def test_corrupted_view_is_flagged_as_outlier():
    K = S.make_K(900.0, SIZE)
    dist = np.array([-0.15, 0.05, 0.0, 0.0, 0.0])
    views = _clean_views(K, dist, 12, seed=5, noise=0.15)
    rng = np.random.default_rng(9)
    bad = views[7]
    corners = bad.corners.copy()
    idx = rng.choice(40, 3, replace=False)
    shift = rng.uniform(2, 5, 3)[:, None] * np.array([[np.cos(a), np.sin(a)] for a in rng.uniform(0, 2 * np.pi, 3)])
    corners[idx] += shift
    views[7] = View(bad.id, bad.corner_ids, corners, None, bad.rvec, bad.tvec)
    c = calibrate(views, "k5", SIZE)
    diag = residual_analysis(c, views, TH, SIZE)
    assert diag.outliers == (7,)
    assert diag.residual_field.shape == (9, 16, 2)
    clean = calibrate(_clean_views(K, dist, 12, seed=5, noise=0.15), "k5", SIZE)
    assert residual_analysis(clean, views, TH).outliers == ()


def _underfit_fires(family, camera_seed):
    cam = make_camera(family, camera_seed)
    views = S.camera_views(cam, np.random.default_rng(5000 + camera_seed), 14, noise=0.25)
    c = calibrate(views, "k5", cam.size)
    p = radial_trend_p(c, views)
    cv_k5 = cv_heldout_rms(views, "k5", cam.size, init=c)
    cv_rat = cv_heldout_rms(views, "rational", cam.size, init=c)
    return bool(len(views) >= TH.model_switch_min_views and p < TH.underfit_p and cv_rat < (1 - TH.model_switch_cv_gain) * cv_k5)


def test_underfit_evidence_separates_lens_families():
    """Spec section 13: R8 evidence (radial trend + 3-fold CV gain of rational) fires on at least
    8 of 10 simulator ``wide`` cameras and at most 1 of 10 ``normal`` cameras (14 views each,
    partial boards allowed, 0.25 px corner noise).

    The simulator ``wide`` family is the Pade-like k1/k4 lens of ``sim/lens.py``, not spec 6.1's
    coefficient ranges: k5 fits those to the noise floor, so R8 has nothing to detect there and
    must stay silent (tests/test_policy.py::test_r8_cannot_fire_on_spec_range_wide_lenses)."""
    wide = sum(_underfit_fires("wide", i) for i in range(10))
    normal = sum(_underfit_fires("normal", i) for i in range(10))
    assert wide >= 8
    assert normal <= 1


def test_out_of_scope_from_rational_hfov():
    """R10 evidence: a 130 deg fisheye fitted with the rational model shows HFOV > 110 deg,
    wide cameras (95 to 110 deg) do not, and a k5 model never fires."""
    for family, expect in (("fisheye130", True), ("wide", False)):
        for s in range(3):
            cam = make_camera(family, s)
            views = S.camera_views(cam, np.random.default_rng(7000 + s), 14, noise=0.25)
            rat = calibrate(views, "rational", cam.size)
            fires, hfov = out_of_scope(rat, cam.size, TH)
            assert fires is expect
            assert hfov == pytest.approx(cam.hfov_deg, abs=3.0)
            assert residual_analysis(rat, views, TH, cam.size).hfov_deg == pytest.approx(hfov)
    assert out_of_scope(calibrate(views, "k5", cam.size), cam.size, TH)[0] is False


def test_warm_start_matches_cold_fit():
    cam = make_camera("wide", 0)
    views = S.camera_views(cam, np.random.default_rng(7000), 14, noise=0.25)
    prev = calibrate(views[:-1], "k5", cam.size)
    for model in ("k5", "rational"):
        cold = calibrate(views, model, cam.size)
        warm = calibrate(views, model, cam.size, init=prev)
        assert warm.rms == pytest.approx(cold.rms, rel=1e-3)
        np.testing.assert_allclose(warm.K, cold.K, rtol=1e-3)
    assert cold.rms < 0.4  # rational; on these views a start from the raw k5 init stalls at 6 px
    # a tuple start is just one more start for the multi-start search
    tup = calibrate(views, "k5", cam.size, init=(prev.K, prev.dist))
    assert tup.rms == pytest.approx(calibrate(views, "k5", cam.size).rms, rel=1e-6)


def test_cv_heldout_rms_tracks_noise():
    K = S.make_K(900.0, SIZE)
    dist = np.array([-0.15, 0.05, 0.0, 0.0, 0.0])
    views = _clean_views(K, dist, 12, seed=6, noise=0.2)
    c = calibrate(views, "k5", SIZE)
    cv = cv_heldout_rms(views, "k5", SIZE)
    assert c.rms <= cv < 1.3 * c.rms
    assert math.isnan(cv_heldout_rms(views[:5], "k5", SIZE))
    new = _clean_views(K, dist, 1, seed=60, noise=0.2)[0]
    assert prequential_rms(new, c) == pytest.approx(0.2 * math.sqrt(2), rel=0.35)


@pytest.mark.parametrize("sag_mm", [0.0, 3.0])
def test_warp_score_and_ro_gain(sag_mm):
    K = S.make_K(900.0, SIZE)
    dist = np.array([-0.12, 0.04, 0.0, 0.0, 0.0])
    obj = board.object_points()
    xc, half = board.BOARD_SIZE_M[0] / 2, board.BOARD_SIZE_M[0] / 2
    warped = obj.copy()
    warped[:, 2] = sag_mm / 1000 * ((obj[:, 0] - xc) / half) ** 2
    rng = np.random.default_rng(7)
    views = []
    for i in range(8):
        rv, tv = S.random_pose(K, SIZE, rng, dist, z_range=(0.35, 0.7))
        p = S.project(K, dist, rv, tv, obj=warped)
        views.append(View(i, np.arange(40, dtype=np.int32), p + rng.normal(0, 0.15, p.shape), None, rv, tv))
    full = list(views)
    part = np.arange(24, dtype=np.int32)  # top three corner rows: partial boards, unusable for RO
    for i in range(8, 12):
        rv, tv = S.random_pose(K, SIZE, rng, dist, z_range=(0.35, 0.7))
        p = S.project(K, dist, rv, tv, obj=warped)[part]
        views.append(View(i, part, p + rng.normal(0, 0.15, p.shape), None, rv, tv))
    c = calibrate(views, "k5", SIZE)
    score, n_full = warp_score(c, views)
    assert n_full == 8
    gain, cv_ro, cv_k5 = ro_cv_gain(views, SIZE)
    # both scores cover the same 8 full-board views, not k5 over all 12
    assert cv_ro == pytest.approx(cv_heldout_rms(views, "ro", SIZE), rel=1e-9)
    assert cv_k5 == pytest.approx(cv_heldout_rms(full, "k5", SIZE), rel=1e-9)
    assert gain == pytest.approx(1 - cv_ro / cv_k5, rel=1e-12)
    if sag_mm == 0:
        assert score < 0.15 and gain < TH.warp_cv_gain
    else:
        assert score >= TH.warp_explained_var and gain >= TH.warp_cv_gain
        ro = calibrate(views, "ro", SIZE)
        assert ro.K[0, 0] == pytest.approx(900.0, rel=0.005)


def test_drift_flag():
    def vs(vals):
        return [View(i, np.zeros(0, np.int32), np.zeros((0, 2)), None, np.zeros(3), np.zeros(3), v) for i, v in enumerate(vals)]

    assert not drift_flag(vs([None, 0.2, 0.25, 0.22, 0.3, 0.21]))
    assert drift_flag(vs([None, 0.2, 0.25, 0.22, 0.9, 1.0, 0.8]))
    assert not drift_flag(vs([None, 0.2, 0.25, 0.22, 0.9, 0.3, 0.8]))
    assert not drift_flag(vs([0.2, 0.9, 1.0, 0.8]))


def test_chi2_tail_and_hfov():
    assert chi2_sf_even(0.0, 8) == 1.0
    assert chi2_sf_even(15.507, 8) == pytest.approx(0.05, abs=1e-4)
    assert chi2_sf_even(20.090, 8) == pytest.approx(0.01, abs=1e-4)
    assert chi2_sf_even(5.991, 2) == pytest.approx(0.05, abs=1e-4)
    K = S.make_K(640.0 / math.tan(math.radians(30)), SIZE)
    assert estimate_hfov_deg(K, np.zeros(5), SIZE) == pytest.approx(60.0, abs=0.1)
