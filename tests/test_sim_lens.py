import math

import cv2
import numpy as np
import pytest

from nextview.sim.lens import LENS_FAMILIES, GTCamera, _radial_pinhole, make_camera, sample_camera

FAMILIES = list(LENS_FAMILIES)
N_DIST = {"k5": 5, "rational": 8, "fisheye": 4}


@pytest.mark.parametrize("family", FAMILIES)
def test_family_ranges(family):
    spec = LENS_FAMILIES[family]
    for seed in range(30):
        cam = make_camera(family, seed)
        assert cam.model == spec.model and len(cam.dist) == N_DIST[spec.model]
        assert cam.size == (1280, 720)
        lo, hi = spec.hfov_deg
        assert lo - 0.6 <= cam.hfov_deg <= hi + 0.6, (seed, cam.hfov_deg)
        assert abs(cam.cx - 640) <= spec.pp_jitter_px[0] and abs(cam.cy - 360) <= spec.pp_jitter_px[1]
        assert cam.fy / cam.fx == pytest.approx(1.0, abs=0.0051)
        assert cam.noise_sigma == spec.noise_sigma and cam.jpeg_quality == spec.jpeg_quality


def test_family_noise_and_jpeg_table():
    assert (LENS_FAMILIES["normal"].noise_sigma, LENS_FAMILIES["normal"].jpeg_quality) == (1.0, 95)
    assert (LENS_FAMILIES["wide"].noise_sigma, LENS_FAMILIES["wide"].jpeg_quality) == (2.0, 95)
    assert (LENS_FAMILIES["webcam"].noise_sigma, LENS_FAMILIES["webcam"].jpeg_quality) == (4.0, 70)
    assert LENS_FAMILIES["webcam"].z_near == 0.25
    assert LENS_FAMILIES["fisheye130"].failure_only


def test_wide_coefficient_ranges_and_monotonic_radial_map():
    for seed in range(40):
        cam = make_camera("wide", seed)
        k1, k2, p1, p2, k3, k4, k5, k6 = cam.dist
        assert 0.10 <= k1 <= 0.30 and 0 <= k2 <= 0.02 and abs(k3) <= 0.005
        assert 0.40 <= k4 - k1 <= 0.55 and 0 <= k5 <= 0.03 and 0 <= k6 <= 0.01
        assert abs(p1) <= 5e-4 and abs(p2) <= 5e-4
        # undistorted radius of the farthest image corner, then the radial map must rise up to it
        corners = np.array([[-0.5, -0.5], [1279.5, -0.5], [-0.5, 719.5], [1279.5, 719.5]])
        r_max = np.max(np.linalg.norm(cam.pixel_to_normalized(corners), axis=1))
        r = np.linspace(0, r_max, 2000)
        rd = r * _radial_pinhole(r * r, cam.dist)
        assert np.all(np.diff(rd) > 0)


def test_normal_and_tele_coefficients():
    for seed in range(20):
        d = make_camera("normal", seed).dist
        assert -0.15 <= d[0] <= 0.05 and -0.05 <= d[1] <= 0.10 and abs(d[2]) <= 1e-3 and abs(d[3]) <= 1e-3
        assert -0.03 <= d[4] <= 0.02
        t = make_camera("tele", seed).dist
        assert abs(t[0]) <= 0.05 and np.all(t[1:] == 0)


@pytest.mark.parametrize("family", FAMILIES)
def test_pixel_ray_round_trip(family):
    cam = make_camera(family, 3)
    rng = np.random.default_rng(0)
    uv = rng.uniform([0, 0], [1279, 719], (2000, 2))
    n = cam.pixel_to_normalized(uv)
    assert np.all(np.isfinite(n))
    back = cam.project_camera_points(np.column_stack([n, np.ones(len(n))]))
    assert np.max(np.linalg.norm(back - uv, axis=1)) < 1e-6


def test_make_camera_is_deterministic_and_seed_dependent():
    a, b, c = make_camera("wide", 7), make_camera("wide", 7), make_camera("wide", 8)
    assert np.array_equal(a.K, b.K) and np.array_equal(a.dist, b.dist)
    assert not np.array_equal(a.K, c.K)
    assert a.seed == 7 and a.key() == b.key()


def test_pinhole_projection_matches_opencv():
    cam = make_camera("wide", 2)
    X = np.array([[0.1, -0.05, 0.6], [-0.3, 0.2, 0.5], [0.0, 0.0, 1.0]])
    p, _ = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.dist)
    np.testing.assert_allclose(cam.project_camera_points(X), p.reshape(-1, 2), atol=1e-9)


def test_fisheye_model_and_angles():
    cam = make_camera("fisheye130", 0)
    assert cam.hfov_deg == pytest.approx(130.0, abs=0.6)
    ang = cam.ray_angle_deg(np.array([[cam.cx, cam.cy], [cam.cx - 640.0, cam.cy]]))
    assert ang[0] == pytest.approx(0.0, abs=1e-6)
    assert ang[1] == pytest.approx(65.0, abs=1e-6)  # HFOV is defined 640 px either side of the principal point
    X = np.array([[math.tan(math.radians(40)), 0.0, 1.0]])
    p, _ = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), cam.K, cam.dist)
    np.testing.assert_allclose(cam.project_camera_points(X), p.reshape(-1, 2), atol=1e-9)


def test_project_checked_rejects_folded_rays():
    # r (1 - 0.3 r^2) peaks at r = 1.054; a ray at r = 1.6 folds back to r_d = 0.371, inside the frame
    K = np.array([[600.0, 0, 640], [0, 600.0, 360], [0, 0, 1]])
    cam = GTCamera.pinhole(K, [-0.3, 0, 0, 0, 0])
    X = np.array([[1.6, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.0, -1.0]])
    uv, ok = cam.project_checked(X)
    assert 0 <= uv[0, 0] <= 1279  # it does land in the image...
    assert ok.tolist() == [False, True, False]  # ...but the camera cannot see it there
    # a sampled wide camera keeps rising (k4..k6 in the denominator), so far rays stay valid but off-image
    wide = make_camera("wide", 1)
    uv, ok = wide.project_checked(np.array([[4.0, 0.0, 1.0]]))
    assert ok[0] and uv[0, 0] > 1279


def test_drifted_camera():
    cam = make_camera("webcam", 0)
    d = cam.drifted()
    assert d.fx == pytest.approx(cam.fx * 1.015) and d.fy == pytest.approx(cam.fy * 1.015)
    assert d.dist[0] == pytest.approx(cam.dist[0] + 0.01)
    np.testing.assert_array_equal(d.dist[1:], cam.dist[1:])
    assert cam.fx != d.fx  # original untouched


def test_pinhole_constructor_and_validation():
    K0 = np.array([[1152.0, 0, 640], [0, 1152.0, 360], [0, 0, 1]])
    c = GTCamera.pinhole(K0)
    assert c.model == "k5" and np.all(c.dist == 0)
    assert GTCamera.pinhole(K0, np.zeros(8)).model == "rational"
    with pytest.raises(ValueError):
        GTCamera("x", "k5", K0, np.zeros(4))
    with pytest.raises(ValueError):
        make_camera("ultrawide", 0)


def _radial_rises_to_corners(cam) -> bool:
    corners = np.array([[-0.5, -0.5], [1279.5, -0.5], [-0.5, 719.5], [1279.5, 719.5]])
    n = cam.pixel_to_normalized(corners)
    if not np.all(np.isfinite(n)):
        return False
    r = np.linspace(0, np.max(np.linalg.norm(n, axis=1)), 2000)
    rd = r * _radial_pinhole(r * r, cam.dist)
    return bool(np.all(np.diff(rd) > 0))


def test_sampler_rejects_impossible_family_draws(monkeypatch):
    import nextview.sim.lens as lens_mod

    # every returned pinhole camera rises monotonically out to the image corners
    rng = np.random.default_rng(123)
    cams = [sample_camera("wide", rng) for _ in range(200)]
    assert all(95 - 0.6 <= c.hfov_deg <= 110 + 0.6 for c in cams)
    assert all(_radial_rises_to_corners(c) for c in cams)
    assert max(c.meta["sample_attempts"] for c in cams) > 1  # rejection does happen on real draws
    for fam in ("normal", "webcam", "tele"):
        assert all(_radial_rises_to_corners(make_camera(fam, s)) for s in range(30))

    # a draw that folds back inside the image is redrawn: r (1 - 0.6 r^2) peaks at r = 0.75
    real = lens_mod._sample_dist
    calls = {"n": 0}

    def folding_first(family, g):
        calls["n"] += 1
        d = real(family, g)
        return np.array([-0.6, 0, 0, 0, 0, 0, 0, 0]) if calls["n"] == 1 else d

    monkeypatch.setattr(lens_mod, "_sample_dist", folding_first)
    cam = sample_camera("wide", np.random.default_rng(0))
    assert cam.meta["sample_attempts"] == 2 and cam.dist[0] != -0.6 and _radial_rises_to_corners(cam)
