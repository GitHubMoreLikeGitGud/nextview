import math

import cv2
import numpy as np
import pytest

from nextview.board import BOARD_SIZE_M
from nextview.sim.nuisance import (
    LAMP_CAM_M,
    NuisanceConfig,
    add_noise,
    apply_corruption,
    apply_psf,
    combine_kernels,
    defocus_kernel,
    defocus_radius,
    glare_hotspot,
    jpeg_roundtrip,
    motion_kernel,
    sample_corruption,
    sample_occluder,
)
from nextview.sim.poses import pose_for_view
from nextview.sim.lens import make_camera


def _moments(k):
    ys, xs = np.mgrid[0 : k.shape[0], 0 : k.shape[1]]
    c = (k.shape[1] // 2, k.shape[0] // 2)
    mx = float((k * xs).sum()) - c[0]
    my = float((k * ys).sum()) - c[1]
    return mx, my, xs - c[0], ys - c[1]


@pytest.mark.parametrize("L,ang", [(0.6, 0.0), (3.0, 30.0), (8.0, 0.0), (10.0, 117.0)])
def test_motion_kernel_centred_and_oriented(L, ang):
    k = motion_kernel(L, ang)
    assert k.dtype == np.float32 and k.sum() == pytest.approx(1.0, abs=1e-6)
    mx, my, X, Y = _moments(k)
    assert abs(mx) < 1e-6 and abs(my) < 1e-6  # no image shift
    a = math.radians(ang)
    along = X * math.cos(a) + Y * math.sin(a)
    across = -X * math.sin(a) + Y * math.cos(a)
    var_along = float((k * along**2).sum())
    var_across = float((k * across**2).sum())
    assert var_along > var_across
    if L >= 3:  # uniform segment: L^2 / 12, plus about 1/6 px^2 from bilinear splatting
        assert var_along == pytest.approx(L * L / 12 + 1 / 6, rel=0.1)
    assert motion_kernel(0.01, 0.0) is None


def test_defocus_radius_formula_and_kernel():
    assert defocus_radius(0.30, 0.25) == 0.0
    assert defocus_radius(0.20, 0.25) == pytest.approx(1.6)
    assert defocus_radius(0.05, 0.25) == 4.0
    assert defocus_radius(0.10, 0.0) == 0.0
    k = defocus_kernel(2.5)
    assert k.sum() == pytest.approx(1.0, abs=1e-6)
    mx, my, X, Y = _moments(k)
    assert abs(mx) < 1e-6 and abs(my) < 1e-6
    # uniform disk: E[r^2] = R^2 / 2
    assert float((k * (X**2 + Y**2)).sum()) == pytest.approx(2.5**2 / 2, rel=0.05)
    assert defocus_kernel(0.1) is None


def test_combined_psf_keeps_edges_in_place():
    img = np.zeros((80, 80), np.float32)
    img[:, 40:] = 200.0
    k = combine_kernels(motion_kernel(6.0, 0.0), defocus_kernel(1.5))
    out = apply_psf(img, k)
    row = out[40]
    # the 50 % crossing stays at the step (39.5) for symmetric kernels
    i = int(np.nonzero(row >= 100.0)[0][0])
    x = i - 1 + (100.0 - row[i - 1]) / (row[i] - row[i - 1])
    assert x == pytest.approx(39.5, abs=0.05)
    assert combine_kernels(None, None) is None


def test_occluder_sampler():
    rng = np.random.default_rng(0)
    specs = [sample_occluder(rng, 0.15) for _ in range(3000)]
    rate = np.mean([s.active for s in specs])
    assert rate == pytest.approx(0.15, abs=0.025)
    board_area = BOARD_SIZE_M[0] * BOARD_SIZE_M[1]
    for s in specs[:300]:
        poly = s.polygon_m.astype(np.float32)
        assert 4 <= len(poly) <= 6
        assert cv2.isContourConvex(poly.reshape(-1, 1, 2))
        assert cv2.contourArea(poly) / board_area == pytest.approx(s.area_frac, rel=1e-3)
        assert 0.05 <= s.area_frac <= 0.20 and 60 <= s.grey <= 120


def test_glare_hotspot_obeys_reflection_law():
    cam = make_camera("webcam", 0)
    found = 0
    rng = np.random.default_rng(3)
    for _ in range(400):
        c = rng.uniform([100, 0], [1180, 400])
        rv, tv = pose_for_view(cam, c, rng.uniform(0.4, 0.8), rng.uniform(0, 40), rng.uniform(0, 360))
        b = glare_hotspot(rv, tv)
        if b is None:
            continue
        found += 1
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
        P = R @ np.array([b[0], b[1], 0.0]) + tv
        n = R[:, 2]
        a = (LAMP_CAM_M - P) / np.linalg.norm(LAMP_CAM_M - P)
        e = (np.zeros(3) - P) / np.linalg.norm(P)
        # incoming and outgoing directions make equal angles with the normal, in one plane
        assert abs(a @ n) == pytest.approx(abs(e @ n), abs=1e-9)
        assert np.linalg.norm(np.cross(a + e, n)) < 1e-9
    assert found > 10


def test_corruption_moves_three_detected_corners():
    rng = np.random.default_rng(1)
    specs = [sample_corruption(rng, 0.05) for _ in range(4000)]
    assert np.mean([s.active for s in specs]) == pytest.approx(0.05, abs=0.012)
    spec = next(s for s in specs if s.active)
    ids = np.array([3, 4, 10, 11, 20, 33, 39])
    corners = np.arange(14, dtype=float).reshape(7, 2) * 10
    out, moved = apply_corruption(ids, corners, spec)
    shift = np.linalg.norm(out - corners, axis=1)
    assert (shift > 0).sum() == 3 and len(moved) == 3
    assert np.all((shift[shift > 0] >= 2.0 - 1e-9) & (shift[shift > 0] <= 5.0 + 1e-9))
    # the same detection always gets the same corruption
    out2, moved2 = apply_corruption(ids, corners, spec)
    np.testing.assert_array_equal(out, out2)
    inactive = next(s for s in specs if not s.active)
    out3, moved3 = apply_corruption(ids, corners, inactive)
    np.testing.assert_array_equal(out3, corners)
    assert len(moved3) == 0


def test_noise_stream_alignment_and_jpeg():
    img = np.full((40, 60), 100.0, np.float32)
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    add_noise(img, 0.0, r1)
    add_noise(img, 3.0, r2)
    assert r1.uniform() == r2.uniform()  # same number of draws whatever sigma is
    noisy = add_noise(img, 4.0, np.random.default_rng(0))
    assert float(np.std(noisy - img)) == pytest.approx(4.0, rel=0.1)
    u8 = np.clip(noisy, 0, 255).astype(np.uint8)
    j = jpeg_roundtrip(u8, 70)
    assert j.dtype == np.uint8 and j.shape == u8.shape
    assert jpeg_roundtrip(u8, None) is u8


def test_nuisance_config():
    cam = make_camera("webcam", 0)
    c = NuisanceConfig.for_camera(cam, ["glare"])
    assert c.glare and not c.autofocus_drift and c.noise_sigma == 4.0 and c.jpeg_quality == 70 and c.z_near == 0.25
    assert c.occluder_p == 0.15 and c.corruption_p == 0.05
    with pytest.raises(ValueError):
        NuisanceConfig.for_camera(cam, ["rain"])
    clean = NuisanceConfig.clean()
    assert clean.noise_sigma == 0 and clean.jpeg_quality is None and clean.occluder_p == 0 and clean.corruption_p == 0
