import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

import _synth as S
from nextview.tools.detect import detect_board
from nextview.tools.esf import (
    GAP_FRAC,
    RISE_PER_SIGMA,
    axial_mean_deg,
    erf,
    fit_edge_profile,
    measure_edge,
    measure_edges,
    norm_cdf,
)
from nextview.tools.quality import assess_frame

SIZE = (1280, 720)
K = S.make_K(900.0, SIZE)


def test_erf_matches_math_erf():
    x = np.linspace(-4, 4, 801)
    ref = np.array([math.erf(v) for v in x])
    assert np.max(np.abs(erf(x) - ref)) < 2e-7


def _analytic_edge(sigma, angle_deg=5.0, size=(240, 240), lo=30.0, hi=220.0, gap=None):
    """Point-sampled Gaussian-blurred step (optionally a white bar of width ``gap``) along a slanted line."""
    h, w = size
    c = np.array([w / 2.0, h / 2.0])
    u = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    n = np.array([-u[1], u[0]])
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    d = (xs - c[0]) * n[0] + (ys - c[1]) * n[1]
    shape = norm_cdf(d / sigma)
    if gap is not None:
        shape = shape - norm_cdf((d - gap) / sigma)
    img = lo + (hi - lo) * shape
    return img, c - 50 * u, c + 50 * u


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.0])
def test_slanted_edge_recovers_gaussian_sigma(sigma):
    img, p0, p1 = _analytic_edge(sigma)
    fit = measure_edge(img, p0, p1, across_px_per_square=100.0, gap_frac=None)
    assert fit.ok
    assert fit.rise == pytest.approx(RISE_PER_SIGMA * sigma, rel=0.10)


@pytest.mark.parametrize("sigma", [1.0, 2.0, 3.0])
def test_marker_border_term_removes_bias(sigma):
    # Square of 60 px: the marker border is 8 px from the edge, inside the reach of a 3 px blur.
    s = 60.0
    img, p0, p1 = _analytic_edge(sigma, angle_deg=-7.0, gap=GAP_FRAC * s)
    with_gap = measure_edge(img, p0, p1, across_px_per_square=s)
    assert with_gap.ok
    assert with_gap.sigma == pytest.approx(sigma, rel=0.10)


def test_fit_edge_profile_on_exact_samples():
    d = np.arange(-6, 6, 0.25) + 0.125
    y = 10 + 200 * norm_cdf((d - 0.3) / 1.3)
    lo, amp, mu, sig, ok = fit_edge_profile(d, y)
    assert ok
    assert (lo, amp, mu, sig) == pytest.approx((10, 200, 0.3, 1.3), rel=1e-4, abs=1e-4)


POSES = [
    ("fronto", dict(distance=0.5, tilt_deg=0.0, azimuth_deg=0.0, roll_deg=4.0)),
    ("tilted", dict(distance=0.55, tilt_deg=35.0, azimuth_deg=200.0, roll_deg=10.0)),
]


@pytest.fixture(scope="module")
def board_frames():
    out = {}
    for name, kw in POSES:
        rv, tv = S.pose_at(K, (640, 360), **kw)
        sharp = S.render(K, None, rv, tv, SIZE, ss=4)
        det = detect_board(sharp)
        assert det.ok and len(det.ids) == 40
        out[name] = (rv, tv, sharp, det)
    return out


@pytest.mark.parametrize("pose", [p[0] for p in POSES])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.0])
def test_board_esf_recovers_added_blur(board_frames, pose, sigma):
    rv, tv, sharp, det = board_frames[pose]
    base = measure_edges(sharp, det)
    assert base.n_edges >= 60
    s0 = base.edge_rise_px / RISE_PER_SIGMA  # pixel aperture plus texture interpolation
    img = S.render(K, None, rv, tv, SIZE, ss=4, blur_sigma=sigma, noise=1.0, rng=np.random.default_rng(1))
    r = measure_edges(img, det)
    assert r.n_edges >= 40
    s_meas = r.edge_rise_px / RISE_PER_SIGMA
    recovered = math.sqrt(max(s_meas**2 - s0**2, 0.0))
    assert recovered == pytest.approx(sigma, rel=0.10)
    assert r.anisotropy < 1.25 and r.blur_dir_deg is None


@pytest.mark.parametrize("angle", [0.0, 90.0])
def test_motion_blur_direction(board_frames, angle):
    rv, tv, _, det = board_frames["fronto"]
    img = S.render(K, None, rv, tv, SIZE, ss=4, motion=(8.0, angle), noise=1.0, rng=np.random.default_rng(2))
    q = assess_frame(img, det)
    assert q.anisotropy >= 1.5
    assert q.blur_dir_deg is not None
    diff = abs((q.blur_dir_deg - angle + 90.0) % 180.0 - 90.0)
    assert diff <= 15.0
    assert q.edge_rise_px > 3.0


def test_motion_blur_direction_on_rolled_board():
    # Edge normals at 25 and 115 deg: the 45 deg bin centre would be 20 deg off a blur along 25 deg.
    rv, tv = S.pose_at(K, (640, 360), 0.5, 0.0, 0.0, 25.0)
    det = detect_board(S.render(K, None, rv, tv, SIZE, ss=4))
    img = S.render(K, None, rv, tv, SIZE, ss=4, motion=(8.0, 25.0), noise=1.0, rng=np.random.default_rng(4))
    q = assess_frame(img, det)
    assert q.anisotropy >= 1.5
    assert abs((q.blur_dir_deg - 25.0 + 90.0) % 180.0 - 90.0) <= 5.0


def test_axial_mean():
    d = axial_mean_deg([170.0, 10.0])  # wraps through 0, not 90
    assert min(d, 180.0 - d) < 1e-9
    assert axial_mean_deg([20.0, 30.0]) == pytest.approx(25.0)


def test_sharp_frame_quality(board_frames):
    rv, tv, sharp, det = board_frames["fronto"]
    q = assess_frame(sharp, det)
    assert q.edge_rise_px < 1.2
    assert q.n_edges == 67
    assert q.saturation_frac == 0.0 and q.glare_centroid is None
    quad = S.project(K, None, rv, tv, obj=np.array([[0, 0, 0], [0.27, 0, 0], [0.27, 0.18, 0], [0, 0.18, 0]], float))
    x, y = quad[:, 0], quad[:, 1]
    true_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / (SIZE[0] * SIZE[1])
    assert q.board_area_frac == pytest.approx(true_area, rel=0.02)


def test_glare_hotspot_is_measured(board_frames):
    rv, tv, sharp, det = board_frames["fronto"]
    img = sharp.astype(np.float32)
    centre = (700.0, 330.0)
    ys, xs = np.mgrid[0 : SIZE[1], 0 : SIZE[0]]
    img += 400.0 * np.exp(-((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) / (2 * 25.0**2))
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = assess_frame(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), det)
    assert q.saturation_frac > 0.02
    assert q.glare_centroid is not None
    assert np.hypot(q.glare_centroid[0] - centre[0], q.glare_centroid[1] - centre[1]) < 5.0


def test_too_few_edges_gives_nan(board_frames):
    _, _, sharp, det = board_frames["fronto"]
    from nextview.types import Detection

    keep = np.array([0, 1, 8, 9, 30, 39])  # two adjacent pairs and two isolated corners: 4 edges
    sub = Detection(det.ids[keep], det.corners[keep], det.obj[keep], det.n_markers, True)
    q = assess_frame(sharp, sub)
    assert math.isnan(q.edge_rise_px) and q.n_edges < 8


def test_undetected_frame_quality():
    img = np.full((SIZE[1], SIZE[0]), 90, np.uint8)
    det = detect_board(img)
    assert not det.ok
    q = assess_frame(img, det)
    assert math.isnan(q.edge_rise_px) and q.board_area_frac == 0.0 and q.n_edges == 0


@pytest.mark.slow
def test_blur_curve():
    """Corner error against measured edge rise on rendered frames; written to results/blur_curve.csv."""
    rng = np.random.default_rng(11)
    dist = np.array([-0.12, 0.04, 0.0, 0.0, 0.0])
    poses = [S.random_pose(K, SIZE, rng, dist, z_range=(0.4, 0.8)) for _ in range(8)]
    rows = []
    for sigma in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        errs, rises = [], []
        for rv, tv in poses:
            img = S.render(K, dist, rv, tv, SIZE, ss=2, blur_sigma=sigma, noise=1.0, rng=rng)
            det = detect_board(img)
            if not det.ok or len(det.ids) < 12:
                continue
            q = assess_frame(img, det)
            gt = S.project(K, dist, rv, tv)[det.ids]
            errs.append(float(np.median(np.linalg.norm(det.corners - gt, axis=1))))
            rises.append(q.edge_rise_px)
        rows.append((sigma, float(np.nanmedian(rises)) if rises else math.nan, float(np.median(errs)) if errs else math.nan, len(errs)))
    out = Path(__file__).resolve().parents[1] / "results" / "blur_curve.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["blur_sigma_px", "median_edge_rise_px", "median_corner_err_px", "n_frames"])
        w.writerows(rows)
    rises = [r[1] for r in rows if np.isfinite(r[1])]
    assert all(b >= a - 0.05 for a, b in zip(rises, rises[1:]))  # rise grows with blur
