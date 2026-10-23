import math

import cv2
import numpy as np
import pytest

import _synth as S
from nextview.config import StopCalibration
from nextview.tools.calibrate import calibrate
from nextview.tools.uncertainty import (
    covariance,
    grid_covariances,
    grid_jacobian,
    information_matrix,
    noise_variance,
    projection_jacobian,
    u95_from_cov,
    uncertainty_map,
    view_information,
)
from nextview.geometry import grid_centres
from nextview.types import View

SIZE = (1280, 720)
K_GT = S.make_K(900.0, SIZE, aspect=1.001, pp_offset=(10.0, -6.0))
D_GT = np.array([-0.2, 0.06, 0.0007, -0.0004, -0.01])
NOISE = 0.25
CHI2_2_95 = 5.991464547107979


def _poses(n, seed=21):
    rng = np.random.default_rng(seed)
    return [S.random_pose(K_GT, SIZE, rng, D_GT) for _ in range(n)]


POSES = _poses(10)
CLEAN = [S.project(K_GT, D_GT, rv, tv) for rv, tv in POSES]


def _views(seed, noise=NOISE, poses=POSES, clean=CLEAN):
    rng = np.random.default_rng(seed)
    return [View(i, np.arange(40, dtype=np.int32), c + rng.normal(0, noise, c.shape), None, rv, tv) for i, ((rv, tv), c) in enumerate(zip(poses, clean))]


def test_schur_sum_equals_full_marginal():
    poses = POSES[:4]
    p = 9
    blocks, F_sum = [], np.zeros((p, p))
    for rv, tv in poses:
        _, Jp, Jt = projection_jacobian(K_GT, D_GT, rv, tv, S.OBJ)
        blocks.append((Jp, Jt))
        F_sum += view_information(K_GT, D_GT, rv, tv, S.OBJ)
    V = len(poses)
    rows = sum(Jp.shape[0] for Jp, _ in blocks)
    J = np.zeros((rows, 6 * V + p))
    r = 0
    for v, (Jp, Jt) in enumerate(blocks):
        J[r : r + Jp.shape[0], 6 * v : 6 * v + 6] = Jp
        J[r : r + Jp.shape[0], 6 * V :] = Jt
        r += Jp.shape[0]
    H = J.T @ J
    Hpp, Hpt, Htt = H[: 6 * V, : 6 * V], H[: 6 * V, 6 * V :], H[6 * V :, 6 * V :]
    marginal = Htt - Hpt.T @ np.linalg.solve(Hpp, Hpt)
    assert np.linalg.norm(F_sum - marginal) / np.linalg.norm(marginal) < 1e-6
    np.testing.assert_allclose(information_matrix(K_GT, D_GT, *zip(*poses), [S.OBJ] * V), F_sum)


def test_jacobian_columns():
    _, Jp, Jt = projection_jacobian(K_GT, D_GT, *POSES[0], S.OBJ)
    assert Jp.shape == (80, 6) and Jt.shape == (80, 9)
    _, _, Jt8 = projection_jacobian(K_GT, np.zeros(8), *POSES[0], S.OBJ)
    assert Jt8.shape == (80, 12)


def test_std_matches_opencv_std_deviations():
    c = calibrate(_views(1), "k5", SIZE)
    ours = np.sqrt(np.diag(c.cov))
    ocv = c.std_intr[:9]
    assert np.all(np.abs(ours / ocv - 1) < 0.15)
    rat = calibrate(_views(1), "rational", SIZE)
    assert np.all(np.abs(np.sqrt(np.diag(rat.cov)) / rat.std_intr[:12] - 1) < 0.15)


def test_noise_variance_formula():
    assert noise_variance(0.5, 400, 9, 10) == pytest.approx(0.25 * 400 / (800 - 9 - 60))
    assert math.isnan(noise_variance(0.5, 10, 9, 2))
    c = calibrate(_views(2), "k5", SIZE)
    assert c.sigma2 == pytest.approx(NOISE**2, rel=0.15)


def _monte_carlo(n_draws):
    thetas, covs, coverage = [], [], []
    u = grid_centres(SIZE, (32, 18))
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
    r_gt = cv2.undistortPoints(u.reshape(-1, 1, 2), K_GT, D_GT, criteria=crit).reshape(-1, 2)
    r_gt = np.column_stack([r_gt, np.ones(len(r_gt))])
    for seed in range(n_draws):
        c = calibrate(_views(1000 + seed), "k5", SIZE)
        thetas.append(np.r_[c.K[0, 0], c.K[1, 1], c.K[0, 2], c.K[1, 2], c.dist])
        covs.append(c.cov)
        q, _ = cv2.projectPoints(r_gt, np.zeros(3), np.zeros(3), c.K, c.dist)
        e = q.reshape(-1, 2) - u
        C = grid_covariances(grid_jacobian(c.K, c.dist, SIZE), c.cov)
        m = np.einsum("mi,mij,mj->m", e, np.linalg.inv(C), e)
        coverage.append(np.mean(m <= CHI2_2_95))
    return np.array(thetas), np.array(covs), np.array(coverage)


@pytest.fixture(scope="module")
def mc30():
    return _monte_carlo(30)


def test_predicted_std_matches_empirical_spread(mc30):
    thetas, covs, _ = mc30
    empirical = thetas.std(axis=0, ddof=1)
    predicted = np.sqrt(np.diag(covs.mean(axis=0)))
    ratio = predicted / empirical
    assert np.all((ratio > 1 / 1.5) & (ratio < 1.5)), ratio


def test_noise_only_coverage_30_draws(mc30):
    _, _, coverage = mc30
    assert abs(coverage.mean() - 0.95) < 0.08


def test_monte_carlo_200_draws():
    thetas, covs, coverage = _monte_carlo(200)
    emp = np.cov(thetas[:, :4], rowvar=False)
    pred = covs.mean(axis=0)[:4, :4]
    assert np.linalg.norm(emp - pred) / np.linalg.norm(pred) < 0.20
    assert abs(coverage.mean() - 0.95) < 0.05


def test_uncertainty_map_contract():
    views = _views(3)
    c = calibrate(views, "k5", SIZE)
    m = uncertainty_map(c, views, SIZE, None)
    assert m.grid == (32, 18) and m.sigma_px.shape == (18, 32)
    assert m.U95 == pytest.approx(np.percentile(m.sigma_px, 95))
    assert math.isnan(m.p_exceed)
    # corners of the image are less certain than the centre
    assert m.sigma_px[0, 0] > m.sigma_px[9, 16] and m.sigma_px[-1, -1] > m.sigma_px[9, 16]
    # rebuilding the covariance from the views gives the same map
    c2 = calibrate(views, "k5", SIZE)
    c2.cov = None
    np.testing.assert_allclose(uncertainty_map(c2, views, SIZE, None).sigma_px, m.sigma_px, rtol=1e-9)


def test_p_exceed_from_stop_calibration():
    views = _views(4)
    c = calibrate(views, "k5", SIZE)
    q = tuple(np.linspace(0.0, 2.0, 101))  # r uniform on [0, 2]
    stop = StopCalibration(calibrated=True, quantiles={"robot": q, "human": q})
    m = uncertainty_map(c, views, SIZE, stop, actuator="robot", tau_px=0.5)
    expected = 1.0 - min(max(0.5 / m.U95 / 2.0, 0.0), 1.0)
    assert m.p_exceed == pytest.approx(expected, abs=1e-9)
    uncal = StopCalibration(calibrated=False, quantiles={})
    assert math.isnan(uncertainty_map(c, views, SIZE, uncal, actuator="human").p_exceed)
    with pytest.raises(ValueError):
        uncertainty_map(c, views, SIZE, stop)  # the stop quantiles depend on the actuator


def test_added_view_reduces_u95():
    views = _views(5)
    c = calibrate(views[:6], "k5", SIZE)
    G = grid_jacobian(c.K, c.dist, SIZE)
    before = u95_from_cov(G, covariance(c.F, c.sigma2))
    S_new = sum(view_information(c.K, c.dist, rv, tv, S.OBJ) for rv, tv in POSES[6:])
    after = u95_from_cov(G, covariance(c.F + S_new, c.sigma2))
    assert after < before


def test_covariance_handles_badly_scaled_information():
    rng = np.random.default_rng(8)
    A = rng.normal(size=(9, 9))
    base = A @ A.T + 9 * np.eye(9)
    scale = np.logspace(0, 6, 9)  # parameters in very different units
    F = base * np.outer(scale, scale)
    exact = np.linalg.inv(base) / np.outer(scale, scale)
    np.testing.assert_allclose(covariance(F, 1.0), exact, rtol=1e-6)
