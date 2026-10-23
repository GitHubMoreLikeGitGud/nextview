"""Statistics and replay helpers of the benchmark on fixtures with known answers."""

import math

import numpy as np
import pytest

from nextview.bench import stats
from nextview.bench.analysis import frames_to, p_exceed, replay_stop, value_at
from nextview.config import StopCalibration


def test_holm_known_answer():
    adj = stats.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_is_monotone():
    adj = stats.holm([0.5, 0.4, 0.9])
    assert max(adj) <= 1.0
    order = np.argsort([0.5, 0.4, 0.9])
    assert all(adj[order[i]] <= adj[order[i + 1]] for i in range(2))


def test_wilcoxon_exact_all_positive():
    a = np.arange(1, 11, dtype=float)
    r = stats.wilcoxon_paired(a, np.zeros(10))
    assert r["p"] == pytest.approx(2.0 / 1024.0)
    assert r["n_nonzero"] == 10


def test_wilcoxon_drops_zero_differences():
    r = stats.wilcoxon_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r["p"] == 1.0 and r["n_nonzero"] == 0


def test_bootstrap_constant_difference_has_degenerate_ci():
    r = stats.bootstrap_median_diff(np.full(50, 3.0), np.full(50, 1.0), n_boot=500)
    assert r["median_diff"] == 2.0 and r["ci"] == [2.0, 2.0]


def test_bootstrap_ci_covers_median_and_is_deterministic():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 100), rng.normal(0.5, 1, 100)
    r1 = stats.bootstrap_median_diff(a, b, n_boot=2000, seed=0)
    r2 = stats.bootstrap_median_diff(a, b, n_boot=2000, seed=0)
    assert r1 == r2
    assert r1["ci"][0] <= r1["median_diff"] <= r1["ci"][1]


def test_paired_endpoint_counts():
    r = stats.paired_endpoint([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 1.0, 5.0], n_boot=200)
    assert (r["wins"], r["ties"], r["losses"]) == (2, 1, 1)


def test_km_curve_known():
    c = stats.km_curve([2, 2, 5, 30], [True, True, True, False], t_max=30)
    s = {p["t"]: p["S"] for p in c}
    assert s[1] == 1.0 and s[2] == pytest.approx(0.5) and s[5] == pytest.approx(0.25) and s[30] == pytest.approx(0.25)
    assert stats.km_median([2, 2, 5, 30], [True, True, True, False]) == 2.0
    assert stats.km_median([30, 30], [False, False]) is None


def test_quantiles101_monotone():
    q = stats.quantiles101(np.random.default_rng(0).lognormal(0, 1, 500))
    assert len(q) == 101 and all(b >= a for a, b in zip(q, q[1:]))


def test_p_exceed_matches_config_stop_calibration():
    q = stats.quantiles101(np.random.default_rng(2).lognormal(0.2, 0.5, 1000))
    sc = StopCalibration(calibrated=True, quantiles={"robot": tuple(q)})
    for u in (0.1, 0.4, 0.75, 1.3, 5.0):
        assert p_exceed(u, q, 0.75) == pytest.approx(sc.p_exceed(u, "robot", 0.75))


def _row(n, views, U95, stable=1, flags=0, p95=1.0):
    return {"n": n, "views": views, "U95": U95, "stable": stable, "open_flags": flags, "true_p95": p95}


def test_replay_stop_rules():
    q = np.linspace(0.5, 2.0, 101)  # r between 0.5 and 2
    rows = [_row(5, 5, 0.1), _row(6, 6, 0.1, stable=0), _row(7, 7, 0.1, flags=1), _row(8, 8, 1.0), _row(9, 9, 0.2)]
    s = replay_stop(rows, q, tau=0.75, threshold=0.10)
    assert s is not None and s["n"] == 9  # 5: < 6 views; 6: unstable; 7: open flag; 8: U95 too large
    assert replay_stop(rows[:4], q, 0.75, 0.10) is None


def test_value_at_and_frames_to():
    rows = [{"n": 3, "true_p95": 4.0}, {"n": 4, "true_p95": 0.9}, {"n": 6, "true_p95": 0.5}]
    assert math.isnan(value_at(rows, 2))
    assert value_at(rows, 5) == 0.9 and value_at(rows, 30) == 0.5
    assert frames_to(rows, 1.0) == (4, True)
    assert frames_to(rows, 0.1) == (30, False)
