"""Planner policy (spec 2.3): table-driven rule firing, instruction templates, R8 on spec-range lenses."""

import math

import numpy as np
import pytest

import _synth as S
from nextview.agent.approvals import ApprovalKind
from nextview.agent.instructions import REJECT_TEMPLATES, describe_target, edge_away, glare_tilt, position_word
from nextview.agent.policy import (
    PRIMARY_ORDER,
    RULES,
    Finding,
    decide,
    gate,
    perception,
    post_accept,
    primary,
    rotation_angle_deg,
    tilted_target,
    warp_seek,
)
from nextview.board import object_points
from nextview.config import load_thresholds
from nextview.geometry import rvec_from_tilt, tilt_azimuth_of, tilt_of
from nextview.sim import lens as L
from nextview.tools.calibrate import calibrate, cv_heldout_rms
from nextview.tools.residuals import residual_analysis
from nextview.types import CalibResult, Detection, Diagnostics, PoseResult, Quality

TH = load_thresholds()
SIZE = (1280, 720)
OBJ = object_points()


def _det(ids, corners=None, n_markers=20, ok=None):
    ids = np.asarray(ids, np.int32)
    if corners is None:
        corners = np.column_stack([100 + 40 * (ids % 8), 100 + 40 * (ids // 8)]).astype(float)
    ok = (len(ids) >= 4) if ok is None else ok
    return Detection(ids, np.asarray(corners, float), OBJ[ids], n_markers, ok)


def _q(rise=1.2, aniso=1.1, bdir=None, sat=0.0, area=0.2, glare=None, edges=30):
    return Quality(rise, (rise,) * 4, aniso, bdir, sat, area, glare, edges)


ALL = np.arange(40)
GATE_CASES = [
    # (name, detection, quality, expected rule or None, expected instruction)
    ("clean", _det(ALL), _q(), None, None),
    ("nothing", _det([], np.zeros((0, 2)), n_markers=0, ok=False), _q(), "R0_NO_BOARD", "Board not found — bring the whole board into view"),
    ("few corners", _det(ALL[:10]), _q(), "R0_NO_BOARD", "Board not found — bring the whole board into view, closer"),
    ("one row", _det(ALL[:8], n_markers=8), _q(), "R0_NO_BOARD", "Board not found — bring the whole board into view, closer"),
    ("tiny", _det(ALL), _q(area=0.03), "R1_SMALL", "Move closer: board should fill ≥ 1/4 of the frame width"),
    ("no edges", _det(ALL), _q(rise=math.nan, edges=5), "R0_NO_BOARD", "Board not found — bring the whole board into view, closer"),
    ("motion blur", _det(ALL), _q(rise=4.04, aniso=2.0, bdir=90.0), "R2_BLUR", "Hold still — motion blur along ~90° (edge rise 4.0 px)"),
    ("soft", _det(ALL), _q(rise=3.6, aniso=1.2), "R2_BLUR", "Image is soft (3.6 px) — hold still and check focus / move back slightly"),
    ("glare right", _det(ALL), _q(sat=0.05, glare=(500.0, 180.0)), "R3_GLARE",
     "Reflection on the board — tilt ~15° about the vertical axis (left edge away) or dim the screen/light"),
    ("glare top", _det(ALL), _q(sat=0.05, glare=(240.0, 20.0)), "R3_GLARE",
     "Reflection on the board — tilt ~15° about the horizontal axis (bottom edge away) or dim the screen/light"),
    ("blur beats glare", _det(ALL), _q(rise=3.5, sat=0.05, glare=(500.0, 180.0)), "R2_BLUR", None),
    ("at the threshold", _det(ALL[:12] + 0), _q(rise=3.0, sat=0.02, area=0.04), None, None),
]


@pytest.mark.parametrize("name,det,q,rule,text", GATE_CASES, ids=[c[0] for c in GATE_CASES])
def test_gate_table(name, det, q, rule, text):
    f = gate(det, q, TH)
    if rule is None:
        assert f is None
        return
    assert f.rule == rule and f.action.startswith("reject")
    if text is not None:
        assert f.instruction == text
    line = f.trace(7)
    assert line.startswith(f"#7 {rule}: ") and " → " in line


def _pose(rvec, tvec, ippe=0.2):
    return PoseResult(np.asarray(rvec, float), np.asarray(tvec, float), 0.1, ippe, tilt_of(rvec), 0.5)


def test_perception_r4_and_r5():
    K = S.make_K(900.0, SIZE)
    rv, tv = S.pose_at(K, (640, 360), 0.5, 30, 0)
    target = {"rvec": rv.tolist(), "tvec": tv.tolist()}
    assert perception(_pose(rv, tv), TH, actuator="human", target=target, K=K, dist=np.zeros(5), size=SIZE) == []
    f = perception(_pose(rv, tv, ippe=0.9), TH, actuator="robot", target=target, K=K, dist=np.zeros(5), size=SIZE)
    assert [x.rule for x in f] == ["R4_AMBIGUOUS"]
    assert f[0].instruction == "Pose ambiguous (IPPE ratio 0.90) — next view needs a steeper tilt"
    rv2, tv2 = S.pose_at(K, (640, 360), 0.5, 30, 180)  # 60 deg away
    f = perception(_pose(rv2, tv2), TH, actuator="human", target=target, K=K, dist=np.zeros(5), size=SIZE)
    assert [x.rule for x in f] == ["R5_OFF_TARGET"] and f[0].instruction.startswith("Off target by 60°")
    rv3, tv3 = S.pose_at(K, (640 + 0.25 * 1280, 360), 0.5, 30, 0)  # 25 % of the width away
    assert [x.rule for x in perception(_pose(rv3, tv3), TH, actuator="human", target=target, K=K, dist=np.zeros(5), size=SIZE)] == ["R5_OFF_TARGET"]
    assert perception(_pose(rv2, tv2), TH, actuator="robot", target=target, K=K, dist=np.zeros(5), size=SIZE) == []


def _calib(model="k5", fx=900.0, per_view=(0.2, 0.21, 0.19, 0.2)):
    K = S.make_K(fx, SIZE)
    n = 8 if model == "rational" else 5
    return CalibResult(model, K, np.zeros(n), 0.2, np.asarray(per_view, float), [], [], np.zeros(4 + n), 0.02,
                       np.eye(4 + n), np.eye(4 + n))


def _diag(radial_p=0.5, outliers=(), warp=0.05):
    return Diagnostics(radial_p, tuple(outliers), warp, False, None)


class Calls:
    def __init__(self, value):
        self.n, self.value = 0, value

    def __call__(self):
        self.n += 1
        return self.value


def _post(**kw):
    base = dict(model={"name": "k5", "pinned": False}, n_views=14, n_full_views=4, calib=_calib(), diag=_diag(),
                preq_history=[None, None, None] + [0.1] * 8, th=TH, size=SIZE, cv_pair=Calls((0.4, 0.3)),
                ro_gain=Calls((0.3, 0.2, 0.3)))
    base.update(kw)
    return post_accept(**base), base


def test_post_accept_priority_and_lazy_evidence():
    f, b = _post()
    assert f is None and b["cv_pair"].n == 0 and b["ro_gain"].n == 0
    f, _ = _post(preq_history=[None] * 3 + [0.1] * 6 + [0.5, 0.6, 0.7], diag=_diag(radial_p=1e-5, outliers=(3,)))
    assert f.rule == "R6_DRIFT" and f.approval is ApprovalKind.DRIFT_RESTART
    assert f.evidence["prequential_rms_series"][-1] == 0.7
    f, _ = _post(diag=_diag(radial_p=1e-5, outliers=(3,)))
    assert f.rule == "R7_OUTLIER" and f.evidence["view_ids"] == [3] and f.approval is None
    f, b = _post(diag=_diag(radial_p=1e-5))
    assert f.rule == "R8_UNDERFIT" and f.approval is ApprovalKind.SWITCH_MODEL and b["cv_pair"].n == 1
    assert f.evidence["cv_rms"] == {"k5": 0.4, "rational": 0.3}


@pytest.mark.parametrize("change", [dict(n_views=11), dict(model={"name": "k5", "pinned": True}),
                                    dict(model={"name": "rational", "pinned": False}), dict(diag=_diag(radial_p=0.02))])
def test_r8_preconditions_skip_the_cv_pair(change):
    kw = dict(diag=_diag(radial_p=1e-5))
    kw.update(change)
    f, b = _post(**kw)
    assert (f is None or f.rule != "R8_UNDERFIT") and b["cv_pair"].n == 0


def test_r8_needs_the_cv_gain_and_defers_when_late():
    notes = []
    f, b = _post(diag=_diag(radial_p=1e-5), cv_pair=Calls((0.4, 0.37)), notes=notes)
    assert f is None and b["cv_pair"].n == 1 and notes and notes[0].startswith("R8 not fired")
    notes = []
    f, b = _post(diag=_diag(radial_p=1e-5), have_time=lambda: False, notes=notes)
    assert f is None and b["cv_pair"].n == 0 and notes[0].startswith("R8 deferred")


def test_r9_and_r10():
    f, b = _post(n_full_views=6, diag=_diag(warp=0.4))
    assert f.rule == "R9_WARP" and f.approval is ApprovalKind.USE_RO and b["ro_gain"].n == 1
    f, b = _post(n_full_views=6, diag=_diag(warp=0.4), ro_gain=Calls((0.05, 0.29, 0.3)))
    assert f is None
    f, b = _post(n_full_views=5, diag=_diag(warp=0.4))
    assert f is None and b["ro_gain"].n == 0
    # R10: rational model whose fitted HFOV is > 110 deg (fx = 640 / tan(60 deg) gives 120 deg)
    wide_fx = 640 / math.tan(math.radians(60))
    f, _ = _post(model={"name": "rational", "pinned": False}, calib=_calib("rational", fx=wide_fx))
    assert f.rule == "R10_OUT_OF_SCOPE" and f.approval is ApprovalKind.OUT_OF_SCOPE and f.evidence["hfov_deg"] > 110
    f, _ = _post(model={"name": "k5", "pinned": True}, calib=_calib("k5", fx=wide_fx))
    assert f is None  # a k5 model never fires R10


def _decide(**kw):
    base = dict(n_views=10, p_exceed=0.05, u95=0.3, prequential_last=[0.25, 0.28], rms=0.2, flags=[], attempts=12,
                budget=30, max_attempts=60, th=TH)
    base.update(kw)
    return decide(**base)


def test_decide_stop_budget_nbv():
    f = _decide()
    assert f.rule == "R11_STOP" and f.approval is ApprovalKind.EXPORT
    for change in (dict(p_exceed=math.nan), dict(p_exceed=0.1), dict(n_views=5), dict(flags=["UNDERFIT"]),
                   dict(prequential_last=[0.25, 0.31]), dict(prequential_last=[0.25]), dict(allow_stop=False)):
        assert _decide(**change).rule == "R13_NBV", change
    f = _decide(p_exceed=0.5, n_views=30)
    assert f.rule == "R12_BUDGET" and f.approval is ApprovalKind.ACCEPT_WITH_WARNINGS
    assert _decide(p_exceed=0.5, attempts=60).rule == "R12_BUDGET"
    assert _decide(n_views=30).rule == "R11_STOP"  # a stop that is due beats the budget


def test_primary_rule_order_and_rule_table():
    assert set(RULES) == set(PRIMARY_ORDER) and len(RULES) == 14
    fs = [Finding("R13_NBV", "", ""), Finding("R5_OFF_TARGET", "", ""), Finding("R4_AMBIGUOUS", "", ""), Finding("R7_OUTLIER", "", "")]
    assert primary(fs).rule == "R7_OUTLIER"
    assert primary(fs + [Finding("R8_UNDERFIT", "", "")]).rule == "R8_UNDERFIT"
    assert primary(fs + [Finding("R2_BLUR", "", "")]).rule == "R2_BLUR"
    for rid, r in RULES.items():
        assert (r.approval is not None) == (rid in ("R6_DRIFT", "R8_UNDERFIT", "R9_WARP", "R10_OUT_OF_SCOPE", "R11_STOP", "R12_BUDGET"))


def test_instruction_wording():
    K = S.make_K(900.0, SIZE)
    rv, tv = S.pose_at(K, (200, 600), 900 * 0.27 / (0.45 * 1280), 35, 180)
    assert describe_target(rv, tv, K, None, SIZE, sim=True) == "Board to bottom-left, ≈45% of frame width, tilt ~35° (left edge away), ≈42 cm"
    rv, tv = S.pose_at(K, (640, 360), 0.5, 2, 0, roll_deg=25)
    assert describe_target(rv, tv, K, None, SIZE) == "Board at the centre, ≈40% of frame width, facing the camera, rotated ~25° clockwise"
    assert [position_word((x, y), SIZE) for x, y in ((10, 10), (640, 360), (1270, 710))] == ["top-left", "centre", "bottom-right"]
    assert edge_away(0) == "right edge" and edge_away(90) == "bottom edge" and edge_away(225) == "top-left corner"
    assert glare_tilt((50, 10)) == ("vertical", 180.0) and glare_tilt((-5, -40)) == ("horizontal", 90.0)
    assert set(REJECT_TEMPLATES) >= {"R0_NO_BOARD", "R1_SMALL", "R3_GLARE", "R4_AMBIGUOUS", "R5_OFF_TARGET"}


def test_glare_retarget_tilts_about_the_board_centre():
    rv = rvec_from_tilt(0.0, 0.0)
    tv = np.array([-0.135, -0.09, 0.5])
    rv2, tv2 = tilted_target(rv, tv, 15.0, 180.0)
    assert rotation_angle_deg(rv, rv2) == pytest.approx(15.0) and tilt_azimuth_of(rv2) == pytest.approx(180.0)
    from nextview.geometry import BOARD_CENTRE_M, as_rotation

    c1 = as_rotation(rv) @ BOARD_CENTRE_M + tv
    c2 = as_rotation(rv2) @ BOARD_CENTRE_M + tv2
    np.testing.assert_allclose(c1, c2, atol=1e-12)


# ---- R8 on the spec's own 'wide' coefficient ranges --------------------------------------------

def _spec_wide_dist(rng):
    """Spec 6.1 'wide': k1 [-0.35, -0.20], k2 [0.03, 0.12], k3 +-0.01, k4 [0, 0.15], k5 [0, 0.03], k6 [0, 0.01]."""
    return np.array([rng.uniform(-0.35, -0.20), rng.uniform(0.03, 0.12), rng.uniform(-1e-3, 1e-3), rng.uniform(-1e-3, 1e-3),
                     rng.uniform(-0.01, 0.01), rng.uniform(0, 0.15), rng.uniform(0, 0.03), rng.uniform(0, 0.01)])


def _r8_fires(cam, seed, n=14, noise=0.25):
    views = S.camera_views(cam, np.random.default_rng(5000 + seed), n, noise=noise)
    c = calibrate(views, "k5", cam.size)
    diag = residual_analysis(c, views, TH, cam.size)
    f = post_accept(model={"name": "k5", "pinned": False}, n_views=len(views), n_full_views=0, calib=c, diag=diag,
                    preq_history=[], th=TH, size=cam.size,
                    cv_pair=lambda: (cv_heldout_rms(views, "k5", cam.size, init=c), cv_heldout_rms(views, "rational", cam.size, init=c)),
                    ro_gain=lambda: (math.nan, math.nan, math.nan), skip=("R6_DRIFT", "R7_OUTLIER"))
    return f is not None and f.rule == "R8_UNDERFIT"


def test_r8_cannot_fire_on_spec_range_wide_lenses(monkeypatch):
    """Why spec 13's "R8 fires on >= 8/10 wide cameras" is unreachable with spec 6.1's 'wide' ranges.

    Those lenses are mostly k5-shaped (k4..k6 small), so the k5 model fits them to within the noise:
    noise-free, a k5 fit on 24 views leaves a median 0.012 px per-corner residual (9 of 10 cameras
    below 0.031 px; one camera stalls at 1.22 px for both k5 and rational, a solver start problem,
    not under-fit), against 0.25 px corner noise. With nothing to detect, the radial-trend p is
    0.11-1.0 and rational does not beat k5 out of sample, so R8 correctly stays silent (0/10).
    The simulator's 'wide' family (sim/lens.py) therefore uses a Pade-like k1/k4 pair that k5
    cannot follow (noise-free k5 residual median 0.354 px), and the real criterion is tested on it:
    >= 8/10 there and <= 1/10 on 'normal' (tests/test_calibrate.py), plus end to end on the
    wide_robot_handheld preset (tests/test_loop.py).
    """
    orig = L._sample_dist
    monkeypatch.setattr(L, "_sample_dist", lambda f, r: _spec_wide_dist(r) if f == "wide" else orig(f, r))
    fires, floors = 0, []
    for s in range(10):
        cam = L.make_camera("wide", s)
        assert cam.model == "rational" and 95 <= cam.hfov_deg <= 111
        fires += _r8_fires(cam, s)
        clean = S.camera_views(cam, np.random.default_rng(9000 + s), 24, noise=0.0)
        floors.append(calibrate(clean, "k5", cam.size).rms)
    assert fires <= 1
    assert float(np.median(floors)) < 0.05 and sum(f < 0.05 for f in floors) >= 9


def test_r8_fires_on_simulator_wide_family_through_the_policy():
    fires = sum(_r8_fires(L.make_camera("wide", s), s) for s in range(10))
    assert fires >= 8


def test_r6_is_not_raised_again_while_drift_is_flagged():
    drift = [None] * 3 + [0.1] * 6 + [0.5, 0.6, 0.7]
    f, _ = _post(preq_history=drift)
    assert f.rule == "R6_DRIFT"
    f, _ = _post(preq_history=drift, flags=["DRIFT"])
    assert f is None


def test_r9_cached_check_and_incremental_result():
    warp = dict(diag=_diag(warp=0.4))
    done = {"done": True, "n_full": 6, "gain": 0.04, "ids": list(range(6))}
    notes = []
    f, b = _post(n_full_views=8, r9_last=done, notes=notes, **warp)
    assert f is None and b["ro_gain"].n == 0 and notes[0].startswith("R9 not re-checked")
    f, b = _post(n_full_views=9, r9_last=done, **warp)
    assert f.rule == "R9_WARP" and b["ro_gain"].n == 1 and f.evidence["cv_full_views"] == 6
    running = {"done": False, "n_full": 6, "ids": list(range(6)), "fold": 1}
    f, b = _post(n_full_views=7, r9_last=running, ro_gain=Calls(None), **warp)
    assert f is None and b["ro_gain"].n == 1  # a running check continues whatever the count; None = not complete


def test_r7_keeps_full_board_views_while_a_warp_is_suspected():
    f, _ = _post(diag=_diag(outliers=(3, 7), warp=0.4), full_view_ids=[1, 2, 3])
    assert f.rule == "R7_OUTLIER" and f.evidence["view_ids"] == [7]
    notes = []
    f, _ = _post(diag=_diag(outliers=(3,), warp=0.4), full_view_ids=[1, 2, 3], notes=notes)
    assert (f is None or f.rule != "R7_OUTLIER") and notes[0].startswith("R7 holds full-board view(s) [3]")
    f, _ = _post(diag=_diag(outliers=(3,), warp=0.1), full_view_ids=[1, 2, 3])
    assert f.rule == "R7_OUTLIER"  # no warp suspected: dropped as usual
    f, _ = _post(diag=_diag(outliers=(3,), warp=0.4), full_view_ids=[1, 3])
    assert f.rule == "R7_OUTLIER"  # fewer than 3 full-board views: the warp score is not trusted yet


def test_warp_seek_window():
    k5 = {"name": "k5", "pinned": False}
    assert [warp_seek(k5, n, 0.4, TH) for n in (2, 3, 5, 6)] == [False, True, True, False]
    assert warp_seek(k5, 7, 0.4, TH, r9_running=True)
    assert not warp_seek(k5, 4, 0.2, TH) and not warp_seek({"name": "k5", "pinned": True}, 4, 0.4, TH)
    assert not warp_seek({"name": "ro", "pinned": False}, 4, 0.4, TH)
