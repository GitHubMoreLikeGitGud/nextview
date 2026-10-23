"""Agent loop end to end on simulated sessions: approvals, stop/escalation paths, token round trips."""

import json

import numpy as np
import pytest

from nextview.agent.approvals import ApprovalError
from nextview.agent.loop import FrameSource, LoopError, apply_approval, make_source, run_step
from nextview.agent.policy import rotation_angle_deg
from nextview.agent.state import decode_token, encode_token, new_session
from nextview.config import StopCalibration, load_thresholds
from nextview.sim import make_simulator
from nextview.sim.poses import pose_for_view

TH = load_thresholds()
UNCAL = StopCalibration(False, {})
KEY = b"t" * 32
# r = true_p95 / U95 taken as exactly 1: P_exceed = 1[U95 > tau]. Exercises R11 only; not a calibration.
IDENTITY = StopCalibration(True, {"robot": tuple([1.0] * 101), "human": tuple([1.0] * 101)})


def _session(preset, actuator, **kw):
    kw.setdefault("baseline", False)
    st = new_session(mode="sim", preset=preset, actuator=actuator, th=TH, **kw)
    return st, make_source(st)


def _run_until(st, src, cond, th=TH, stopcal=UNCAL, max_calls=80):
    for _ in range(max_calls):
        if cond(st):
            return
        assert st.pending_approval is None, st.pending_approval
        run_step(st, src, th=th, stopcal=stopcal)
    raise AssertionError("condition not reached")


def _copy(st):
    return decode_token(encode_token(st, KEY), KEY)


def test_budget_escalates_and_export_needs_approval():
    st, src = _session("inspection_gauge_cam", "robot", seed=0, budget=6)
    results = []
    while st.pending_approval is None:
        results.append(run_step(st, src, th=TH, stopcal=UNCAL))
        assert results[-1].export_files is None
    r = results[-1]
    assert r.status == "escalated" and r.rule == "R12_BUDGET" and r.approval["kind"] == "accept_with_warnings"
    assert "stop probability not calibrated" in r.approval["question"] or "not calibrated" in json.dumps(r.approval)
    assert len(st.views) == 6
    attempt = st.attempt
    w = run_step(st, src, th=TH, stopcal=UNCAL)  # nothing is captured while a card is open
    assert w.status == "escalated" and st.attempt == attempt and w.approval["id"] == r.approval["id"]
    with pytest.raises(ApprovalError):
        apply_approval(st, "ap_999", "approve", source=src, th=TH, stopcal=UNCAL)
    with pytest.raises(ApprovalError):
        apply_approval(st, r.approval["id"], "acknowledge", source=src, th=TH, stopcal=UNCAL)
    other = _copy(st)
    ok = apply_approval(st, r.approval["id"], "approve", "fine for a demo", source=src, th=TH, stopcal=UNCAL, now=123.0)
    assert ok.status == "exported" and set(ok.export_files) == {"yaml", "png", "json"} and ok.export_meta["warnings"] is True
    assert ok.export_files["png"][:4] == b"\x89PNG" and b"camera_matrix" in ok.export_files["yaml"]
    assert st.approvals[-1] == {"id": r.approval["id"], "kind": "accept_with_warnings", "decision": "approve",
                                "reason": "fine for a demo", "ts": 123.0, "step": st.step, "rule": "R12_BUDGET"}
    assert any("APPROVAL" in ln and "accept_with_warnings: approve" in ln for ln in st.trace)
    with pytest.raises(LoopError):
        run_step(st, src, th=TH, stopcal=UNCAL)
    gone = apply_approval(other, r.approval["id"], "deny", source=src, th=TH, stopcal=UNCAL)
    assert gone.status == "discarded" and gone.export_files is None


def test_stop_asks_for_export_and_deny_keeps_capturing():
    st, src = _session("inspection_gauge_cam", "robot", seed=0)
    _run_until(st, src, lambda s: s.pending_approval is not None, stopcal=IDENTITY)
    card = st.pending_approval
    assert st.status == "stopped" and card["kind"] == "export" and card["rule"] == "R11_STOP"
    assert card["evidence"]["p_exceed"] < TH.stop_p_exceed and len(st.views) >= TH.min_views_stop
    keep = _copy(st)
    d = apply_approval(st, card["id"], "deny", "one more please", source=src, th=TH, stopcal=IDENTITY)
    assert d.status == "continue" and d.export_files is None and d.rule == "R13_NBV" and st.pending_approval is None
    n = len(st.views)
    r = run_step(st, src, th=TH, stopcal=IDENTITY)
    assert len(st.views) >= n  # capturing resumed
    e = apply_approval(keep, card["id"], "approve", source=src, th=TH, stopcal=IDENTITY)
    assert e.status == "exported" and e.export_meta["warnings"] is False and "true_field_p95_px" in e.export_meta
    assert r.status in ("continue", "stopped", "escalated")


def test_model_switch_needs_approval_and_deny_pins():
    st, src = _session("wide_robot_handheld", "human", seed=0)
    _run_until(st, src, lambda s: s.pending_approval is not None)
    card = st.pending_approval
    assert card["kind"] == "switch_model" and card["rule"] == "R8_UNDERFIT" and st.status == "awaiting_approval"
    assert st.model == {"name": "k5", "pinned": False} and st.calib["model"] == "k5" and "UNDERFIT" in st.flags
    ev = card["evidence"]
    assert ev["radial_trend_p"] < TH.underfit_p and ev["cv_rms"]["rational"] < 0.9 * ev["cv_rms"]["k5"]
    assert ev["residual_field"]["grid"] == [16, 9] and len(ev["residual_field"]["vectors"]) == 144
    attempt = st.attempt
    assert run_step(st, src, th=TH, stopcal=UNCAL).rule == "R8_UNDERFIT" and st.attempt == attempt and st.model["name"] == "k5"

    approve = _copy(st)
    d = apply_approval(st, card["id"], "deny", "keep k5", source=src, th=TH, stopcal=UNCAL)
    assert d.status == "continue" and st.model == {"name": "k5", "pinned": True} and "UNDERFIT" not in st.flags
    n = len(st.views)
    _run_until(st, src, lambda s: len(s.views) >= n + 3 or s.pending_approval is not None)
    assert st.pending_approval is None or st.pending_approval["kind"] != "switch_model"
    assert st.model["name"] == "k5" and not any(ln.split(": ")[0].endswith("R8_UNDERFIT") for ln in st.trace[-6:])

    a = apply_approval(approve, card["id"], "approve", source=src, th=TH, stopcal=UNCAL)
    assert approve.model == {"name": "rational", "pinned": False} and approve.calib["model"] == "rational"
    assert a.calibration["model"] == "rational" and len(a.calibration["dist"]) == 8
    assert a.truth["field_p95_px"] < 1.0  # the rational refit fixes the under-fit


def test_out_of_scope_is_reachable_on_fisheye130():
    st, src = _session("fisheye_out_of_scope", "robot", seed=0)
    for _ in range(80):
        card = st.pending_approval
        if card is None:
            run_step(st, src, th=TH, stopcal=UNCAL)
        elif card["kind"] == "switch_model":
            res = apply_approval(st, card["id"], "approve", source=src, th=TH, stopcal=UNCAL)
        else:
            break
    card = st.pending_approval
    assert card["kind"] == "out_of_scope" and res.rule == "R10_OUT_OF_SCOPE" and st.status == "escalated"
    assert card["evidence"]["hfov_deg"] > TH.out_of_scope_hfov_deg and "OUT_OF_SCOPE" in st.flags
    assert "use cv2.fisheye" in card["question"] and [o["decision"] for o in card["options"]] == ["acknowledge"]
    assert "> 110°" in card["question"] and "120°" not in card["question"]
    with pytest.raises(ApprovalError):
        apply_approval(st, card["id"], "approve", source=src, th=TH, stopcal=UNCAL)
    end = apply_approval(st, card["id"], "acknowledge", source=src, th=TH, stopcal=UNCAL)
    assert end.status == "ended" and st.terminal


def test_token_round_trip_reproduces_the_in_memory_run():
    a, src = _session("phone_crack_metrology", "human", seed=3, baseline=True)
    b = _copy(a)
    ra = [run_step(a, src, th=TH, stopcal=UNCAL) for _ in range(6)]
    rb = []
    for _ in range(6):
        rb.append(run_step(b, src, th=TH, stopcal=UNCAL))
        b = _copy(b)
    assert [r.trace_all for r in ra] == [r.trace_all for r in rb]
    assert [r.uncertainty and r.uncertainty["U95_px"] for r in ra] == [r.uncertainty and r.uncertainty["U95_px"] for r in rb]
    assert a.to_dict()["views"] == b.to_dict()["views"] and a.baseline["truth"] == b.baseline["truth"]
    resp = ra[-1].to_response()
    text = json.dumps(resp)
    assert len(text) < 1_000_000 and "jpeg" not in text
    assert len(resp["uncertainty"]["sigma_px"]) == 576 and len(resp["coverage"]["counts"]) == 144
    assert resp["baseline"]["policy"] == "random" and resp["truth"]["field_p95_px"] > 0
    tools = {t["tool"] for r in ra for t in r.to_response()["decision"]["tools"]}
    assert tools >= {"sim_capture", "detect_board", "assess_frame", "calibrate", "uncertainty_map", "propose_next_views"}
    planned = [r.to_response() for r in ra if r.rule == "R13_NBV" and r.calibration]
    assert planned and len(planned[-1]["plan"]["candidates"]) == 5 and planned[-1]["plan"]["target"]["instruction"].startswith("Board")


def test_glare_reject_retries_tilted_away():
    st, src = _session("phone_crack_metrology", "human", seed=0)
    for _ in range(20):
        before = dict(st.last_target) if st.last_target else None
        r = run_step(st, src, th=TH, stopcal=UNCAL)
        if r.rule == "R3_GLARE":
            break
    assert r.rule == "R3_GLARE" and r.instruction.startswith("Reflection on the board — tilt ~15° about the")
    assert st.last_target["source"] == "glare_retry"
    assert rotation_angle_deg(before["rvec"], st.last_target["rvec"]) == pytest.approx(15.0, abs=1e-6)
    assert "retry tilted 15°" in r.trace


def test_drift_card_restart_and_continue():
    th = TH.replace(drift_factor=2.5)
    st, src = _session("autofocus_drift", "human", seed=0)
    _run_until(st, src, lambda s: s.pending_approval is not None, th=th)
    card = st.pending_approval
    assert card["kind"] == "drift_restart" and st.n_accepted_total > 10
    keep = _copy(st)
    r = apply_approval(st, card["id"], "approve", source=src, th=th, stopcal=UNCAL)
    assert r.status == "continue" and st.views == [] and st.attempts == 0 and st.last_target["source"] == "opening"
    assert st.attempt > 0 and st.n_accepted_total > 10  # random streams and the drifted camera carry on
    run_step(st, src, th=th, stopcal=UNCAL)
    c = apply_approval(keep, card["id"], "deny", source=src, th=th, stopcal=UNCAL)
    assert "DRIFT" in keep.flags and c.status in ("continue", "escalated")


def test_webcam_frame_and_candidate_override():
    sim = make_simulator("normal", "robot", camera_seed=2, seed=0, occluder_p=0.0, corruption_p=0.0)
    st = new_session(mode="webcam", actuator="human", seed=0, th=TH)
    for k in range(4):
        rv, tv = pose_for_view(sim.camera, (480 + 100 * k, 300 + 30 * k), 0.5, 20 + 5 * k, 90 * k)
        r = run_step(st, FrameSource(sim.render_exact(rv, tv)), th=TH, stopcal=UNCAL)
        assert r.accepted and r.truth is None and r.baseline is None
    assert len(st.views) == 4 and len(st.candidates) == 5
    rv, tv = pose_for_view(sim.camera, (640, 360), 0.5, 30, 45)
    r = run_step(st, FrameSource(sim.render_exact(rv, tv)), th=TH, stopcal=UNCAL, choose_candidate=2)
    assert any(ln.startswith(f"#{r.step} USER: candidate 2") for ln in st.trace)
    with pytest.raises(LoopError):
        run_step(st, FrameSource(np.zeros((480, 640), np.uint8)), th=TH, stopcal=UNCAL)
    with pytest.raises(LoopError):
        run_step(st, FrameSource(sim.render_exact(rv, tv)), th=TH, stopcal=UNCAL, choose_candidate=9)


# ---- review fixes: R9 reachability and cost, R6 after deny, approvals, sources, glare cap ----------

def _sagged_source(base, amp):
    """Sim source whose detections come from a sagged board (spec 6.3 warp, amplitude in metres)."""
    from nextview.agent.loop import Observation, SimSource
    from nextview.sim.warp import sag_object_points
    from nextview.types import Detection

    class Sagged(SimSource):
        def capture(self, rvec, tvec, **kw):
            o = super().capture(rvec, tvec, **kw)
            d = o.det
            if len(d.ids):
                cam = self.camera(kw["n_accepted"])
                p, _ = cam.project_checked(sag_object_points(amp)[np.asarray(d.ids).astype(int)], o.truth.rvec, o.truth.tvec)
                p = p + np.random.default_rng(kw["attempt"] + 99).normal(0, 0.15, p.shape)
                d = Detection(d.ids, p, d.obj, d.n_markers, d.ok)
            return Observation(o.image, d, o.moved_ids, o.truth, o.ms)

    return Sagged(base.sim)


def test_sagged_board_reaches_r9_and_ro_plans_whole_boards():
    st, base = _session("inspection_gauge_cam", "robot", seed=0)
    src = _sagged_source(base, 0.003)
    seen_seek = False
    for _ in range(14):
        if st.pending_approval is not None:
            break
        r = run_step(st, src, th=TH, stopcal=UNCAL)
        seen_seek |= st.seek_full_board
    card = st.pending_approval
    assert seen_seek and card is not None and card["kind"] == "use_ro" and card["rule"] == "R9_WARP"
    assert card["evidence"]["warp_score"] >= TH.warp_explained_var and card["evidence"]["cv_gain"] >= TH.warp_cv_gain
    assert st.r9["done"] and len(st.r9["fold_s"]) == 3 and "WARP" in st.flags
    a = apply_approval(st, card["id"], "approve", source=src, th=TH, stopcal=UNCAL)
    assert st.model["name"] == "ro" and a.calibration["model"] == "ro" and not st.seek_full_board
    nbv_info = next(t for t in a.tools if t["tool"] == "propose_next_views")
    assert nbv_info["full_board_only"] is True
    assert a.truth is not None and a.truth["field_p95_px"] > 0  # sim truth still reaches the agent through the wrapped source


def test_r9_fold_per_step_matches_ro_cv_gain_and_defers_without_time():
    import time

    import _synth as S
    from nextview.agent import loop as L
    from nextview.sim.lens import make_camera
    from nextview.tools.calibrate import calibrate, ro_cv_gain
    from nextview.tools._views import full_board_views

    cam = make_camera("normal", 0)
    views = S.camera_views(cam, np.random.default_rng(7), 20, noise=0.25, min_visible=0.9)
    full = full_board_views(views)
    assert len(full) > 12
    st = new_session(mode="webcam", actuator="robot", th=TH)
    st.views = list(views)
    calib = calibrate(views, "k5", cam.size)
    late = L._Ctx(st, None, TH, None, time.perf_counter(), time.perf_counter() + 1.0, L._Tools())
    assert L._r9_step(late, calib) is None and st.r9["fold"] == 0 and "deferred" in late.notes[0]
    ctx = L._Ctx(st, None, TH, None, time.perf_counter(), time.perf_counter() + 1e6, L._Tools())
    out = [L._r9_step(ctx, calib) for _ in range(3)]
    assert out[0] is None and out[1] is None and st.r9["done"] and len(st.r9["ids"]) == 12
    assert st.r9["ids"] == [int(v.id) for v in full[-12:]]
    ref = ro_cv_gain(full[-12:], cam.size, init=calib)
    np.testing.assert_allclose(out[2], ref, rtol=1e-9)


def test_drift_deny_is_not_asked_again():
    th = TH.replace(drift_factor=2.5)
    st, src = _session("autofocus_drift", "human", seed=0)
    _run_until(st, src, lambda s: s.pending_approval is not None, th=th)
    card = st.pending_approval
    assert card["kind"] == "drift_restart"
    apply_approval(st, card["id"], "deny", source=src, th=th, stopcal=UNCAL)
    _run_until(st, src, lambda s: s.pending_approval is not None, th=th)
    assert st.pending_approval["kind"] == "accept_with_warnings" and "DRIFT" in st.flags
    assert [a["kind"] for a in st.approvals] == ["drift_restart"]


def test_budget_card_without_a_calibration_offers_discard_only():
    th = TH.replace(max_attempts=2)
    st = new_session(mode="webcam", actuator="human", seed=0, th=th)
    blank = np.zeros((720, 1280), np.uint8)
    r1 = run_step(st, FrameSource(blank), th=th, stopcal=UNCAL)
    assert r1.rule == "R0_NO_BOARD" and r1.status == "continue"
    r2 = run_step(st, FrameSource(blank), th=th, stopcal=UNCAL)
    card = st.pending_approval
    assert r2.status == "escalated" and r2.rule == "R12_BUDGET" and r2.instruction == card["question"]
    assert r2.trace.startswith(f"#{r2.step} R12_BUDGET") and any(" R0_NO_BOARD:" in ln for ln in r2.trace_all)
    assert [o["decision"] for o in card["options"]] == ["deny"] and "no calibration to export" in card["question"]
    before = st.to_dict()
    with pytest.raises(ApprovalError):
        apply_approval(st, card["id"], "approve", th=th, stopcal=UNCAL)
    assert st.to_dict() == before  # validated before anything changed
    assert apply_approval(st, card["id"], "deny", th=th, stopcal=UNCAL).status == "discarded"


def test_frame_source_must_match_the_session_mode():
    st, src = _session("inspection_gauge_cam", "robot", seed=0)
    img = np.zeros((720, 1280), np.uint8)
    with pytest.raises(LoopError):
        run_step(st, FrameSource(img), th=TH, stopcal=UNCAL)
    with pytest.raises(LoopError):
        run_step(st, th=TH, stopcal=UNCAL, image=img)
    web = new_session(mode="webcam", actuator="human", seed=0, th=TH)
    with pytest.raises(LoopError):
        run_step(web, src, th=TH, stopcal=UNCAL)
    assert st.step == 0 and web.step == 0


def test_glare_retry_stays_under_the_tilt_cap():
    from nextview.agent.loop import _glare_turn
    from nextview.agent.policy import tilted_target
    from nextview.geometry import rvec_from_tilt, tilt_of

    tv = [-0.135, -0.09, 0.6]
    for tilt0, az in ((50.0, 0.0), (55.0, 0.0), (59.0, 0.0), (40.0, 0.0), (55.0, 90.0)):
        target = {"rvec": rvec_from_tilt(tilt0, 0.0).tolist(), "tvec": tv}
        turn, az2, flipped = _glare_turn(target, az, TH.max_tilt_deg)
        rv, _ = tilted_target(target["rvec"], target["tvec"], turn, az2)
        assert tilt_of(rv) < TH.max_tilt_deg, (tilt0, az, turn, az2)
        assert turn > 0
    assert _glare_turn({"rvec": rvec_from_tilt(40.0, 0.0).tolist(), "tvec": tv}, 0.0, TH.max_tilt_deg) == (15.0, 0.0, False)
