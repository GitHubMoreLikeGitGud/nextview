"""Planner policy (spec 2.3): deterministic, priority-ordered rules R0-R13.

Every rule that fires produces a ``Finding`` whose trace line reads
``#<step> <RULE>: <measurement> → <action>``. Exactly one primary rule decides each step's
outcome (``loop.run_step`` picks it: a reject, then an approval / stop / escalation, then an R7
drop, then R4, then R5, else R13); every other rule that fired is logged as a secondary line.

* ``gate`` (R0-R3) rejects a frame before it is accepted.
* ``perception`` (R4, R5) accepts and flags.
* ``post_accept`` (R6-R10) runs after the refit and returns the first rule that needs action;
  the loop executes an R7 drop, refits, and calls it again without R6/R7.
* ``decide`` (R11-R13) chooses stop, escalate or plan.

Rule functions are pure: they read measurements and return findings. The loop applies them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np

from ..geometry import as_rotation, board_centre_px
from ..tools.detect import corner_span
from ..tools.residuals import drift_flag, out_of_scope
from ..types import CalibResult, Detection, Diagnostics, PoseResult, Quality
from .approvals import ApprovalKind
from .instructions import describe_reject

__all__ = ["Rule", "RULES", "Finding", "gate", "perception", "post_accept", "decide", "underfit_evidence",
           "warp_seek", "OPEN_FLAGS", "PRIMARY_ORDER", "primary", "tilted_target", "rotation_angle_deg",
           "GLARE_TILT_DEG"]

GLARE_TILT_DEG = 15.0
# R9 cost control (the RO cross-validation is the most expensive optional step):
WARP_SEEK_MIN_FULL_VIEWS = 3  # warp score trusted from 3 full-board views (pure noise gives about 1 / (2 V))
RO_CV_MAX_FULL_VIEWS = 12  # the RO cross-validation uses the last 12 full-board views
R9_RECHECK_FULL_VIEWS = 3  # a completed negative R9 check is repeated only after 3 more full-board views


@dataclass(frozen=True)
class Rule:
    id: str
    stage: str  # gate | perception | post_accept | decide
    trigger: str
    action: str
    approval: str | None = None


RULES: dict[str, Rule] = {r.id: r for r in (
    Rule("R0_NO_BOARD", "gate", "n_corners < min_corners, corners span < 2 rows or < 2 cols, or < 8 measurable edges", "reject; next candidate"),
    Rule("R1_SMALL", "gate", "board_area_frac < min_board_area_frac", "reject; next candidate"),
    Rule("R2_BLUR", "gate", "edge_rise_px > max_edge_rise_px", "reject; retry the same target"),
    Rule("R3_GLARE", "gate", "saturation_frac > max_saturation_frac inside the board hull", "reject; retry tilted ~15° away from the hotspot"),
    Rule("R4_AMBIGUOUS", "perception", "ippe_ratio > ippe_ambiguity_ratio", "accept, flag AMBIGUOUS, next target tilt >= 25 deg"),
    Rule("R5_OFF_TARGET", "perception", "human mode: realised pose off target by > off_target_deg or > off_target_frac x W", "accept, log, re-plan"),
    Rule("R6_DRIFT", "post_accept", "last 3 accepted frames have prequential RMS > drift_factor x the session median "
         "(not raised again while DRIFT is set)", "pause", "drift_restart"),
    Rule("R7_OUTLIER", "post_accept", "per-view RMS > median + 3 x 1.4826 x MAD and > 0.5 px", "drop view(s), refit"),
    Rule("R8_UNDERFIT", "post_accept", ">= 12 views, radial-trend p < underfit_p, 3-fold CV RMS(rational) < 0.90 x RMS(k5)", "propose rational model", "switch_model"),
    Rule("R9_WARP", "post_accept", ">= 6 full-board views, warp score >= 0.30, 3-fold CV gain of RO >= 10 % (last 12 "
         "full-board views, one fold per step; re-checked after 3 more)", "propose release-object model", "use_ro"),
    Rule("R10_OUT_OF_SCOPE", "post_accept", "model rational and estimated HFOV > out_of_scope_hfov_deg", "escalate and stop", "out_of_scope"),
    Rule("R11_STOP", "decide", "views >= 6, P_exceed < 0.10, last 2 prequential RMS <= 1.5 x in-sample RMS, no open flags", "stop", "export"),
    Rule("R12_BUDGET", "decide", "accepted views = budget or attempts = max_attempts", "escalate", "accept_with_warnings"),
    Rule("R13_NBV", "decide", "otherwise", "plan next view"),
)}

OPEN_FLAGS = ("UNDERFIT", "WARP", "DRIFT", "AMBIGUOUS", "OUT_OF_SCOPE")
# Primary-rule priority within a step (lower index wins).
PRIMARY_ORDER = ("R0_NO_BOARD", "R1_SMALL", "R2_BLUR", "R3_GLARE", "R6_DRIFT", "R8_UNDERFIT", "R9_WARP",
                 "R10_OUT_OF_SCOPE", "R11_STOP", "R12_BUDGET", "R7_OUTLIER", "R4_AMBIGUOUS", "R5_OFF_TARGET", "R13_NBV")


@dataclass
class Finding:
    rule: str
    measurement: str
    action: str
    metric: str | None = None
    value: float | None = None
    instruction: str | None = None
    approval: ApprovalKind | None = None
    evidence: dict = field(default_factory=dict)

    def trace(self, step: int) -> str:
        return f"#{step} {self.rule}: {self.measurement} → {self.action}"


def primary(findings: Sequence[Finding]) -> Finding | None:
    """The finding that decides the step (``PRIMARY_ORDER``)."""
    if not findings:
        return None
    return min(findings, key=lambda f: PRIMARY_ORDER.index(f.rule))


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ---- R0-R3 ---------------------------------------------------------------------------------

def gate(det: Detection, q: Quality, th) -> Finding | None:
    """Perception gate on a new frame: the first of R0-R3 that fires, or None (frame passes)."""
    n = int(len(det.ids))
    rows, cols = corner_span(det.ids)
    span = int(getattr(th, "min_corner_span", 2))
    if not det.ok or n < th.min_corners or rows < span or cols < span:
        closer = n > 0 or det.n_markers > 0
        return Finding("R0_NO_BOARD", f"{n} corners over {rows}×{cols} rows×cols (need {th.min_corners}, ≥{span}×{span})",
                       "reject", "n_corners", float(n), describe_reject("R0_NO_BOARD", closer=closer))
    if not (q.board_area_frac >= th.min_board_area_frac):
        return Finding("R1_SMALL", f"board area {100 * q.board_area_frac:.1f}% of frame (min {100 * th.min_board_area_frac:.0f}%)",
                       "reject", "board_area_frac", float(q.board_area_frac), describe_reject("R1_SMALL"))
    if not _finite(q.edge_rise_px):
        return Finding("R0_NO_BOARD", f"only {q.n_edges} measurable square edges (need 8)", "reject", "n_edges",
                       float(q.n_edges), describe_reject("R0_NO_BOARD", closer=True))
    if q.edge_rise_px > th.max_edge_rise_px:
        motion = _finite(q.anisotropy) and q.anisotropy >= th.motion_anisotropy and q.blur_dir_deg is not None
        meas = f"edge rise {q.edge_rise_px:.2f} px > {th.max_edge_rise_px} px, anisotropy {q.anisotropy:.2f}"
        if motion:
            meas += f", blur along {q.blur_dir_deg:.0f}°"
        return Finding("R2_BLUR", meas, "reject", "edge_rise_px", float(q.edge_rise_px),
                       describe_reject("R2_BLUR", rise=q.edge_rise_px, anisotropy=q.anisotropy, blur_dir=q.blur_dir_deg,
                                       motion_anisotropy=th.motion_anisotropy))
    if q.saturation_frac > th.max_saturation_frac:
        centre = np.asarray(det.corners, float).mean(axis=0)
        off = (0.0, 0.0) if q.glare_centroid is None else (q.glare_centroid[0] - centre[0], q.glare_centroid[1] - centre[1])
        return Finding("R3_GLARE", f"{100 * q.saturation_frac:.1f}% of the board clipped (max {100 * th.max_saturation_frac:.0f}%), "
                       f"hotspot offset ({off[0]:+.0f}, {off[1]:+.0f}) px", "reject, retry tilted away from the hotspot",
                       "saturation_frac", float(q.saturation_frac), describe_reject("R3_GLARE", glare_offset=off),
                       evidence={"glare_offset": [float(off[0]), float(off[1])]})
    return None


# ---- R4, R5 --------------------------------------------------------------------------------

def tilted_target(rvec, tvec, tilt_deg: float, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """The target pose turned by ``tilt_deg`` about the board centre, sending the ``azimuth_deg``
    edge away from the camera (same convention as ``geometry.rotation_from_tilt``)."""
    import cv2

    from ..geometry import BOARD_CENTRE_M, rotation_from_tilt

    R = as_rotation(rvec)
    c = R @ BOARD_CENTRE_M + np.asarray(tvec, float).ravel()
    R2 = rotation_from_tilt(tilt_deg, azimuth_deg) @ R
    r2, _ = cv2.Rodrigues(R2)
    return r2.ravel(), c - R2 @ BOARD_CENTRE_M


def rotation_angle_deg(rvec_a, rvec_b) -> float:
    Ra, Rb = as_rotation(rvec_a), as_rotation(rvec_b)
    c = (float(np.trace(Ra @ Rb.T)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def perception(pose: PoseResult, th, *, actuator: str, target: dict | None, K, dist, size) -> list[Finding]:
    """R4 (ambiguous IPPE pose) and R5 (human mode, realised pose off target)."""
    out = []
    if pose.ippe_ratio > th.ippe_ambiguity_ratio:
        out.append(Finding("R4_AMBIGUOUS", f"IPPE ratio {pose.ippe_ratio:.2f} > {th.ippe_ambiguity_ratio} at tilt {pose.tilt_deg:.0f}°",
                           f"accept, next target tilt ≥ {getattr(th, 'ambiguous_min_tilt_deg', 25):.0f}°", "ippe_ratio",
                           float(pose.ippe_ratio), describe_reject("R4_AMBIGUOUS", ratio=pose.ippe_ratio)))
    if actuator == "human" and target is not None:
        trv, ttv = np.asarray(target["rvec"], float), np.asarray(target["tvec"], float)
        ang = rotation_angle_deg(pose.rvec, trv)
        c_real = board_centre_px(pose.rvec, pose.tvec, K, dist)
        c_tgt = board_centre_px(trv, ttv, K, dist)
        frac = float(np.hypot(*(c_real - c_tgt))) / float(size[0]) if np.all(np.isfinite(c_real)) and np.all(np.isfinite(c_tgt)) else 0.0
        if ang > th.off_target_deg or frac > th.off_target_frac:
            out.append(Finding("R5_OFF_TARGET", f"realised pose {ang:.0f}° and {100 * frac:.0f}% of the width from target",
                               "accept, re-plan from the new view", "off_target_deg", float(ang),
                               describe_reject("R5_OFF_TARGET", deg=ang, frac=frac)))
    return out


# ---- R6-R10 --------------------------------------------------------------------------------

def _field_vectors(diag: Diagnostics) -> dict | None:
    f = diag.residual_field
    if f is None:
        return None
    rows, cols = f.shape[:2]
    vec = [[None, None] if not np.all(np.isfinite(v)) else [round(float(v[0]), 3), round(float(v[1]), 3)] for v in f.reshape(-1, 2)]
    return {"grid": [cols, rows], "vectors": vec}


def underfit_evidence(model: dict, n_views: int, radial_p: float, th) -> bool:
    """The cheap R8 preconditions: model k5 and not pinned, >= model_switch_min_views views, p < underfit_p.
    The 3-fold CV pair is computed only when these hold."""
    return (model.get("name") == "k5" and not model.get("pinned") and n_views >= th.model_switch_min_views
            and _finite(radial_p) and radial_p < th.underfit_p)


def _warp_suspected(model: dict, n_full_views: int, warp_score: float, th) -> bool:
    return (model.get("name") == "k5" and not model.get("pinned") and n_full_views >= WARP_SEEK_MIN_FULL_VIEWS
            and _finite(warp_score) and warp_score >= th.warp_explained_var)


def warp_seek(model: dict, n_full_views: int, warp_score: float, th, r9_running: bool = False) -> bool:
    """R9 pre-check: k5 model not pinned, warp score >= ``warp_explained_var`` on 3 to 5 full-board
    views, or an R9 cross-validation still running. The NBV then plans whole-board views only, so
    the 6 full-board views R9 needs arrive and the views captured during the check add to them."""
    return (model.get("name") == "k5" and not model.get("pinned")
            and (WARP_SEEK_MIN_FULL_VIEWS <= n_full_views < th.warp_min_full_views or r9_running)
            and _finite(warp_score) and warp_score >= th.warp_explained_var)


def post_accept(*, model: dict, n_views: int, n_full_views: int, calib: CalibResult, diag: Diagnostics,
                preq_history: Sequence[float], th, size, cv_pair: Callable[[], tuple[float, float]],
                ro_gain: Callable[[], tuple[float, float, float] | None], skip: Sequence[str] = (),
                have_time: Callable[[], bool] = lambda: True, notes: list | None = None, flags: Sequence[str] = (),
                r9_last: dict | None = None, full_view_ids: Sequence[int] = ()) -> Finding | None:
    """R6-R10 in priority order; returns the first rule that needs action, else None.

    ``cv_pair() -> (cv_k5, cv_rational)`` and ``ro_gain() -> (gain, cv_ro, cv_k5)`` are evaluated
    lazily, only when the cheap preconditions hold; ``cv_pair`` only while ``have_time()``. A
    deferred or negative evaluation is appended to ``notes``. ``ro_gain`` may return None: the
    loop spreads the RO cross-validation over steps (one fold per step, each fold only when the
    step has time for it), so None means "not complete yet".

    R6 is skipped while the DRIFT flag is set (the operator already chose to continue). While a
    warp is suspected (k5 not pinned, warp score >= ``warp_explained_var`` on >= 3 full-board
    views), R7 does not drop full-board views (``full_view_ids``): under a warped board they are
    the worst-fitting views because of the warp, and they are the evidence R9 needs. After a
    completed R9 check (``r9_last`` with ``done``, cached in the session state) R9 is not
    evaluated again until the full-board count reaches ``r9_last["n_full"] + 3``.
    """
    notes = [] if notes is None else notes
    if "R6_DRIFT" not in skip and "DRIFT" not in flags:
        hist = [SimpleNamespace(prequential_rms=x) for x in preq_history]
        if drift_flag(hist, th.drift_factor, int(th.drift_consecutive)):
            vals = [float(x) for x in preq_history if x is not None]
            k = int(th.drift_consecutive)
            ref = float(np.median(vals[:-k]))
            return Finding("R6_DRIFT", f"last {k} prequential RMS {', '.join(f'{v:.2f}' for v in vals[-k:])} px > "
                           f"{th.drift_factor:g} × median {ref:.2f} px", "pause, ask to restart", "prequential_rms",
                           float(vals[-1]), approval=ApprovalKind.DRIFT_RESTART,
                           evidence={"prequential_rms_series": [round(v, 4) for v in vals], "median_px": round(ref, 4),
                                     "factor": th.drift_factor})
    outliers = list(diag.outliers)
    if outliers and full_view_ids and _warp_suspected(model, len(full_view_ids), diag.warp_score, th):
        full = {int(i) for i in full_view_ids}
        held = [i for i in outliers if int(i) in full]
        if held:
            outliers = [i for i in outliers if int(i) not in full]
            if "R7_OUTLIER" not in skip:
                notes.append(f"R7 holds full-board view(s) {held}: warp score {diag.warp_score:.2f}, left for the R9 check")
    if "R7_OUTLIER" not in skip and outliers:
        pv = np.asarray(calib.per_view, float)
        return Finding("R7_OUTLIER", f"view(s) {outliers} per-view RMS above median {np.median(pv):.2f} px + "
                       f"{th.outlier_mad_k:g}·1.4826·MAD and > {th.outlier_min_abs_px} px", "drop view(s), refit",
                       "per_view_rms", float(pv.max()), evidence={"view_ids": outliers})
    if underfit_evidence(model, n_views, diag.radial_p, th):
        if not have_time():
            notes.append(f"R8 deferred: radial p {diag.radial_p:.2g} (CV pair postponed by the step deadline)")
        else:
            cv_k5, cv_rat = cv_pair()
            if _finite(cv_k5) and _finite(cv_rat) and cv_rat < (1.0 - th.model_switch_cv_gain) * cv_k5:
                return Finding("R8_UNDERFIT", f"radial-trend p {diag.radial_p:.2g} < {th.underfit_p}, held-out RMS k5 "
                               f"{cv_k5:.3f} vs rational {cv_rat:.3f} px ({100 * (1 - cv_rat / cv_k5):.0f}% better)",
                               "propose CALIB_RATIONAL_MODEL", "radial_trend_p", float(diag.radial_p),
                               approval=ApprovalKind.SWITCH_MODEL,
                               evidence={"radial_trend_p": float(diag.radial_p), "cv_rms": {"k5": cv_k5, "rational": cv_rat},
                                         "residual_field": _field_vectors(diag)})
            notes.append(f"R8 not fired: radial p {diag.radial_p:.2g} but held-out RMS k5 {cv_k5:.3f} vs rational {cv_rat:.3f} px")
    if (model.get("name") == "k5" and not model.get("pinned") and n_full_views >= th.warp_min_full_views
            and _finite(diag.warp_score) and diag.warp_score >= th.warp_explained_var):
        recheck = int(getattr(th, "warp_recheck_full_views", R9_RECHECK_FULL_VIEWS))
        done = r9_last if r9_last and r9_last.get("done") else None
        if done is not None and n_full_views < int(done["n_full"]) + recheck:
            g = done.get("gain")
            gtxt = "n/a" if g is None else f"{float(g):.2f}"
            notes.append(f"R9 not re-checked: RO CV gain {gtxt} at {done['n_full']} full-board views "
                         f"(next check at {int(done['n_full']) + recheck})")
        else:
            res = ro_gain()
            if res is not None:
                gain, cv_ro, cv_k5 = res
                n_used = len(r9_last["ids"]) if r9_last and r9_last.get("ids") else min(
                    n_full_views, int(getattr(th, "ro_cv_max_full_views", RO_CV_MAX_FULL_VIEWS)))
                if _finite(gain) and gain >= th.warp_cv_gain:
                    return Finding("R9_WARP", f"warp score {diag.warp_score:.2f} ≥ {th.warp_explained_var}, RO held-out RMS "
                                   f"{cv_ro:.3f} vs k5 {cv_k5:.3f} px over {n_used} full-board views "
                                   f"({100 * gain:.0f}% better)", "propose calibrateCameraROExtended",
                                   "warp_score", float(diag.warp_score), approval=ApprovalKind.USE_RO,
                                   evidence={"warp_score": float(diag.warp_score), "cv_gain": float(gain),
                                             "cv_rms": {"k5": cv_k5, "ro": cv_ro}, "cv_full_views": n_used,
                                             "residual_field": _field_vectors(diag)})
                notes.append(f"R9 not fired: warp score {diag.warp_score:.2f} but RO CV gain "
                             f"{gain if _finite(gain) else float('nan'):.2f}")
    fires, hfov = out_of_scope(calib, size, th)
    if fires:
        return Finding("R10_OUT_OF_SCOPE", f"model rational, estimated HFOV {hfov:.0f}° > {th.out_of_scope_hfov_deg}°"
                       + (f", radial p {diag.radial_p:.2g}" if _finite(diag.radial_p) else ""), "escalate and stop",
                       "hfov_deg", float(hfov), approval=ApprovalKind.OUT_OF_SCOPE,
                       evidence={"hfov_deg": float(hfov), "hfov_limit_deg": float(th.out_of_scope_hfov_deg),
                                 "radial_trend_p": diag.radial_p if _finite(diag.radial_p) else None, "model": calib.model})
    return None


# ---- R11-R13 -------------------------------------------------------------------------------

def decide(*, n_views: int, p_exceed: float, u95: float | None, prequential_last: Sequence[float], rms: float | None,
           flags: Sequence[str], attempts: int, budget: int, max_attempts: int, th, allow_stop: bool = True) -> Finding:
    """R11 STOP, else R12 BUDGET, else R13 NBV."""
    open_flags = [f for f in flags if f in OPEN_FLAGS]
    pre = [float(x) for x in prequential_last if x is not None and _finite(x)][-2:]
    stable = len(pre) == 2 and rms is not None and all(x <= th.prequential_ratio * rms for x in pre)
    pe = f"{p_exceed:.2f}" if _finite(p_exceed) else "n/a (uncalibrated)"
    if (allow_stop and n_views >= th.min_views_stop and _finite(p_exceed) and p_exceed < th.stop_p_exceed and stable
            and not open_flags):
        return Finding("R11_STOP", f"P(field p95 > {th.tau_px} px) = {pe} < {th.stop_p_exceed}, U95 {u95:.2f} px, "
                       f"prequential {', '.join(f'{x:.2f}' for x in pre)} ≤ {th.prequential_ratio}× RMS {rms:.2f} px",
                       "stop, ask to export", "p_exceed", float(p_exceed), approval=ApprovalKind.EXPORT)
    if n_views >= budget or attempts >= max_attempts:
        why = f"{n_views}/{budget} views" if n_views >= budget else f"{attempts}/{max_attempts} attempts"
        u = f"{u95:.2f} px" if u95 is not None and _finite(u95) else "n/a"
        return Finding("R12_BUDGET", f"{why}, U95 {u}, P(>τ) {pe}" + (f", open flags {open_flags}" if open_flags else ""),
                       "escalate", "views" if n_views >= budget else "attempts",
                       float(n_views if n_views >= budget else attempts), approval=ApprovalKind.ACCEPT_WITH_WARNINGS)
    return Finding("R13_NBV", "", "plan next view")
