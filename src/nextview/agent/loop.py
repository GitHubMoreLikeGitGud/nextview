"""One perceive → decide → act cycle (spec 2.8) and the approval gates (spec 2.7).

``run_step(state, source)``: refit from the views, capture at the current target, detect,
assess, gate (R0-R3), estimate the pose (R4, R5), accept, refit, diagnose (R6-R10), map the
uncertainty, then stop (R11), escalate (R12) or plan the next view (R13). ``apply_approval``
resolves the pending card and re-plans. Both mutate ``state`` and return a ``StepResult``.

Frames come from a source: ``SimSource`` (simulated capture, detection through
``SimFrame.observe`` so post-detection corruption reaches the agent) or ``FrameSource`` (one
uploaded webcam frame). In simulation a shadow random-policy track (``state.baseline``) advances
one frame per step for the race chart, on the same random streams (common random numbers).
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from ..config import load_stop_calibration, load_thresholds
from ..geometry import ghost_quad, tilt_of
from ..tools.calibrate import calibrate, cv_heldout_rms, heldout_view_error, prequential_rms
from ..tools.detect import detect_board
from ..tools.export import export_calibration, intrinsic_std
from ..tools.nbv import focal_from_detection, k_initial, opening_target, propose_next_views
from ..tools.pose import estimate_pose
from ..tools.quality import assess_frame
from ..tools.residuals import residual_analysis
from ..tools.uncertainty import uncertainty_map
from ..tools._views import board_points, full_board_views
from ..types import CalibResult, View
from .approvals import ApprovalError, ApprovalKind, check_decision, make_card
from .instructions import REJECT_TEMPLATES, describe_target, edge_away, glare_tilt, position_word, short_desc
from .policy import (GLARE_TILT_DEG, RO_CV_MAX_FULL_VIEWS, Finding, decide, gate, perception, post_accept, primary,
                     tilted_target, warp_seek)
from .state import SessionState, jsonable, round_corners

__all__ = ["LoopError", "Observation", "SimSource", "FrameSource", "make_source", "StepResult", "run_step",
           "apply_approval", "awaiting"]

BASELINE_STREAM = 5
NBV_STREAM = 3
MAX_FOCAL_OBS = 10
BASELINE_MIN_CORNERS = 6


class LoopError(ValueError):
    """A request the loop cannot serve (ended session, bad candidate index, wrong frame size)."""

    def __init__(self, message: str, code: str = "BAD_REQUEST"):
        super().__init__(message)
        self.code = code


# ---- frame sources -------------------------------------------------------------------------

@dataclass(eq=False)
class Observation:
    image: np.ndarray
    det: object  # Detection
    moved_ids: np.ndarray
    truth: object | None = None  # sim FrameTruth
    ms: dict = field(default_factory=dict)  # tool -> milliseconds


class SimSource:
    """Simulated capture for a session (one simulator per preset, camera and seed)."""

    kind = "sim"

    def __init__(self, simulator):
        self.sim = simulator

    def capture(self, rvec, tvec, *, attempt: int, n_accepted: int, prev_pose: dict | None) -> Observation:
        pr = None if prev_pose is None else np.asarray(prev_pose["rvec"], float)
        pt = None if prev_pose is None else np.asarray(prev_pose["tvec"], float)
        t = time.perf_counter()
        fr = self.sim.capture(np.asarray(rvec, float), np.asarray(tvec, float), attempt=attempt, n_accepted=n_accepted,
                              prev_rvec=pr, prev_tvec=pt)
        t1 = time.perf_counter()
        det, moved = fr.observe()
        ms = {"sim_capture": 1000 * (t1 - t), "detect_board": 1000 * (time.perf_counter() - t1)}
        return Observation(fr.image, det, np.asarray(moved), fr.truth, ms)

    def camera(self, n_accepted: int):
        return self.sim.camera_at(n_accepted)


class FrameSource:
    """One uploaded frame (webcam mode): detected here, never stored."""

    kind = "frame"

    def __init__(self, image: np.ndarray):
        self.image = np.asarray(image)

    def capture(self, rvec, tvec, **_) -> Observation:
        t = time.perf_counter()
        det = detect_board(self.image)
        return Observation(self.image, det, np.zeros(0, int), None, {"detect_board": 1000 * (time.perf_counter() - t)})


@lru_cache(maxsize=4)
def _simulator(preset: str, camera_seed: int, seed: int, actuator: str):
    from ..sim.scenarios import scenario_from_preset

    return scenario_from_preset(preset, camera_seed=camera_seed, seed=seed, actuator=actuator).simulator


def make_source(state: SessionState, image: np.ndarray | None = None):
    """``SimSource`` for a sim session (simulator cached per process), ``FrameSource`` for a frame."""
    if state.mode == "sim":
        if image is not None:
            raise LoopError("a sim session takes no uploaded frame")
        if state.preset is None:
            raise LoopError("sim session has no preset")
        return SimSource(_simulator(state.preset, state.camera_seed, state.seed, state.actuator))
    if image is None:
        raise LoopError("webcam step needs a frame")
    return FrameSource(image)


# ---- result ----------------------------------------------------------------------------------

@dataclass(eq=False)
class StepResult:
    status: str
    step: int
    rule: str | None
    accepted: bool
    trace: str
    trace_all: list
    instruction: str | None
    notes: list
    tools: list
    perception: dict | None
    calibration: dict | None
    uncertainty: dict | None
    coverage: dict | None
    plan: dict | None
    approval: dict | None
    truth: dict | None
    baseline: dict | None
    latency_ms: float
    image: np.ndarray | None = None
    export_files: dict | None = None
    export_meta: dict | None = None

    def to_response(self) -> dict:
        """Spec 7.3 response body without ``state`` and ``frame`` (the handler adds those)."""
        out = {
            "status": self.status,
            "step": self.step,
            "perception": self.perception,
            "decision": {"rule": self.rule, "accepted": self.accepted, "trace": self.trace, "trace_all": self.trace_all,
                         "instruction": self.instruction, "notes": self.notes, "tools": self.tools},
            "calibration": self.calibration,
            "uncertainty": self.uncertainty,
            "coverage": self.coverage,
            "plan": self.plan,
            "approval": self.approval,
            "latency_ms": {"total": round(self.latency_ms, 1)},
        }
        if self.truth is not None:
            out["truth"] = self.truth
        if self.baseline is not None:
            out["baseline"] = self.baseline
        return jsonable(out)

    def log_fields(self) -> dict:
        """Fields for the one-line JSON step log (no corners, no image)."""
        return jsonable({"step": self.step, "status": self.status, "rule": self.rule, "accepted": self.accepted,
                         "views": None if self.calibration is None else self.calibration.get("views"),
                         "U95_px": None if self.uncertainty is None else self.uncertainty.get("U95_px"),
                         "latency_ms": round(self.latency_ms, 1),
                         "tools": [{"tool": t["tool"], "ms": t["ms"]} for t in self.tools]})


class _Tools:
    def __init__(self):
        self.items: list[dict] = []

    @contextmanager
    def time(self, tool: str, **extra):
        t = time.perf_counter()
        info = dict(extra)
        try:
            yield info
        finally:
            self.items.append({"tool": tool, "ms": round(1000 * (time.perf_counter() - t), 1), **info})


# ---- helpers ---------------------------------------------------------------------------------

def _hint(state: SessionState, model: str):
    """Warm start for ``calibrate`` from the (HMAC-covered) previous fit. A switch to RO starts from
    the k5 fit as a warm start (RO only adds the board points); a switch to rational gets the k5
    fit as one more start of the multi-start search."""
    h = state.calib
    if not h:
        return None
    K, d = np.asarray(h["K"], float), np.asarray(h["dist"], float)
    if h.get("model") != model and not (model == "ro" and h.get("model") == "k5"):
        return (K, d)
    return CalibResult(model=model, K=K, dist=d, rms=float(h["rms"]), per_view=np.zeros(0), rvecs=[], tvecs=[],
                       std_intr=np.zeros(0), sigma2=math.nan, F=np.zeros(0), cov=None)


def _store_calib(state: SessionState, calib: CalibResult | None) -> None:
    state.calib = None if calib is None else {"model": calib.model, "K": calib.K.tolist(), "dist": calib.dist.tolist(),
                                              "rms": float(calib.rms)}


def _fit(state: SessionState, views, init=None) -> CalibResult | None:
    if len(views) < 3:
        return None
    model = state.model["name"]
    if model == "ro" and len(full_board_views(views)) < 3:
        model = "k5"  # RO needs 3 full-board views; fall back until there are enough
    return calibrate(views, model, state.image_size, init=init if init is not None else _hint(state, model))


def _focal_hint(obs: list) -> float | None:
    return float(np.median(obs)) if obs else None


def _target(rvec, tvec, K, dist, size, sim: bool, source: str, **extra) -> dict:
    rvec, tvec = np.asarray(rvec, float).ravel(), np.asarray(tvec, float).ravel()
    q = ghost_quad(rvec, tvec, K, dist)
    return {"rvec": [round(float(x), 6) for x in rvec], "tvec": [round(float(x), 6) for x in tvec], "source": source,
            "ghost_quad": [[round(float(a), 1), round(float(b), 1)] for a, b in q],
            "instruction": describe_target(rvec, tvec, K, dist, size, sim=sim),
            "desc": short_desc(rvec, tvec, K, dist, size), **extra}


def _opening(state: SessionState, k: int, focal_obs: list | None = None) -> dict:
    f = _focal_hint(state.focal_obs if focal_obs is None else focal_obs)
    rvec, tvec = opening_target(k, state.camera_seed, state.seed, state.image_size, f)
    K = k_initial(state.image_size, f)
    return _target(rvec, tvec, K, np.zeros(5), state.image_size, state.mode == "sim", "opening", k=int(k),
                   focal=round(float(K[0, 0]), 3), rank=None, pred_gain=None, pred_U95_after=None)


def _cand_dict(c, K, dist, size, sim) -> dict:
    return _target(c.rvec, c.tvec, K, dist, size, sim, "nbv", rank=c.rank, pred_gain=round(c.gain, 5),
                   pred_U95_after=round(c.pred_U95_after, 5), effort=round(c.effort, 4), score=round(c.score, 5))


def _coverage(views, size, grid) -> dict:
    cols, rows = grid
    W, H = size
    counts = np.zeros(rows * cols, int)
    for v in views:
        c = np.asarray(v.corners, float).reshape(-1, 2)
        ix = np.clip((c[:, 0] / W * cols).astype(int), 0, cols - 1)
        iy = np.clip((c[:, 1] / H * rows).astype(int), 0, rows - 1)
        np.add.at(counts, iy * cols + ix, 1)
    return {"grid": [cols, rows], "counts": counts.tolist()}


def _calib_dict(calib: CalibResult | None, views, preq) -> dict | None:
    if calib is None:
        return None
    return {"model": calib.model, "K": np.round(calib.K, 4).tolist(), "dist": np.round(calib.dist, 7).tolist(),
            "rms_px": round(float(calib.rms), 4), "views": len(views),
            "prequential_rms_px": None if preq is None else round(float(preq), 4),
            "std_intrinsics": np.round(intrinsic_std(calib), 6).tolist()}


def _unc_dict(unc) -> dict | None:
    if unc is None:
        return None
    return {"grid": list(unc.grid), "sigma_px": np.round(np.asarray(unc.sigma_px, float), 3).ravel().tolist(),
            "U95_px": round(float(unc.U95), 4), "p_exceed": None if not math.isfinite(unc.p_exceed) else round(float(unc.p_exceed), 4)}


def _truth(source, n_accepted: int, calib: CalibResult | None) -> dict | None:
    if not isinstance(source, SimSource) or calib is None:
        return None
    from ..sim.truth import field_error, fx_rel_err, planar_mm_error

    cam = source.camera(n_accepted)
    fe = field_error(cam, calib.K, calib.dist)
    try:
        mm = planar_mm_error(cam, calib.K, calib.dist, 1.0).median
    except RuntimeError:
        mm = math.nan
    return {"field_p95_px": round(fe.p95, 4), "field_mean_px": round(fe.mean, 4),
            "fx_rel_err_pct": round(fx_rel_err(cam, calib.K), 4), "mm_at_1m_edge": None if not math.isfinite(mm) else round(mm, 4)}


def awaiting(state: SessionState) -> StepResult:
    """Response for a step requested while an approval is pending (nothing is captured)."""
    card = state.pending_approval
    line = f"#{state.step} WAIT: approval {card['id']} ({card['kind']}) is pending → no capture"
    return StepResult(status=state.status, step=state.step, rule=card.get("rule"), accepted=False, trace=line, trace_all=[line],
                      instruction=card["question"], notes=[], tools=[], perception=None, calibration=None, uncertainty=None,
                      coverage=None, plan=None, approval=card, truth=None, baseline=None, latency_ms=0.0)


# ---- the shared decide/act part --------------------------------------------------------------

@dataclass
class _Ctx:
    state: SessionState
    source: object
    th: object
    stopcal: object
    t0: float
    deadline: float
    tools: _Tools
    findings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    calib: CalibResult | None = None
    unc: object = None

    def have_time(self, frac: float = 0.5) -> bool:
        return time.perf_counter() < self.t0 + frac * (self.deadline - self.t0)

    def time_left(self) -> float:
        return self.deadline - time.perf_counter()


def _raise_approval(ctx: _Ctx, f: Finding, extra: dict | None = None) -> str:
    st = ctx.state
    aid = f"ap_{st.next_approval_id}"
    st.next_approval_id += 1
    ev = dict(f.evidence)
    ev.update(extra or {})
    st.pending_approval = make_card(f.approval, aid, jsonable(ev), f.rule, st.step)
    if f.rule == "R8_UNDERFIT":
        st.set_flag("UNDERFIT")
    elif f.rule == "R9_WARP":
        st.set_flag("WARP")
    elif f.rule == "R10_OUT_OF_SCOPE":
        st.set_flag("OUT_OF_SCOPE")
    if f.rule == "R11_STOP":
        return "stopped"
    if f.rule in ("R10_OUT_OF_SCOPE", "R12_BUDGET"):
        return "escalated"
    return "awaiting_approval"


def _drop_views(ctx: _Ctx, ids: list) -> bool:
    st = ctx.state
    keep = [v for v in st.views if v.id not in set(ids)]
    if len(keep) < 3:
        ctx.notes.append(f"R7 not applied: dropping {ids} would leave {len(keep)} views")
        return False
    pv = {v.id: e for v, e in zip(st.views, ctx.calib.per_view)} if ctx.calib.model != "ro" else {}
    for v in st.views:
        if v.id in ids:
            st.rejected.append({"step": st.step, "attempt": st.attempt, "rule": "R7_OUTLIER", "metric": "per_view_rms",
                                "value": round(float(pv.get(v.id, math.nan)), 4) if v.id in pv else None, "view_id": int(v.id)})
    st.views = keep
    return True


R9_FOLDS = 3
R9_FIRST_FOLD_S = 3.0  # one RO fold at 512 MB Lambda before any is measured (k5-type lenses: 1.0-4.2 s)
R9_RESERVE_S = 1.5  # left after a fold for the uncertainty map, the NBV minimum and the response


def _r9_step(ctx: _Ctx, calib: CalibResult) -> tuple[float, float, float] | None:
    """R9 evidence, one fold per step: 3-fold held-out RMS of RO vs k5 on a fixed set (the last 12
    full-board views when the check starts), folds by index modulo 3, each fold fitted on the
    other two thirds and warm-started from the session's k5 fit (the same statistic as
    ``ro_cv_gain``). The running sums live in ``state.r9``; a fold runs only when the time left
    covers the slowest fold measured so far (``R9_FIRST_FOLD_S`` before the first) plus
    ``R9_RESERVE_S``. Returns ``(gain, cv_ro, cv_k5)`` once the third fold is done, else None.
    """
    st, size, th = ctx.state, ctx.state.image_size, ctx.th
    full = full_board_views(st.views)
    r9 = st.r9
    have = {int(v.id) for v in st.views}
    if r9 is None or r9.get("done") or not set(r9.get("ids", ())) <= have:
        use = full[-int(getattr(th, "ro_cv_max_full_views", RO_CV_MAX_FULL_VIEWS)):]
        r9 = st.r9 = {"n_full": len(full), "ids": [int(v.id) for v in use], "fold": 0, "sse_ro": 0.0, "n_ro": 0,
                      "sse_k5": 0.0, "n_k5": 0, "fold_s": [], "done": False}
    k = int(r9["fold"])
    need = (max(r9["fold_s"]) if r9["fold_s"] else R9_FIRST_FOLD_S) + R9_RESERVE_S
    left = ctx.time_left()
    if left < need:
        ctx.notes.append(f"R9 fold {k + 1}/{R9_FOLDS} deferred: needs ~{need:.1f} s, {left:.1f} s left in this step")
        return None
    vmap = {int(v.id): v for v in st.views}
    vs = [vmap[i] for i in r9["ids"]]
    train = [v for i, v in enumerate(vs) if i % R9_FOLDS != k]
    test = [v for i, v in enumerate(vs) if i % R9_FOLDS == k]
    t = time.perf_counter()
    ok = True
    with ctx.tools.time("ro_cv_fold", fold=k + 1, views=len(vs)):
        try:
            ro = calibrate(train, "ro", size, init=calib)
            k5 = calibrate(train, "k5", size, init=calib)
            for v in test:
                a, n = heldout_view_error(v, ro.K, ro.dist, board_points(ro))
                b, m = heldout_view_error(v, k5.K, k5.dist)
                r9["sse_ro"] += float(a)
                r9["n_ro"] += int(n)
                r9["sse_k5"] += float(b)
                r9["n_k5"] += int(m)
        except (ValueError, cv2.error):
            ok = False
    r9["fold_s"].append(round(time.perf_counter() - t, 3))
    r9["fold"] = k + 1
    if ok and r9["fold"] < R9_FOLDS:
        ctx.notes.append(f"R9 check: fold {k + 1}/{R9_FOLDS} done over {len(vs)} full-board views")
        return None
    cv_ro = math.sqrt(r9["sse_ro"] / r9["n_ro"]) if ok and r9["n_ro"] else math.nan
    cv_k5 = math.sqrt(r9["sse_k5"] / r9["n_k5"]) if ok and r9["n_k5"] else math.nan
    gain = 1.0 - cv_ro / cv_k5 if (math.isfinite(cv_ro) and math.isfinite(cv_k5) and cv_k5 > 0) else math.nan
    r9.update({"done": True, "step": st.step, "gain": round(gain, 4) if math.isfinite(gain) else None,
               "cv_ro": round(cv_ro, 4) if math.isfinite(cv_ro) else None,
               "cv_k5": round(cv_k5, 4) if math.isfinite(cv_k5) else None})
    return gain, cv_ro, cv_k5


def _diagnose_and_decide(ctx: _Ctx, *, new_view: bool, allow_stop: bool = True, skip=()) -> str:
    """Refit → R6-R10 → uncertainty → R11/R12/R13 (plan). Returns the status."""
    st, th, size = ctx.state, ctx.th, ctx.state.image_size
    calib = ctx.calib
    if calib is not None:
        with ctx.tools.time("residual_analysis"):
            diag = residual_analysis(calib, st.views, th, size)

        def cv_pair():
            with ctx.tools.time("cv_heldout_rms", models="k5,rational"):
                return (cv_heldout_rms(st.views, "k5", size, init=calib),
                        cv_heldout_rms(st.views, "rational", size, init=calib))

        def ro_gain():
            return _r9_step(ctx, calib)

        sk = tuple(skip) + (() if new_view else ("R6_DRIFT",))
        args = dict(model=st.model, calib=calib, preq_history=st.preq_history, th=th, size=size, cv_pair=cv_pair,
                    ro_gain=ro_gain, have_time=ctx.have_time, notes=ctx.notes, flags=tuple(st.flags), r9_last=st.r9,
                    full_view_ids=[int(v.id) for v in full_board_views(st.views)])
        f = post_accept(n_views=len(st.views), n_full_views=len(full_board_views(st.views)), diag=diag, skip=sk, **args)
        if f is not None and f.rule == "R7_OUTLIER":
            if _drop_views(ctx, list(f.evidence["view_ids"])):
                ctx.findings.append(f)
                with ctx.tools.time("calibrate", reason="R7 refit"):
                    calib = ctx.calib = _fit(st, st.views, init=calib)
                with ctx.tools.time("residual_analysis"):
                    diag = residual_analysis(calib, st.views, th, size)
            args["r9_last"] = st.r9
            args["full_view_ids"] = [int(v.id) for v in full_board_views(st.views)]
            f = post_accept(n_views=len(st.views), n_full_views=len(full_board_views(st.views)), diag=diag,
                            skip=sk + ("R6_DRIFT", "R7_OUTLIER"), **args)
        running = bool(st.r9) and not st.r9.get("done") and set(st.r9.get("ids", ())) <= {int(v.id) for v in st.views}
        seek = warp_seek(st.model, len(full_board_views(st.views)), diag.warp_score, th, r9_running=running)
        if seek and not st.seek_full_board:
            ctx.notes.append(f"warp score {diag.warp_score:.2f} on {len(full_board_views(st.views))} full-board views: "
                             f"planning whole-board views until {th.warp_min_full_views} for the R9 check")
        st.seek_full_board = seek
        _store_calib(st, calib)
        with ctx.tools.time("uncertainty_map"):
            ctx.unc = uncertainty_map(calib, st.views, size, ctx.stopcal, actuator=st.actuator, tau_px=st.tau,
                                      grid=tuple(th.grid_uncertainty))
        if f is not None:
            ev = {}
            if f.rule == "R6_DRIFT":
                ev = {"views": len(st.views)}
            ctx.findings.append(f)
            return _raise_approval(ctx, f, ev)
    u95 = None if ctx.unc is None else float(ctx.unc.U95)
    pex = math.nan if ctx.unc is None else float(ctx.unc.p_exceed)
    d = decide(n_views=len(st.views), p_exceed=pex, u95=u95, prequential_last=[v.prequential_rms for v in st.views[-2:]],
               rms=None if calib is None else float(calib.rms), flags=st.flags, attempts=st.attempts, budget=st.budget,
               max_attempts=st.max_attempts, th=th, allow_stop=allow_stop)
    if d.approval is not None:
        ev = {"views": len(st.views), "attempts": st.attempts, "U95_px": u95, "p_exceed": pex if math.isfinite(pex) else None,
              "tau_px": st.tau, "model": st.model["name"], "flags": list(st.flags)}
        if d.rule == "R12_BUDGET":
            ev["label"] = ("likely within target" if math.isfinite(pex) and pex < 0.5
                           else "likely above target — recommend recapture" if math.isfinite(pex) else "stop probability not calibrated")
        if calib is not None:
            ev.update({"K": np.round(calib.K, 4).tolist(), "dist": np.round(calib.dist, 7).tolist(), "rms_px": round(calib.rms, 4)})
        ctx.findings.append(d)
        return _raise_approval(ctx, d, ev)
    _plan(ctx, d)
    return "continue"


def _plan(ctx: _Ctx, d: Finding) -> None:
    """R13: the next target (an opening pose before 3 views, else the NBV top candidate)."""
    st, th, size = ctx.state, ctx.th, ctx.state.image_size
    sim = st.mode == "sim"
    if ctx.calib is None:
        k = len(st.views)
        st.last_target = _opening(st, k)
        st.candidates = []
        d.measurement = f"{len(st.views)} views, no calibration yet"
        d.action = f"opening pose {k + 1}/3: {st.last_target['desc']}"
        d.instruction = st.last_target["instruction"]
        ctx.findings.append(d)
        return
    calib = ctx.calib
    rng = np.random.default_rng([st.camera_seed, st.seed, st.attempt, NBV_STREAM])
    nbv_deadline = ctx.t0 + float(th.nbv_time_budget_s)
    with ctx.tools.time("propose_next_views") as info:
        cands = propose_next_views(st, calib, size, th, rng, k=5, deadline=nbv_deadline)
        info.update(cands.stats())
    if cands:
        st.candidates = [_cand_dict(c, calib.K, calib.dist, size, sim) for c in cands]
        st.last_target = st.candidates[0]
    else:  # no feasible lattice point (should not happen): fall back to an opening pose under the estimate
        st.candidates = []
        k = len(st.views) % 3
        rvec, tvec = opening_target(k, st.camera_seed, st.seed, size, float(calib.K[0, 0]))
        st.last_target = _target(rvec, tvec, calib.K, calib.dist, size, sim, "fallback", rank=None, pred_gain=None,
                                 pred_U95_after=None)
        ctx.notes.append("NBV found no feasible candidate; fell back to an opening pose")
    s = np.asarray(ctx.unc.sigma_px, float)
    r, c = np.unravel_index(int(np.argmax(s)), s.shape)
    cols, rows = ctx.unc.grid
    W, H = size
    where = position_word(((c + 0.5) * W / cols, (r + 0.5) * H / rows), size)
    pe = "n/a" if not math.isfinite(ctx.unc.p_exceed) else f"{ctx.unc.p_exceed:.2f}"
    d.measurement = f"largest σ {where} {float(s.max()):.2f} px, U95 {ctx.unc.U95:.2f} px, P(>τ)={pe}"
    tgt = st.last_target
    d.action = f"target {tgt['desc']}" + ("" if tgt.get("pred_gain") is None else f" (predicted ΔU95 {tgt['pred_gain']:.3f} px)")
    d.instruction = tgt["instruction"]
    ctx.findings.append(d)


def _retarget_after_reject(ctx: _Ctx, g: Finding, target: dict, calib_prev: CalibResult | None) -> None:
    """How a rejected frame changes the next target.

    R3 GLARE: the same target tilted ~15° about the axis perpendicular to the hotspot offset, the
    edge opposite the hotspot sent away (the wording of the R3 instruction), kept under the
    ``max_tilt_deg`` cap (``_glare_turn``). R0 / R1 in the
    planning phase: the next-ranked stored candidate (the target is likely not reachable or too
    small under the true lens), and a fresh NBV plan once the list is used up. R2 BLUR keeps the
    target (hold still and retry). Before the first calibration the opening pose is re-placed
    with the updated focal estimate.
    """
    st, size, sim = ctx.state, ctx.state.image_size, ctx.state.mode == "sim"
    K = calib_prev.K if calib_prev is not None else k_initial(size, _focal_hint(st.focal_obs))
    dist = calib_prev.dist if calib_prev is not None else np.zeros(5)
    if g.rule == "R3_GLARE":
        axis, az = glare_tilt(g.evidence.get("glare_offset", (0.0, 0.0)))
        turn, az, flipped = _glare_turn(target, az, float(getattr(ctx.th, "max_tilt_deg", 60.0)))
        rv, tv = tilted_target(target["rvec"], target["tvec"], turn, az)
        if flipped:
            g.instruction = REJECT_TEMPLATES["R3_GLARE"].format(axis=axis, edge=edge_away(az))
        keep = {k: target.get(k) for k in ("rank", "pred_gain", "pred_U95_after", "k", "focal") if k in target}
        st.last_target = _target(rv, tv, K, dist, size, sim, "glare_retry", **keep)
        g.action = (f"reject, retry tilted {turn:.0f}° about the {axis} axis ({edge_away(az)} away"
                    + (", the other way: the usual way passes the tilt cap" if flipped else "") + ")")
        return
    if calib_prev is None:
        st.last_target = _opening(st, len(st.views))
        return
    if g.rule in ("R0_NO_BOARD", "R1_SMALL"):
        rank = target.get("rank")
        nxt = [c for c in st.candidates if rank is not None and c.get("rank") is not None and c["rank"] > rank]
        if nxt:
            st.last_target = nxt[0]
            g.action = f"reject, next candidate (rank {nxt[0]['rank']}): {nxt[0]['desc']}"
            return
        d = Finding("R13_NBV", "", "plan next view")
        _plan(ctx, d)
        ctx.findings.remove(d)
        g.action = f"reject, candidates used up, re-planned: {st.last_target['desc']}"


GLARE_MIN_TURN_DEG = 5.0


def _glare_turn(target: dict, az: float, max_tilt: float) -> tuple[float, float, bool]:
    """(turn deg, azimuth, flipped) for the R3 retry, keeping the retry target's tilt under ``max_tilt``.

    The usual turn is 15° sending the edge opposite the hotspot away. When that passes the tilt
    cap the turn is shortened; when less than 5° would remain, the board is turned 15° the
    other way instead (the opposite edge away), which lowers the tilt.
    """
    rv0, tv0 = target["rvec"], target["tvec"]
    cap = max_tilt - 0.5
    if tilt_of(tilted_target(rv0, tv0, GLARE_TILT_DEG, az)[0]) <= cap:
        return GLARE_TILT_DEG, az, False
    lo, hi = 0.0, GLARE_TILT_DEG
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if tilt_of(tilted_target(rv0, tv0, mid, az)[0]) <= cap:
            lo = mid
        else:
            hi = mid
    if lo >= GLARE_MIN_TURN_DEG:
        return lo, az, False
    az2 = (az + 180.0) % 360.0
    if tilt_of(tilted_target(rv0, tv0, GLARE_TILT_DEG, az2)[0]) <= cap:
        return GLARE_TILT_DEG, az2, True
    return lo, az, False


def _finish(ctx: _Ctx, status: str, *, accepted: bool, perception_dict, image=None, preq=None, baseline=None,
            truth=None, export_files=None, export_meta=None, pred_gain=None) -> StepResult:
    st = ctx.state
    st.status = status
    lines = [f.trace(st.step) for f in ctx.findings]
    for ln in lines:
        st.add_trace(ln)
    pf = primary(ctx.findings)
    if status != "continue" and st.pending_approval is not None:
        pf = next((f for f in ctx.findings if f.rule == st.pending_approval.get("rule")), pf)
    rule = None if pf is None else pf.rule
    instruction = None
    if pf is not None and pf.rule in ("R0_NO_BOARD", "R1_SMALL", "R2_BLUR", "R3_GLARE"):
        instruction = pf.instruction
    elif st.pending_approval is not None:
        instruction = st.pending_approval["question"]
    elif st.last_target is not None and status == "continue":
        instruction = st.last_target["instruction"]
    notes = [f.instruction for f in ctx.findings if f.rule in ("R4_AMBIGUOUS", "R5_OFF_TARGET") and f.instruction] + ctx.notes
    plan = None
    if st.last_target is not None and status == "continue":
        t = st.last_target
        plan = {"target": {k: t[k] for k in ("rvec", "tvec", "ghost_quad", "instruction", "source", "desc")},
                "candidates": [{"rank": c.get("rank"), "desc": c.get("desc"), "pred_U95_after": c.get("pred_U95_after"),
                                "gain": c.get("pred_gain"), "effort": c.get("effort"), "score": c.get("score")}
                               for c in st.candidates]}
        if st.history:
            plan["last_realised"] = st.history[-1].get("realised")
    unc = _unc_dict(ctx.unc)
    latency = 1000 * (time.perf_counter() - ctx.t0)
    hist = {"step": st.step, "rule": rule, "accepted": accepted, "views": len(st.views), "attempts": st.attempts,
            "model": st.model["name"], "U95": None if ctx.unc is None else round(float(ctx.unc.U95), 4),
            "p_exceed": None if unc is None else unc["p_exceed"], "latency_ms": round(latency, 1), "status": status}
    if truth is not None:
        hist["true_p95"] = truth["field_p95_px"]
        hist["fx_rel_err_pct"] = truth["fx_rel_err_pct"]
    if baseline is not None:
        hist["baseline_views"] = baseline["views"]
        hist["baseline_true_p95"] = (baseline.get("truth") or {}).get("field_p95_px")
    realised = _realised(st, accepted, hist, pred_gain)
    if realised is not None:
        hist["realised"] = realised
        if plan is not None:
            plan["last_realised"] = realised
    st.history.append(hist)
    return StepResult(
        status=status, step=st.step, rule=rule, accepted=accepted,
        trace=lines[[f.rule for f in ctx.findings].index(rule)] if rule else "", trace_all=lines,
        instruction=instruction, notes=notes, tools=ctx.tools.items, perception=perception_dict,
        calibration=_calib_dict(ctx.calib, st.views, preq), uncertainty=unc,
        coverage=_coverage(st.views, st.image_size, tuple(ctx.th.grid_coverage)) if st.views else None,
        plan=plan, approval=st.pending_approval, truth=truth, baseline=baseline, latency_ms=latency, image=image,
        export_files=export_files, export_meta=export_meta,
    )


def _realised(st: SessionState, accepted: bool, hist: dict, pred_gain) -> dict | None:
    """Predicted vs realised U95 drop (and true p95 drop in sim) for the view just accepted."""
    if not accepted or not st.history:
        return None
    prev = next((h for h in reversed(st.history) if h.get("U95") is not None), None)
    if prev is None or hist.get("U95") is None:
        return None
    out = {"pred_gain": pred_gain, "realised_gain": round(prev["U95"] - hist["U95"], 5)}
    if prev.get("true_p95") is not None and hist.get("true_p95") is not None:
        out["true_gain"] = round(prev["true_p95"] - hist["true_p95"], 5)
    return out


def _perception_dict(det, q, pose) -> dict:
    d = {"detected": bool(det.ok), "n_corners": int(len(det.ids)), "ids": np.asarray(det.ids).astype(int).tolist(),
         "corners": round_corners(det.corners).tolist(), "quality": None}
    if q is not None:
        d["quality"] = {"edge_rise_px": q.edge_rise_px, "anisotropy": q.anisotropy, "blur_dir_deg": q.blur_dir_deg,
                        "saturation_frac": q.saturation_frac, "board_area_frac": q.board_area_frac,
                        "ippe_ratio": None if pose is None else pose.ippe_ratio,
                        "tilt_deg": None if pose is None else pose.tilt_deg,
                        "distance_m": None if pose is None else pose.distance}
        d["quality"] = {k: (None if v is None else round(float(v), 4)) for k, v in d["quality"].items()}
    return d


# ---- baseline track (sim only) ----------------------------------------------------------------

def _advance_baseline(st: SessionState, source: SimSource, th, tools: _Tools) -> dict | None:
    """B0 shadow track: the 3 shared opening poses, then random feasible poses (simulator sampler),
    no gate beyond detection, k5 model. One capture per step."""
    from ..sim.poses import random_pose
    from ..sim.truth import field_error

    b = st.baseline
    views = b["views"]
    if len(views) >= st.budget:
        return _baseline_view(b)
    size = st.image_size
    attempt = int(b["attempt"])
    b["attempt"] = attempt + 1
    n_acc = int(b.get("n_accepted", 0))
    with tools.time("baseline"):
        if len(views) < 3:
            f = _focal_hint(b["focal_obs"])
            rvec, tvec = opening_target(len(views), st.camera_seed, st.seed, size, f)
            f_used = float(k_initial(size, f)[0, 0])
        else:
            rng = np.random.default_rng([st.camera_seed, st.seed, attempt, BASELINE_STREAM])
            try:
                rvec, tvec = random_pose(source.camera(n_acc), rng)
            except RuntimeError:
                return _baseline_view(b)
            f_used = None
        obs = source.capture(rvec, tvec, attempt=attempt, n_accepted=n_acc, prev_pose=b.get("sim_prev_pose"))
        b["sim_prev_pose"] = {"rvec": obs.truth.rvec.tolist(), "tvec": obs.truth.tvec.tolist()}
        det = obs.det
        if f_used is not None and len(det.ids) >= 4:
            fo = focal_from_detection(det.ids, det.corners, rvec, tvec, f_used, size)
            if fo is not None:
                b["focal_obs"] = (b["focal_obs"] + [round(fo, 3)])[-MAX_FOCAL_OBS:]
        if det.ok and len(det.ids) >= BASELINE_MIN_CORNERS:
            vid = int(b.get("next_view_id", 0))
            b["next_view_id"] = vid + 1
            views.append(View(vid, det.ids, round_corners(det.corners), None, np.asarray(rvec, float), np.asarray(tvec, float)))
            b["n_accepted"] = n_acc + 1
            if len(views) >= 3:
                h = b.get("calib")
                init = None if not h else CalibResult("k5", np.asarray(h["K"]), np.asarray(h["dist"]), float(h["rms"]),
                                                     np.zeros(0), [], [], np.zeros(0), math.nan, np.zeros(0), None)
                try:
                    c = calibrate(views, "k5", size, init=init)
                except ValueError:
                    c = None
                if c is not None:
                    b["calib"] = {"K": c.K.tolist(), "dist": c.dist.tolist(), "rms": float(c.rms)}
                    fe = field_error(source.camera(b["n_accepted"]), c.K, c.dist)
                    b["truth"] = {"field_p95_px": round(fe.p95, 4), "field_mean_px": round(fe.mean, 4)}
    return _baseline_view(b)


def _baseline_view(b: dict) -> dict:
    return {"policy": "random", "views": len(b["views"]), "truth": b.get("truth")}


# ---- run_step / apply_approval ----------------------------------------------------------------

def _defaults(th, stopcal):
    th = load_thresholds() if th is None else th
    stopcal = load_stop_calibration() if stopcal is None else stopcal
    return th, stopcal


def run_step(state: SessionState, source=None, *, th=None, stopcal=None, deadline: float | None = None,
             choose_candidate: int | None = None, image: np.ndarray | None = None, baseline: bool = True) -> StepResult:
    """One perceive → decide → act cycle. Mutates ``state``; returns the step result.

    ``source``: a ``SimSource`` / ``FrameSource`` (built from ``state`` and ``image`` when None).
    ``deadline``: ``time.perf_counter()`` value (default now + ``step_deadline_s``); the NBV stops
    at ``nbv_time_budget_s``, and optional work (R8/R9 cross-validation, the baseline track) is
    postponed when the step runs late. ``choose_candidate``: 1-5, a human override of the target.
    """
    t0 = time.perf_counter()
    th, stopcal = _defaults(th, stopcal)
    st = state
    if st.terminal:
        raise LoopError(f"session has ended ({st.status})")
    if st.pending_approval is not None:
        return awaiting(st)
    if source is None:
        source = make_source(st, image)
    elif image is not None:
        raise LoopError("pass a frame source or an image, not both")
    kind = getattr(source, "kind", None)
    if (st.mode == "sim") != (kind == "sim"):
        raise LoopError(f"a {st.mode} session cannot take a {kind or 'unknown'} frame source")
    if isinstance(source, FrameSource):
        h, w = source.image.shape[:2]
        if (w, h) != tuple(st.image_size):
            raise LoopError(f"frame is {w}x{h}, session expects {st.image_size[0]}x{st.image_size[1]}")
    deadline = t0 + float(th.step_deadline_s) if deadline is None else deadline
    ctx = _Ctx(st, source, th, stopcal, t0, deadline, _Tools())
    size = st.image_size
    st.step += 1

    if st.last_target is None:
        st.last_target = _opening(st, len(st.views))
    target = st.last_target
    if choose_candidate is not None:
        k = int(choose_candidate)
        if not 1 <= k <= len(st.candidates):
            raise LoopError(f"choose_candidate must be 1..{len(st.candidates)}, got {choose_candidate}")
        target = dict(st.candidates[k - 1], source="user")
        st.last_target = target
        st.add_trace(f"#{st.step} USER: candidate {k} chosen → target {target['desc']}")
    pred_gain = target.get("pred_gain")

    calib_prev = None
    if len(st.views) >= 3:
        with ctx.tools.time("calibrate", reason="rebuild"):
            calib_prev = _fit(st, st.views)
    ctx.calib = calib_prev

    obs = source.capture(target["rvec"], target["tvec"], attempt=st.attempt, n_accepted=st.n_accepted_total,
                         prev_pose=st.sim_prev_pose)
    ctx.tools.items.extend({"tool": k, "ms": round(v, 1)} for k, v in obs.ms.items())
    det = obs.det
    st.attempt += 1
    st.attempts += 1
    if obs.truth is not None:
        st.sim_prev_pose = {"rvec": obs.truth.rvec.tolist(), "tvec": obs.truth.tvec.tolist()}
    with ctx.tools.time("assess_frame"):
        q = assess_frame(obs.image, det, anisotropy_threshold=th.motion_anisotropy)

    if calib_prev is None and target.get("focal") is not None and len(det.ids) >= 4:
        fo = focal_from_detection(det.ids, det.corners, target["rvec"], target["tvec"], target["focal"], size)
        if fo is not None:
            st.focal_obs = (st.focal_obs + [round(fo, 3)])[-MAX_FOCAL_OBS:]

    K_cur = calib_prev.K if calib_prev is not None else k_initial(size, _focal_hint(st.focal_obs))
    d_cur = calib_prev.dist if calib_prev is not None else np.zeros(5)
    g = gate(det, q, th)
    pose = None
    if g is None:
        try:
            with ctx.tools.time("estimate_pose"):
                pose = estimate_pose(det, K_cur, d_cur)
        except ValueError as e:
            g = Finding("R0_NO_BOARD", f"pose not solvable ({e})", "reject", "n_corners", float(len(det.ids)),
                        "Board not found — bring the whole board into view")
    accepted = g is None
    preq = None
    if not accepted:
        ctx.findings.append(g)
        st.rejected.append({"step": st.step, "attempt": st.attempt - 1, "rule": g.rule, "metric": g.metric,
                            "value": None if g.value is None or not math.isfinite(g.value) else round(float(g.value), 4)})
    else:
        pf = perception(pose, th, actuator=st.actuator, target=target if target.get("source") != "opening" else None,
                        K=K_cur, dist=d_cur, size=size)
        ctx.findings.extend(pf)
        if any(f.rule == "R4_AMBIGUOUS" for f in pf):
            st.set_flag("AMBIGUOUS")
        else:
            st.clear_flag("AMBIGUOUS")
        view = View(st.next_view_id, np.asarray(det.ids, np.int32), round_corners(det.corners), q, pose.rvec, pose.tvec)
        if calib_prev is not None:
            with ctx.tools.time("prequential"):
                preq = prequential_rms(view, calib_prev)
            view = View(view.id, view.corner_ids, view.corners, q, pose.rvec, pose.tvec, round(float(preq), 4))
        st.views.append(view)
        st.next_view_id += 1
        st.n_accepted_total += 1
        st.preq_history.append(None if preq is None else round(float(preq), 4))
        st.last_realised = {"rvec": pose.rvec.tolist(), "tvec": pose.tvec.tolist()}

    if accepted:
        if len(st.views) >= 3:
            with ctx.tools.time("calibrate"):
                try:
                    ctx.calib = _fit(st, st.views, init=calib_prev)
                except ValueError as e:  # degenerate geometry: undo the view, keep the previous fit
                    st.views.pop()
                    st.preq_history.pop()
                    ctx.calib = calib_prev
                    ctx.notes.append(f"calibration failed with the new view ({e}); view dropped")
        status = _diagnose_and_decide(ctx, new_view=True)
    else:
        if calib_prev is not None:
            with ctx.tools.time("uncertainty_map"):
                ctx.unc = uncertainty_map(calib_prev, st.views, size, stopcal, actuator=st.actuator, tau_px=st.tau,
                                          grid=tuple(th.grid_uncertainty))
        d = decide(n_views=len(st.views), p_exceed=math.nan, u95=None if ctx.unc is None else ctx.unc.U95,
                   prequential_last=[], rms=None, flags=st.flags, attempts=st.attempts, budget=st.budget,
                   max_attempts=st.max_attempts, th=th, allow_stop=False)
        if d.approval is not None:
            ctx.findings.append(d)
            status = _raise_approval(ctx, d, {"views": len(st.views), "attempts": st.attempts, "tau_px": st.tau,
                                              "U95_px": None if ctx.unc is None else ctx.unc.U95, "p_exceed": None,
                                              "label": "stop probability not available"})
        else:
            _retarget_after_reject(ctx, g, target, calib_prev)
            status = "continue"

    truth = None
    if isinstance(source, SimSource):
        with ctx.tools.time("truth"):
            truth = _truth(source, st.n_accepted_total, ctx.calib)
    base = None
    if baseline and isinstance(source, SimSource) and st.baseline is not None:
        if time.perf_counter() < t0 + 0.75 * (deadline - t0):
            base = _advance_baseline(st, source, th, ctx.tools)
        else:
            ctx.notes.append("baseline frame postponed by the step deadline")
            base = _baseline_view(st.baseline)
    return _finish(ctx, status, accepted=accepted, perception_dict=_perception_dict(det, q, pose), image=obs.image,
                   preq=preq, baseline=base, truth=truth, pred_gain=pred_gain)


def apply_approval(state: SessionState, approval_id: str, decision: str, reason: str = "", source=None, *, th=None,
                   stopcal=None, deadline: float | None = None, now: float | None = None) -> StepResult:
    """Resolve the pending approval, record it (state + trace), act on it and re-plan."""
    t0 = time.perf_counter()
    th, stopcal = _defaults(th, stopcal)
    st = state
    if st.terminal:
        raise LoopError(f"session has ended ({st.status})")
    kind = check_decision(st.pending_approval, approval_id, decision)
    card = st.pending_approval
    if kind in (ApprovalKind.EXPORT, ApprovalKind.ACCEPT_WITH_WARNINGS) and decision == "approve" and len(st.views) < 3:
        raise ApprovalError("nothing to export: fewer than 3 views")
    st.step += 1
    deadline = t0 + float(th.step_deadline_s) if deadline is None else deadline
    ctx = _Ctx(st, source, th, stopcal, t0, deadline, _Tools())
    ts = time.time() if now is None else float(now)
    reason = str(reason or "")[:500]
    st.approvals.append({"id": approval_id, "kind": kind.value, "decision": decision, "reason": reason, "ts": round(ts, 3),
                         "step": st.step, "rule": card.get("rule")})
    st.pending_approval = None
    if source is None and st.mode == "sim":
        source = ctx.source = make_source(st)
    action = {
        (ApprovalKind.SWITCH_MODEL, "approve"): "refit with CALIB_RATIONAL_MODEL",
        (ApprovalKind.SWITCH_MODEL, "deny"): "model pinned to k5, continue",
        (ApprovalKind.USE_RO, "approve"): "refit with calibrateCameraROExtended",
        (ApprovalKind.USE_RO, "deny"): "model pinned, continue",
        (ApprovalKind.DRIFT_RESTART, "approve"): "clear views and restart",
        (ApprovalKind.DRIFT_RESTART, "deny"): "continue with DRIFT flag",
        (ApprovalKind.ACCEPT_WITH_WARNINGS, "approve"): "export with warnings",
        (ApprovalKind.ACCEPT_WITH_WARNINGS, "deny"): "session discarded",
        (ApprovalKind.EXPORT, "approve"): "export calibration",
        (ApprovalKind.EXPORT, "deny"): "session stays open, continue",
        (ApprovalKind.OUT_OF_SCOPE, "acknowledge"): "session ended (out of scope)",
    }[(kind, decision)]
    line = f"#{st.step} APPROVAL {approval_id} {kind.value}: {decision}" + (f" ({reason})" if reason else "") + f" → {action}"
    st.add_trace(line)

    export_files = export_meta = None
    status = None
    if kind is ApprovalKind.SWITCH_MODEL:
        st.clear_flag("UNDERFIT")
        if decision == "approve":
            st.model = {"name": "rational", "pinned": False}
        else:
            st.model = dict(st.model, pinned=True)
    elif kind is ApprovalKind.USE_RO:
        st.clear_flag("WARP")
        st.seek_full_board = False
        if decision == "approve":
            st.model = {"name": "ro", "pinned": False}
        else:
            st.model = dict(st.model, pinned=True)
    elif kind is ApprovalKind.DRIFT_RESTART:
        if decision == "approve":
            st.views, st.preq_history, st.focal_obs, st.candidates = [], [], [], []
            st.calib = st.last_realised = st.r9 = None
            st.seek_full_board = False
            st.attempts = 0
            st.clear_flag("AMBIGUOUS")
            if st.baseline is not None:
                st.baseline.update({"views": [], "calib": None, "focal_obs": [], "truth": None})
            st.last_target = _opening(st, 0)
            status = "continue"
        else:
            st.set_flag("DRIFT")
    elif kind is ApprovalKind.OUT_OF_SCOPE:
        status = "ended"
    elif kind is ApprovalKind.ACCEPT_WITH_WARNINGS and decision == "deny":
        status = "discarded"

    if status is None:
        with ctx.tools.time("calibrate", reason="after approval"):
            ctx.calib = _fit(st, st.views)
        if (kind in (ApprovalKind.EXPORT, ApprovalKind.ACCEPT_WITH_WARNINGS)) and decision == "approve":
            if ctx.calib is None:
                raise ApprovalError("nothing to export: fewer than 3 views")
            with ctx.tools.time("uncertainty_map"):
                ctx.unc = uncertainty_map(ctx.calib, st.views, st.image_size, stopcal, actuator=st.actuator,
                                          tau_px=st.tau, grid=tuple(th.grid_uncertainty))
            warn = kind is ApprovalKind.ACCEPT_WITH_WARNINGS
            export_meta = {"session": st.sid, "preset": st.preset, "mode": st.mode, "approval_id": approval_id,
                           "warnings": warn, "U95_px": round(float(ctx.unc.U95), 4), "tau_px": st.tau,
                           "p_exceed": ctx.unc.p_exceed if math.isfinite(ctx.unc.p_exceed) else None,
                           "views": len(st.views), "flags": ",".join(st.flags) or "none"}
            truth = _truth(source, st.n_accepted_total, ctx.calib)
            if truth is not None:
                export_meta["true_field_p95_px"] = truth["field_p95_px"]
            with ctx.tools.time("export_calibration"):
                export_files = export_calibration(ctx.calib, ctx.unc, st.trace, size=st.image_size, meta=export_meta)
            status = "exported"
        else:
            status = _diagnose_and_decide(ctx, new_view=False, allow_stop=kind is not ApprovalKind.EXPORT)
    truth = _truth(source, st.n_accepted_total, ctx.calib) if source is not None else None
    base = None if st.baseline is None else _baseline_view(st.baseline)
    res = _finish(ctx, status, accepted=False, perception_dict=None, truth=truth, baseline=base,
                  export_files=export_files, export_meta=export_meta)
    res.trace_all = [line] + res.trace_all
    if not res.trace:
        res.trace = line
    if res.rule is None:
        res.rule = "APPROVAL"
    return res
