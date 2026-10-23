"""Benchmark policies (spec 11.3): the agent A, its ablations, and baselines B0-B4.

Every policy sees the same simulated camera, actuator and nuisance streams for a given
(camera, seed, actuator): captures are indexed by the attempt number (common random numbers,
spec 6.4). Every run goes to ``budget`` accepted frames (or 60 attempts) and records one metric
row per accepted frame (``common.metric_row``). On the failure split the agent instead runs until
its own terminal card (R10, R12 on views or attempts, R6, or a clean R11 stop), so the
abstention metric is not cut short by the harness.

A and its ablations run through the real agent loop (``agent.loop.run_step`` /
``apply_approval``), with every frame detected through ``SimFrame.observe``:

* ``A``: approve_all for model changes; at the first R11 STOP card the stop point is recorded,
  the export is denied ("session stays open") and the stop rule is switched off for the rest of
  the run. The trajectory then continues to the budget with the same poses, fx, fy, cx, cy and
  true field p95 as without a stop; the extra refit after the deny can move the distortion
  coefficients and U95 by under 1e-4 relative and the true field mean by about 1e-3 relative
  (measured on TEST camera 1045, seed 2, robot; three other stopped runs were identical).

Outlier bookkeeping: a view counts as corrupted when its capture had corners moved by the
corruption nuisance and it entered the set, whether or not R7 dropped it in the same step.
* ``A-gate``: no R1-R3 rejection; a frame is rejected only when detection fails (fewer than 6
  corners or a degenerate span), the same acceptance as B0.
* ``A-outlier``: R7 skipped.
* ``A-model``: model-change cards denied (approval_policy = deny_all), spec "equivalent to deny_all".
* ``A-nbv``: the NBV order replaced by a random order over the same feasible candidate set
  (same lattice sample, same feasibility rules, same AMBIGUOUS / full-board constraints).

The benchmark passes no wall-clock budget to the loop (deadline and ``nbv_time_budget_s`` set to
1e5 s), so every run is deterministic: all sampled NBV candidates are evaluated and the R8/R9
cross-validation is never postponed. On Lambda those budgets can bind; they do not change rules.

Baselines (no agent gate, k5 model, ``tools.calibrate.calibrate`` warm-started from the previous
fit, except B2 which uses the interactive-calibration app's own calibrateCamera call):

* ``B0``: the 3 shared opening poses, then random feasible poses from the simulator sampler
  (``sim.poses.random_pose``), exactly the UI race-chart shadow track. Accept: >= 6 corners.
* ``B1``: fixed 12-pose textbook template (own order), jittered repeats beyond 12.
* ``B2``: OpenCV ``apps/interactive-calibration`` (tag 5.0.0) emulation over the B0 random pose
  stream, with the app's own calibrateCamera call; see ``B2App``.
* ``B2-k5`` (descriptive only, not a pre-registered endpoint): B2's acceptance and stop rules with
  the bench's k5 solver, to separate the app's rules from its 30-iteration solver.
* ``B3``: coverage-greedy over the NBV lattice sample under the current estimate.
* ``B4``: dense 45-pose grid in farthest-point order.

B1 and B4 are defined in image terms (board centre, size fraction, tilt); they are placed with
the GT camera (``sim.poses.pose_for_view``), i.e. the operator frames them by looking at the
screen. B0 uses the simulator's own sampler, also GT-framed. A and B3 place poses under their
current estimate.
"""

from __future__ import annotations

import dataclasses
import math
import time
from contextlib import contextmanager

import cv2
import numpy as np

from ..agent import loop as agent_loop
from ..agent.loop import Observation, SimSource, apply_approval, run_step
from ..agent.policy import OPEN_FLAGS, Finding
from ..agent.state import new_session, round_corners
from ..board import N_CORNERS, object_points
from ..sim.poses import pose_for_view, random_pose
from ..sim.scenarios import make_simulator
from ..tools.calibrate import calibrate
from ..tools.detect import corner_span
from ..tools.nbv import (CandidateList, focal_from_detection, k_initial, lattice, opening_target, propose_next_views)
from ..tools.uncertainty import covariance, information_matrix, noise_variance
from ..types import CalibResult, Detection, View
from .common import A_FAMILY, Job, heldout_views, metric_row

__all__ = ["run_job", "B_MIN_CORNERS", "BASELINE_STREAM", "NBV_STREAM", "template_b1", "grid_b4", "B2App"]

B_MIN_CORNERS = 6  # B-policy acceptance: det.ok and >= 6 corners (= the loop's shadow track)
BASELINE_STREAM = agent_loop.BASELINE_STREAM
NBV_STREAM = agent_loop.NBV_STREAM
RANDOM_NBV_STREAM = 11
B1_JITTER_STREAM = 7
B4_JITTER_STREAM = 8
NO_DEADLINE_S = 1e5
MAX_CALLS = 400


# ---- sources -------------------------------------------------------------------------------------

class RecordingSource(SimSource):
    """``SimSource`` that remembers the last observation (for corruption tracking) and can replace
    the detected corners by those of a sagged board (spec 6.3 warp, corner level)."""

    def __init__(self, simulator, sag_m: float | None = None):
        super().__init__(simulator)
        self.sag_m = sag_m
        self.last: Observation | None = None

    def capture(self, rvec, tvec, **kw):
        o = super().capture(rvec, tvec, **kw)
        if self.sag_m:
            from ..sim.warp import sag_object_points

            d = o.det
            if len(d.ids):
                cam = self.camera(kw["n_accepted"])
                p, _ = cam.project_checked(sag_object_points(self.sag_m)[np.asarray(d.ids).astype(int)],
                                           o.truth.rvec, o.truth.tvec)
                p = p + np.random.default_rng([self.sim.camera_seed, self.sim.seed, kw["attempt"], 99]).normal(0, 0.15, p.shape)
                o = Observation(o.image, Detection(d.ids, p, d.obj, d.n_markers, d.ok), np.zeros(0, int), o.truth, o.ms)
        self.last = o
        return o


# ---- ablation patches (module globals of agent.loop, restored after the run) -----------------------

def _gate_detection_only(det, q, th):
    """A-gate: reject only a failed detection (the B0 acceptance rule)."""
    n = int(len(det.ids))
    rows, cols = corner_span(det.ids)
    if not det.ok or n < B_MIN_CORNERS or rows < 2 or cols < 2:
        return Finding("R0_NO_BOARD", f"{n} corners (detection failed)", "reject", "n_corners", float(n),
                       "Board not found — bring the whole board into view")
    return None


def _make_post_accept_no_r7(orig):
    def post_accept(**kw):
        kw["skip"] = tuple(kw.get("skip", ())) + ("R7_OUTLIER",)
        return orig(**kw)
    return post_accept


def _random_proposals(state, calib, size, th=None, rng=None, k=5, deadline=None, **kw):
    """A-nbv: the feasible candidates of the NBV sample, in a random order."""
    allc = propose_next_views(state, calib, size, th, rng, k=10 ** 6, deadline=None, **kw)
    out = CandidateList()
    for name in ("n_sampled", "n_feasible", "n_evaluated", "u95_now", "ms", "stopped_by_deadline", "full_board_only"):
        setattr(out, name, getattr(allc, name))
    if not allc:
        return out
    r2 = np.random.default_rng([int(state.camera_seed), int(state.seed), int(state.attempt), RANDOM_NBV_STREAM])
    for rank, j in enumerate(r2.permutation(len(allc))[:k], start=1):
        out.append(dataclasses.replace(allc[j], rank=rank))
    return out


class _UncHook:
    """Wraps ``agent.loop.uncertainty_map`` to keep the CalibResult (with covariance) it was given."""

    def __init__(self, orig):
        self.orig = orig
        self.calib = None
        self.unc = None

    def __call__(self, calib, views, size, stopcal=None, **kw):
        u = self.orig(calib, views, size, stopcal, **kw)
        self.calib, self.unc = calib, u
        return u


@contextmanager
def _patched(policy: str):
    saved = {k: getattr(agent_loop, k) for k in ("gate", "post_accept", "propose_next_views", "uncertainty_map")}
    hook = _UncHook(saved["uncertainty_map"])
    try:
        agent_loop.uncertainty_map = hook
        if policy == "A-gate":
            agent_loop.gate = _gate_detection_only
        elif policy == "A-outlier":
            agent_loop.post_accept = _make_post_accept_no_r7(saved["post_accept"])
        elif policy == "A-nbv":
            agent_loop.propose_next_views = _random_proposals
        yield hook
    finally:
        for k, v in saved.items():
            setattr(agent_loop, k, v)


# ---- the agent runs ------------------------------------------------------------------------------

def _decision(job: Job, kind: str) -> str:
    over = (job.variant or {}).get("approvals", {})
    if kind in over:
        return over[kind]
    if job.policy == "A-model" and kind in ("switch_model", "use_ro"):
        return "deny"
    return "approve"


def _stable_and_flags(st, calib, th) -> tuple[bool, int]:
    """The R11 prequential-stability test and the open-flag count, exactly as ``policy.decide``."""
    pre = [float(v.prequential_rms) for v in st.views[-2:] if v.prequential_rms is not None and math.isfinite(v.prequential_rms)]
    rms = None if calib is None else float(calib.rms)
    stable = len(pre) == 2 and rms is not None and all(x <= th.prequential_ratio * rms for x in pre)
    return stable, sum(1 for f in st.flags if f in OPEN_FLAGS)


def _r7_dropped(st) -> set[int]:
    """Ids of the views R7 has dropped from the set so far."""
    return {int(r["view_id"]) for r in st.rejected if r.get("rule") == "R7_OUTLIER" and r.get("view_id") is not None}


def run_agent(job: Job, th, stopcal) -> tuple[list[dict], dict]:
    t_start = time.perf_counter()
    th_run = th.replace(nbv_time_budget_s=NO_DEADLINE_S)
    th_nostop = th_run.replace(stop_p_exceed=-1.0)
    sim = make_simulator(job.family, job.actuator, job.camera, job.seed, job.nuisances)
    src = RecordingSource(sim, sag_m=(job.variant or {}).get("sag_m"))
    held = heldout_views(sim.camera)
    st = new_session(mode="sim", preset=None, actuator=job.actuator, seed=job.seed, camera_seed=job.camera,
                     budget=job.budget, th=th, baseline=False, now=0.0, sid=f"bench_{job.camera}_{job.seed}")
    failure = job.split == "failure"
    rows: list[dict] = []
    approvals, call_ms = [], []
    corrupted_ids: set[int] = set()
    restart_cleared: set[int] = set()
    stop = None
    end = None
    th_cur = th_run
    recorded = 0
    with _patched(job.policy) as hook:
        calls = 0
        while calls < MAX_CALLS and not st.terminal:
            card = st.pending_approval
            t = time.perf_counter()
            if card is None:
                # the failure split runs until the agent's own terminal card (R10, R12 on views
                # or attempts, R6, or a clean R11 stop), so abstention is measured, not truncated
                if not failure and st.n_accepted_total >= job.budget:
                    end = end or {"rule": "BUDGET_REACHED", "status": st.status}
                    break
                res = run_step(st, src, th=th_cur, stopcal=stopcal, deadline=time.perf_counter() + NO_DEADLINE_S)
                if res.accepted and src.last is not None and len(src.last.moved_ids):
                    # the corrupted view entered the set as id next_view_id - 1; R7 may already have
                    # dropped it in this same step, so it need not still be the last view
                    vid = st.next_view_id - 1
                    if any(int(v.id) == vid for v in st.views) or vid in _r7_dropped(st):
                        corrupted_ids.add(vid)
            else:
                kind, rule = card["kind"], card.get("rule")
                n_now = st.n_accepted_total
                if kind == "export":
                    if stop is None:
                        stop = {"n": n_now, "attempts": st.attempt, "views": len(st.views), "step": st.step}
                    if failure:
                        end = {"rule": rule, "status": "stopped", "kind": kind}
                        break
                    d = "deny"
                    th_cur = th_nostop
                elif kind in ("accept_with_warnings", "out_of_scope") or (failure and kind == "drift_restart"):
                    end = {"rule": rule, "status": st.status, "kind": kind}
                    if failure:
                        end.update(n=n_now, views=len(st.views), attempts=st.attempts)
                    if recorded < n_now:
                        rows.append(_agent_row(job, st, hook, sim, held, th_cur))
                        recorded = n_now
                    break
                else:
                    d = _decision(job, kind)
                allowed = [o["decision"] for o in card["options"]]
                if d not in allowed:
                    d = allowed[0]
                if kind == "drift_restart" and d == "approve":
                    restart_cleared.update(int(v.id) for v in st.views)
                approvals.append({"rule": rule, "kind": kind, "decision": d, "n": n_now})
                res = apply_approval(st, card["id"], d, "bench", src, th=th_cur, stopcal=stopcal,
                                     deadline=time.perf_counter() + NO_DEADLINE_S, now=0.0)
            call_ms.append(1000 * (time.perf_counter() - t))
            calls += 1
            # a frame's row is written once its approvals are resolved (the R11 export card is
            # resolved by a deny that does not change the fit, so the row is written before it)
            if st.n_accepted_total > recorded and (st.pending_approval is None
                                                  or st.pending_approval["kind"] == "export"):
                rows.append(_agent_row(job, st, hook, sim, held, th_cur))
                recorded = st.n_accepted_total
            if st.attempts >= st.max_attempts and st.pending_approval is None and not st.terminal:
                end = end or {"rule": "ATTEMPTS", "status": st.status}
                break
    if end is None:
        end = {"rule": "TERMINAL" if st.terminal else "MAX_CALLS", "status": st.status}
    dropped = _r7_dropped(st)
    rej = {}
    for r in st.rejected:
        rej[r["rule"]] = rej.get(r["rule"], 0) + 1
    sess = {
        "split": job.split, "camera": job.camera, "family": job.family, "seed": job.seed, "actuator": job.actuator,
        "policy": job.policy, "case": (job.variant or {}).get("case"), "n_final": st.n_accepted_total,
        "attempts_final": st.attempt, "views_final": len(st.views), "model_final": st.model, "flags_final": list(st.flags),
        "stop": stop, "end": end, "approvals": approvals, "rejected": rej,
        "corrupted_accepted": len(corrupted_ids), "corrupted_dropped": len(corrupted_ids & dropped),
        "clean_dropped": len(dropped - corrupted_ids),
        "corrupted_cleared_by_restart": len((corrupted_ids & restart_cleared) - dropped), "calls": len(call_ms),
        "call_ms_median": round(float(np.median(call_ms)), 1) if call_ms else None,
        "call_ms_max": round(float(np.max(call_ms)), 1) if call_ms else None,
        "wall_s": round(time.perf_counter() - t_start, 2), "trace": list(st.trace),
        # predicted vs realised U95 drop (and true p95 drop) per accepted NBV view (spec 11.5 item 9)
        "realised": [dict(h["realised"], views=h.get("views")) for h in st.history if h.get("realised")
                     and h["realised"].get("pred_gain") is not None],
    }
    return rows, sess


def _agent_row(job, st, hook, sim, held, th) -> dict:
    calib = hook.calib if len(st.views) >= 3 else None
    stable, nflags = _stable_and_flags(st, calib, th)
    pex = None if hook.unc is None or calib is None else hook.unc.p_exceed
    return metric_row(job, st.n_accepted_total, st.attempt, len(st.views), sim.camera_at(st.n_accepted_total), calib, held,
                      grid=tuple(th.grid_uncertainty), p_exceed_loop=pex, stable=stable, open_flags=nflags)


# ---- baselines -------------------------------------------------------------------------------------

def template_b1() -> list[tuple[float, float, float, float, float, float]]:
    """The 12-pose textbook template: (centre x frac, centre y frac, s, tilt, azimuth, roll).

    centre frontal; 4 corners frontal at s = 0.4; centre +-30 deg yaw; centre +-30 deg pitch;
    near s = 0.8; far s = 0.3; centre rolled 45 deg. Azimuth 0/180 tilts about the vertical axis
    (yaw), 90/270 about the horizontal axis (pitch) in the ``geometry`` convention.
    """
    c, s = 0.5, 0.5
    return [
        (c, c, s, 0.0, 0.0, 0.0),
        (0.21, 0.25, 0.4, 0.0, 0.0, 0.0), (0.79, 0.25, 0.4, 0.0, 0.0, 0.0),
        (0.21, 0.75, 0.4, 0.0, 0.0, 0.0), (0.79, 0.75, 0.4, 0.0, 0.0, 0.0),
        (c, c, s, 30.0, 0.0, 0.0), (c, c, s, 30.0, 180.0, 0.0),
        (c, c, s, 30.0, 90.0, 0.0), (c, c, s, 30.0, 270.0, 0.0),
        (c, c, 0.8, 0.0, 0.0, 0.0), (c, c, 0.3, 0.0, 0.0, 0.0),
        (c, c, s, 0.0, 0.0, 45.0),
    ]


def grid_b4() -> list[tuple[float, float, float, float, float, float]]:
    """45 poses: 5 x 3 cell centres x {frontal, yaw 30, pitch 30}, s = 0.5, farthest-point order.

    Distance = sqrt(dx^2 + dy^2 + do^2), centres in image fractions and orientation coordinates
    frontal (0, 0), yaw (0.5, 0), pitch (0, 0.5); starts at the centre frontal pose, ties go to the
    lower index.
    """
    pts, feats = [], []
    for j in range(3):
        for i in range(5):
            x, y = (i + 0.5) / 5, (j + 0.5) / 3
            for tilt, az, o in ((0.0, 0.0, (0.0, 0.0)), (30.0, 0.0, (0.5, 0.0)), (30.0, 90.0, (0.0, 0.5))):
                pts.append((x, y, 0.5, tilt, az, 0.0))
                feats.append((x, y, o[0], o[1]))
    F = np.asarray(feats)
    start = int(np.argmin(np.hypot(F[:, 0] - 0.5, F[:, 1] - 0.5) + F[:, 2] + F[:, 3]))
    order = [start]
    dmin = np.linalg.norm(F - F[start], axis=1)
    while len(order) < len(pts):
        dmin[order] = -1.0
        k = int(np.argmax(dmin))
        order.append(k)
        dmin = np.minimum(dmin, np.linalg.norm(F - F[k], axis=1))
    return [pts[k] for k in order]


def _jitter(p, rng):
    x, y, s, tilt, az, roll = p
    u = rng.uniform(-1.0, 1.0, 5)
    return (float(np.clip(x + 0.05 * u[0], 0.05, 0.95)), float(np.clip(y + 0.05 * u[1], 0.05, 0.95)),
            s * (1.0 + 0.1 * u[2]), max(0.0, tilt + 5.0 * u[3]), az, roll + 5.0 * u[4])


def _cell_counts(views, size, grid=(16, 9)) -> np.ndarray:
    cols, rows = grid
    W, H = size
    counts = np.zeros(rows * cols, int)
    for v in views:
        c = np.asarray(v.corners, float).reshape(-1, 2)
        ix = np.clip((c[:, 0] / W * cols).astype(int), 0, cols - 1)
        iy = np.clip((c[:, 1] / H * rows).astype(int), 0, rows - 1)
        np.add.at(counts, iy * cols + ix, 1)
    return counts


def _b3_target(views, calib, size, th, rng):
    """B3: the sampled lattice candidate whose predicted visible corners fall in the most 16 x 9
    cells that hold no accepted corner yet (ties: sum of 1 / (1 + count) over its cells)."""
    from ..tools import nbv

    K, dist = np.asarray(calib.K, float), np.asarray(calib.dist, float).ravel()
    lat = lattice(size)
    idx = rng.choice(len(lat), size=min(int(th.nbv_candidates), len(lat)), replace=False)
    pts = lat[idx]
    rolls = rng.uniform(-nbv.ROLL_MAX_DEG, nbv.ROLL_MAX_DEG, len(idx))
    R, t = nbv._poses_from_points(pts, rolls, K, dist, size)
    px, ok, _J, vis = nbv._evaluate(R, t, object_points(), K, dist, size, float(th.nbv_margin_px))
    good = (vis.mean(axis=1) >= float(th.nbv_min_visible_frac)) & (pts[:, 3] < float(th.max_tilt_deg))
    if good.any():
        area = np.zeros(len(idx))
        area[good] = nbv._area_fracs(px[good], ok[good], vis[good], size)
        good &= area >= float(th.min_board_area_frac)
    counts = _cell_counts(views, size)
    cols, rows = 16, 9
    W, H = size
    best, best_key = None, None
    for i in np.nonzero(good)[0]:
        c = px[i, :N_CORNERS][vis[i]]
        ix = np.clip((c[:, 0] / W * cols).astype(int), 0, cols - 1)
        iy = np.clip((c[:, 1] / H * rows).astype(int), 0, rows - 1)
        cells = np.unique(iy * cols + ix)
        key = (int(np.sum(counts[cells] == 0)), float(np.sum(1.0 / (1.0 + counts[cells]))))
        if best_key is None or key > best_key:
            best, best_key = i, key
    if best is None:
        return None
    rv, _ = cv2.Rodrigues(R[best])
    return rv.ravel(), t[best].copy()


class B2App:
    """OpenCV ``apps/interactive-calibration`` (tag 5.0.0) rules, replicated over a pose stream.

    Sources (https://github.com/opencv/opencv/tree/5.0.0/apps/interactive-calibration):

    * Frame acceptance: a ChArUco frame is a template when more than 3 corners are detected
      (frameProcessor.cpp:85). The stability test - board centre moving < diag/20 over
      ``captureDelay * fps`` frames (frameProcessor.cpp:271-272, 341-344) - always passes here:
      each simulated capture is one still frame.
    * Bad-angle rejection: after saving, ``checkLastFrame`` solves PnP (default iterative) with the
      current camera matrix, or fx = fy = 20000 with cx = H/2, cy = W/2 before the first
      calibration (frameProcessor.cpp:237-247, swapped as in the source), converts with
      ``RodriguesToEuler`` (pitch = atan2(-R12, R22), yaw = asin(R02); rotationConverters.cpp:38-42)
      and drops the frame when |pitch| > 40 or |yaw| > 40 deg (frameProcessor.cpp:250-259).
    * Calibration after every captured frame (calibration_step = 1, calibCommon.hpp:106;
      main.cpp:189-205): ``calibrateCamera`` with the controller's current flags, no initial flags
      (main.cpp:136), criteria COUNT+EPS 30 / 1e-7 (calibCommon.hpp:128-129; main.cpp:131-132),
      camera matrix and distortion passed in and out (main.cpp:195-200).
    * Flag auto-tuning (``ft`` = true, main.cpp:59) once frames > min_frames_num = 10
      (calibController.cpp:84-122, calibCommon.hpp:109): fix aspect ratio when |fx - fy| < 3 sigma
      of both; zero tangential distortion when |p1|, |p2| < 0.005; fix k1, k2, k3 when |k| < 0.005.
    * Worst-frame dropping: once frames >= max_frames_num = 30 the frame maximising
      alpha * per-view error + (1 - alpha) * grid-quality delta is removed, alpha = 0.1
      (calibController.cpp:201-239, calibCommon.hpp:108, 134). With a 30-frame budget this
      first acts after the 30th frame, so it never changes a recorded calibration here.
    * Stop ("calibration OK"): frames > 10 and 1.96-sigma relative confidence of fx, fy, cx, cy
      < 5 % with every distortion std <= |coefficient| (calibController.cpp:62-79) and RMS < 0.5 px
      (calibController.cpp:143-146) and coverage quality mean/std over a 10 x 10 grid > 1.8
      (calibController.cpp:18-45, 81-82); all four (calibController.cpp:126-131).
    """

    MIN_FRAMES = 10
    MAX_FRAMES = 30
    ALPHA = 0.1
    SIGMA_MULT = 1.96
    BAD_ANGLE = 40.0
    CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-7)

    def __init__(self, size):
        self.size = (int(size[0]), int(size[1]))
        self.flags = 0
        self.K = None
        self.dist = None
        self.std = None
        self.rms = math.nan
        self.per_view = None
        self.rvecs = self.tvecs = None
        self.coverage_ok = False
        self.conf_ok = False
        self.stop_n = None

    def angle_ok(self, obj, img) -> bool:
        W, H = self.size
        if self.K is None:
            K = np.eye(3)
            K[0, 0] = K[1, 1] = 20000.0
            K[0, 2], K[1, 2] = H // 2, W // 2
            d = np.zeros(5)
        else:
            K, d = self.K, self.dist
        ok, r, _t = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64), K, d)
        if not ok:
            return False
        R, _ = cv2.Rodrigues(r)
        if R[0, 2] < -0.998:
            pitch, yaw = -math.atan2(R[1, 0], R[1, 1]), -math.pi / 2
        elif R[0, 2] > 0.998:
            pitch, yaw = math.atan2(R[1, 0], R[1, 1]), math.pi / 2
        else:
            pitch, yaw = math.atan2(-R[1, 2], R[2, 2]), math.asin(R[0, 2])
        return abs(math.degrees(pitch)) <= self.BAD_ANGLE and abs(math.degrees(yaw)) <= self.BAD_ANGLE

    def _grid_quality(self, views, exclude=None) -> float:
        W, H = self.size
        g = 10
        xs, ys = W // g, H // g
        counts = np.zeros(g * g, int)
        for k, v in enumerate(views):
            if k == exclude:
                continue
            c = np.asarray(v.corners, float).reshape(-1, 2)
            i = np.clip((c[:, 0] / xs).astype(int), 0, g - 1)
            j = np.clip((c[:, 1] / ys).astype(int), 0, g - 1)
            np.add.at(counts, i * g + j, 1)
        return float(counts.mean() / (counts.std() + 1e-7))

    def calibrate(self, views) -> bool:
        obj = [object_points()[np.asarray(v.corner_ids, int)].astype(np.float32) for v in views]
        img = [np.asarray(v.corners, np.float32).reshape(-1, 1, 2) for v in views]
        K = np.eye(3) if self.K is None else self.K.copy()
        d = np.zeros(5) if self.dist is None else self.dist.copy()
        try:
            out = cv2.calibrateCameraExtended(obj, img, self.size, K, d, flags=self.flags, criteria=self.CRITERIA)
        except cv2.error:
            return False
        rms, K, d, rvecs, tvecs, std_intr, _std_ext, per_view = out
        if not (np.all(np.isfinite(K)) and np.all(np.isfinite(d))):
            return False
        self.rms, self.K, self.dist = float(rms), np.asarray(K, float), np.asarray(d, float).ravel()[:5]
        self.std = np.asarray(std_intr, float).ravel()
        self.rvecs = [np.asarray(r, float).ravel() for r in rvecs]
        self.tvecs = [np.asarray(t, float).ravel() for t in tvecs]
        self.per_view = np.asarray(per_view, float).ravel()
        self._update_state(len(views), views)
        return True

    def _update_state(self, n, views):
        K, d, s, eps = self.K, self.dist, self.std, 0.05
        f_ok = (self.SIGMA_MULT * s[0] / K[0, 0] < eps) and (self.SIGMA_MULT * s[1] / K[1, 1] < eps)
        c_ok = (self.SIGMA_MULT * s[2] / K[0, 2] < eps) and (self.SIGMA_MULT * s[3] / K[1, 2] < eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = s[4:9] / np.abs(d[:5])
        d_ok = not bool(np.any(ratio > 1))  # NaN compares False, as in C++
        self.conf_ok = bool(f_ok and c_ok and d_ok)
        frames_ok = n > self.MIN_FRAMES
        if frames_ok:
            self.coverage_ok = self._grid_quality(views) > 1.8
            if not (self.flags & cv2.CALIB_FIX_ASPECT_RATIO):
                fdiff = abs(K[0, 0] - K[1, 1])
                if fdiff < 3 * s[0] and fdiff < 3 * s[1]:
                    self.flags |= cv2.CALIB_FIX_ASPECT_RATIO
                    self.K[0, 0] = self.K[1, 1]
            if not (self.flags & cv2.CALIB_ZERO_TANGENT_DIST) and abs(d[2]) < 0.005 and abs(d[3]) < 0.005:
                self.flags |= cv2.CALIB_ZERO_TANGENT_DIST
            if not (self.flags & cv2.CALIB_FIX_K1) and abs(d[0]) < 0.005:
                self.flags |= cv2.CALIB_FIX_K1
            if not (self.flags & cv2.CALIB_FIX_K2) and abs(d[1]) < 0.005:
                self.flags |= cv2.CALIB_FIX_K2
            if not (self.flags & cv2.CALIB_FIX_K3) and abs(d[4]) < 0.005:
                self.flags |= cv2.CALIB_FIX_K3
        if self.stop_n is None and frames_ok and self.conf_ok and self.rms < 0.5 and self.coverage_ok:
            self.stop_n = n

    def adopt(self, calib, views) -> None:
        """B2-k5: take the bench's k5 fit as the app's current estimate, then run the app's state
        update (flag auto-tuning is recorded but not used by the bench solver)."""
        if calib is None:
            return
        self.rms, self.K, self.dist = float(calib.rms), np.asarray(calib.K, float).copy(), np.asarray(calib.dist, float).ravel()[:5]
        std = np.zeros(9)
        s = np.asarray(calib.std_intr, float).ravel()
        std[: min(9, s.size)] = s[:9]
        self.std = std
        self.per_view = np.asarray(calib.per_view, float).ravel()
        self._update_state(len(views), views)

    def filter_frames(self, views) -> list:
        n = len(views)
        if n < self.MAX_FRAMES or self.per_view is None or len(self.per_view) != n:
            return views
        max_q = self._grid_quality(views, exclude=n)
        vals = [self.per_view[i] * self.ALPHA + (self._grid_quality(views, exclude=i) - max_q) * (1 - self.ALPHA)
                for i in range(n)]
        worst = int(np.argmax(vals))
        return views[:worst] + views[worst + 1:]

    def calib_result(self, views) -> CalibResult | None:
        """The app's estimate as a CalibResult, with a covariance built the same way as ``calibrate``."""
        if self.K is None or len(views) < 3:
            return None
        objs = [object_points()[np.asarray(v.corner_ids, int)] for v in views]
        F = information_matrix(self.K, self.dist, self.rvecs, self.tvecs, objs)
        n_pts = int(sum(len(o) for o in objs))
        s2 = noise_variance(self.rms, n_pts, 9, len(views), 0)
        cov = covariance(F, s2) if math.isfinite(s2) else None
        return CalibResult(model="k5", K=self.K.copy(), dist=self.dist.copy(), rms=self.rms, per_view=self.per_view,
                           rvecs=self.rvecs, tvecs=self.tvecs, std_intr=self.std, sigma2=s2, F=F, cov=cov)


def run_baseline(job: Job, th) -> tuple[list[dict], dict]:
    t_start = time.perf_counter()
    sim = make_simulator(job.family, job.actuator, job.camera, job.seed, job.nuisances)
    held = heldout_views(sim.camera)
    size = tuple(sim.camera.size)
    max_attempts = int(th.max_attempts)
    views: list[View] = []
    calib = None
    focal_obs: list[float] = []
    prev = (None, None)
    n_acc = attempt = 0
    rows: list[dict] = []
    template = template_b1() if job.policy == "B1" else grid_b4() if job.policy == "B4" else None
    app = B2App(size) if job.policy in ("B2", "B2-k5") else None
    app_k5 = job.policy == "B2-k5"
    app_views: list[View] = []
    corrupted = rejected = 0
    while n_acc < job.budget and attempt < max_attempts:
        cam = sim.camera_at(n_acc)
        f_used = None
        if template is not None:
            p = template[attempt % len(template)]
            if attempt >= len(template):
                salt = B1_JITTER_STREAM if job.policy == "B1" else B4_JITTER_STREAM
                p = _jitter(p, np.random.default_rng([job.camera, job.seed, attempt, salt]))
            W, H = size
            try:
                rvec, tvec = pose_for_view(cam, (p[0] * (W - 1), p[1] * (H - 1)), p[2], p[3], p[4], p[5])
            except ValueError:
                attempt += 1
                rejected += 1
                continue
        elif len(views) < 3:
            f = float(np.median(focal_obs)) if focal_obs else None
            rvec, tvec = opening_target(len(views), job.camera, job.seed, size, f)
            f_used = float(k_initial(size, f)[0, 0])
        elif job.policy == "B3":
            tgt = _b3_target(views, calib, size, th, np.random.default_rng([job.camera, job.seed, attempt, NBV_STREAM]))
            if tgt is None:
                try:
                    tgt = random_pose(cam, np.random.default_rng([job.camera, job.seed, attempt, BASELINE_STREAM]))
                except RuntimeError:
                    attempt += 1
                    rejected += 1
                    continue
            rvec, tvec = tgt
        else:  # B0, B2: the simulator's random feasible pose stream
            try:
                rvec, tvec = random_pose(cam, np.random.default_rng([job.camera, job.seed, attempt, BASELINE_STREAM]))
            except RuntimeError:
                attempt += 1
                rejected += 1
                continue
        fr = sim.capture(rvec, tvec, attempt=attempt, n_accepted=n_acc, prev_rvec=prev[0], prev_tvec=prev[1])
        attempt += 1
        prev = (fr.truth.rvec, fr.truth.tvec)
        det, moved = fr.observe()
        if f_used is not None and len(det.ids) >= 4:
            fo = focal_from_detection(det.ids, det.corners, rvec, tvec, f_used, size)
            if fo is not None:
                focal_obs = (focal_obs + [round(fo, 3)])[-10:]
        ids = np.asarray(det.ids, np.int32)
        if app is not None:
            accept = len(ids) > 3 and det.ok
        else:
            accept = bool(det.ok and len(ids) >= B_MIN_CORNERS)
        if not accept:
            rejected += 1
            continue
        view = View(n_acc, ids, round_corners(det.corners), None, np.asarray(rvec, float), np.asarray(tvec, float))
        if app is not None:
            if not app.angle_ok(object_points()[ids], view.corners):
                rejected += 1
                continue
        if app is not None and not app_k5:
            cand = app_views + [view]
            if not app.calibrate(cand) and len(cand) >= 3:
                rejected += 1
                continue
            app_views = app.filter_frames(cand)
            views.append(view)
            n_acc += 1
            corrupted += int(len(moved) > 0)
            rows.append(metric_row(job, n_acc, attempt, len(app_views), sim.camera_at(n_acc), app.calib_result(cand), held,
                                   grid=tuple(th.grid_uncertainty)))
            continue
        cand = views + [view]
        new = None
        if len(cand) >= 3:
            try:
                new = calibrate(cand, "k5", size, init=calib)
            except (ValueError, cv2.error):
                rejected += 1
                continue
        if app_k5:  # B2-k5: the app's state machine (angle check, stop) fed by the bench k5 fit
            app.adopt(new, cand)
        views = cand
        calib = new
        n_acc += 1
        corrupted += int(len(moved) > 0)
        rows.append(metric_row(job, n_acc, attempt, len(views), sim.camera_at(n_acc), calib, held,
                               grid=tuple(th.grid_uncertainty)))
    sess = {
        "split": job.split, "camera": job.camera, "family": job.family, "seed": job.seed, "actuator": job.actuator,
        "policy": job.policy, "case": (job.variant or {}).get("case"), "n_final": n_acc, "attempts_final": attempt,
        "views_final": len(app_views) if app is not None else len(views), "model_final": {"name": "k5"},
        "stop": None if app is None or app.stop_n is None else {"n": app.stop_n, "rule": "B2_APP_OK"},
        "end": {"rule": "BUDGET_REACHED" if n_acc >= job.budget else "ATTEMPTS"}, "rejected": {"not_accepted": rejected},
        "corrupted_accepted": corrupted, "wall_s": round(time.perf_counter() - t_start, 2),
        "b2_flags_final": None if app is None else int(app.flags),
    }
    return rows, sess


def run_job(job: Job, th, stopcal) -> tuple[list[dict], dict]:
    """Run one job; never raises (an error is returned in the session record)."""
    try:
        if job.policy in A_FAMILY:
            return run_agent(job, th, stopcal)
        return run_baseline(job, th)
    except Exception as e:  # noqa: BLE001 - one failed run must not stop a 6000-run benchmark
        import traceback

        return [], {"split": job.split, "camera": job.camera, "family": job.family, "seed": job.seed,
                    "actuator": job.actuator, "policy": job.policy, "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-2000:]}
