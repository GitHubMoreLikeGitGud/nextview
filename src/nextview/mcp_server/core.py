"""In-process session hub behind the MCP server (spec section 9).

Every method calls the same ``nextview.tools`` / ``nextview.agent`` functions as the HTTP API and
the benchmark. The hub only keeps the per-session objects between tool calls (the agent's
``SessionState``, the simulator and the last few captured frames) and turns results into small
JSON payloads with a one-line summary, so a tool-calling client chooses the actions while the
tools do all the measurement and maths.

A method returns a ``Reply``, raises ``Refusal`` (a well-formed call that the session state does
not allow, with a hint), or, for the gated actions (a lens-model change, an export), returns an
``ApprovalRequest`` that the server resolves through MCP elicitation before anything changes.
"""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..agent.approvals import ApprovalKind, make_card
from ..agent.instructions import describe_target, position_word, short_desc
from ..agent.policy import decide, gate, perception, post_accept, warp_seek
from ..agent.state import SessionState, jsonable, new_session, round_corners
from ..config import StopCalibration, load_presets, load_stop_calibration, load_thresholds
from ..geometry import ghost_quad
from ..tools._views import full_board_views
from ..tools.calibrate import calibrate as fit_calibration
from ..tools.calibrate import cv_heldout_rms, prequential_rms, ro_cv_gain
from ..tools.detect import corner_span
from ..tools.detect import detect_board as detect
from ..tools.export import export_calibration as export_bytes
from ..tools.export import intrinsic_std
from ..tools.nbv import focal_from_detection, k_initial, opening_target
from ..tools.nbv import propose_next_views as plan_views
from ..tools.pose import estimate_pose
from ..tools.quality import assess_frame as assess
from ..tools.residuals import outlier_views
from ..tools.residuals import residual_analysis as analyse
from ..tools.uncertainty import uncertainty_map as unc_map
from ..types import CalibResult, View

__all__ = ["Refusal", "Reply", "ApprovalRequest", "Hub", "identity_stopcal", "load_stopcal", "MODELS"]

NBV_STREAM = 3  # the agent loop's NBV random stream: equal states give equal candidate lists
MAX_FRAMES = 8  # captured frames kept per session (about 1 MB each)
COARSE_GRID = (8, 5)  # (cols, rows) of the uncertainty grid returned to the client
MODELS = ("k5", "rational", "ro")
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class Refusal(Exception):
    """A well-formed call the session state does not allow."""

    def __init__(self, code: str, message: str, hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "hint": self.hint}}

    def text(self) -> str:
        return f"refused ({self.code}): {self.message}" + (f" Hint: {self.hint}" if self.hint else "")


@dataclass
class Reply:
    summary: str
    data: dict

    def payload(self) -> dict:
        return jsonable(self.data)

    def text(self) -> str:
        return self.summary + "\n" + json.dumps(self.payload(), separators=(",", ":"), ensure_ascii=False)


@dataclass
class ApprovalRequest:
    """A gated action waiting for a human decision (``resolve(decision, reason)`` applies it)."""

    session_id: str
    card: dict
    resolve: Callable[[str, str], Reply]

    def needs_approval(self) -> Reply:
        return Reply(f"needs_approval: {self.card['question']} This client cannot ask the approver, so nothing was changed.",
                     {"status": "needs_approval", "session_id": self.session_id,
                      "approval": {k: self.card[k] for k in ("id", "kind", "question", "options")}})


def identity_stopcal() -> StopCalibration:
    """P_exceed = 1 if U95 > tau else 0 (r = true_p95 / U95 taken as exactly 1). Exercises the
    stop rule before ``bench/fit_stop.py`` has fitted the real calibration; it is not one."""
    q = tuple([1.0] * 101)
    return StopCalibration(calibrated=True, quantiles={"robot": q, "human": q}, note="identity (r = 1), not a calibration")


def load_stopcal(spec: str | None) -> StopCalibration:
    """``"config"`` (or None): ``config/stop_calibration.json``; ``"identity"``; else a JSON path."""
    if spec in (None, "", "config"):
        return load_stop_calibration()
    if spec == "identity":
        return identity_stopcal()
    return load_stop_calibration(spec)


def _r(x, nd: int = 4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


@dataclass(eq=False)
class Frame:
    id: int
    image: np.ndarray | None
    det: object  # Detection
    quality: object  # Quality
    finding: object  # gate Finding or None
    target: dict
    truth: object | None
    step: int
    view_id: int | None = None


@dataclass(eq=False)
class Session:
    state: SessionState
    sim: object | None  # Simulator (sim mode)
    frames: OrderedDict = field(default_factory=OrderedDict)
    next_frame_id: int = 0
    calib: CalibResult | None = None
    calib_key: tuple | None = None
    plan_key: tuple | None = None
    captures: list = field(default_factory=list)  # evaluator log: one row per capture
    accepts: list = field(default_factory=list)  # evaluator log: one row per accepted view
    exported: dict | None = None

    @property
    def sid(self) -> str:
        return self.state.sid

    def view_key(self) -> tuple:
        return tuple(int(v.id) for v in self.state.views)


class Hub:
    """The MCP tools' state: ``sessions[session_id] -> Session``."""

    def __init__(self, th=None, stopcal: StopCalibration | str | None = None, presets=None,
                 export_dir: str | Path | None = None):
        self.th = load_thresholds() if th is None else th
        self.stopcal = stopcal if isinstance(stopcal, StopCalibration) else load_stopcal(stopcal)
        self.presets = load_presets() if presets is None else presets
        self.export_dir = None if export_dir is None else Path(export_dir)
        self.sessions: dict[str, Session] = {}

    # ---- lookups ----------------------------------------------------------------------------
    def _session(self, session_id: str, *, open_only: bool = True, sim: bool | None = None) -> Session:
        s = self.sessions.get(str(session_id).strip())
        if s is None:
            known = ", ".join(self.sessions) or "none"
            raise Refusal("UNKNOWN_SESSION", f"no session {session_id!r} (known: {known})",
                          "use the session_id returned by sim_new_session")
        if open_only and s.state.terminal:
            raise Refusal("SESSION_ENDED", f"session {session_id} has ended ({s.state.status})",
                          "start a new session to calibrate again")
        if sim is True and s.state.mode != "sim":
            raise Refusal("NOT_SIM", "this is an image session; sim tools need a sim session", "use observe_image")
        if sim is False and s.state.mode == "sim":
            raise Refusal("NOT_IMAGE", "this is a sim session; use sim_capture", None)
        return s

    def _frame(self, s: Session, frame_id) -> Frame:
        try:
            fid = int(frame_id)
        except (TypeError, ValueError):
            raise Refusal("BAD_ARGUMENT", f"frame_id must be an integer, got {frame_id!r}") from None
        f = s.frames.get(fid)
        if f is None:
            kept = list(s.frames)
            raise Refusal("UNKNOWN_FRAME", f"no frame {fid} in session {s.sid} (kept: {kept})",
                          "use the frame_id returned by the last capture")
        return f

    # ---- calibration and uncertainty of the current views --------------------------------------
    def _fit_model(self, s: Session) -> str:
        m = s.state.model["name"]
        if m == "ro" and len(full_board_views(s.state.views)) < 3:
            return "k5"  # RO needs 3 full-board views (as in the agent loop)
        return m

    def _calib(self, s: Session) -> CalibResult | None:
        """Calibration of the current views with the session model (refit only when the views or
        the model changed since the last fit)."""
        st = s.state
        if len(st.views) < 3:
            return None
        key = (s.view_key(), self._fit_model(s))
        if s.calib is not None and s.calib_key == key:
            return s.calib
        model = key[1]
        init = s.calib if (s.calib is not None and s.calib.model == model) else (
            None if s.calib is None else (s.calib.K, s.calib.dist))
        calib = fit_calibration(st.views, model, st.image_size, init=init)
        s.calib, s.calib_key = calib, key
        st.calib = {"model": calib.model, "K": calib.K.tolist(), "dist": calib.dist.tolist(), "rms": float(calib.rms)}
        return calib

    def _need_calib(self, s: Session) -> CalibResult:
        n = len(s.state.views)
        if n < 3:
            raise Refusal("NEEDS_VIEWS", f"{n} accepted view(s); a calibration needs at least 3",
                          "capture and accept more views (propose_next_views gives the opening poses)")
        try:
            return self._calib(s)
        except (ValueError, cv2.error) as e:
            raise Refusal("CALIBRATION_FAILED", f"calibration failed on the current views: {e}",
                          "drop the last view or capture a different pose") from None

    def _unc(self, s: Session, calib: CalibResult):
        st = s.state
        return unc_map(calib, st.views, st.image_size, self.stopcal, actuator=st.actuator, tau_px=st.tau,
                       grid=tuple(self.th.grid_uncertainty))

    def _decision(self, s: Session, calib: CalibResult | None, unc):
        st = s.state
        pex = math.nan if unc is None else float(unc.p_exceed)
        return decide(n_views=len(st.views), p_exceed=pex, u95=None if unc is None else float(unc.U95),
                      prequential_last=[v.prequential_rms for v in st.views[-2:]],
                      rms=None if calib is None else float(calib.rms), flags=st.flags, attempts=st.attempts,
                      budget=st.budget, max_attempts=st.max_attempts, th=self.th)

    @staticmethod
    def _calib_dict(c: CalibResult) -> dict:
        std = intrinsic_std(c)
        return {"model": c.model, "rms_px": _r(c.rms), "fx": _r(c.K[0, 0], 2), "fy": _r(c.K[1, 1], 2),
                "cx": _r(c.K[0, 2], 2), "cy": _r(c.K[1, 2], 2), "dist": [_r(x, 5) for x in c.dist],
                "std_fx_px": _r(std[0], 3) if len(std) else None}

    def _status(self, s: Session) -> dict:
        st = s.state
        return {"views": len(st.views), "budget": st.budget, "attempts": st.attempts, "model": st.model["name"]}

    # ---- targets ---------------------------------------------------------------------------
    @staticmethod
    def _target(st: SessionState, rvec, tvec, K, dist, source: str, **extra) -> dict:
        rvec, tvec = np.asarray(rvec, float).ravel(), np.asarray(tvec, float).ravel()
        sim = st.mode == "sim"
        return {"rvec": [round(float(x), 6) for x in rvec], "tvec": [round(float(x), 6) for x in tvec],
                "source": source, "desc": short_desc(rvec, tvec, K, dist, st.image_size),
                "instruction": describe_target(rvec, tvec, K, dist, st.image_size, sim=sim),
                "ghost_quad": [[round(float(a), 1), round(float(b), 1)] for a, b in ghost_quad(rvec, tvec, K, dist)],
                **extra}

    def _opening(self, s: Session) -> dict:
        st = s.state
        f = float(np.median(st.focal_obs)) if st.focal_obs else None
        k = len(st.views)
        rvec, tvec = opening_target(k, st.camera_seed, st.seed, st.image_size, f)
        K = k_initial(st.image_size, f)
        return self._target(st, rvec, tvec, K, np.zeros(5), "opening", rank=1, k=int(k),
                            focal=round(float(K[0, 0]), 3), pred_gain=None, pred_U95_after=None)

    @staticmethod
    def _cand_public(c: dict) -> dict:
        return {"rank": c.get("rank"), "desc": c.get("desc"), "pred_gain_px": c.get("pred_gain"),
                "pred_U95_after_px": c.get("pred_U95_after"), "effort": c.get("effort"), "score": c.get("score"),
                "instruction": c.get("instruction")}

    # ---- tools: sessions ------------------------------------------------------------------------
    def sim_new_session(self, preset: str = "inspection_gauge_cam", actuator: str | None = "robot", seed: int = 0,
                        budget: int = 30, camera_seed: int | None = None) -> Reply:
        from ..sim.scenarios import scenario_from_preset

        if preset not in self.presets:
            raise Refusal("BAD_ARGUMENT", f"unknown preset {preset!r}", f"one of {list(self.presets.ids)}")
        p = self.presets.get(preset)
        act = p.actuator if actuator in (None, "") else str(actuator)
        if act not in ("robot", "human"):
            raise Refusal("BAD_ARGUMENT", f"actuator must be 'robot' or 'human', got {actuator!r}")
        try:
            budget = int(budget)
            seed = int(seed)
            cam_seed = seed if camera_seed is None else int(camera_seed)
        except (TypeError, ValueError):
            raise Refusal("BAD_ARGUMENT", "budget, seed and camera_seed must be integers") from None
        if not 3 <= budget <= int(self.th.budget_frames):
            raise Refusal("BAD_ARGUMENT", f"budget must be 3..{self.th.budget_frames}, got {budget}")
        sc = scenario_from_preset(p, camera_seed=cam_seed, seed=seed, presets=self.presets, actuator=act)
        st = new_session(mode="sim", preset=p.id, actuator=act, seed=seed, camera_seed=cam_seed, budget=budget,
                         th=self.th, image_size=tuple(sc.camera.size), baseline=False)
        s = Session(st, sc.simulator)
        self.sessions[st.sid] = s
        st.add_trace(f"#0 SESSION: sim preset {p.id} ({p.family}, {act}), camera {cam_seed}, seed {seed}, budget {budget}")
        stop = ("P_exceed < 0.10 with stable prequential error and no open flags" if self.stopcal.calibrated
                else "stop calibration not fitted: P_exceed is not available, so stop at the budget")
        return Reply(f"session {st.sid} ready: {p.label}, {act} actuator, budget {budget} views. "
                     f"Next: propose_next_views or sim_capture.",
                     {"session_id": st.sid, "preset": {"id": p.id, "label": p.label, "family": p.family},
                      "actuator": act, "image_size": list(st.image_size), "budget": budget, "tau_px": st.tau,
                      "stop_rule": stop})

    def image_new_session(self, width: int = 1280, height: int = 720, actuator: str = "human", budget: int = 30) -> Reply:
        try:
            size = (int(width), int(height))
            budget = int(budget)
        except (TypeError, ValueError):
            raise Refusal("BAD_ARGUMENT", "width, height and budget must be integers") from None
        if not (64 <= size[0] <= 4096 and 64 <= size[1] <= 4096):
            raise Refusal("BAD_ARGUMENT", f"image size {size} out of range 64..4096")
        if actuator not in ("robot", "human"):
            raise Refusal("BAD_ARGUMENT", f"actuator must be 'robot' or 'human', got {actuator!r}")
        if not 3 <= budget <= int(self.th.budget_frames):
            raise Refusal("BAD_ARGUMENT", f"budget must be 3..{self.th.budget_frames}, got {budget}")
        st = new_session(mode="webcam", preset=None, actuator=actuator, seed=0, budget=budget, th=self.th,
                         image_size=size, baseline=False)
        self.sessions[st.sid] = Session(st, None)
        st.add_trace(f"#0 SESSION: image frames {size[0]}x{size[1]}, {actuator}, budget {budget}")
        return Reply(f"session {st.sid} ready for {size[0]}x{size[1]} image files. Next: propose_next_views, "
                     f"then observe_image with each new photo of the board.",
                     {"session_id": st.sid, "image_size": list(size), "budget": budget, "tau_px": st.tau})

    # ---- tools: perception ------------------------------------------------------------------------
    def _record_frame(self, s: Session, image, det, target: dict, truth, moved_n: int = 0) -> tuple[Frame, Reply]:
        st, th = s.state, self.th
        q = assess(image, det, anisotropy_threshold=th.motion_anisotropy)
        g = gate(det, q, th)
        if len(st.views) < 3:  # opening phase: refine the focal guess that places the opening poses
            if target.get("focal") is not None and len(det.ids) >= 4:
                fo = focal_from_detection(det.ids, det.corners, target["rvec"], target["tvec"], target["focal"],
                                          st.image_size)
                if fo is not None:
                    st.focal_obs = (st.focal_obs + [round(fo, 3)])[-10:]
        fid = s.next_frame_id
        s.next_frame_id += 1
        f = Frame(fid, image, det, q, g, target, truth, st.step)
        s.frames[fid] = f
        while len(s.frames) > MAX_FRAMES:
            s.frames.popitem(last=False)
        if g is not None:
            st.rejected.append({"step": st.step, "attempt": st.attempt - 1, "rule": g.rule, "metric": g.metric,
                                "value": _r(g.value)})
            st.add_trace(f"#{st.step} {g.rule}: {g.measurement} → reject frame {fid}")
        s.captures.append({"frame_id": fid, "step": st.step, "source": target.get("source"), "rank": target.get("rank"),
                           "chosen": target.get("chosen"), "gate": None if g is None else g.rule,
                           "n_corners": int(len(det.ids)), "views_before": len(st.views)})
        data = {"frame_id": fid, "session_id": s.sid, "target": {k: target.get(k) for k in ("source", "rank", "desc")},
                **self._detection_dict(det), "quality": self._quality_dict(q), "gate": self._gate_dict(g),
                **self._status(s)}
        if g is None:
            summ = (f"frame {fid}: {len(det.ids)} corners, edge rise {_r(q.edge_rise_px, 2)} px, board "
                    f"{_r(100 * q.board_area_frac, 1)}% of frame → passes the gate. Next: accept_view(frame_id={fid}).")
        else:
            summ = (f"frame {fid}: {g.rule} → rejected ({g.measurement}). Instruction: {g.instruction}. "
                    f"Do not accept it; capture again.")
        return f, Reply(summ, data)

    @staticmethod
    def _detection_dict(det) -> dict:
        rows, cols = corner_span(det.ids)
        return {"detection": {"n_corners": int(len(det.ids)), "n_markers": int(det.n_markers), "ok": bool(det.ok),
                              "rows": rows, "cols": cols}}

    @staticmethod
    def _quality_dict(q) -> dict:
        return {"edge_rise_px": _r(q.edge_rise_px, 3), "anisotropy": _r(q.anisotropy, 3),
                "blur_dir_deg": _r(q.blur_dir_deg, 1), "saturation_frac": _r(q.saturation_frac, 4),
                "board_area_frac": _r(q.board_area_frac, 4), "n_edges": int(q.n_edges)}

    @staticmethod
    def _gate_dict(g) -> dict:
        if g is None:
            return {"pass": True, "rule": None}
        return {"pass": False, "rule": g.rule, "measurement": g.measurement, "instruction": g.instruction}

    def sim_capture(self, session_id: str, candidate_rank: int | None = None, pose: dict | None = None) -> Reply:
        s = self._session(session_id, sim=True)
        st = s.state
        if st.attempts >= st.max_attempts:
            raise Refusal("ATTEMPT_CAP", f"{st.attempts} captures used (cap {st.max_attempts})",
                          "export_calibration (accept with warnings) or end the session")
        if pose is not None and candidate_rank is not None:
            raise Refusal("BAD_ARGUMENT", "pass candidate_rank or pose, not both")
        if pose is not None:
            target = self._pose_target(s, pose)
        else:
            target = self._planned_target(s, candidate_rank)
        st.step += 1
        st.last_target = target
        prev = st.sim_prev_pose
        fr = s.sim.capture(np.asarray(target["rvec"], float), np.asarray(target["tvec"], float), attempt=st.attempt,
                           n_accepted=st.n_accepted_total,
                           prev_rvec=None if prev is None else np.asarray(prev["rvec"], float),
                           prev_tvec=None if prev is None else np.asarray(prev["tvec"], float))
        det, moved = fr.observe()
        st.attempt += 1
        st.attempts += 1
        st.sim_prev_pose = {"rvec": fr.truth.rvec.tolist(), "tvec": fr.truth.tvec.tolist()}
        _, rep = self._record_frame(s, fr.image, det, target, fr.truth, len(moved))
        return rep

    def _pose_target(self, s: Session, pose) -> dict:
        if not isinstance(pose, dict) or "rvec" not in pose or "tvec" not in pose:
            raise Refusal("BAD_ARGUMENT", "pose must be {\"rvec\": [3 floats], \"tvec\": [3 floats]} (board in camera frame, m)")
        try:
            rv = np.asarray(pose["rvec"], float).ravel()
            tv = np.asarray(pose["tvec"], float).ravel()
        except (TypeError, ValueError):
            raise Refusal("BAD_ARGUMENT", "pose rvec/tvec must be numbers") from None
        if rv.size != 3 or tv.size != 3 or not (np.all(np.isfinite(rv)) and np.all(np.isfinite(tv))):
            raise Refusal("BAD_ARGUMENT", "pose rvec and tvec need 3 finite numbers each")
        if not 0.05 <= float(tv[2]) <= 10.0:
            raise Refusal("BAD_ARGUMENT", f"tvec z = {tv[2]:.3f} m is outside 0.05..10 m")
        c = s.calib
        K = c.K if c is not None else k_initial(s.state.image_size, None)
        d = c.dist if c is not None else np.zeros(5)
        return self._target(s.state, rv, tv, K, d, "pose", rank=None, chosen="pose", pred_gain=None)

    def _planned_target(self, s: Session, rank) -> dict:
        st = s.state
        chosen = "explicit"
        if rank is None:
            if not st.candidates and len(st.views) < 3:
                return dict(self._opening(s), chosen="default_opening")
            rank, chosen = 1, "default_rank1"
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            raise Refusal("BAD_ARGUMENT", f"candidate_rank must be an integer, got {rank!r}") from None
        if not st.candidates:
            raise Refusal("NO_PLAN", "there is no plan yet", "call propose_next_views first")
        if s.plan_key != s.view_key():
            raise Refusal("STALE_PLAN", "the candidate list was planned before the last change to the accepted views",
                          "call propose_next_views for a fresh plan, then sim_capture(candidate_rank=...)")
        if not 1 <= rank <= len(st.candidates):
            raise Refusal("BAD_ARGUMENT", f"candidate_rank must be 1..{len(st.candidates)}, got {rank}")
        return dict(st.candidates[rank - 1], chosen=chosen)

    def observe_image(self, session_id: str, path: str) -> Reply:
        s = self._session(session_id, sim=False)
        st = s.state
        p = Path(str(path))
        if not p.is_file():
            raise Refusal("BAD_ARGUMENT", f"no file at {str(path)!r}")
        if p.stat().st_size > MAX_IMAGE_BYTES:
            raise Refusal("BAD_ARGUMENT", f"file is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Refusal("BAD_ARGUMENT", f"{p.name} is not a readable image")
        h, w = img.shape[:2]
        if (w, h) != tuple(st.image_size):
            raise Refusal("BAD_ARGUMENT", f"image is {w}x{h}, session expects {st.image_size[0]}x{st.image_size[1]}")
        st.step += 1
        st.attempt += 1
        st.attempts += 1
        target = dict(st.last_target or {"source": "operator"}, chosen="operator")
        _, rep = self._record_frame(s, img, detect(img), target, None)
        return rep

    def detect_board(self, session_id: str, frame_id: int) -> Reply:
        s = self._session(session_id, open_only=False)
        f = self._frame(s, frame_id)
        d = self._detection_dict(f.det)
        n = d["detection"]["n_corners"]
        return Reply(f"frame {f.id}: {n} ChArUco corners from {f.det.n_markers} markers over "
                     f"{d['detection']['rows']}x{d['detection']['cols']} corner rows x cols.",
                     {"frame_id": f.id, **d})

    def assess_frame(self, session_id: str, frame_id: int) -> Reply:
        s = self._session(session_id, open_only=False)
        f = self._frame(s, frame_id)
        g = self._gate_dict(f.finding)
        verdict = "passes the gate" if g["pass"] else f"fails {g['rule']}: {g['instruction']}"
        return Reply(f"frame {f.id}: edge rise {_r(f.quality.edge_rise_px, 2)} px, saturation "
                     f"{_r(100 * f.quality.saturation_frac, 2)}%, board {_r(100 * f.quality.board_area_frac, 1)}% → {verdict}.",
                     {"frame_id": f.id, "quality": self._quality_dict(f.quality), "gate": g})

    # ---- tools: accept / drop -------------------------------------------------------------------------
    def accept_view(self, session_id: str, frame_id: int) -> Reply:
        s = self._session(session_id)
        st, th = s.state, self.th
        f = self._frame(s, frame_id)
        if f.view_id is not None:
            raise Refusal("ALREADY_ACCEPTED", f"frame {f.id} is already view {f.view_id}", "capture a new frame")
        if f.finding is not None:
            raise Refusal("GATE_FAILED", f"frame {f.id} failed {f.finding.rule}: {f.finding.measurement}",
                          f"{f.finding.instruction}; capture again")
        if len(st.views) >= st.budget:
            raise Refusal("BUDGET_REACHED", f"{len(st.views)}/{st.budget} views accepted",
                          "call export_calibration")
        calib_prev = self._calib(s) if len(st.views) >= 3 else None
        K = calib_prev.K if calib_prev is not None else k_initial(
            st.image_size, float(np.median(st.focal_obs)) if st.focal_obs else None)
        dist = calib_prev.dist if calib_prev is not None else np.zeros(5)
        try:
            pose = estimate_pose(f.det, K, dist)
        except ValueError as e:
            raise Refusal("POSE_FAILED", f"pose not solvable for frame {f.id} ({e})", "capture again") from None
        tgt = f.target if f.target.get("source") not in ("opening", "operator") and "rvec" in f.target else None
        findings = perception(pose, th, actuator=st.actuator, target=tgt, K=K, dist=dist, size=st.image_size)
        if any(x.rule == "R4_AMBIGUOUS" for x in findings):
            st.set_flag("AMBIGUOUS")
        else:
            st.clear_flag("AMBIGUOUS")
        view = View(st.next_view_id, np.asarray(f.det.ids, np.int32), round_corners(f.det.corners), f.quality,
                    pose.rvec, pose.tvec)
        preq = None
        if calib_prev is not None:
            preq = float(prequential_rms(view, calib_prev))
            view = View(view.id, view.corner_ids, view.corners, f.quality, pose.rvec, pose.tvec, round(preq, 4))
        st.views.append(view)
        try:
            calib = self._calib(s)
        except (ValueError, cv2.error) as e:
            st.views.pop()
            raise Refusal("CALIBRATION_FAILED", f"calibration fails with frame {f.id} added ({e}); not accepted",
                          "capture a different pose") from None
        st.next_view_id += 1
        st.n_accepted_total += 1
        st.preq_history.append(None if preq is None else round(preq, 4))
        st.last_realised = {"rvec": pose.rvec.tolist(), "tvec": pose.tvec.tolist()}
        f.view_id = int(view.id)
        st.add_trace(f"#{st.step} ACCEPT: frame {f.id}, {len(f.det.ids)} corners, tilt {pose.tilt_deg:.0f}°"
                     + ("" if preq is None else f", prequential {preq:.2f} px") + f" → view {view.id}")
        for x in findings:
            st.add_trace(x.trace(st.step))
        unc = None if calib is None else self._unc(s, calib)
        s.accepts.append({"view_id": int(view.id), "n": st.n_accepted_total, "views": len(st.views), "frame_id": f.id,
                          "U95": None if unc is None else _r(unc.U95), "truth": self._truth(s, calib)})
        outliers = []  # the cheap R7 test (residual_analysis applies the full R6-R10 rules)
        if calib is not None and calib.model != "ro":
            outliers = [int(i) for i in outlier_views(calib, st.views, th.outlier_mad_k, th.outlier_min_abs_px)]
            if len(st.views) - len(outliers) < 3:
                outliers = []
        sugg = None if not outliers else {
            "rule": "R7_OUTLIER", "outlier_views": outliers,
            "suggested_call": " and ".join(f"drop_view(view_id={i}, reason='R7 outlier')" for i in outliers)}
        data = {"view_id": int(view.id), "frame_id": f.id, **self._status(s),
                "pose": {"tilt_deg": _r(pose.tilt_deg, 1), "distance_m": _r(pose.distance, 3),
                         "ippe_ratio": _r(pose.ippe_ratio, 3)},
                "prequential_rms_px": _r(preq), "flags": list(st.flags),
                "notes": [x.instruction for x in findings if x.instruction], "suggestion": sugg,
                "calibration": None if calib is None else self._calib_dict(calib)}
        msg = f"view {view.id} accepted ({len(st.views)}/{st.budget} views)"
        if calib is not None:
            msg += f"; refit {calib.model} RMS {calib.rms:.3f} px, fx {calib.K[0, 0]:.1f}"
        if sugg is not None:
            msg += (f". R7: view(s) {outliers} have per-view RMS far above the others (outliers) → "
                    f"{sugg['suggested_call']}")
        elif len(st.views) >= st.budget:
            msg += ". Budget reached: call export_calibration."
        elif calib is None:
            msg += f". {3 - len(st.views)} more view(s) before the first calibration."
        else:
            msg += ". Next: uncertainty_map or propose_next_views."
        return Reply(msg, data)

    def drop_view(self, session_id: str, view_id: int, reason: str = "") -> Reply:
        s = self._session(session_id)
        st = s.state
        try:
            vid = int(view_id)
        except (TypeError, ValueError):
            raise Refusal("BAD_ARGUMENT", f"view_id must be an integer, got {view_id!r}") from None
        ids = [int(v.id) for v in st.views]
        if vid not in ids:
            raise Refusal("UNKNOWN_VIEW", f"no view {vid} (views: {ids})")
        if len(ids) == 3:
            raise Refusal("NEEDS_VIEWS", "dropping a view would leave 2; a calibration needs at least 3",
                          "capture and accept more views first")
        pv = None
        if s.calib is not None and s.calib.model != "ro" and s.calib_key and s.calib_key[0] == s.view_key():
            pv = _r(s.calib.per_view[ids.index(vid)])
        st.views = [v for v in st.views if int(v.id) != vid]
        reason = str(reason or "")[:200]
        st.rejected.append({"step": st.step, "attempt": st.attempt, "rule": "DROP", "metric": "per_view_rms",
                            "value": pv, "view_id": vid, "reason": reason})
        st.add_trace(f"#{st.step} DROP: view {vid}" + (f" ({reason})" if reason else "") + " → refit")
        calib = None
        try:
            calib = self._calib(s)
        except (ValueError, cv2.error):
            calib = None
        if calib is not None:  # evaluator log: the estimate at this accepted-frame count is now the refit
            s.accepts.append({"view_id": None, "n": st.n_accepted_total, "views": len(st.views), "after_drop": vid,
                              "U95": _r(self._unc(s, calib).U95), "truth": self._truth(s, calib)})
        return Reply(f"view {vid} dropped; {len(st.views)} views remain" +
                     ("" if calib is None else f"; refit RMS {calib.rms:.3f} px") + ".",
                     {"dropped": vid, **self._status(s), "calibration": None if calib is None else self._calib_dict(calib)})

    # ---- tools: calibrate (gated model change) ----------------------------------------------------------
    def calibrate(self, session_id: str, model: str = "k5") -> Reply | ApprovalRequest:
        s = self._session(session_id)
        st = s.state
        if model not in MODELS:
            raise Refusal("BAD_ARGUMENT", f"model must be one of {list(MODELS)}, got {model!r}")
        calib = self._need_calib(s)
        cur = st.model["name"]
        if model == cur:
            return self._fit_reply(s, calib, f"{calib.model} fit")
        if st.model.get("pinned"):
            raise Refusal("MODEL_PINNED", f"the approver pinned the model to {cur}", "keep calibrating with the current model")
        if model == "k5":
            raise Refusal("BAD_ARGUMENT", f"the session model is {cur} (approved); switching back is not offered",
                          "start a new session for a k5 calibration")
        size = st.image_size
        if model == "rational":
            diag = analyse(calib, st.views, self.th, size)
            cv_k5 = cv_heldout_rms(st.views, "k5", size, init=calib)
            cv_rat = cv_heldout_rms(st.views, "rational", size, init=calib)
            ev = {"radial_trend_p": _r(diag.radial_p, 6), "cv_rms": {"k5": _r(cv_k5), "rational": _r(cv_rat)},
                  "views": len(st.views)}
            kind, rule = ApprovalKind.SWITCH_MODEL, "R8_UNDERFIT"
        else:
            n_full = len(full_board_views(st.views))
            if n_full < 3:
                raise Refusal("NEEDS_VIEWS", f"the RO model needs 3 full-board views, have {n_full}",
                              "capture views with the whole board inside the frame")
            diag = analyse(calib, st.views, self.th, size)
            gain, cv_ro, cv_k5 = ro_cv_gain(st.views, size, init=calib)
            ev = {"warp_score": _r(diag.warp_score), "cv_gain": _r(gain), "cv_rms": {"k5": _r(cv_k5), "ro": _r(cv_ro)},
                  "views": len(st.views), "full_board_views": n_full}
            kind, rule = ApprovalKind.USE_RO, "R9_WARP"
        aid = f"ap_{st.next_approval_id}"
        st.next_approval_id += 1
        card = make_card(kind, aid, jsonable(ev), rule, st.step)

        def resolve(decision: str, reason: str) -> Reply:
            self._record_approval(s, card, decision, reason)
            if decision == "approve":
                st.model = {"name": model, "pinned": False}
                st.clear_flag("UNDERFIT" if model == "rational" else "WARP")
                c = self._need_calib(s)
                return self._fit_reply(s, c, f"approved ({aid}): refit with {model}")
            st.model = dict(st.model, pinned=True)
            return self._fit_reply(s, calib, f"denied ({aid}): model pinned to {cur}")

        return ApprovalRequest(s.sid, card, resolve)

    def _fit_reply(self, s: Session, calib: CalibResult, head: str) -> Reply:
        st = s.state
        pv = np.asarray(calib.per_view, float)
        data = {**self._status(s), "calibration": self._calib_dict(calib),
                "per_view_rms_px": {"median": _r(np.median(pv)) if pv.size else None,
                                    "max": _r(pv.max()) if pv.size else None,
                                    "worst_view": int(st.views[int(np.argmax(pv))].id) if pv.size == len(st.views) and pv.size else None}}
        return Reply(f"{head}: {calib.model} over {len(st.views)} views, RMS {calib.rms:.3f} px, fx {calib.K[0, 0]:.1f}, "
                     f"cx {calib.K[0, 2]:.1f}, cy {calib.K[1, 2]:.1f}.", data)

    def _record_approval(self, s: Session, card: dict, decision: str, reason: str) -> None:
        st = s.state
        reason = str(reason or "")[:500]
        st.approvals.append({"id": card["id"], "kind": card["kind"], "decision": decision, "reason": reason,
                             "ts": round(time.time(), 3), "step": st.step, "rule": card.get("rule"), "via": "mcp_elicit"})
        st.add_trace(f"#{st.step} APPROVAL {card['id']} {card['kind']}: {decision}" + (f" ({reason})" if reason else ""))

    # ---- tools: diagnostics and planning ----------------------------------------------------------------
    def uncertainty_map(self, session_id: str) -> Reply:
        s = self._session(session_id)
        st = s.state
        calib = self._need_calib(s)
        unc = self._unc(s, calib)
        d = self._decision(s, calib, unc)
        sig = np.asarray(unc.sigma_px, np.float32)
        coarse = cv2.resize(sig, COARSE_GRID, interpolation=cv2.INTER_AREA)
        r, c = np.unravel_index(int(np.argmax(sig)), sig.shape)
        cols, rows = unc.grid
        W, H = st.image_size
        where = position_word(((c + 0.5) * W / cols, (r + 0.5) * H / rows), st.image_size)
        pex = None if not math.isfinite(unc.p_exceed) else round(float(unc.p_exceed), 4)
        if d.rule == "R11_STOP":
            rec = "stop: call export_calibration"
        elif d.rule == "R12_BUDGET":
            rec = "budget used: call export_calibration (it will be stamped with warnings)"
        else:
            rec = "continue: propose_next_views, then sim_capture"
        data = {**self._status(s), "U95_px": _r(unc.U95), "p_exceed": pex, "tau_px": st.tau,
                "grid": list(COARSE_GRID), "sigma_px": np.round(coarse, 3).tolist(),
                "largest_sigma": {"where": where, "px": _r(sig.max(), 3)}, "stop_ready": d.rule == "R11_STOP",
                "rule": d.rule, "check": d.measurement or None, "flags": list(st.flags), "recommendation": rec}
        pe = "n/a (stop calibration not fitted)" if pex is None else f"{pex:.2f}"
        return Reply(f"U95 {unc.U95:.3f} px (target τ {st.tau} px), P_exceed {pe}, largest σ {sig.max():.2f} px at {where}; "
                     f"{len(st.views)}/{st.budget} views → {rec}.", data)

    def residual_analysis(self, session_id: str) -> Reply:
        s = self._session(session_id)
        st, th, size = s.state, self.th, s.state.image_size
        calib = self._need_calib(s)
        diag = analyse(calib, st.views, th, size)
        notes: list[str] = []
        full_ids = [int(v.id) for v in full_board_views(st.views)]

        def cv_pair():
            return (cv_heldout_rms(st.views, "k5", size, init=calib), cv_heldout_rms(st.views, "rational", size, init=calib))

        def ro_gain():
            g = ro_cv_gain(st.views, size, init=calib)
            st.r9 = {"done": True, "n_full": len(full_ids), "ids": full_ids[-12:], "gain": _r(g[0]),
                     "cv_ro": _r(g[1]), "cv_k5": _r(g[2]), "step": st.step}
            return g

        kw = dict(model=st.model, n_views=len(st.views), n_full_views=len(full_ids), calib=calib, diag=diag,
                  preq_history=st.preq_history, th=th, size=size, cv_pair=cv_pair, ro_gain=ro_gain, notes=notes,
                  flags=tuple(st.flags), r9_last=st.r9, full_view_ids=full_ids)
        f = post_accept(**kw)
        if f is not None and f.rule == "R7_OUTLIER" and len(st.views) - len(f.evidence.get("view_ids", ())) < 3:
            notes.append(f"R7 not suggested: dropping {list(f.evidence['view_ids'])} would leave fewer than 3 views")
            f = post_accept(**kw, skip=("R6_DRIFT", "R7_OUTLIER"))
        sugg = None
        if f is not None:
            call = {"R6_DRIFT": "focus or zoom may have changed: stop and start a new session with focus locked",
                    "R7_OUTLIER": " and ".join(f"drop_view(view_id={i}, reason='R7 outlier')" for i in f.evidence.get("view_ids", ())),
                    "R8_UNDERFIT": "calibrate(model='rational') (asks the approver)",
                    "R9_WARP": "calibrate(model='ro') (asks the approver)",
                    "R10_OUT_OF_SCOPE": "lens is outside the pinhole/rational scope: stop; do not export"}.get(f.rule)
            sugg = {"rule": f.rule, "finding": f.measurement, "action": f.action, "suggested_call": call}
            if f.rule == "R7_OUTLIER":
                sugg["view_ids"] = [int(i) for i in f.evidence.get("view_ids", ())]
            if f.rule == "R8_UNDERFIT":
                st.set_flag("UNDERFIT")
            elif f.rule == "R9_WARP":
                st.set_flag("WARP")
            elif f.rule == "R10_OUT_OF_SCOPE":
                st.set_flag("OUT_OF_SCOPE")
            st.add_trace(f.trace(st.step))
        data = {**self._status(s), "radial_trend_p": _r(diag.radial_p, 6), "outlier_views": [int(i) for i in diag.outliers],
                "warp_score": _r(diag.warp_score), "drift": bool(diag.drift), "hfov_deg": _r(diag.hfov_deg, 1),
                "suggestion": sugg, "notes": notes, "flags": list(st.flags)}
        head = (f"radial-trend p {_r(diag.radial_p, 4)}, outliers {list(diag.outliers) or 'none'}, warp score "
                f"{_r(diag.warp_score, 2)}")
        tail = f" → {f.rule}: {sugg['suggested_call']}" if f is not None else " → nothing to fix; continue planning views"
        return Reply(head + tail + ".", data)

    def propose_next_views(self, session_id: str, k: int = 5) -> Reply:
        s = self._session(session_id)
        st, th = s.state, self.th
        try:
            k = max(1, min(5, int(k)))
        except (TypeError, ValueError):
            raise Refusal("BAD_ARGUMENT", f"k must be an integer 1..5, got {k!r}") from None
        if len(st.views) < 3:
            t = self._opening(s)
            st.candidates = [t]
            s.plan_key = s.view_key()
            if st.mode != "sim":  # image sessions: the operator is asked for the top candidate
                st.last_target = t
            return Reply(f"opening pose {len(st.views) + 1}/3 (shared seeded opening, no calibration yet): {t['desc']}. "
                         f"Next: sim_capture(candidate_rank=1).",
                         {**self._status(s), "candidates": [self._cand_public(t)], "phase": "opening"})
        calib = self._need_calib(s)
        # R9 pre-check, as in the agent loop: while a warp is suspected on 3-5 full-board views the
        # planner asks for whole-board views only, so the R9 check gets the views it needs
        n_full = len(full_board_views(st.views))
        st.seek_full_board = warp_seek(st.model, n_full, analyse(calib, st.views, th, st.image_size).warp_score, th)
        rng = np.random.default_rng([st.camera_seed, st.seed, st.attempt, NBV_STREAM])
        t0 = time.perf_counter()
        cands = plan_views(st, calib, st.image_size, th, rng, k=k, deadline=t0 + float(th.nbv_time_budget_s))
        stats = cands.stats()
        out = []
        for c in cands:
            out.append(self._target(st, c.rvec, c.tvec, calib.K, calib.dist, "nbv", rank=int(c.rank),
                                    pred_gain=round(float(c.gain), 5), pred_U95_after=round(float(c.pred_U95_after), 5),
                                    effort=round(float(c.effort), 4), score=round(float(c.score), 5)))
        if not out:
            raise Refusal("NO_FEASIBLE_VIEW", "no feasible candidate pose under the current estimate",
                          "capture with an explicit pose or call residual_analysis")
        st.candidates = out
        s.plan_key = s.view_key()
        top = out[0]
        if st.mode != "sim":
            st.last_target = top
        data = {**self._status(s), "U95_now_px": stats.get("u95_now"), "n_evaluated": stats.get("n_candidates"),
                "candidates": [self._cand_public(c) for c in out], "phase": "nbv"}
        return Reply(f"{len(out)} candidates from {stats.get('n_candidates')} evaluated; rank 1: {top['desc']} "
                     f"(predicted U95 {stats.get('u95_now')} → {top['pred_U95_after']:.3f} px). "
                     f"Next: sim_capture(candidate_rank=1).", data)

    # ---- tools: export (gated) --------------------------------------------------------------------------
    def export_calibration(self, session_id: str) -> ApprovalRequest:
        s = self._session(session_id)
        st = s.state
        calib = self._need_calib(s)
        unc = self._unc(s, calib)
        d = self._decision(s, calib, unc)
        pex = float(unc.p_exceed)
        kind = ApprovalKind.EXPORT if d.rule == "R11_STOP" else ApprovalKind.ACCEPT_WITH_WARNINGS
        ev = {"views": len(st.views), "attempts": st.attempts, "U95_px": _r(unc.U95), "tau_px": st.tau,
              "p_exceed": _r(pex), "model": calib.model, "flags": list(st.flags), "rms_px": _r(calib.rms),
              "K": np.round(calib.K, 3).tolist(), "dist": np.round(calib.dist, 6).tolist()}
        if kind is ApprovalKind.ACCEPT_WITH_WARNINGS:
            ev["label"] = ("likely within target" if math.isfinite(pex) and pex < 0.5 else
                           "likely above target — recommend recapture" if math.isfinite(pex) else
                           "stop probability not calibrated")
            ev["why_not_clean_stop"] = d.measurement or d.rule
        aid = f"ap_{st.next_approval_id}"
        st.next_approval_id += 1
        card = make_card(kind, aid, jsonable(ev), "R11_STOP" if kind is ApprovalKind.EXPORT else "EXPORT_REQUEST", st.step)

        def resolve(decision: str, reason: str) -> Reply:
            self._record_approval(s, card, decision, reason)
            if decision != "approve":
                if kind is ApprovalKind.ACCEPT_WITH_WARNINGS:
                    st.status = "discarded"
                    return Reply(f"export denied ({aid}): session discarded.", {"exported": False, "status": st.status})
                return Reply(f"export denied ({aid}): the session stays open; keep capturing.",
                             {"exported": False, "status": st.status, **self._status(s)})
            meta = {"session": st.sid, "preset": st.preset, "mode": st.mode, "approval_id": aid,
                    "warnings": kind is ApprovalKind.ACCEPT_WITH_WARNINGS, "U95_px": _r(unc.U95), "tau_px": st.tau,
                    "p_exceed": _r(pex), "views": len(st.views), "flags": ",".join(st.flags) or "none", "via": "mcp"}
            files = export_bytes(calib, unc, st.trace, size=st.image_size, meta=meta)
            paths = {}
            if self.export_dir is not None:
                out = self.export_dir / st.sid
                out.mkdir(parents=True, exist_ok=True)
                for ext, b in files.items():
                    p = out / f"calibration.{ext}"
                    p.write_bytes(b)
                    paths[ext] = str(p)
            st.status = "exported"
            s.exported = {"approval_id": aid, "warnings": meta["warnings"], "views": len(st.views)}
            st.add_trace(f"#{st.step} EXPORT: {aid} → {', '.join(sorted(files))}")
            return Reply(f"calibration exported ({aid}{', with warnings' if meta['warnings'] else ''}): {calib.model}, "
                         f"{len(st.views)} views, U95 {unc.U95:.3f} px. The session is finished.",
                         {"exported": True, "status": st.status, "warnings": meta["warnings"], "files": paths,
                          "bytes": {k: len(v) for k, v in files.items()}, "calibration": self._calib_dict(calib)})

        return ApprovalRequest(s.sid, card, resolve)

    # ---- status and evaluator-only tools -----------------------------------------------------------------
    def session_status(self, session_id: str) -> Reply:
        s = self._session(session_id, open_only=False)
        st = s.state
        fresh = bool(st.candidates) and s.plan_key == s.view_key()
        calib = s.calib if (s.calib is not None and s.calib_key and s.calib_key[0] == s.view_key()) else None
        data = {"session_id": s.sid, "status": st.status, **self._status(s), "flags": list(st.flags),
                "plan": {"candidates": len(st.candidates), "fresh": fresh},
                "frames_kept": list(s.frames), "last_frame": None if not s.frames else self._last_frame(s),
                "calibration": None if calib is None else self._calib_dict(calib),
                "approvals": [{k: a[k] for k in ("id", "kind", "decision")} for a in st.approvals],
                "trace_tail": st.trace[-5:]}
        return Reply(f"session {s.sid} {st.status}: {len(st.views)}/{st.budget} views, {st.attempts} captures, "
                     f"model {st.model['name']}, plan {'fresh' if fresh else 'none or stale'}.", data)

    def _last_frame(self, s: Session) -> dict:
        f = next(reversed(s.frames.values()))
        return {"frame_id": f.id, "gate": self._gate_dict(f.finding)["rule"], "accepted_as_view": f.view_id}

    def _truth(self, s: Session, calib: CalibResult | None) -> dict | None:
        if s.sim is None or calib is None:
            return None
        from ..sim.truth import field_error, intrinsics_error, planar_mm_error

        cam = s.sim.camera_at(s.state.n_accepted_total)
        fe = field_error(cam, calib.K, calib.dist)
        try:
            mm = planar_mm_error(cam, calib.K, calib.dist, 1.0).median
        except RuntimeError:
            mm = math.nan
        ie = intrinsics_error(cam, calib.K)
        return {"field_p95_px": _r(fe.p95), "field_mean_px": _r(fe.mean), "fx_rel_err_pct": _r(ie["fx_rel_pct"]),
                "fy_rel_err_pct": _r(ie["fy_rel_pct"]), "cx_err_px": _r(ie["cx_err_px"], 3),
                "cy_err_px": _r(ie["cy_err_px"], 3), "mm_at_1m_edge": _r(mm)}

    def ground_truth_error(self, session_id: str) -> Reply:
        """Evaluator only (sim): the true error of the current calibration. Never shown to an LLM."""
        s = self._session(session_id, open_only=False, sim=True)
        calib = None
        if len(s.state.views) >= 3:
            try:
                calib = self._calib(s)
            except (ValueError, cv2.error):
                calib = None
        t = self._truth(s, calib)
        unc = None if calib is None else self._unc(s, calib)
        data = {"session_id": s.sid, "views": len(s.state.views), "truth": t,
                "U95_px": None if unc is None else _r(unc.U95),
                "p_exceed": None if unc is None or not math.isfinite(unc.p_exceed) else _r(unc.p_exceed)}
        if t is None:
            return Reply("no calibration yet (fewer than 3 views)", data)
        return Reply(f"true field p95 {t['field_p95_px']} px, fx error {t['fx_rel_err_pct']}%", data)

    def session_log(self, session_id: str) -> Reply:
        """Evaluator only: per-capture and per-accept rows, approvals, rejections and the trace."""
        s = self._session(session_id, open_only=False)
        st = s.state
        return Reply(f"log of {s.sid}: {len(s.captures)} captures, {len(s.accepts)} accepts",
                     {"session_id": s.sid, "status": st.status, "captures": s.captures, "accepts": s.accepts,
                      "approvals": st.approvals, "rejected": st.rejected, "trace": st.trace, "model": st.model,
                      "flags": list(st.flags), "views": len(st.views), "attempts": st.attempts, "exported": s.exported,
                      "camera_seed": st.camera_seed, "seed": st.seed, "preset": st.preset, "actuator": st.actuator})
