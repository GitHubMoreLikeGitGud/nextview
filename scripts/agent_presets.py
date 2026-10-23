"""Run the agent end to end (locally, no Lambda) on every preset and summarise each session.

Each session starts from ``new_session`` and loops ``run_step`` / ``apply_approval`` until the
first stop or escalation card (R10, R11 or R12), which is then resolved (export approved,
accept-with-warnings approved, out-of-scope acknowledged) so the terminal path runs too.
Model-change and drift cards follow ``--approval`` (approve_all or deny_all). With
``--token`` (default) the state goes through the signed token after every call, as on Lambda.
``--sag-mm`` replaces the detected corners with those of a sagged board (spec 6.3 warp), the
R9 / release-object path.

    python scripts/agent_presets.py --out results/agent/presets.json
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from collections import Counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

from nextview.agent.loop import Observation, SimSource, apply_approval, make_source, run_step  # noqa: E402
from nextview.agent.state import decode_token, encode_token, new_session  # noqa: E402
from nextview.config import StopCalibration, load_presets, load_stop_calibration, load_thresholds  # noqa: E402
from nextview.sim.truth import fx_rel_err  # noqa: E402
from nextview.tools._views import full_board_views  # noqa: E402

TERMINAL_CARDS = {"export": "approve", "accept_with_warnings": "approve", "out_of_scope": "acknowledge"}


class SaggedSource(SimSource):
    """A sim source whose detected corners are those of a board sagging by ``amp_m`` (spec 6.3),
    projected with the true camera, plus 0.15 px noise."""

    def __init__(self, simulator, amp_m: float):
        super().__init__(simulator)
        self.amp = float(amp_m)

    def capture(self, rvec, tvec, **kw):
        from nextview.sim.warp import sag_object_points
        from nextview.types import Detection

        o = super().capture(rvec, tvec, **kw)
        d = o.det
        if len(d.ids):
            cam = self.camera(kw["n_accepted"])
            p, _ = cam.project_checked(sag_object_points(self.amp)[np.asarray(d.ids).astype(int)], o.truth.rvec, o.truth.tvec)
            p = p + np.random.default_rng(kw["attempt"] + 99).normal(0, 0.15, p.shape)
            d = Detection(d.ids, p, d.obj, d.n_markers, d.ok)
        return Observation(o.image, d, o.moved_ids, o.truth, o.ms)


def identity_stopcal() -> StopCalibration:
    """P_exceed = 1 if U95 > tau else 0 (r = true_p95 / U95 taken as exactly 1). Exercises R11 only;
    NOT a calibration: bench/fit_stop.py fits the real one on the DEV split."""
    q = tuple([1.0] * 101)
    return StopCalibration(calibrated=True, quantiles={"robot": q, "human": q}, note="identity (r = 1), for R11 path tests only")


def run_session(preset: str, *, camera_seed: int = 0, seed: int = 0, approval: str = "approve_all", stopcal=None,
                th=None, token: bool = True, max_calls: int = 250, actuator: str | None = None, sag_mm: float = 0.0) -> dict:
    th = th or load_thresholds()
    stopcal = stopcal or load_stop_calibration()
    p = load_presets().get(preset)
    st = new_session(mode="sim", preset=preset, actuator=actuator or p.actuator, seed=seed, camera_seed=camera_seed, th=th)
    key = secrets.token_bytes(32)
    src = make_source(st)
    if sag_mm:
        src = SaggedSource(src.sim, sag_mm / 1000.0)
    calls, step_ms, token_bytes = [], [], []
    decision = None
    first_cards = []
    while len(calls) < max_calls and not st.terminal:
        t = time.perf_counter()
        card = st.pending_approval
        if card is None:
            res = run_step(st, src, th=th, stopcal=stopcal)
        else:
            kind = card["kind"]
            allowed = [o["decision"] for o in card["options"]]
            if kind in TERMINAL_CARDS:
                if decision is None:
                    decision = {"rule": card["rule"], "kind": kind, "status": st.status, "step": st.step}
                d = TERMINAL_CARDS[kind]
            else:
                d = "approve" if approval == "approve_all" else "deny"
            if d not in allowed:  # e.g. a budget card over < 3 views offers only "deny" (Discard)
                d = allowed[0]
            first_cards.append(f"{card['rule']}:{kind}->{d}@{st.step}")
            res = apply_approval(st, card["id"], d, f"auto {approval}", src, th=th, stopcal=stopcal)
        ms = 1000 * (time.perf_counter() - t)
        step_ms.append(ms)
        calls.append({"step": res.step, "rule": res.rule, "status": res.status, "accepted": res.accepted, "ms": round(ms, 1),
                      "tools": res.tools, "U95": None if res.uncertainty is None else res.uncertainty["U95_px"],
                      "true_p95": None if res.truth is None else res.truth["field_p95_px"], "trace": res.trace_all})
        if token:
            tok = encode_token(st, key)
            token_bytes.append(len(tok))
            st = decode_token(tok, key)
    hist = [h for h in st.history if h.get("U95") is not None]
    cam = src.camera(st.n_accepted_total)
    K = np.asarray(st.calib["K"]) if st.calib else None
    rej = Counter(r["rule"] for r in st.rejected)
    slow = max(range(len(calls)), key=lambda i: calls[i]["ms"])
    return {
        "preset": preset, "family": p.family, "actuator": st.actuator, "camera_seed": camera_seed, "seed": seed,
        "approval_policy": approval, "board_sag_mm": sag_mm, "stop_calibration": "calibrated" if stopcal.calibrated else "uncalibrated (P_exceed = NaN)",
        "calls": len(calls), "capture_steps": st.attempt, "approvals": first_cards,
        "accepted_frames": st.n_accepted_total, "views_final": len(st.views),
        "full_board_views_final": len(full_board_views(st.views)),
        "r9_check": None if not st.r9 else {k: st.r9.get(k) for k in ("n_full", "step", "gain", "cv_ro", "cv_k5", "fold_s")},
        "rejected": dict(rej), "decision": decision, "final_status": st.status, "model": st.model,
        "final_U95_px": hist[-1]["U95"] if hist else None, "final_p_exceed": hist[-1]["p_exceed"] if hist else None,
        "final_true_p95_px": hist[-1].get("true_p95") if hist else None,
        "fx_rel_err_pct": None if K is None else round(fx_rel_err(cam, K), 4),
        "baseline": st.baseline and {"views": len(st.baseline["views"]), "truth": st.baseline.get("truth")},
        "slowest_call_ms": round(calls[slow]["ms"], 1), "slowest_call": {k: calls[slow][k] for k in ("step", "rule", "tools")},
        "median_call_ms": round(float(np.median(step_ms)), 1), "max_token_bytes": max(token_bytes) if token_bytes else None,
        "flags": st.flags, "trace_tail": st.trace[-6:], "calls_detail": calls,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--presets", nargs="*", default=None)
    ap.add_argument("--camera-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--approval", choices=("approve_all", "deny_all"), default="approve_all")
    ap.add_argument("--identity-stopcal", action="store_true", help="exercise R11 with r = 1 (not a calibration)")
    ap.add_argument("--no-token", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a threshold for this run only (e.g. drift_factor=2.5); recorded in the output")
    ap.add_argument("--sag-mm", type=float, default=0.0, help="sagged board amplitude in mm (R9 / RO path)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--detail", action="store_true", help="keep per-call detail in the output JSON")
    a = ap.parse_args(argv)
    presets = a.presets or list(load_presets().ids)
    stopcal = identity_stopcal() if a.identity_stopcal else None
    th = load_thresholds()
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = float(v) if "." in v else int(v)
    if overrides:
        th = th.replace(**overrides)
    out = []
    for pid in presets:
        r = run_session(pid, camera_seed=a.camera_seed, seed=a.seed, approval=a.approval, stopcal=stopcal, th=th,
                        token=not a.no_token, sag_mm=a.sag_mm)
        r["threshold_overrides"] = overrides
        if not a.detail:
            r.pop("calls_detail")
        out.append(r)
        d = r["decision"] or {}
        print(f"{pid:24s} calls {r['calls']:3d} captures {r['capture_steps']:3d} accepted {r['accepted_frames']:3d} "
              f"rejected {r['rejected']} decision {d.get('rule')}->{r['final_status']} model {r['model']['name']} "
              f"U95 {r['final_U95_px']} true_p95 {r['final_true_p95_px']} fx_err% {r['fx_rel_err_pct']} "
              f"slowest {r['slowest_call_ms']} ms (median {r['median_call_ms']}) token {r['max_token_bytes']} B "
              f"approvals {r['approvals']}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"opencv": cv2.__version__, "threads": cv2.getNumThreads(), "sessions": out}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
