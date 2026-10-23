"""Failure gallery F1-F11 (spec 12) -> ``results/failure_gallery.json`` + figure E12.

    python -m nextview.bench.gallery --out results

Measured numbers come from the result files (``synth_failure``, ``synth_test``, ``pool``); the
example frames are re-rendered by running the agent on the named camera (deterministic, so the
frame shown is the frame the benchmark saw).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

__all__ = ["main"]


def _drive(family, camera, seed, actuator, nuisances, until, max_calls=200, approvals=None):
    """Run the agent until ``until(res, st)`` is true; returns (res, state, frames {step: image})."""
    from nextview.agent.loop import apply_approval, run_step
    from nextview.agent.state import new_session
    from nextview.bench.policies import RecordingSource
    from nextview.config import load_stop_calibration, load_thresholds
    from nextview.sim.scenarios import make_simulator

    th = load_thresholds().replace(nbv_time_budget_s=1e5)
    sc = load_stop_calibration()
    sim = make_simulator(family, actuator, camera, seed, nuisances)
    src = RecordingSource(sim)
    st = new_session(mode="sim", actuator=actuator, seed=seed, camera_seed=camera, th=th, baseline=False, now=0.0, sid="gallery")
    approvals = approvals or {}
    import time

    for _ in range(max_calls):
        if st.terminal:
            break
        card = st.pending_approval
        if card is None:
            res = run_step(st, src, th=th, stopcal=sc, deadline=time.perf_counter() + 1e5)
        else:
            if until(None, st):
                return None, st
            d = approvals.get(card["kind"], "approve" if card["kind"] != "out_of_scope" else "acknowledge")
            allowed = [o["decision"] for o in card["options"]]
            res = apply_approval(st, card["id"], d if d in allowed else allowed[0], "gallery", src, th=th, stopcal=sc,
                                 deadline=time.perf_counter() + 1e5, now=0.0)
        if until(res, st):
            return res, st
    return None, st


def _panel_frames():
    """(title, image, caption) for F1, F2, F4, F5 and F6."""
    import cv2

    from nextview.agent.policy import gate, perception
    from nextview.sim.poses import pose_for_view
    from nextview.sim.scenarios import make_simulator
    from nextview.tools.detect import detect_board
    from nextview.tools.pose import estimate_pose
    from nextview.tools.quality import assess_frame
    from nextview.config import load_thresholds

    th = load_thresholds()
    out = []
    # F1 fisheye: run until the out-of-scope card
    res, st = _drive("fisheye130", 2000, 0, "robot", (), lambda r, s: s.pending_approval is not None
                     and s.pending_approval["kind"] == "out_of_scope")
    if st.pending_approval is not None:
        ev = st.pending_approval.get("evidence", {})
        sim = make_simulator("fisheye130", "robot", 2000, 0)
        img = sim.render_exact(np.asarray(st.sim_prev_pose["rvec"]), np.asarray(st.sim_prev_pose["tvec"]))
        sw = [a for a in st.approvals if a["kind"] == "switch_model"]
        out.append(("F1 fisheye 130°", img, f"R8 switch at step {sw[0]['step'] if sw else '-'}, then R10: "
                    f"est. HFOV {ev.get('hfov_deg', float('nan')):.0f}° > {ev.get('hfov_limit_deg', 110):.0f}°"))
    # F2 glare: first R3 rejection on a webcam camera of the TEST split
    for cam in range(1075, 1100):
        res, st = _drive("webcam", cam, 0, "human", ("glare",), lambda r, s: r is not None and r.rule == "R3_GLARE", max_calls=60)
        if res is not None:
            out.append(("F2 glare (R3)", res.image, res.instruction))
            break
    # F4 small / far board: s = 0.12 frontal
    sim = make_simulator("normal", "robot", 1000, 0, (), clean=True)
    rv, tv = pose_for_view(sim.camera, (640, 360), 0.14, 0.0, 0.0, 0.0)
    img = sim.render_exact(rv, tv)
    det = detect_board(img)
    q = assess_frame(img, det)
    g = gate(det, q, th)
    out.append(("F4 small board (R0/R1)", img, f"{g.rule if g else 'pass'}: {g.instruction if g else ''} "
                f"(area {100 * q.board_area_frac:.1f} %)"))
    # F5 IPPE ambiguity: small fronto-parallel board
    best = None
    for s in (0.3, 0.25, 0.2):
        rv, tv = pose_for_view(sim.camera, (640, 360), s, 0.0, 0.0, 0.0)
        img = sim.render_exact(rv, tv)
        det = detect_board(img)
        q = assess_frame(img, det)
        if gate(det, q, th) is None:
            pose = estimate_pose(det, sim.camera.K, sim.camera.dist)
            f = [x for x in perception(pose, th, actuator="robot", target=None, K=sim.camera.K, dist=sim.camera.dist,
                                       size=sim.camera.size) if x.rule == "R4_AMBIGUOUS"]
            best = (img, pose, f)
            if f:
                break
    if best is not None:
        img, pose, f = best
        out.append(("F5 IPPE ambiguity (R4)", img, (f[0].instruction if f else "no R4") + f" (ratio {pose.ippe_ratio:.2f})"))
    return out


def _numbers(results: Path) -> dict:
    from nextview.bench.analysis import replay_stop, sessions_of
    from nextview.bench.files import load_runs, load_sessions

    g = {}
    if (results / "synth_failure").exists():
        S = load_sessions(results / "synth_failure")
        for case in ("fisheye130", "autofocus_drift", "warped_board"):
            ss = [s for s in S if s.get("case") == case]
            ends = Counter((s.get("end") or {}).get("rule") for s in ss)
            strict = sum(ends.get(k, 0) for k in ("R10_OUT_OF_SCOPE", "R12_BUDGET", "R6_DRIFT"))
            cut = sum(ends.get(k, 0) for k in ("BUDGET_REACHED", "ATTEMPTS", "MAX_CALLS", "TERMINAL"))
            r12 = Counter(("views" if (s.get("end") or {}).get("views", 0) >= s.get("budget", 30) else "attempts")
                          for s in ss if (s.get("end") or {}).get("rule") == "R12_BUDGET")
            g[case] = {"runs": len(ss), "end_rules": dict(ends), "clean_stops": ends.get("R11_STOP", 0),
                       "correct_abstentions": strict, "cut_by_harness": cut, "r12_on": dict(r12),
                       "correct_abstention_strict": strict / len(ss) if ss else None,
                       "no_clean_stop": (len(ss) - ends.get("R11_STOP", 0)) / len(ss) if ss else None,
                       "approvals": dict(Counter(f"{a['rule']}:{a['decision']}" for s in ss for a in s.get("approvals", [])))}
    sp = results / "synth_test" / "summary.json"
    if sp.exists():
        sm = json.loads(sp.read_text(encoding="utf-8"))
        g["F3_robot_vs_human"] = {e["id"]: {"actuator": e["actuator"], "median_diff": e["median_diff"], "ci": e["ci"]}
                                  for e in sm["primary_endpoints"]}
        h = sm["uncertainty_honesty_E5"]
        g["F7_under_coverage"] = {k: {"cov95_by_family": v.get("cov95_by_family"), "r_by_family": v.get("r_by_family")}
                                  for k, v in h.items() if k.startswith("A/")}
        tau = sm["tau_px"]
        rows = load_runs(results / "synth_test")
        sc = json.loads(Path("config/stop_calibration.json").read_text(encoding="utf-8"))
        stops = []
        for key, srows in sessions_of(rows, {"A"}).items():
            r = replay_stop(srows, sc["quantiles"][key[3]], tau, 0.10)
            if r is not None:
                stops.append({"camera": key[0], "family": key[1], "seed": key[2], "actuator": key[3], "n": r["n"],
                              "true_p95": r["true_p95"], "U95": r["U95"], "p_exceed": r["p_exceed_replay"]})
        stops.sort(key=lambda d: -d["true_p95"])
        g["F9_worst_false_stops"] = {"tau_px": tau, "false_stop_threshold_px": 2 * tau, "worst3": stops[:3],
                                     "false_stops": sum(1 for d in stops if d["true_p95"] > 2 * tau), "stops": len(stops)}
        ev = Counter()
        for s in load_sessions(results / "synth_test"):
            if s.get("policy") == "A":
                for k, v in (s.get("rejected") or {}).items():
                    ev[f"{s['family']}/{s['actuator']}/{k}"] += v
        g["gate_rejections_A_test"] = dict(sorted(ev.items()))
    pd = results / "pool"
    for f in sorted(pd.glob("metric_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        g.setdefault("F11_metric", {})[d["subset"]] = [
            {"camera": c["camera"], "frames": c["frames"], "detected": c["detected"], "full_board": c["full_board"],
             "gate_diagnostics": c["gate_diagnostics"], "r8_check": c.get("r8_check")} for c in d["cameras"]]
    g["F10_rolling_shutter"] = "not modelled; stated as a limitation"
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args(argv)
    import cv2

    cv2.setNumThreads(1)
    res = Path(a.out)
    nums = _numbers(res)
    (res / "failure_gallery.json").write_text(json.dumps(nums, indent=1, default=float), encoding="utf-8")
    if not a.no_images:
        from nextview.bench.plots import INK2, _mpl, _save

        plt = _mpl()
        panels = _panel_frames()
        fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.4))
        for ax, (title, img, cap) in zip(np.atleast_1d(axes), panels):
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            import textwrap

            ax.set_xlabel("\n".join(textwrap.wrap(cap or "", 52)), fontsize=7, color=INK2)
        fig.suptitle("E12  Failure gallery: what the agent measured and did", fontsize=10, x=0.01, ha="left")
        fig.tight_layout()
        print(_save(fig, res / "figures", "E12_failure_gallery"))
    print(f"wrote {res / 'failure_gallery.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
