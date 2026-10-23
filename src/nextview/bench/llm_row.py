"""LLM row (spec section 9): a local LLM drives the MCP tools on the simulator, next to the
deterministic planner.

On the same cameras (default: the first 10 ``normal`` TEST cameras, 1000-1009, robot actuator,
policy seed 0, budget 10 accepted frames) it runs:

* ``A`` and ``B0`` through the benchmark's own runner (``bench.policies.run_job``): the agent
  loop, with its stop point recorded, and the random-pose baseline, to the budget;
* ``MCP-scripted``: a fixed client of the MCP server that always captures candidate 1, accepts
  every frame that passes the gate and exports at ``stop_ready`` or the budget (checks that the
  MCP tools reproduce the loop's maths);
* one LLM row per model through ``mcp_server.llm_driver`` (Ollama, temperature 0), with
  approvals answered by the recorded policy ``approve_all``.

Per run it records tool-call validity (over parsed tool calls and over attempted calls, which
also count replies holding a tool call Ollama could not parse; whitespace-repaired calls are
counted apart), steps to stop, agreement with the planner's rank-1 candidate (spec 9: share of
all captures; a capture at a model-supplied pose is a non-agreement and is also counted on its
own), the final U95 and the true field p95 / fx error against ground truth, and a sha256 of the
source tree that produced it. By default the model is not offered ``sim_capture``'s explicit
``pose`` argument (model flag ``@pose`` offers it).

Output: ``<out>/summary.json``, ``<out>/table.md``, ``<out>/runs.csv``, ``<out>/runs.json`` (the
full per-run records, the input of ``--from-runs``) and one JSONL call log per LLM episode at
``<out>/<model-slug>/run_<camera_seed>.jsonl`` (spec 9 named ``results/llm/run_<seed>.jsonl``;
the logs are kept per model and per camera).

    python -m nextview.bench.llm_row --models qwen2.5:7b nemotron-3-nano:4b@nothink --out results/llm_row
    python -m nextview.bench.llm_row --from-runs results/llm_row/runs.json --out results/llm_row
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["run_reference", "run_scripted", "run_llm", "aggregate", "main"]

REPO = Path(__file__).resolve().parents[3]


def _f(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _med(xs):
    v = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return round(float(np.median(v)), 4) if v else None


# ---- deterministic reference (the benchmark's runner) ---------------------------------------------

def run_reference(camera: int, policy: str, *, budget: int, seed: int, actuator: str, family: str, th, stopcal) -> dict:
    """``A`` or ``B0`` through ``bench.policies.run_job`` (identical to the TEST-split runner)."""
    from .common import SPLIT_NUISANCES, Job
    from .policies import run_job

    job = Job("test", camera, family, seed, actuator, policy, budget, SPLIT_NUISANCES[family])
    t = time.perf_counter()
    rows, sess = run_job(job, th, stopcal)
    by_n = {int(r["n"]): r for r in rows}
    last = rows[-1] if rows else {}
    stop = sess.get("stop") if policy == "A" else None
    at_stop = by_n.get(int(stop["n"])) if stop else None
    out = {"policy": policy, "camera_seed": camera, "seed": seed, "actuator": actuator, "family": family,
           "budget": budget, "final_views": last.get("views"), "accepted_views": last.get("n"),
           "captures": last.get("attempts"), "final_U95_px": _f(last.get("U95")), "final_true_p95_px": _f(last.get("true_p95")),
           "fx_rel_err_pct": _f(last.get("fx_rel_pct")), "final_p_exceed": _f(last.get("p_exceed_loop")),
           "model_final": last.get("model"),
           "true_p95_by_views": {int(r["n"]): _f(r.get("true_p95")) for r in rows},
           "U95_by_views": {int(r["n"]): _f(r.get("U95")) for r in rows},
           "wall_s": round(time.perf_counter() - t, 2), "end": sess.get("end"), "approvals": sess.get("approvals")}
    if policy == "A":
        # the session ends at the R11 stop card when it fires, else at the budget (R12)
        out.update({"stopped": stop is not None, "stop_views": None if not stop else stop["n"],
                    "stop_captures": None if not stop else stop["attempts"],
                    "stop_p_exceed": None if not at_stop else _f(at_stop.get("p_exceed_loop"))})
        if at_stop is not None:
            out.update({"final_views": at_stop.get("views"), "accepted_views": at_stop.get("n"),
                        "captures": at_stop.get("attempts"), "final_U95_px": _f(at_stop.get("U95")),
                        "final_p_exceed": _f(at_stop.get("p_exceed_loop")),
                        "final_true_p95_px": _f(at_stop.get("true_p95")), "fx_rel_err_pct": _f(at_stop.get("fx_rel_pct"))})
    return out


# ---- scripted MCP client ---------------------------------------------------------------------------

async def _scripted(camera: int, *, budget: int, seed: int, actuator: str, preset: str, stopcal: str) -> dict:
    from mcp import types
    from mcp.shared.memory import create_connected_server_and_client_session

    from ..mcp_server.server import build_server

    async def approve(ctx, params):
        return types.ElicitResult(action="accept", content={"approve": True, "reason": "scripted approve_all"})

    srv = build_server(stopcal=stopcal, evaluator=True)
    n_calls, t0 = 0, time.perf_counter()
    async with create_connected_server_and_client_session(srv, elicitation_callback=approve) as c:
        async def call(name, **a):
            nonlocal n_calls
            n_calls += 1
            r = await c.call_tool(name, a)
            if r.isError:
                raise RuntimeError(f"{name}: {r.content[0].text}")
            return r.structuredContent

        s = await call("sim_new_session", preset=preset, actuator=actuator, seed=seed, budget=budget, camera_seed=camera)
        sid = s["session_id"]
        n_calls = 0
        stopped_at = None
        dropped = 0
        for _ in range(60):
            await call("propose_next_views", session_id=sid)
            f = await call("sim_capture", session_id=sid, candidate_rank=1)
            if not f["gate"]["pass"]:
                continue
            a = await call("accept_view", session_id=sid, frame_id=f["frame_id"])
            if a["views"] >= 3:
                d = await call("residual_analysis", session_id=sid)
                sg = d.get("suggestion") or {}
                if sg.get("rule") == "R7_OUTLIER":
                    for vid in sg["view_ids"]:
                        await call("drop_view", session_id=sid, view_id=vid, reason="R7 outlier")
                        dropped += 1
                elif sg.get("rule") in ("R8_UNDERFIT", "R9_WARP"):
                    await call("calibrate", session_id=sid, model="rational" if sg["rule"] == "R8_UNDERFIT" else "ro")
                u = await call("uncertainty_map", session_id=sid)
                if u["stop_ready"] or u["views"] >= budget:
                    stopped_at = {"views": u["views"], "rule": u["rule"], "p_exceed": u["p_exceed"]}
                    break
        e = await call("export_calibration", session_id=sid)
        g = (await c.call_tool("ground_truth_error", {"session_id": sid})).structuredContent
        lg = (await c.call_tool("session_log", {"session_id": sid})).structuredContent
    t = g.get("truth") or {}
    acc = lg.get("accepts", [])  # per accepted frame, plus a row after each drop_view refit
    return {"policy": "MCP-scripted", "camera_seed": camera, "seed": seed, "actuator": actuator, "budget": budget,
            "tool_calls": n_calls, "validity": 1.0, "captures": len(lg.get("captures", [])),
            "accepted_views": sum(1 for a in acc if a.get("after_drop") is None),
            "final_views": lg.get("views"), "exported": bool(e.get("exported")), "export_warnings": e.get("warnings"),
            "dropped_views": dropped,
            "stop": stopped_at, "final_U95_px": g.get("U95_px"), "final_p_exceed": g.get("p_exceed"),
            "final_true_p95_px": t.get("field_p95_px"), "fx_rel_err_pct": t.get("fx_rel_err_pct"),
            "true_p95_by_views": {a.get("n", a["views"]): (a.get("truth") or {}).get("field_p95_px") for a in acc},
            "U95_by_views": {a.get("n", a["views"]): a.get("U95") for a in acc}, "wall_s": round(time.perf_counter() - t0, 2),
            "trace": lg.get("trace")}


def run_scripted(camera: int, **kw) -> dict:
    return asyncio.run(_scripted(camera, **kw))


# ---- LLM ------------------------------------------------------------------------------------------

def parse_model(spec: str) -> dict:
    """"name[@flag+flag]" -> EpisodeConfig overrides plus ``label``. Flags: ``think`` / ``nothink``
    (Ollama reasoning flag), ``nohints`` (hide the tools' "Next: ..." hints), ``hN`` (keep only
    the last N messages of history after the system prompt and the task), ``pose`` (also offer
    sim_capture's explicit pose argument to the model)."""
    name, _, opts = spec.partition("@")
    out = {"model": name, "think": None, "hints": True, "offer_pose": False, "label": spec}
    for f in filter(None, opts.split("+")):
        if f in ("think", "nothink"):
            out["think"] = f == "think"
        elif f == "nohints":
            out["hints"] = False
        elif f == "pose":
            out["offer_pose"] = True
        elif f[:1] == "h" and f[1:].isdigit() and int(f[1:]) >= 2:
            out["max_history"] = int(f[1:])
            out["trim_chunk"] = max(1, min(16, int(f[1:]) // 2))
        else:
            raise ValueError(f"unknown model flag {f!r} in {spec!r}")
    return out


def _slug(label: str) -> str:
    return label.replace(":", "_").replace("/", "_").replace("@", "_").replace("+", "_")


def run_llm(camera: int, model_spec: str, *, budget: int, seed: int, actuator: str, preset: str, stopcal: str,
            max_calls: int, out: Path, num_ctx: int, num_predict: int | None = 2048, llm_timeout_s: float = 120.0) -> dict:
    from ..mcp_server.llm_driver import EpisodeConfig, run_episode

    opts = parse_model(model_spec)
    label = opts.pop("label")
    log = out / _slug(label) / f"run_{camera}.jsonl"
    cfg = EpisodeConfig(preset=preset, actuator=actuator, camera_seed=camera, seed=seed, budget=budget,
                        max_calls=max_calls, approval="approve_all", stopcal=stopcal, num_ctx=num_ctx,
                        num_predict=num_predict, llm_timeout_s=llm_timeout_s, log_path=str(log), **opts)
    s = asyncio.run(run_episode(cfg))
    s["policy"] = f"LLM {label}"
    s["max_history"] = cfg.max_history
    s["log"] = _rel(log)
    return s


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(path).as_posix()


# ---- runs recorded before metrics v2 --------------------------------------------------------------

def upgrade_legacy(r: dict, runs_dir: Path) -> dict:
    """An LLM run recorded before metrics v2 (no ``metrics_version``), brought to v2.

    Those runs offered the model sim_capture's ``pose`` argument, so the label gets the ``pose``
    flag. Call validity is rescored from the run's JSONL log (incomplete-call turns, repaired
    calls); agreement is rebuilt from the recorded capture counts: opening captures = captures -
    NBV captures - pose captures, all rank 1 (one candidate), and rank-1 NBV captures = the old
    NBV-only agreement x NBV captures."""
    from ..mcp_server.llm_driver import METRICS_VERSION, call_metrics

    if not r.get("policy", "").startswith("LLM ") or r.get("metrics_version"):
        return r
    r = dict(r)
    old = r["policy"][4:]
    name, _, opts = old.partition("@")
    new = old if "pose" in opts.split("+") else (f"{old}+pose" if opts else f"{old}@pose")
    log = next((q for q in (runs_dir / _slug(new) / f"run_{r['camera_seed']}.jsonl",
                            runs_dir / _slug(old) / f"run_{r['camera_seed']}.jsonl") if q.exists()), None)
    if log is not None:
        recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        calls = [x for x in recs if x.get("kind") == "tool"]
        turns = [x for x in recs if x.get("kind") == "llm"]
        r.update(call_metrics(calls, turns))
        r["log"] = _rel(log)
    caps, nbv = int(r.get("captures") or 0), int(r.get("nbv_captures") or 0)
    pose = int((r.get("rank_choices") or {}).get("pose", 0))
    opening = caps - nbv - pose
    rank1_nbv = round((r.get("agreement_rank1") or 0) * nbv)
    r.update({"opening_captures": opening, "pose_captures": pose, "rank1_nbv_captures": rank1_nbv,
              "rank1_captures": opening + rank1_nbv,
              "agreement_rank1": round((opening + rank1_nbv) / caps, 4) if caps else None,
              "agreement_rank1_after_opening": round(rank1_nbv / (caps - opening), 4) if caps - opening else None,
              "policy": f"LLM {new}", "offer_pose": True, "metrics_version": METRICS_VERSION,
              "rescored_from_log": log is not None,
              "legacy": {"label": old, "recorded_before": "metrics v2; source tree not recorded (code uncommitted)"}})
    return r


# ---- aggregation ---------------------------------------------------------------------------------

def _at(d: dict | None, n: int):
    if not d:
        return None
    return d.get(n, d.get(str(n)))


def aggregate(runs: list[dict], budget: int) -> dict:
    """Per-policy medians and pooled call validity; paired LLM-vs-A differences on the same cameras."""
    pols: dict[str, list[dict]] = {}
    for r in runs:
        pols.setdefault(r["policy"], []).append(r)
    ref = {r["camera_seed"]: r for r in pols.get("A", [])}
    b0 = {r["camera_seed"]: r for r in pols.get("B0", [])}
    out = {}
    for p, rs in pols.items():
        a = {"runs": len(rs), "cameras": sorted(r["camera_seed"] for r in rs)}
        if p.startswith("LLM") or p == "MCP-scripted":
            calls = sum(r.get("tool_calls") or 0 for r in rs)
            ok = sum(r.get("valid_calls") or 0 for r in rs) if p.startswith("LLM") else calls
            a.update({"tool_calls_total": calls, "valid_calls_total": ok,
                      "validity_pooled": round(ok / calls, 4) if calls else None,
                      "exported": sum(1 for r in rs if r.get("exported")),
                      "median_tool_calls": _med(r.get("tool_calls") for r in rs)})
        if p.startswith("LLM"):
            inv = sum(r["tool_calls"] - r["valid_calls"] for r in rs)

            def tot(k):
                return sum(r.get(k) or 0 for r in rs)

            att, caps, after = tot("attempted_calls"), tot("captures"), tot("captures") - tot("opening_captures")
            cats, codes = {}, {}
            for r in rs:
                for k, v in (r.get("categories") or {}).items():
                    cats[k] = cats.get(k, 0) + v
                for k, v in (r.get("refusal_codes") or {}).items():
                    codes[k] = codes.get(k, 0) + v
            a.update({"invalid_calls_total": inv, "invalid_attempts_total": att - ok, "categories": cats,
                      "refusal_codes": codes,
                      "offer_pose": sorted({bool(r.get("offer_pose")) for r in rs}),
                      "incomplete_call_turns_total": tot("incomplete_call_turns"), "attempted_calls_total": att,
                      "repaired_calls_total": tot("repaired_calls"),
                      "validity_attempted_pooled": round(ok / att, 4) if att else None,
                      "validity_incl_repaired_pooled": round((ok + tot("repaired_calls")) / calls, 4) if calls else None,
                      "schema_valid_pooled": round(1 - (cats.get("unknown_tool", 0) + cats.get("bad_args", 0)) /
                                                   max(1, sum(cats.values())), 4),
                      "captures_total": caps, "nbv_captures": tot("nbv_captures"),
                      "opening_captures": tot("opening_captures"), "pose_captures_total": tot("pose_captures"),
                      "pose_arg_calls_total": tot("pose_arg_calls"), "pose_arg_calls_refused": tot("pose_arg_calls_refused"),
                      "agreement_rank1_pooled": round(tot("rank1_captures") / caps, 4) if caps else None,
                      "agreement_rank1_after_opening_pooled": round(tot("rank1_nbv_captures") / after, 4) if after else None,
                      "end_reasons": {k: sum(1 for r in rs if r.get("end_reason") == k) for k in
                                      sorted({r.get("end_reason") for r in rs}, key=str)},
                      "nudges_total": sum(r.get("nudges") or 0 for r in rs),
                      "nudges_incomplete_call_total": sum(r.get("nudges_incomplete_call") or 0 for r in rs),
                      "whitespace_stripped_calls_total": sum(r.get("whitespace_stripped_calls") or 0 for r in rs),
                      "exported_within_40_calls": sum(1 for r in rs if r.get("exported_within_40")),
                      "median_llm_turns": _med(r.get("llm_turns") for r in rs),
                      "median_wall_s": _med(r.get("wall_s") for r in rs),
                      "median_llm_ms_per_turn": _med(r.get("llm_ms_median") for r in rs)})
        # session end: A at its R11 stop card or the budget; MCP clients at the export (or where they gave up)
        if p == "A":
            a["stopped_by_rule"] = sum(1 for r in rs if r.get("stopped"))
            a["median_stop_p_exceed"] = _med(r.get("stop_p_exceed") for r in rs)
        elif p.startswith("LLM") or p == "MCP-scripted":
            a["exported_clean_stop"] = sum(1 for r in rs if r.get("exported") and r.get("export_warnings") is False)
        a["median_final_p_exceed"] = _med(r.get("final_p_exceed") for r in rs)
        fp = [r.get("final_p_exceed") for r in rs if r.get("final_p_exceed") is not None]
        a["final_p_exceed_range"] = [min(fp), max(fp)] if fp else None
        a.update({
            "median_end_accepted_frames": _med(r.get("accepted_views") for r in rs),
            "median_end_views": _med(r.get("final_views") for r in rs),
            "median_captures": _med(r.get("captures") for r in rs),
            "median_final_U95_px": _med(r.get("final_U95_px") for r in rs),
            "median_final_true_p95_px": _med(r.get("final_true_p95_px") for r in rs),
            "median_fx_rel_err_pct": _med(r.get("fx_rel_err_pct") for r in rs),
            f"median_true_p95_at_{budget}_frames_px": _med(_at(r.get("true_p95_by_views"), budget) for r in rs),
            f"reached_{budget}_frames": sum(1 for r in rs if _at(r.get("true_p95_by_views"), budget) is not None),
        })
        if p != "A" and ref:
            d_end, d_n, d_fx = [], [], []
            for r in rs:
                A = ref.get(r["camera_seed"])
                if A is None:
                    continue
                if r.get("final_true_p95_px") is not None and A.get("final_true_p95_px") is not None and (
                        p == "B0" or r.get("exported")):
                    d_end.append(r["final_true_p95_px"] - A["final_true_p95_px"])
                    if r.get("fx_rel_err_pct") is not None and A.get("fx_rel_err_pct") is not None:
                        d_fx.append(r["fx_rel_err_pct"] - A["fx_rel_err_pct"])
                x, y = _at(r.get("true_p95_by_views"), budget), _at(A.get("true_p95_by_views"), budget)
                if x is not None and y is not None:
                    d_n.append(x - y)
            a["paired_median_diff_true_p95_vs_A_at_end_px"] = _med(d_end)
            a["paired_median_diff_fx_err_vs_A_at_end_pct"] = _med(d_fx)
            a["paired_n_at_end"] = len(d_end)
            a[f"paired_median_diff_true_p95_vs_A_at_{budget}_frames_px"] = _med(d_n)
            a[f"paired_n_at_{budget}_frames"] = len(d_n)
        if b0 and p != "B0":
            d = [(_at(r.get("true_p95_by_views"), budget), _at(b0[r["camera_seed"]].get("true_p95_by_views"), budget))
                 for r in rs if r["camera_seed"] in b0]
            d = [x - y for x, y in d if x is not None and y is not None]
            a[f"paired_median_diff_true_p95_vs_B0_at_{budget}_px"] = _med(d)
        out[p] = a
    return out


CSV_FIELDS = ("policy", "camera_seed", "end_reason", "stopped", "tool_calls", "incomplete_call_turns", "attempted_calls",
              "valid_calls", "repaired_calls", "validity", "validity_attempted", "captures", "pose_captures",
              "accepted_views", "final_views", "exported", "export_warnings", "agreement_rank1",
              "agreement_rank1_after_opening", "dropped_views", "final_U95_px", "final_p_exceed", "final_true_p95_px",
              "fx_rel_err_pct", "true_p95_at_budget", "nudges", "nudges_incomplete_call", "wall_s", "code", "log")


def _pct(x):
    return "–" if x is None else f"{100 * x:.0f} %"


def _num(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def table_md(agg: dict, budget: int) -> str:
    """The report table, straight from the aggregate (no hand copying)."""
    head = ("| Policy | Runs | Valid / tool calls | Valid / attempted | Repaired | Rank-1 agreement (all / after opening) "
            "| Pose captures | Exported | Median end views | Median true p95 at end (px) "
            f"| Median true p95 at {budget} frames (px) | Median final P_exceed |")
    rows = [head, "|" + "---|" * 12]
    for p, a in agg.items():
        if p.startswith("LLM") or p == "MCP-scripted":
            v = f"{a.get('valid_calls_total')}/{a.get('tool_calls_total')} ({_pct(a.get('validity_pooled'))})"
            att = ("–" if p == "MCP-scripted" else
                   f"{a.get('valid_calls_total')}/{a.get('attempted_calls_total')} ({_pct(a.get('validity_attempted_pooled'))})")
            rep = "–" if p == "MCP-scripted" else str(a.get("repaired_calls_total"))
            agr = "–" if p == "MCP-scripted" else (f"{_pct(a.get('agreement_rank1_pooled'))} / "
                                                   f"{_pct(a.get('agreement_rank1_after_opening_pooled'))}")
            pose = "–" if p == "MCP-scripted" else str(a.get("pose_captures_total"))
            exp = f"{a.get('exported')}/{a['runs']}"
        else:
            v = att = rep = agr = pose = exp = "–"
        rows.append(f"| {p} | {a['runs']} | {v} | {att} | {rep} | {agr} | {pose} | {exp} | "
                    f"{_num(a.get('median_end_views'), 1)} | {_num(a.get('median_final_true_p95_px'))} | "
                    f"{_num(a.get(f'median_true_p95_at_{budget}_frames_px'))} | {_num(a.get('median_final_p_exceed'))} |")
    return "\n".join(rows) + "\n"


def _env(models: list[str]) -> dict:
    import httpx

    from ..config import config_dir
    from .run_synthetic import code_manifest

    env = {"opencv": cv2.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}
    try:
        env["git_head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                         text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        env["git_head"] = None
    code = code_manifest()
    env["source_tree_sha256"] = code["source_tree_sha256"]  # every nextview/*.py, committed or not
    env["git_dirty_files"] = code["git"].get("dirty_files")
    for name in ("thresholds.json", "stop_calibration.json"):
        p = Path(config_dir()) / name
        env[f"{name}_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None
    try:
        env["ollama"] = httpx.get("http://localhost:11434/api/version", timeout=5).json().get("version")
        tags = httpx.get("http://localhost:11434/api/tags", timeout=5).json().get("models", [])
        want = {parse_model(m)["model"] for m in models}
        env["ollama_models"] = {t["name"]: t.get("digest", "")[:12] for t in tags if t["name"] in want}
    except Exception:  # noqa: BLE001 - Ollama not running: recorded as unknown
        env["ollama"] = None
    return env


def _cameras(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nextview.bench.llm_row", description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", default=["qwen2.5:7b", "nemotron-3-nano:4b@nothink"],
                    help="Ollama models; suffix @think / @nothink sets the reasoning flag, @nohints hides the "
                         "tools' 'Next: ...' hints, @pose offers sim_capture's explicit pose argument, @hN keeps "
                         "N messages of history (flags join with +)")
    ap.add_argument("--cameras", default="1000-1009", help="camera seeds, e.g. 1000-1009 (normal TEST cameras)")
    ap.add_argument("--family", default="normal")
    ap.add_argument("--preset", default="inspection_gauge_cam", help="MCP preset with the same family / actuator")
    ap.add_argument("--actuator", default="robot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--max-calls", type=int, default=60, help="spec: 40; 60 leaves room for recoverable errors "
                                                              "(views and export within 40 calls are reported too)")
    ap.add_argument("--stopcal", default="config")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--skip-scripted", action="store_true")
    ap.add_argument("--num-predict", type=int, default=2048, help="token cap per LLM reply (reasoning included)")
    ap.add_argument("--llm-timeout", type=float, default=120.0,
                    help="seconds per Ollama request; a timeout ends the episode (llm_error), it is not retried")
    ap.add_argument("--from-runs", nargs="*", default=None, metavar="RUNS_JSON",
                    help="run nothing: aggregate runs.json / summary.json file(s) of earlier invocations (the first "
                         "copy of a policy x camera wins; runs from before metrics v2 are rescored from their logs)")
    ap.add_argument("--out", default="results/llm_row")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cv2.setNumThreads(1)
    from ..config import load_thresholds
    from ..mcp_server.core import load_stopcal

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    th = load_thresholds()
    stopcal = load_stopcal(a.stopcal)
    cams = _cameras(a.cameras)
    common = dict(budget=a.budget, seed=a.seed, actuator=a.actuator)
    runs: list[dict] = []
    t0 = time.perf_counter()
    env = _env(a.models)
    code = env["source_tree_sha256"][:16]
    source_envs = {}

    def note(r):
        r["code"] = code
        runs.append(r)
        print(f"[{time.perf_counter() - t0:7.1f}s] {r['policy']:<32} cam {r['camera_seed']}: frames "
              f"{r.get('accepted_views')} views {r.get('final_views')}, U95 {r.get('final_U95_px')}, true p95 "
              f"{r.get('final_true_p95_px')} px, fx err {r.get('fx_rel_err_pct')}%, calls {r.get('tool_calls', '-')}, "
              f"valid {r.get('validity', '-')}, end {r.get('end_reason', r.get('end'))}", flush=True)
        (out / "runs.json").write_text(json.dumps(runs, indent=1, default=str), encoding="utf-8")

    if a.from_runs is not None:
        seen = set()
        for path in a.from_runs:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(doc, dict):  # a summary.json: its runs, and the environment that produced them
                source_envs[_rel(Path(path))] = doc.get("env")
                doc = doc["runs"]
            for r in doc:
                r = upgrade_legacy(r, Path(path).parent)
                r.setdefault("code", None)  # None: recorded before source hashes were stamped on runs
                key = (r["policy"], r["camera_seed"])
                if key not in seen:  # A / B0 are deterministic: keep one copy per camera
                    seen.add(key)
                    runs.append(r)
        cams = sorted({r["camera_seed"] for r in runs})
        a.models = sorted({r["policy"][4:] for r in runs if r["policy"].startswith("LLM ")})
    for cam in ([] if a.from_runs is not None else cams):
        if not a.skip_reference:
            for pol in ("A", "B0"):
                note(run_reference(cam, pol, family=a.family, th=th, stopcal=stopcal, **common))
        if not a.skip_scripted:
            note(run_scripted(cam, preset=a.preset, stopcal=a.stopcal, **common))
    for m in ([] if a.from_runs is not None else a.models):  # one model at a time: the GPU may be shared
        for cam in cams:
            note(run_llm(cam, m, preset=a.preset, stopcal=a.stopcal, max_calls=a.max_calls, out=out, num_ctx=a.num_ctx,
                         num_predict=a.num_predict, llm_timeout_s=a.llm_timeout, **common))
    agg = aggregate(runs, a.budget)
    summary = {"what": "LLM row (spec 9): a local LLM drives the MCP tools; A and B0 from bench.policies.run_job",
               "config": {"cameras": cams, "family": a.family, "preset": a.preset, "actuator": a.actuator, "seed": a.seed,
                          "budget": a.budget, "max_calls": a.max_calls, "stopcal": a.stopcal,
                          "stopcal_calibrated": stopcal.calibrated, "tau_px": th.tau_px, "approval_policy": "approve_all",
                          "models": a.models, "temperature": 0, "num_ctx": a.num_ctx, "num_predict": a.num_predict,
                          "llm_timeout_s": a.llm_timeout, "aggregated_from": a.from_runs},
               "env": env, "source_envs": source_envs or None,
               "codes": {p: sorted({str(r.get("code")) for r in runs if r["policy"] == p}) for p in agg},
               "aggregate": agg, "wall_s": round(time.perf_counter() - t0, 1),
               "runs": [{k: v for k, v in r.items() if k not in ("true_p95_by_views", "U95_by_views", "approvals", "end", "trace")}
                        | {"true_p95_by_views": r.get("true_p95_by_views")} for r in runs]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({**r, "true_p95_at_budget": _at(r.get("true_p95_by_views"), a.budget)})
    (out / "runs.json").write_text(json.dumps(runs, indent=1, default=str), encoding="utf-8")
    (out / "table.md").write_text(table_md(agg, a.budget), encoding="utf-8")
    print(json.dumps(agg, indent=1, default=str))
    print(table_md(agg, a.budget))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
