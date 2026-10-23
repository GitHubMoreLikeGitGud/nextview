"""A local LLM (Ollama, over HTTP with httpx) drives the NextView MCP server (spec section 9).

The driver starts ``nextview.mcp_server.server`` as an MCP stdio subprocess, converts the
server's tool schemas to Ollama's tool format and loops ``POST /api/chat`` (``stream=false``,
``temperature=0``). The model only chooses tool calls; every measurement, fit, uncertainty map,
stop check and candidate pose comes from the tools. Approvals asked by gated tools (MCP
elicitation) go to the terminal operator (``--approval ask``) or, for benchmarks, to a fixed
policy (``approve_all`` / ``deny_all``) that is recorded with the results.

The harness itself (not the model) opens the session, so every policy is run on the same
simulated camera, and reads the evaluator-only tools (``ground_truth_error``, ``session_log``)
after the episode: the server is started with ``--evaluator`` for that, and those tools are never
offered to the model. By default the model is not offered ``sim_capture``'s explicit ``pose``
argument either, so it can only choose among the planner's candidates (``offer_pose=True``
restores it); a call that still passes it is refused by the driver (``ARG_NOT_OFFERED``).

Every LLM turn and tool call is logged as one JSON line. Validity is reported over the tool
calls Ollama parsed and over attempted calls (tool calls plus replies holding a tool call that
Ollama could not parse); calls whose string arguments the driver had to strip of whitespace
count as ``repaired``, not as valid.

    python -m nextview.mcp_server.llm_driver --model qwen2.5:7b --camera-seed 1000 --budget 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

__all__ = ["EpisodeConfig", "ollama_tools", "simplify_schema", "system_prompt", "run_episode", "classify_result",
           "call_metrics", "capture_metrics", "summarise", "main"]

OLLAMA_URL = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")
LLM_TOOLS = ("sim_capture", "detect_board", "assess_frame", "accept_view", "calibrate", "uncertainty_map",
             "residual_analysis", "propose_next_views", "drop_view", "export_calibration", "session_status")
HIDDEN_TOOLS = ("ground_truth_error", "session_log")  # evaluator only, never offered to the model
POSE_ARGS = {"sim_capture": ("pose",)}  # model-supplied poses: offered only with offer_pose
METRICS_VERSION = 2  # 2: spec agreement over all captures, attempted-call validity, repaired calls
NUDGE = ("Continue by calling the next tool. When the calibration is finished, call export_calibration; "
         "after the export reply DONE.")
NUDGE_INCOMPLETE = "Your last reply was an incomplete tool call. Send it again as one complete tool call with all its arguments."


def system_prompt(budget: int) -> str:
    """The spec's prompt (section 9) plus the operating rules of the tool interface."""
    return ("You calibrate a camera. Each turn: capture, assess, accept or reject, calibrate, check uncertainty, "
            "choose the next view from propose_next_views. "
            f"Stop when P_exceed<0.10 or after {budget} accepted views.\n"
            "Rules: act only by calling tools; the tools do every measurement and calculation, so never invent "
            "numbers. Use the session_id you are given. sim_capture already detects and assesses the frame; "
            "accept_view only frames whose gate passed. Candidate lists are planned for the current views, so call "
            "propose_next_views again after every accepted view. To stop, call export_calibration (a human "
            "approves it), then reply DONE.")


def simplify_schema(schema: Any) -> Any:
    """Tool input schema for small models: ``anyOf [X, null]`` -> X, no titles."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "title":
                continue
            out[k] = simplify_schema(v)
        any_of = out.get("anyOf")
        if isinstance(any_of, list):
            non_null = [s for s in any_of if not (isinstance(s, dict) and s.get("type") == "null")]
            if len(non_null) == 1:
                out.pop("anyOf")
                merged = dict(non_null[0])
                merged.update({k: v for k, v in out.items() if k not in merged})
                out = merged
        if out.get("default", 0) is None:
            out.pop("default")
        return out
    if isinstance(schema, list):
        return [simplify_schema(s) for s in schema]
    return schema


def ollama_tools(tools: list[types.Tool], allow: tuple[str, ...] = LLM_TOOLS,
                 hidden_args: dict[str, tuple[str, ...]] | None = None) -> list[dict]:
    """MCP ``list_tools()`` result -> Ollama ``tools`` (only ``allow``, never the evaluator tools).
    ``hidden_args`` (tool -> argument names) are removed from the schema and from the description."""
    out = []
    for t in tools:
        if t.name not in allow or t.name in HIDDEN_TOOLS:
            continue
        params = simplify_schema(t.inputSchema or {"type": "object", "properties": {}})
        desc = t.description or ""
        for arg in (hidden_args or {}).get(t.name, ()):
            (params.get("properties") or {}).pop(arg, None)
            if arg in (params.get("required") or []):
                params["required"] = [r for r in params["required"] if r != arg]
            desc = re.sub(rf"\s*\b{re.escape(arg)}: [^.]*\.", "", desc)
        out.append({"type": "function", "function": {"name": t.name, "description": desc, "parameters": params}})
    return out


def classify_result(res: types.CallToolResult) -> tuple[str, str | None]:
    """("ok" | "refused" | "needs_approval" | "bad_args" | "server_error", error code)."""
    sc = res.structuredContent or {}
    if res.isError:
        err = sc.get("error") if isinstance(sc, dict) else None
        if isinstance(err, dict):
            code = err.get("code")
            return ("server_error" if code == "INTERNAL" else "refused"), code
        return "bad_args", "VALIDATION"
    if isinstance(sc, dict) and sc.get("status") == "needs_approval":
        return "needs_approval", None
    return "ok", None


def _text(res: types.CallToolResult) -> str:
    return "\n".join(c.text for c in res.content if getattr(c, "type", None) == "text")


@dataclass
class EpisodeConfig:
    model: str = "qwen2.5:7b"
    preset: str = "inspection_gauge_cam"
    actuator: str = "robot"
    camera_seed: int = 1000
    seed: int = 0
    budget: int = 10
    max_calls: int = 40
    approval: str = "ask"  # ask | approve_all | deny_all
    stopcal: str = "config"
    ollama_url: str = OLLAMA_URL
    num_ctx: int = 16384
    num_predict: int | None = 2048  # tokens per reply, reasoning included
    think: bool | None = None
    temperature: float = 0.0
    max_history: int = 48  # messages kept after the system prompt and the task
    trim_chunk: int = 16
    max_nudges: int = 40
    max_consecutive_text: int = 4  # text-only replies in a row before the episode counts as stalled
    llm_timeout_s: float = 300.0  # a timed-out request is not retried (temperature 0 replays it)
    offer_pose: bool = False  # offer sim_capture's explicit pose argument to the model
    log_path: str | None = None
    export_dir: str | None = None
    debug_dir: str | None = None  # dump the request when a reply holds an unparsed tool call
    hints: bool = True  # False: drop the tools' "Next: ..." hints from what the model sees
    extra: dict = field(default_factory=dict)


class _Log:
    def __init__(self, path: str | None):
        self.f = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "w", encoding="utf-8")

    def write(self, rec: dict) -> None:
        if self.f:
            self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self.f.flush()

    def close(self) -> None:
        if self.f:
            self.f.close()


def _elicitation_callback(policy: str, record: list):
    async def cb(context, params) -> types.ElicitResult:
        msg = getattr(params, "message", "")
        if policy == "approve_all":
            ans = {"approve": True, "reason": "benchmark policy approve_all"}
        elif policy == "deny_all":
            ans = {"approve": False, "reason": "benchmark policy deny_all"}
        else:
            import anyio

            print(f"\n[APPROVAL] {msg}\nApprove? [y/n] ", end="", file=sys.stderr, flush=True)
            line = (await anyio.to_thread.run_sync(sys.stdin.readline)).strip().lower()
            ans = {"approve": line.startswith("y"), "reason": f"operator answered {line or 'n'!r}"}
        record.append({"message": msg, "answer": ans, "policy": policy})
        return types.ElicitResult(action="accept", content=ans)

    return cb


def _trim(messages: list[dict], head: int, cfg: EpisodeConfig) -> list[dict]:
    """Keep the system prompt and the task, and the most recent turns (dropped in chunks so the
    prompt prefix, and Ollama's cache of it, changes rarely). Never starts on a tool message."""
    rest = messages[head:]
    if len(rest) <= cfg.max_history:
        return messages
    cut = len(rest) - cfg.max_history + cfg.trim_chunk
    while cut < len(rest) and rest[cut].get("role") == "tool":
        cut += 1
    return messages[:head] + rest[cut:]


async def _chat(client: httpx.AsyncClient, cfg: EpisodeConfig, messages: list[dict], tools: list[dict]) -> dict:
    opts: dict[str, Any] = {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx, "seed": 0}
    if cfg.num_predict:
        opts["num_predict"] = int(cfg.num_predict)  # caps runaway reasoning (done_reason "length")
    body: dict[str, Any] = {"model": cfg.model, "messages": messages, "tools": tools, "stream": False, "options": opts}
    if cfg.think is not None:
        body["think"] = bool(cfg.think)
    last = None
    for attempt in range(3):
        try:
            r = await client.post(f"{cfg.ollama_url}/api/chat", json=body, timeout=cfg.llm_timeout_s)
            r.raise_for_status()
            return r.json()
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
            # at temperature 0 and a fixed seed the same request runs away the same way again
            raise RuntimeError(f"Ollama /api/chat timed out after {cfg.llm_timeout_s:.0f} s "
                               f"({type(e).__name__}); not retried") from None
        except (httpx.HTTPError, ValueError) as e:  # connection errors, 5xx, a broken body
            last = e
            await asyncio.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Ollama /api/chat failed: {last}")


def _args_of(call: dict) -> tuple[dict | None, str | None]:
    fn = call.get("function") or {}
    a = fn.get("arguments", {})
    if isinstance(a, str):
        try:
            a = json.loads(a) if a.strip() else {}
        except json.JSONDecodeError as e:
            return None, f"arguments are not valid JSON ({e})"
    if not isinstance(a, dict):
        return None, f"arguments must be an object, got {type(a).__name__}"
    return a, None


_DONE = re.compile(r"\bDONE\b")
_NEXT_HINT = re.compile(r"\s*Next: [^\n]*")


def strip_hints(text: str) -> str:
    """Tool result without its procedural "Next: ..." hint (the hint ablation). Measurements,
    gate verdicts, rule suggestions and the stop-rule recommendation stay."""
    return _NEXT_HINT.sub("", text, count=1)
_PARTIAL = re.compile(r"<tool_call>|<function=")


def _incomplete_call(content: str | None) -> bool:
    """A reply whose text is a tool call that Ollama's parser could not complete (seen with
    nemotron-3-nano: ``<tool_call>\\n<function=NAME>`` and nothing after it)."""
    return bool(content) and bool(_PARTIAL.search(content))


def _strip_strings(a: dict | None) -> tuple[dict | None, bool]:
    """Strip leading/trailing whitespace from string arguments. Ollama's parser for the XML-style
    tool format keeps the newlines around each parameter value (``" \\nnv_1a2b "``); the number
    of affected calls is reported."""
    if not isinstance(a, dict):
        return a, False
    out, hit = {}, False
    for k, v in a.items():
        if isinstance(v, str) and v != v.strip():
            v, hit = v.strip(), True
        out[k] = v
    return out, hit


async def run_episode(cfg: EpisodeConfig) -> dict:
    """One LLM-driven calibration session. Returns the per-episode summary."""
    log = _Log(cfg.log_path)
    approvals: list = []
    args = ["-m", "nextview.mcp_server.server", "--stopcal", cfg.stopcal, "--evaluator"]
    if cfg.export_dir:
        args += ["--export-dir", cfg.export_dir]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    params = StdioServerParameters(command=sys.executable, args=args, env=env)
    t_start = time.perf_counter()
    calls: list[dict] = []
    llm_turns: list[dict] = []
    end_reason = None
    nudges = nudges_incomplete = ws_args = 0
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, elicitation_callback=_elicitation_callback(cfg.approval, approvals)) as mcp:
                await mcp.initialize()
                listed = (await mcp.list_tools()).tools
                hidden_args = {} if cfg.offer_pose else POSE_ARGS
                tools = ollama_tools(listed, hidden_args=hidden_args)
                allowed = {t["function"]["name"] for t in tools}
                res = await mcp.call_tool("sim_new_session", {"preset": cfg.preset, "actuator": cfg.actuator,
                                                              "seed": cfg.seed, "budget": cfg.budget,
                                                              "camera_seed": cfg.camera_seed})
                if res.isError:
                    raise RuntimeError(f"sim_new_session failed: {_text(res)}")
                sess = res.structuredContent
                sid = sess["session_id"]
                task = (f"Session {sid} is ready: {sess['preset']['label']} ({sess['preset']['family']} lens), "
                        f"{sess['actuator']} actuator, budget {cfg.budget} accepted views, target U95 tau "
                        f"{sess['tau_px']} px. Stop rule: {sess['stop_rule']}. Calibrate the camera now.")
                messages = [{"role": "system", "content": system_prompt(cfg.budget)}, {"role": "user", "content": task}]
                log.write({"kind": "start", "config": {k: v for k, v in cfg.__dict__.items() if k != "extra"},
                           "session_id": sid, "tools": sorted(allowed), "hidden_args": hidden_args,
                           "system": messages[0]["content"], "task": task})
                consecutive_text = 0
                async with httpx.AsyncClient() as http:
                    while end_reason is None:
                        if len(calls) >= cfg.max_calls:
                            end_reason = "call_cap"
                            break
                        messages = _trim(messages, 2, cfg)
                        t = time.perf_counter()
                        try:
                            resp = await _chat(http, cfg, messages, tools)
                            msg = resp.get("message") or {}
                        except RuntimeError as e:
                            end_reason = "llm_error"
                            log.write({"kind": "llm_error", "error": str(e)})
                            break
                        ms = 1000 * (time.perf_counter() - t)
                        tcs = msg.get("tool_calls") or []
                        turn = {"kind": "llm", "turn": len(llm_turns), "ms": round(ms, 1),
                                "prompt_tokens": resp.get("prompt_eval_count"), "eval_tokens": resp.get("eval_count"),
                                "done_reason": resp.get("done_reason"), "content": (msg.get("content") or "")[:2000],
                                "n_tool_calls": len(tcs), "thinking_chars": len(msg.get("thinking") or ""),
                                "incomplete_call": not tcs and _incomplete_call(msg.get("content"))}
                        llm_turns.append(turn)
                        log.write(turn)
                        if cfg.debug_dir and not tcs and "<tool_call>" in (msg.get("content") or ""):
                            Path(cfg.debug_dir).mkdir(parents=True, exist_ok=True)
                            Path(cfg.debug_dir, f"unparsed_{cfg.camera_seed}_{turn['turn']}.json").write_text(
                                json.dumps({"messages": messages, "tools": tools, "response": resp},
                                           ensure_ascii=False), encoding="utf-8")
                        keep = {"role": "assistant", "content": msg.get("content") or ""}
                        if tcs:
                            keep["tool_calls"] = tcs
                        messages.append(keep)
                        if not tcs:
                            consecutive_text += 1
                            if _DONE.search(msg.get("content") or ""):
                                end_reason = "done"
                            elif consecutive_text >= cfg.max_consecutive_text or nudges >= cfg.max_nudges:
                                end_reason = "stalled"
                            else:
                                nudges += 1
                                inc = turn["incomplete_call"]
                                nudges_incomplete += int(inc)
                                messages.append({"role": "user", "content": NUDGE_INCOMPLETE if inc else NUDGE})
                                log.write({"kind": "nudge", "n": nudges, "incomplete_call": inc})
                            continue
                        consecutive_text = 0
                        for tc in tcs:
                            if len(calls) >= cfg.max_calls:
                                end_reason = "call_cap"
                                break
                            name = (tc.get("function") or {}).get("name")
                            a, perr = _args_of(tc)
                            a, ws = _strip_strings(a)
                            ws_args += int(ws)
                            rec = {"kind": "tool", "i": len(calls), "turn": turn["turn"], "tool": name, "args": a,
                                   "ws_stripped": ws}
                            t = time.perf_counter()
                            not_offered = [k for k in hidden_args.get(name, ()) if (a or {}).get(k) is not None]
                            if name not in allowed:
                                cat, code = "unknown_tool", None
                                text = f"error: unknown tool {name!r}; available tools: {', '.join(sorted(allowed))}"
                                sc = None
                            elif perr is not None:
                                cat, code, text, sc = "bad_args", "JSON", f"error: {perr}", None
                            elif not_offered:
                                cat, code, sc = "bad_args", "ARG_NOT_OFFERED", None
                                text = (f"error: {name} takes no {', '.join(not_offered)} argument here; choose a "
                                        f"candidate_rank from propose_next_views")
                            else:
                                res = await mcp.call_tool(name, a)
                                cat, code = classify_result(res)
                                text, sc = _text(res), res.structuredContent
                                if cat == "ok" and ws:
                                    cat = "ok_repaired"  # it worked only after the driver stripped whitespace
                            rec.update({"category": cat, "code": code, "ms": round(1000 * (time.perf_counter() - t), 1),
                                        "result": text[:1500]})
                            calls.append(rec)
                            log.write(rec)
                            tool_msg = {"role": "tool", "content": text if cfg.hints else strip_hints(text),
                                        "tool_name": name or ""}
                            if tc.get("id"):
                                tool_msg["tool_call_id"] = tc["id"]
                            messages.append(tool_msg)
                            if isinstance(sc, dict) and (sc.get("exported") is True or sc.get("status") == "discarded"):
                                end_reason = "exported" if sc.get("exported") else "discarded"
                                break
                gt = await mcp.call_tool("ground_truth_error", {"session_id": sid})
                lg = await mcp.call_tool("session_log", {"session_id": sid})
                truth = gt.structuredContent or {}
                slog = lg.structuredContent or {}
    finally:
        log.close()
    summary = summarise(calls, llm_turns, slog, truth, approvals)
    summary.update({"model": cfg.model, "camera_seed": cfg.camera_seed, "seed": cfg.seed, "preset": cfg.preset,
                    "actuator": cfg.actuator, "budget": cfg.budget, "stopcal": cfg.stopcal, "approval_policy": cfg.approval,
                    "max_calls": cfg.max_calls, "end_reason": end_reason, "nudges": nudges, "nudges_incomplete_call": nudges_incomplete,
                    "whitespace_stripped_calls": ws_args, "session_id": sid,
                    "wall_s": round(time.perf_counter() - t_start, 1), "think": cfg.think, "hints": cfg.hints,
                    "offer_pose": cfg.offer_pose, "llm_timeout_s": cfg.llm_timeout_s})
    if cfg.log_path:
        with open(cfg.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "summary", **summary}, ensure_ascii=False, default=str) + "\n")
    return summary


WORKED = ("ok", "ok_repaired")


def _category(c: dict) -> str:
    """The call's category; logs written before ``ok_repaired`` existed marked a call that worked
    only after whitespace stripping as plain ``ok``."""
    cat = c.get("category")
    return "ok_repaired" if cat == "ok" and c.get("ws_stripped") else cat


def _r4(a: int, b: int):
    return round(a / b, 4) if b else None


def call_metrics(calls: list[dict], turns: list[dict]) -> dict:
    """Validity of the model's calls. ``valid`` = the server accepted the call as sent;
    ``repaired`` = it worked only after the driver stripped whitespace from string arguments;
    ``attempted`` = parsed tool calls plus replies holding a tool call Ollama could not parse
    (those never reach the server, so they are invalid attempts)."""
    n = len(calls)
    cats: dict[str, int] = {}
    codes: dict[str, int] = {}
    by_tool: dict[str, int] = {}
    for c in calls:
        cat = _category(c)
        cats[cat] = cats.get(cat, 0) + 1
        if c.get("code"):
            codes[c["code"]] = codes.get(c["code"], 0) + 1
        by_tool[c.get("tool") or "?"] = by_tool.get(c.get("tool") or "?", 0) + 1
    incomplete = sum(1 for x in turns if not x.get("n_tool_calls")
                     and (x.get("incomplete_call") or _incomplete_call(x.get("content"))))
    ok, rep, att = cats.get("ok", 0), cats.get("ok_repaired", 0), n + incomplete
    pose_calls = [c for c in calls if c.get("tool") == "sim_capture" and isinstance(c.get("args"), dict)
                  and c["args"].get("pose") is not None]
    return {
        "tool_calls": n,
        "llm_turns": len(turns),
        "incomplete_call_turns": incomplete,
        "attempted_calls": att,
        "valid_calls": ok,
        "repaired_calls": rep,
        "validity": _r4(ok, n),
        "validity_attempted": _r4(ok, att),
        "validity_incl_repaired": _r4(ok + rep, n),
        "schema_valid": _r4(n - cats.get("unknown_tool", 0) - cats.get("bad_args", 0), n),
        "categories": cats,
        "refusal_codes": codes,
        "calls_by_tool": by_tool,
        "pose_arg_calls": len(pose_calls),
        "pose_arg_calls_refused": sum(1 for c in pose_calls if _category(c) not in WORKED),
    }


def capture_metrics(caps: list[dict]) -> dict:
    """Agreement with the planner (spec 9: % of captures where the model took candidate rank 1).
    A capture at a model-supplied pose (``source == "pose"``) is a non-agreement. The opening
    captures (one candidate, before 3 views) count as in the spec; ``agreement_rank1_after_opening``
    leaves them out."""
    by_src: dict[str, list] = {}
    for c in caps:
        by_src.setdefault(c.get("source"), []).append(c)
    nbv, opening, pose = by_src.get("nbv", []), by_src.get("opening", []), by_src.get("pose", [])
    rank1 = sum(1 for c in nbv + opening if c.get("rank") == 1)
    rank1_nbv = sum(1 for c in nbv if c.get("rank") == 1)
    after_opening = len(caps) - len(opening)
    return {
        "captures": len(caps),
        "opening_captures": len(opening),
        "nbv_captures": len(nbv),
        "pose_captures": len(pose),
        "rank1_captures": rank1,
        "rank1_nbv_captures": rank1_nbv,
        "agreement_rank1": _r4(rank1, len(caps)),
        "agreement_rank1_after_opening": _r4(rank1_nbv, after_opening),
        "rank_choices": {k: sum(1 for c in caps if c.get("chosen") == k)
                         for k in sorted({c.get("chosen") for c in caps if c.get("chosen")})},
    }


def summarise(calls: list[dict], turns: list[dict], slog: dict, truth: dict, approvals: list) -> dict:
    """Tool-call validity, steps, agreement with the planner's rank 1, and the final errors."""
    caps = slog.get("captures", [])
    rows = slog.get("accepts", [])  # one row per accepted frame, plus a row after each drop_view refit
    accepted = [a for a in rows if a.get("after_drop") is None]
    calls_at_40 = calls[:40]
    t = (truth or {}).get("truth") or {}
    gate_failed_accepts = sum(1 for c in calls if c["tool"] == "accept_view" and c.get("code") == "GATE_FAILED")
    return {
        "metrics_version": METRICS_VERSION,
        **call_metrics(calls, turns),
        **capture_metrics(caps),
        "accepted_views": len(accepted),
        "final_views": slog.get("views"),
        "views_by_call_40": sum(1 for c in calls_at_40 if c["tool"] == "accept_view" and _category(c) in WORKED),
        "exported_within_40": any(c["tool"] == "export_calibration" and _category(c) in WORKED for c in calls_at_40),
        "gate_failed_frames": sum(1 for c in caps if c.get("gate")),
        "accept_attempts_on_failed_frames": gate_failed_accepts,
        "exported": bool(slog.get("exported")),
        "export_warnings": None if not slog.get("exported") else slog["exported"].get("warnings"),
        "approvals": [{k: a.get(k) for k in ("kind", "decision")} for a in slog.get("approvals", [])],
        "model_final": (slog.get("model") or {}).get("name"),
        "final_U95_px": truth.get("U95_px"),
        "final_p_exceed": truth.get("p_exceed"),
        "final_true_p95_px": t.get("field_p95_px"),
        "fx_rel_err_pct": t.get("fx_rel_err_pct"),
        "mm_at_1m_edge": t.get("mm_at_1m_edge"),
        # keyed by accepted frames (dropped views included), as the benchmark's metric rows
        "true_p95_by_views": {a.get("n", a["views"]): (a.get("truth") or {}).get("field_p95_px") for a in rows},
        "U95_by_views": {a.get("n", a["views"]): a.get("U95") for a in rows},
        "dropped_views": sum(1 for c in calls if c["tool"] == "drop_view" and _category(c) in WORKED),
        "outlier_suggestions": sum(1 for c in calls if c["tool"] == "accept_view" and "R7: view" in (c.get("result") or "")),
        "llm_ms_total": round(sum(x["ms"] for x in turns), 1),
        "llm_ms_median": sorted(x["ms"] for x in turns)[len(turns) // 2] if turns else None,
        "tool_ms_total": round(sum(c["ms"] for c in calls), 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nextview.mcp_server.llm_driver", description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--preset", default="inspection_gauge_cam")
    ap.add_argument("--actuator", default="robot", choices=("robot", "human"))
    ap.add_argument("--camera-seed", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--approval", default="ask", choices=("ask", "approve_all", "deny_all"))
    ap.add_argument("--stopcal", default="config", help="config | identity | PATH (see the server)")
    ap.add_argument("--think", default=None, choices=("true", "false"), help="Ollama 'think' flag for reasoning models")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--log", default=None, help="JSONL log path (default results/llm_row/run_<camera_seed>.jsonl; "
                                                "spec 9 named results/llm/run_<seed>.jsonl)")
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    ap.add_argument("--debug-dir", default=None, help="dump requests whose reply holds an unparsed tool call")
    ap.add_argument("--no-hints", action="store_true", help="hide the tools' 'Next: ...' hints from the model")
    ap.add_argument("--offer-pose", action="store_true",
                    help="also offer sim_capture's explicit pose argument (default: candidates only)")
    ap.add_argument("--llm-timeout", type=float, default=300.0, help="seconds per Ollama request (not retried)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = EpisodeConfig(model=a.model, preset=a.preset, actuator=a.actuator, camera_seed=a.camera_seed, seed=a.seed,
                        budget=a.budget, max_calls=a.max_calls, approval=a.approval, stopcal=a.stopcal,
                        think=None if a.think is None else a.think == "true", num_ctx=a.num_ctx,
                        log_path=a.log or f"results/llm_row/run_{a.camera_seed}.jsonl", export_dir=a.export_dir,
                        ollama_url=a.ollama_url, debug_dir=a.debug_dir, hints=not a.no_hints,
                        offer_pose=a.offer_pose, llm_timeout_s=a.llm_timeout)
    s = asyncio.run(run_episode(cfg))
    print(json.dumps({k: v for k, v in s.items() if k not in ("true_p95_by_views", "U95_by_views")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
