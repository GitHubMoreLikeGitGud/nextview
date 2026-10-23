"""LLM driver (spec 9): schema conversion, result classification, and a full episode over MCP
stdio with a scripted stand-in for Ollama (no model needed)."""

import asyncio
import json

from mcp import types

from nextview.mcp_server import llm_driver as drv


def test_simplify_schema_drops_null_union_and_titles():
    s = {"title": "X", "type": "object", "properties": {
        "session_id": {"title": "Session Id", "type": "string"},
        "candidate_rank": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None, "title": "Candidate Rank"}},
        "required": ["session_id"]}
    out = drv.simplify_schema(s)
    assert out["properties"]["candidate_rank"] == {"type": "integer"}
    assert "title" not in out and "title" not in out["properties"]["session_id"]
    assert out["required"] == ["session_id"]


def test_ollama_tools_hide_evaluator_tools():
    tools = [types.Tool(name=n, description=n, inputSchema={"type": "object", "properties": {}})
             for n in ("sim_capture", "ground_truth_error", "session_log", "observe_image", "accept_view")]
    names = [t["function"]["name"] for t in drv.ollama_tools(tools)]
    assert names == ["sim_capture", "accept_view"]


def test_ollama_tools_hide_pose_argument():
    cap = types.Tool(name="sim_capture", description="Capture. candidate_rank: 1-5 (default 1). pose: optional explicit "
                                                     "{rvec:[3], tvec:[3]} in metres. Returns frame_id.",
                     inputSchema={"type": "object", "required": ["session_id"], "properties": {
                         "session_id": {"type": "string"}, "candidate_rank": {"type": "integer"},
                         "pose": {"type": "object"}}})
    hidden = drv.ollama_tools([cap], hidden_args=drv.POSE_ARGS)[0]["function"]
    assert set(hidden["parameters"]["properties"]) == {"session_id", "candidate_rank"}
    assert hidden["description"] == "Capture. candidate_rank: 1-5 (default 1). Returns frame_id."
    offered = drv.ollama_tools([cap])[0]["function"]
    assert "pose" in offered["parameters"]["properties"] and "rvec" in offered["description"]


def test_call_and_capture_metrics():
    calls = [{"tool": "sim_capture", "category": "ok", "args": {"candidate_rank": 1}},
             {"tool": "accept_view", "category": "ok", "ws_stripped": True},  # legacy log: worked after stripping
             {"tool": "accept_view", "category": "ok_repaired", "ws_stripped": True},
             {"tool": "sim_capture", "category": "ok", "args": {"pose": {"rvec": [0, 0, 0], "tvec": [0, 0, 1]}}},
             {"tool": "sim_capture", "category": "refused", "code": "BAD_ARGUMENT", "args": {"pose": {"rvec": [1]}}}]
    turns = [{"n_tool_calls": 1, "content": ""}, {"n_tool_calls": 0, "content": "<tool_call>\n<function=accept_view>"},
             {"n_tool_calls": 0, "content": "", "incomplete_call": True}, {"n_tool_calls": 0, "content": "DONE"}]
    m = drv.call_metrics(calls, turns)
    assert (m["tool_calls"], m["incomplete_call_turns"], m["attempted_calls"]) == (5, 2, 7)
    assert (m["valid_calls"], m["repaired_calls"]) == (2, 2)
    assert m["validity"] == 0.4 and m["validity_attempted"] == round(2 / 7, 4) and m["validity_incl_repaired"] == 0.8
    assert (m["pose_arg_calls"], m["pose_arg_calls_refused"]) == (2, 1)
    caps = ([{"source": "opening", "rank": 1, "chosen": "explicit"}] * 3 + [{"source": "nbv", "rank": 1, "chosen": "explicit"}] * 4
            + [{"source": "nbv", "rank": 2, "chosen": "explicit"}] + [{"source": "pose", "rank": None, "chosen": "pose"}] * 2)
    c = drv.capture_metrics(caps)
    assert (c["captures"], c["opening_captures"], c["nbv_captures"], c["pose_captures"]) == (10, 3, 5, 2)
    assert c["agreement_rank1"] == 0.7  # spec 9: 7 of 10 captures took rank 1; pose captures count against
    assert c["agreement_rank1_after_opening"] == round(4 / 7, 4)
    assert c["rank_choices"] == {"explicit": 8, "pose": 2}


def test_classify_and_hygiene():
    ok = types.CallToolResult(content=[types.TextContent(type="text", text="x")], structuredContent={"a": 1})
    ref = types.CallToolResult(content=[types.TextContent(type="text", text="r")], isError=True,
                               structuredContent={"error": {"code": "STALE_PLAN"}})
    bad = types.CallToolResult(content=[types.TextContent(type="text", text="validation error")], isError=True)
    need = types.CallToolResult(content=[types.TextContent(type="text", text="n")], structuredContent={"status": "needs_approval"})
    assert drv.classify_result(ok) == ("ok", None)
    assert drv.classify_result(ref) == ("refused", "STALE_PLAN")
    assert drv.classify_result(bad) == ("bad_args", "VALIDATION")
    assert drv.classify_result(need) == ("needs_approval", None)
    assert drv._strip_strings({"session_id": " \nnv_1 ", "k": 5}) == ({"session_id": "nv_1", "k": 5}, True)
    assert drv._incomplete_call("<tool_call>\n<function=accept_view>")
    assert not drv._incomplete_call("DONE: exported")
    assert "after 10 accepted views" in drv.system_prompt(10)
    t = "frame 3: 38 corners → passes the gate. Next: accept_view(frame_id=3).\n{\"frame_id\":3}"
    assert drv.strip_hints(t) == "frame 3: 38 corners → passes the gate.\n{\"frame_id\":3}"


def test_llm_row_model_spec():
    from nextview.bench.llm_row import parse_model

    assert parse_model("qwen2.5:7b") == {"model": "qwen2.5:7b", "think": None, "hints": True, "offer_pose": False,
                                         "label": "qwen2.5:7b"}
    p = parse_model("nemotron-3-nano:4b@nothink+nohints+h8")
    assert (p["model"], p["think"], p["hints"], p["max_history"], p["trim_chunk"]) == ("nemotron-3-nano:4b", False, False, 8, 4)
    assert parse_model("qwen2.5:7b@nohints+pose")["offer_pose"] is True


def test_llm_row_upgrades_legacy_runs(tmp_path):
    from nextview.bench.llm_row import aggregate, upgrade_legacy

    d = tmp_path / "qwen2.5_7b_nohints"
    d.mkdir()
    recs = [{"kind": "start"}, {"kind": "llm", "n_tool_calls": 0, "content": "<tool_call>\n<function=sim_capture>"},
            {"kind": "llm", "n_tool_calls": 2, "content": ""},
            {"kind": "tool", "tool": "sim_capture", "category": "ok", "args": {"pose": {"rvec": [0, 0, 0], "tvec": [0, 0, 1]}}},
            {"kind": "tool", "tool": "accept_view", "category": "ok", "ws_stripped": True}, {"kind": "summary"}]
    (d / "run_1001.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    old = {"policy": "LLM qwen2.5:7b@nohints", "camera_seed": 1001, "tool_calls": 2, "valid_calls": 2, "captures": 10,
           "nbv_captures": 6, "agreement_rank1": 1.0, "rank_choices": {"explicit": 9, "pose": 1}}
    r = upgrade_legacy(old, tmp_path)
    assert r["policy"] == "LLM qwen2.5:7b@nohints+pose" and r["offer_pose"] and r["rescored_from_log"]
    assert (r["valid_calls"], r["repaired_calls"], r["attempted_calls"]) == (1, 1, 3)
    assert (r["opening_captures"], r["pose_captures"], r["rank1_captures"]) == (3, 1, 9)
    assert r["agreement_rank1"] == 0.9 and r["agreement_rank1_after_opening"] == round(6 / 7, 4)
    assert upgrade_legacy(r, tmp_path) == r  # idempotent
    agg = aggregate([r], 10)["LLM qwen2.5:7b@nohints+pose"]
    assert agg["validity_attempted_pooled"] == round(1 / 3, 4) and agg["pose_captures_total"] == 1
    assert agg["agreement_rank1_pooled"] == 0.9


def test_chat_does_not_retry_timeouts():
    import httpx

    class Client:
        n = 0

        async def post(self, *a, **k):
            Client.n += 1
            raise httpx.ReadTimeout("slow")

    cfg = drv.EpisodeConfig(llm_timeout_s=1.0)
    try:
        asyncio.run(drv._chat(Client(), cfg, [], []))
    except RuntimeError as e:
        assert "not retried" in str(e)
    else:
        raise AssertionError("expected a RuntimeError")
    assert Client.n == 1


class FakeLLM:
    """Plays a tool-calling model: follows the 'Next:' hints, makes one wrong call, one text reply,
    one reply with a tool call Ollama could not parse and one capture at a made-up pose."""

    def __init__(self):
        self.n = 0
        self.sid = None

    def __call__(self, messages):
        self.n += 1
        if self.sid is None:
            self.sid = messages[1]["content"].split()[1]
        last = messages[-1]
        if self.n == 2:
            return {"role": "assistant", "content": "I will capture the first frame now."}  # -> nudge
        if self.n == 3:
            return self._call("capture_frame", {"session_id": self.sid})  # unknown tool
        if self.n == 4:
            return {"role": "assistant", "content": "<tool_call>\n<function=propose_next_views>"}  # unparsed
        if self.n == 5:  # pose is not offered: the driver refuses it
            return self._call("sim_capture", {"session_id": self.sid, "pose": {"rvec": [0, 0, 0], "tvec": [0, 0, 0.6]}})
        text = last["content"] if last["role"] == "tool" else ""
        data = json.loads(text.split("\n", 1)[1]) if "\n{" in text else {}
        if "exported" in data:
            return {"role": "assistant", "content": "DONE"}
        if "frame_id" in data and "gate" in data:
            return self._call("accept_view", {"session_id": self.sid, "frame_id": data["frame_id"]})
        if data.get("views", 0) >= 3 and "view_id" in data:
            return self._call("export_calibration", {"session_id": " " + self.sid + "\n"})  # whitespace: stripped
        if "candidates" in data:
            return self._call("sim_capture", {"session_id": self.sid, "candidate_rank": 1})
        return self._call("propose_next_views", {"session_id": self.sid})

    @staticmethod
    def _call(name, args):
        return {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": name, "arguments": args}}]}


def test_episode_over_stdio_with_fake_llm(tmp_path, monkeypatch):
    fake = FakeLLM()

    async def chat(client, cfg, messages, tools):
        assert {t["function"]["name"] for t in tools} == set(drv.LLM_TOOLS)
        cap = next(t["function"] for t in tools if t["function"]["name"] == "sim_capture")
        assert "pose" not in cap["parameters"]["properties"]
        return {"message": fake(messages), "prompt_eval_count": 1, "eval_count": 1}

    monkeypatch.setattr(drv, "_chat", chat)
    log = tmp_path / "run.jsonl"
    cfg = drv.EpisodeConfig(model="fake", camera_seed=1000, budget=10, max_calls=40, approval="approve_all",
                            log_path=str(log))
    s = asyncio.run(drv.run_episode(cfg))
    assert s["end_reason"] == "exported" and s["exported"]
    assert s["accepted_views"] == 3 and s["final_views"] == 3
    assert s["categories"].get("unknown_tool") == 1 and s["nudges"] == 2 and s["nudges_incomplete_call"] == 1
    assert s["refusal_codes"].get("ARG_NOT_OFFERED") == 1 and s["pose_arg_calls_refused"] == 1 and s["pose_captures"] == 0
    assert s["whitespace_stripped_calls"] == 1 and s["repaired_calls"] == 1 and s["categories"].get("ok_repaired") == 1
    assert s["incomplete_call_turns"] == 1 and s["attempted_calls"] == s["tool_calls"] + 1
    assert s["valid_calls"] == s["tool_calls"] - 3  # unknown tool, pose not offered, repaired export
    assert s["approvals"] == [{"kind": "accept_with_warnings", "decision": "approve"}]
    assert s["final_true_p95_px"] is not None and s["fx_rel_err_pct"] is not None
    assert s["agreement_rank1"] == 1.0 and s["agreement_rank1_after_opening"] is None  # three opening poses only
    recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert recs[0]["kind"] == "start" and recs[-1]["kind"] == "summary"
    assert "ground_truth_error" not in recs[0]["tools"] and recs[0]["hidden_args"] == {"sim_capture": ["pose"]}
    assert sum(1 for r in recs if r["kind"] == "llm" and r["incomplete_call"]) == 1
