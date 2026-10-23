"""MCP server (spec 9 / 13): in-process FastMCP client, tool list, sim session flow, gates."""

import asyncio

import numpy as np
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session

from nextview.mcp_server.core import Hub
from nextview.mcp_server.server import EVALUATOR_TOOLS, build_server

SPEC_TOOLS = {"sim_new_session", "sim_capture", "detect_board", "assess_frame", "accept_view", "calibrate",
              "uncertainty_map", "residual_analysis", "propose_next_views", "drop_view", "export_calibration"}


def _run(coro):
    return asyncio.run(coro)


async def _approve(ctx, params):
    return types.ElicitResult(action="accept", content={"approve": True, "reason": "test"})


async def _deny(ctx, params):
    return types.ElicitResult(action="accept", content={"approve": False, "reason": "test deny"})


async def _session(c, **kw):
    r = await c.call_tool("sim_new_session", {"preset": "inspection_gauge_cam", "camera_seed": 1000, "budget": 10, **kw})
    assert not r.isError, r.content[0].text
    return r.structuredContent["session_id"]


async def _views(c, sid, n):
    """Capture and accept ``n`` views along the planner's rank-1 candidates."""
    for _ in range(n):
        p = await c.call_tool("propose_next_views", {"session_id": sid})
        assert not p.isError, p.content[0].text
        f = await c.call_tool("sim_capture", {"session_id": sid, "candidate_rank": 1})
        assert not f.isError, f.content[0].text
        assert f.structuredContent["gate"]["pass"], f.content[0].text
        a = await c.call_tool("accept_view", {"session_id": sid, "frame_id": f.structuredContent["frame_id"]})
        assert not a.isError, a.content[0].text
    return a.structuredContent


def test_list_tools_and_schemas():
    async def go():
        async with create_connected_server_and_client_session(build_server()) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert SPEC_TOOLS <= set(tools)
            assert not set(EVALUATOR_TOOLS) & set(tools)  # a generic client never sees the evaluator tools
            r = await c.call_tool("ground_truth_error", {"session_id": "nv_x"})
            assert r.isError
        async with create_connected_server_and_client_session(build_server(evaluator=True)) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert set(EVALUATOR_TOOLS) | SPEC_TOOLS <= set(tools)
            assert tools["sim_capture"].inputSchema["required"] == ["session_id"]
            assert "candidate_rank" in tools["sim_capture"].inputSchema["properties"]
            assert all(t.description for t in tools.values())

    _run(go())


def test_sim_session_capture_accept_calibrate():
    async def go():
        async with create_connected_server_and_client_session(build_server(evaluator=True)) as c:
            sid = await _session(c)
            p = await c.call_tool("propose_next_views", {"session_id": sid})
            assert p.structuredContent["phase"] == "opening" and len(p.structuredContent["candidates"]) == 1
            f = await c.call_tool("sim_capture", {"session_id": sid, "candidate_rank": 1})
            sc = f.structuredContent
            assert not f.isError and sc["frame_id"] == 0 and sc["detection"]["n_corners"] >= 12
            assert sc["gate"]["pass"] and sc["quality"]["edge_rise_px"] > 0
            text = f.content[0].text
            assert text.splitlines()[0].startswith("frame 0:")  # one-line summary, then JSON
            d = await c.call_tool("detect_board", {"session_id": sid, "frame_id": 0})
            assert d.structuredContent["detection"]["n_corners"] == sc["detection"]["n_corners"]
            q = await c.call_tool("assess_frame", {"session_id": sid, "frame_id": 0})
            assert q.structuredContent["gate"]["pass"]
            a = await c.call_tool("accept_view", {"session_id": sid, "frame_id": 0})
            assert a.structuredContent["views"] == 1 and a.structuredContent["calibration"] is None
            e = await c.call_tool("calibrate", {"session_id": sid})
            assert e.isError and e.structuredContent["error"]["code"] == "NEEDS_VIEWS"
            last = await _views(c, sid, 3)
            assert last["views"] == 4 and last["calibration"]["model"] == "k5"
            k = await c.call_tool("calibrate", {"session_id": sid, "model": "k5"})
            cal = k.structuredContent["calibration"]
            assert not k.isError and cal["rms_px"] < 0.5 and 900 < cal["fx"] < 2000
            u = await c.call_tool("uncertainty_map", {"session_id": sid})
            us = u.structuredContent
            assert us["grid"] == [8, 5] and np.asarray(us["sigma_px"]).shape == (5, 8) and us["U95_px"] > 0
            assert us["rule"] in ("R11_STOP", "R12_BUDGET", "R13_NBV")
            r = await c.call_tool("residual_analysis", {"session_id": sid})
            assert not r.isError and "radial_trend_p" in r.structuredContent
            n = await c.call_tool("propose_next_views", {"session_id": sid, "k": 3})
            cands = n.structuredContent["candidates"]
            assert n.structuredContent["phase"] == "nbv" and len(cands) == 3 and cands[0]["rank"] == 1
            assert cands[0]["pred_gain_px"] >= cands[-1]["pred_gain_px"] - 1e-9  # robot: score = gain
            g = await c.call_tool("ground_truth_error", {"session_id": sid})
            assert g.structuredContent["truth"]["field_p95_px"] > 0

    _run(go())


def test_refusals_are_structured():
    async def go():
        async with create_connected_server_and_client_session(build_server()) as c:
            r = await c.call_tool("sim_capture", {"session_id": "nv_nope"})
            assert r.isError and r.structuredContent["error"]["code"] == "UNKNOWN_SESSION"
            sid = await _session(c)
            # a board 5 m away is far too small: the gate rejects it and accept_view refuses it
            f = await c.call_tool("sim_capture", {"session_id": sid, "pose": {"rvec": [0, 0, 0], "tvec": [-0.1, -0.07, 5.0]}})
            assert not f.isError and not f.structuredContent["gate"]["pass"]
            assert f.structuredContent["gate"]["rule"] in ("R0_NO_BOARD", "R1_SMALL")
            a = await c.call_tool("accept_view", {"session_id": sid, "frame_id": f.structuredContent["frame_id"]})
            assert a.isError and a.structuredContent["error"]["code"] == "GATE_FAILED"
            await _views(c, sid, 1)
            # the plan was made before the last accepted view: capturing from it is refused
            s = await c.call_tool("sim_capture", {"session_id": sid, "candidate_rank": 1})
            assert s.isError and s.structuredContent["error"]["code"] == "STALE_PLAN"
            b = await c.call_tool("sim_capture", {"session_id": sid, "candidate_rank": "x"})
            assert b.isError  # schema validation (not a refusal): no structured error
            assert not (b.structuredContent or {}).get("error")

    _run(go())


def test_gated_export_without_elicitation_needs_approval():
    async def go():
        srv = build_server()
        async with create_connected_server_and_client_session(srv) as c:  # client cannot elicit
            sid = await _session(c)
            await _views(c, sid, 3)
            e = await c.call_tool("export_calibration", {"session_id": sid})
            assert not e.isError and e.structuredContent["status"] == "needs_approval"
            assert e.structuredContent["approval"]["kind"] in ("export", "accept_with_warnings")
            m = await c.call_tool("calibrate", {"session_id": sid, "model": "rational"})
            assert m.structuredContent["status"] == "needs_approval"
            st = srv.hub.sessions[sid].state
            assert st.status == "ready" and st.model["name"] == "k5" and st.approvals == []
            again = await c.call_tool("propose_next_views", {"session_id": sid})
            assert not again.isError  # nothing changed, the session is still open

    _run(go())


def test_gated_export_with_elicitation_records_approval(tmp_path):
    async def go():
        srv = build_server(export_dir=tmp_path, evaluator=True)
        async with create_connected_server_and_client_session(srv, elicitation_callback=_approve) as c:
            sid = await _session(c)
            await _views(c, sid, 3)
            e = await c.call_tool("export_calibration", {"session_id": sid})
            assert not e.isError and e.structuredContent["exported"] is True, e.content[0].text
            files = e.structuredContent["files"]
            assert set(files) == {"yaml", "png", "json"}
            assert (tmp_path / sid / "calibration.yaml").read_bytes().startswith(b"%YAML")
            log = (await c.call_tool("session_log", {"session_id": sid})).structuredContent
            assert [a["decision"] for a in log["approvals"]] == ["approve"]
            assert log["approvals"][0]["via"] == "mcp_elicit"
            assert any("APPROVAL" in t for t in log["trace"])
            after = await c.call_tool("sim_capture", {"session_id": sid})
            assert after.isError and after.structuredContent["error"]["code"] == "SESSION_ENDED"

    _run(go())


def test_model_switch_deny_pins_model():
    async def go():
        srv = build_server()
        async with create_connected_server_and_client_session(srv, elicitation_callback=_deny) as c:
            sid = await _session(c)
            await _views(c, sid, 3)
            r = await c.call_tool("calibrate", {"session_id": sid, "model": "rational"})
            assert not r.isError and "denied" in r.content[0].text
            st = srv.hub.sessions[sid].state
            assert st.model == {"name": "k5", "pinned": True}
            assert st.approvals[-1]["kind"] == "switch_model" and st.approvals[-1]["decision"] == "deny"
            again = await c.call_tool("calibrate", {"session_id": sid, "model": "rational"})
            assert again.isError and again.structuredContent["error"]["code"] == "MODEL_PINNED"

    _run(go())


def test_rank1_client_matches_agent_loop():
    """The MCP tools are the loop's functions: always taking candidate 1 reproduces run_step."""
    from nextview.agent.loop import make_source, run_step
    from nextview.agent.state import new_session
    from nextview.config import load_stop_calibration, load_thresholds
    from nextview.sim.truth import field_error

    th, stopcal = load_thresholds(), load_stop_calibration()
    st = new_session(mode="sim", preset="inspection_gauge_cam", actuator="robot", seed=0, camera_seed=1000, budget=10,
                     th=th, baseline=False)
    src = make_source(st)
    while st.n_accepted_total < 5:
        run_step(st, src, th=th, stopcal=stopcal)
    K_loop = np.asarray(st.calib["K"])

    async def go():
        srv = build_server()
        async with create_connected_server_and_client_session(srv) as c:
            sid = await _session(c)
            await _views(c, sid, 5)
            return srv.hub.sessions[sid]

    s = _run(go())
    assert len(s.state.views) == 5
    np.testing.assert_allclose(s.calib.K, K_loop, rtol=1e-6)
    cam = s.sim.camera_at(5)
    assert abs(field_error(cam, s.calib.K, s.calib.dist).p95 - field_error(cam, K_loop, np.asarray(st.calib["dist"])).p95) < 1e-6


def test_hub_identity_stopcal_and_image_session(tmp_path):
    import cv2

    hub = Hub(stopcal="identity")
    assert hub.stopcal.calibrated
    rep = hub.image_new_session(width=1280, height=720)
    sid = rep.data["session_id"]
    from nextview.sim.scenarios import make_simulator

    sim = make_simulator("normal", "robot", 1000, 0, clean=True)
    from nextview.tools.nbv import opening_target

    rv, tv = opening_target(0, 1000, 0, (1280, 720), sim.camera.fx)
    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), sim.render_exact(rv, tv))
    f = hub.observe_image(sid, str(p))
    assert f.data["gate"]["pass"] and f.data["detection"]["n_corners"] >= 12
    a = hub.accept_view(sid, f.data["frame_id"])
    assert a.data["views"] == 1
