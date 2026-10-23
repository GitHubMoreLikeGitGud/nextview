"""NextView MCP server (spec section 9): FastMCP over stdio, sessions kept in process.

The tools are the same functions as the HTTP API and the benchmark (``nextview.tools`` and
``nextview.agent``), exposed one step at a time so that any MCP client, or an LLM through
``llm_driver``, chooses the next action while the tools do the measurement and the maths:

    sim_new_session -> propose_next_views -> sim_capture -> accept_view -> uncertainty_map /
    residual_analysis -> ... -> export_calibration

The API routes map onto them: ``POST /api/session`` = ``sim_new_session`` (``image_new_session``
for real frames), ``/api/step`` = ``propose_next_views`` + ``sim_capture`` + ``accept_view``,
``/api/frame`` = ``observe_image`` + ``accept_view``, ``/api/approve`` = the approval asked
inside the gated tools.

Gated tools (``calibrate`` with a model change, ``export_calibration``) ask the human through
MCP elicitation (``Context.elicit``); the decision is recorded in the session's approvals and
trace. A client that cannot elicit gets ``status: needs_approval`` and nothing changes, so a
model can never approve its own request. ``ground_truth_error`` and ``session_log`` are for the
evaluator only (sim truth and the capture log): they are registered only with ``--evaluator``
(``build_server(evaluator=True)``, used by the benchmark harness), so a generic MCP client never
sees them, and ``llm_driver`` also keeps them out of the model's tool list.

Run: ``python -m nextview.mcp_server.server [--stopcal config|identity|PATH] [--export-dir DIR] [--evaluator]``
"""

import argparse
import json
import logging
import sys
import traceback

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, ClientCapabilities, ElicitationCapability, TextContent
from pydantic import BaseModel, Field

from .core import ApprovalRequest, Hub, Refusal, Reply

__all__ = ["build_server", "main", "EVALUATOR_TOOLS", "IMAGE_TOOLS", "ApprovalForm"]

log = logging.getLogger("nextview.mcp")

EVALUATOR_TOOLS = ("ground_truth_error", "session_log")
IMAGE_TOOLS = ("image_new_session", "observe_image")

INSTRUCTIONS = (
    "NextView calibrates a camera with a ChArUco board by deciding where the board should go next. "
    "Loop: propose_next_views -> sim_capture(candidate_rank) -> accept_view(frame_id) if the gate passed -> "
    "uncertainty_map. Stop with export_calibration when uncertainty_map says stop_ready or the budget is used. "
    "Model changes and exports ask the human approver."
)


class ApprovalForm(BaseModel):
    """What the approver answers (MCP elicitation form)."""

    approve: bool = Field(description="true = approve, false = deny")
    reason: str = Field(default="", description="short reason, kept in the audit trail")


def _ok(rep: Reply) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=rep.text())], structuredContent=rep.payload())


def _err(r: Refusal) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=r.text())], structuredContent=r.to_dict(), isError=True)


def _call(fn, *args, **kw) -> CallToolResult:
    try:
        return _ok(fn(*args, **kw))
    except Refusal as r:
        return _err(r)
    except Exception as e:  # noqa: BLE001 - a tool bug must not kill the server
        log.error("tool %s failed:\n%s", getattr(fn, "__name__", fn), traceback.format_exc())
        return _err(Refusal("INTERNAL", f"{type(e).__name__}: {e}"))


def _can_elicit(ctx: Context | None) -> bool:
    if ctx is None:
        return False
    try:
        return bool(ctx.session.check_client_capability(ClientCapabilities(elicitation=ElicitationCapability())))
    except Exception:  # noqa: BLE001 - no request context (direct call)
        return False


def _elicit_message(card: dict) -> str:
    ev = {k: v for k, v in card.get("evidence", {}).items() if k != "residual_field"}
    return (f"[{card['id']} {card['kind']}] {card['question']}\nEvidence: {json.dumps(ev, ensure_ascii=False)}\n"
            f"Answer approve=true to {card['options'][0]['label'].lower()}, approve=false to "
            f"{card['options'][-1]['label'].lower() if len(card['options']) > 1 else 'decline'}.")


async def _gated(ctx: Context | None, fn, *args, **kw) -> CallToolResult:
    try:
        out = fn(*args, **kw)
    except Refusal as r:
        return _err(r)
    except Exception as e:  # noqa: BLE001
        log.error("gated tool failed:\n%s", traceback.format_exc())
        return _err(Refusal("INTERNAL", f"{type(e).__name__}: {e}"))
    if isinstance(out, Reply):
        return _ok(out)
    req: ApprovalRequest = out
    if not _can_elicit(ctx):
        return _ok(req.needs_approval())
    res = await ctx.elicit(message=_elicit_message(req.card), schema=ApprovalForm)
    if res.action != "accept":
        return _err(Refusal("APPROVAL_DECLINED", f"the approver gave no decision ({res.action}); nothing was changed",
                            "continue calibrating or try again later"))
    decision = "approve" if res.data.approve else "deny"
    return _call(req.resolve, decision, res.data.reason)


def build_server(hub: Hub | None = None, *, evaluator: bool = False, **hub_kw) -> FastMCP:
    """A FastMCP server bound to ``hub`` (a fresh ``Hub(**hub_kw)`` by default). The evaluator
    tools (``EVALUATOR_TOOLS``) are registered only when ``evaluator`` is true."""
    hub = hub if hub is not None else Hub(**hub_kw)
    mcp = FastMCP("nextview", instructions=INSTRUCTIONS, log_level="WARNING")
    mcp.hub = hub  # for tests and in-process clients

    @mcp.tool(description="Start a simulated calibration session. preset: inspection_gauge_cam (normal lens), "
                          "wide_robot_handheld, drone_survey_cam, phone_crack_metrology, fisheye_out_of_scope, "
                          "autofocus_drift. actuator: robot or human. Returns session_id.")
    async def sim_new_session(preset: str = "inspection_gauge_cam", actuator: str = "robot", seed: int = 0,
                              budget: int = 30, camera_seed: int | None = None) -> CallToolResult:
        return _call(hub.sim_new_session, preset=preset, actuator=actuator, seed=seed, budget=budget,
                     camera_seed=camera_seed)

    @mcp.tool(description="Move the board to a target pose and capture a simulated frame, then detect the board "
                          "and assess blur, glare and size. candidate_rank: 1-5 from the latest propose_next_views "
                          "(default 1, or the opening pose before 3 views). pose: optional explicit "
                          "{rvec:[3], tvec:[3]} in metres. Returns frame_id and whether the frame passes the gate.")
    async def sim_capture(session_id: str, candidate_rank: int | None = None, pose: dict | None = None) -> CallToolResult:
        return _call(hub.sim_capture, session_id, candidate_rank=candidate_rank, pose=pose)

    @mcp.tool(description="Board detection of a captured frame: ChArUco corner count, markers, corner rows x cols.")
    async def detect_board(session_id: str, frame_id: int) -> CallToolResult:
        return _call(hub.detect_board, session_id, frame_id)

    @mcp.tool(description="Quality of a captured frame (slanted-edge blur in px, saturation, board area) and the "
                          "perception gate verdict with the operator instruction when it fails.")
    async def assess_frame(session_id: str, frame_id: int) -> CallToolResult:
        return _call(hub.assess_frame, session_id, frame_id)

    @mcp.tool(description="Accept a captured frame that passed the gate as a calibration view; estimates its pose "
                          "and refits the calibration. Refused for frames that failed the gate.")
    async def accept_view(session_id: str, frame_id: int) -> CallToolResult:
        return _call(hub.accept_view, session_id, frame_id)

    @mcp.tool(description="Fit the calibration on the accepted views (needs 3). model: k5 (default), rational or "
                          "ro; a change of model asks the human approver first.")
    async def calibrate(session_id: str, model: str = "k5", ctx: Context = None) -> CallToolResult:
        return await _gated(ctx, hub.calibrate, session_id, model)

    @mcp.tool(description="Predicted per-pixel error map of the current calibration: U95 in px, P_exceed (the "
                          "probability that the true error exceeds the target), a coarse 8x5 grid, and whether "
                          "the stop rule holds (stop_ready).")
    async def uncertainty_map(session_id: str) -> CallToolResult:
        return _call(hub.uncertainty_map, session_id)

    @mcp.tool(description="Residual diagnostics: radial-trend p-value, outlier views, board-warp score, focus "
                          "drift; returns a suggested next call (drop_view, calibrate(model=...)) when a rule fires.")
    async def residual_analysis(session_id: str) -> CallToolResult:
        return _call(hub.residual_analysis, session_id)

    @mcp.tool(description="Ranked next board poses (up to k=5) with the predicted drop in U95, effort and an "
                          "operator instruction. Before 3 views it returns the next opening pose.")
    async def propose_next_views(session_id: str, k: int = 5) -> CallToolResult:
        return _call(hub.propose_next_views, session_id, k)

    @mcp.tool(description="Remove an accepted view (for example an outlier found by residual_analysis) and refit.")
    async def drop_view(session_id: str, view_id: int, reason: str = "") -> CallToolResult:
        return _call(hub.drop_view, session_id, view_id, reason)

    @mcp.tool(description="Export the calibration (OpenCV YAML, report-card PNG, JSON). Asks the human approver; "
                          "when the stop rule does not hold the export is stamped with warnings. Ends the session.")
    async def export_calibration(session_id: str, ctx: Context = None) -> CallToolResult:
        return await _gated(ctx, hub.export_calibration, session_id)

    @mcp.tool(description="Session summary: views, budget, captures, model, flags, plan freshness, approvals.")
    async def session_status(session_id: str) -> CallToolResult:
        return _call(hub.session_status, session_id)

    @mcp.tool(description="Start a session for real image files of the board (no simulator). Returns session_id.")
    async def image_new_session(width: int = 1280, height: int = 720, actuator: str = "human",
                                budget: int = 30) -> CallToolResult:
        return _call(hub.image_new_session, width=width, height=height, actuator=actuator, budget=budget)

    @mcp.tool(description="Read one image file of the board (image sessions), detect and assess it. Returns frame_id.")
    async def observe_image(session_id: str, path: str) -> CallToolResult:
        return _call(hub.observe_image, session_id, path)

    if not evaluator:
        return mcp

    @mcp.tool(description="EVALUATOR ONLY (simulation): true error of the current calibration against ground truth.")
    async def ground_truth_error(session_id: str) -> CallToolResult:
        return _call(hub.ground_truth_error, session_id)

    @mcp.tool(description="EVALUATOR ONLY: capture and accept log, approvals, rejections and the trace.")
    async def session_log(session_id: str) -> CallToolResult:
        return _call(hub.session_log, session_id)

    return mcp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nextview.mcp_server.server", description=__doc__.splitlines()[0])
    ap.add_argument("--stopcal", default="config",
                    help="stop calibration: 'config' (config/stop_calibration.json), 'identity' (r = 1, exercises the "
                         "stop rule before fit_stop.py has run; not a calibration) or a JSON path")
    ap.add_argument("--export-dir", default=None, help="write approved exports to DIR/<session_id>/")
    ap.add_argument("--evaluator", action="store_true",
                    help="also register the evaluator-only tools (ground_truth_error, session_log); for benchmark "
                         "harnesses, never for a client whose model plans the session")
    a = ap.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("mcp").setLevel(logging.WARNING)
    import cv2

    cv2.setNumThreads(1)
    mcp = build_server(stopcal=a.stopcal, export_dir=a.export_dir, evaluator=a.evaluator)
    log.info("nextview MCP server on stdio (stopcal=%s, evaluator tools %s)", a.stopcal, "on" if a.evaluator else "off")
    mcp.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
