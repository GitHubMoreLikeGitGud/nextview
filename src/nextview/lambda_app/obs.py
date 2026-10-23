"""Observability: one JSON log line per request, with a CloudWatch Embedded Metric Format block
for the agent-cycle routes (spec 10.6).

The line goes to stdout, which Lambda forwards to CloudWatch Logs unchanged, so Logs Insights
can query its fields (``filter evt="step" | stats pct(latency_ms.total, 50) by mode``) and the
``_aws`` block turns the named fields into metrics in the ``NextView`` namespace.

Only whitelisted fields are logged: never corners, frames, JPEG data or state tokens.
"""

from __future__ import annotations

import json
import math
import sys
import time

__all__ = ["NAMESPACE", "EMF_ROUTES", "EMF_MODES", "request_line", "emit", "error_line"]

NAMESPACE = "NextView"
# Routes that run an agent cycle get metrics; static files, health and session starts are logged only
# (each Mode x Route pair is a separate set of custom metrics). Only 200 responses carry the EMF block, so
# refused requests (4xx, wrong mode for the route) never add a Mode x Route series: at most 4 pairs x 6 metrics.
EMF_ROUTES = ("/api/step", "/api/frame", "/api/approve")
EMF_MODES = ("sim", "webcam")
EMF_METRICS = (
    ("LatencyMs", "Milliseconds"),
    ("NbvMs", "Milliseconds"),
    ("Rejects", "Count"),
    ("ApprovalsRequested", "Count"),
    ("Stops", "Count"),
    ("ColdStart", "Count"),
)
EVT = {"/api/step": "step", "/api/frame": "step", "/api/approve": "approve", "/api/session": "session"}
REJECT_RULES = ("R0_NO_BOARD", "R1_SMALL", "R2_BLUR", "R3_GLARE")


def _num(x, nd: int = 4):
    if x is None or isinstance(x, bool):
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def request_line(route: str, *, method: str, status: int, latency_ms: float, cold: bool, request_id: str | None = None,
                 session: dict | None = None, result=None, handler_ms: dict | None = None, error_code: str | None = None,
                 resp_bytes: int | None = None, now: float | None = None) -> dict:
    """The log record for one request.

    ``session``: ``{"sid", "mode", "preset", "actuator"}`` (when a state was decoded or created).
    ``result``: a ``loop.StepResult`` for the cycle routes (only its whitelisted fields are read).
    ``handler_ms``: extra timings (decode, encode, frame JPEG, S3) added to ``latency_ms``.
    """
    s = session or {}
    mode = s.get("mode") or "none"
    rec: dict = {"evt": EVT.get(route, "request"), "route": route, "method": method, "status": int(status),
                 "mode": mode, "cold": bool(cold)}
    if request_id:
        rec["request_id"] = request_id
    for k in ("sid", "preset", "actuator"):
        if s.get(k) is not None:
            rec[k] = s[k]
    lat = {"total": round(float(latency_ms), 1)}
    nbv_ms = None
    rejects = approvals = stops = 0
    if result is not None:
        tools = [{"tool": str(t.get("tool")), "ms": _num(t.get("ms"), 1)} for t in (result.tools or [])]
        nbv = next((t for t in result.tools or [] if t.get("tool") == "propose_next_views"), None)
        if nbv is not None:
            nbv_ms = _num(nbv.get("ms"), 1)
            rec["n_candidates"] = nbv.get("n_candidates")
            rec["nbv_deadline_hit"] = nbv.get("deadline_hit")
        unc = result.uncertainty or {}
        cal = result.calibration or {}
        rec.update({
            "step": int(result.step), "rule": result.rule, "accepted": bool(result.accepted), "status_agent": result.status,
            "views": cal.get("views"), "model": cal.get("model"), "U95": _num(unc.get("U95_px")),
            "p_exceed": _num(unc.get("p_exceed")), "tools": tools,
        })
        lat["agent"] = round(float(result.latency_ms), 1)
        if nbv_ms is not None:
            lat["nbv"] = nbv_ms
        rejects = int(result.rule in REJECT_RULES)
        card = result.approval
        approvals = int(bool(card) and int(card.get("step", -1)) == int(result.step))  # raised by this call
        stops = int(result.status == "stopped")
        if result.approval:
            rec["approval_kind"] = result.approval.get("kind")
    for k, v in (handler_ms or {}).items():
        lat[k] = round(float(v), 1)
    rec["latency_ms"] = lat
    if resp_bytes is not None:
        rec["resp_bytes"] = int(resp_bytes)
    if error_code:
        rec["error"] = error_code
    if route in EMF_ROUTES and int(status) == 200 and mode in EMF_MODES:
        rec["Mode"] = mode
        rec["Route"] = route
        rec["LatencyMs"] = lat["total"]
        rec["NbvMs"] = 0.0 if nbv_ms is None else nbv_ms
        rec["Rejects"] = rejects
        rec["ApprovalsRequested"] = approvals
        rec["Stops"] = stops
        rec["ColdStart"] = int(bool(cold))
        rec["_aws"] = {
            "Timestamp": int(1000 * (time.time() if now is None else now)),
            "CloudWatchMetrics": [{
                "Namespace": NAMESPACE,
                "Dimensions": [["Mode", "Route"]],
                "Metrics": [{"Name": n, "Unit": u} for n, u in EMF_METRICS],
            }],
        }
    return rec


def error_line(route: str, method: str, exc: BaseException, tb: str, request_id: str | None = None) -> dict:
    rec = {"evt": "error", "route": route, "method": method, "error": type(exc).__name__, "message": str(exc)[:500],
           "traceback": tb[-4000:]}
    if request_id:
        rec["request_id"] = request_id
    return rec


def emit(rec: dict, stream=None) -> None:
    """Write one JSON line (EMF-compatible) to stdout."""
    out = stream or sys.stdout
    out.write(json.dumps(rec, separators=(",", ":"), allow_nan=False, default=str) + "\n")
    out.flush()
