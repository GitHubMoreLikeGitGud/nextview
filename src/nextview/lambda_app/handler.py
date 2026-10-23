"""Lambda entry point (API Gateway HTTP API, payload format 2.0).

Routes live in ``routes``; this module keeps the cold-start flag, maps errors to the spec error
shape (stack traces go to the logs only) and writes one JSON log line per request (``obs``).
"""

from __future__ import annotations

import os
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2  # noqa: E402

from nextview import __version__  # noqa: E402

from . import obs, routes  # noqa: E402

cv2.setNumThreads(1)

CSP = routes.CSP
_cold = True


def _warm() -> None:
    if os.environ.get("NEXTVIEW_SKIP_WARM") == "1":
        return
    t = time.perf_counter()
    try:
        parts = routes.warm_start()
        obs.emit({"evt": "init_warm", "ms": round(1000 * (time.perf_counter() - t), 1), "parts": parts})
    except Exception as e:  # a failed warm-up must never stop the container from serving
        obs.emit({"evt": "init_warm", "error": f"{type(e).__name__}: {e}"[:300]})


_warm()


def _version() -> str:
    if __version__ != "dev":
        return __version__
    return os.environ.get("NEXTVIEW_VERSION", "dev")


def handler(event: dict, context: object = None) -> dict:
    global _cold
    cold, _cold = _cold, False
    t0 = time.perf_counter()
    event = event or {}
    http = (event.get("requestContext") or {}).get("http") or {}
    method = (http.get("method") or "GET").upper()
    path = event.get("rawPath") or http.get("path") or "/"
    request_id = (event.get("requestContext") or {}).get("requestId") or getattr(context, "aws_request_id", None)
    ctx = routes.Ctx(cold=cold, version=_version())
    error_code = None
    try:
        req = routes.Request.from_event(event, context)
        resp = routes.dispatch(req, ctx)
    except Exception as e:  # every failure becomes the spec error shape
        mapped = routes.map_exception(e)
        if mapped is None:
            obs.emit(obs.error_line(path, method, e, traceback.format_exc(), request_id))
            mapped = (500, "INTERNAL", "internal error; see the service logs")
        status, error_code, message = mapped
        resp = routes.error_response(status, error_code, message)
    out = resp.to_event()
    latency_ms = 1000 * (time.perf_counter() - t0)
    try:
        obs.emit(obs.request_line(routes.route_name(path), method=method, status=out["statusCode"], latency_ms=latency_ms,
                                  cold=cold, request_id=request_id, session=ctx.session, result=ctx.result,
                                  handler_ms=ctx.timings, error_code=error_code,
                                  resp_bytes=len(out["body"])))
    except Exception:  # logging must never fail a request
        traceback.print_exc()
    return out
