"""HTTP routes (spec 7.2, 7.3, 10.5): request parsing, caps, the agent routes and static files.

Every route takes a ``Request`` and returns a ``Response``; ``handler.handler`` turns API Gateway
v2 events into requests, logs one JSON line per request (``obs``) and maps errors to the spec
error shape. The session lives in the signed state token, so any warm container can serve any
step.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs

import cv2
import numpy as np

from .. import board as board_mod
from ..agent.approvals import ApprovalError
from ..agent.loop import LoopError, apply_approval, make_source, run_step
from ..agent.state import StateError, decode_token, encode_token, new_session, state_key
from ..config import load_presets, load_stop_calibration, load_thresholds
from . import s3io

__all__ = ["CSP", "CAPS", "MAX_JPEG_BYTES", "HttpError", "Request", "Response", "Ctx", "dispatch", "ROUTES",
           "jpeg_size", "TH", "STOPCAL", "PRESETS"]

CSP = (
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
    "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'"
)
KB = 1024
CAPS = {"/api/session": 4 * KB, "/api/step": 512 * KB, "/api/frame": 512 * KB, "/api/approve": 512 * KB}
MAX_JPEG_BYTES = 300 * KB
MAX_FRAME = (1280, 720)
MIN_FRAME = (160, 120)
SIM_FRAME = (960, 540)
SIM_JPEG_QUALITY = 80
MAX_SEED = 2**31 - 1
REASON_MAX = 500
HANDLER_RESERVE_S = 5.0  # kept back from the Lambda time limit for encoding and the response
WEB_DIR = Path(__file__).resolve().parent / "web"
EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
EVIDENCE_RE = re.compile(r"^/evidence/([A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.png)$")
EXPORT_RE = re.compile(r"^/api/export/([0-9a-f]{24})/([A-Za-z0-9_.-]{1,64})$")

TH = load_thresholds()
STOPCAL = load_stop_calibration()
PRESETS = load_presets()


class HttpError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = int(status), code, message


@dataclass
class Request:
    method: str
    path: str
    query: dict
    headers: dict
    body: bytes
    request_id: str | None = None
    remaining_ms: int | None = None

    @classmethod
    def from_event(cls, event: dict, context=None) -> "Request":
        rc = event.get("requestContext") or {}
        http = rc.get("http") or {}
        method = (http.get("method") or "GET").upper()
        path = event.get("rawPath") or http.get("path") or "/"
        raw_q = event.get("rawQueryString") or ""
        query = {k: v[-1] for k, v in parse_qs(raw_q, keep_blank_values=True).items()}
        headers = {str(k).lower(): v for k, v in (event.get("headers") or {}).items()}
        body = event.get("body")
        cap = CAPS.get(path)
        if body is None:
            data = b""
        elif event.get("isBase64Encoded"):
            if cap is not None and len(body) > 4 * (cap // 3) + 8:
                raise HttpError(413, "PAYLOAD_TOO_LARGE", f"request body over {cap // KB} KB")
            try:
                data = base64.b64decode(body)
            except (binascii.Error, ValueError):
                raise HttpError(400, "BAD_REQUEST", "body is not valid base64") from None
        else:
            if cap is not None and len(body) > cap:  # characters >= bytes / 4: cheap pre-check before encoding
                raise HttpError(413, "PAYLOAD_TOO_LARGE", f"request body over {cap // KB} KB")
            data = body.encode("utf-8")
        if cap is not None and len(data) > cap:
            raise HttpError(413, "PAYLOAD_TOO_LARGE", f"request body is {len(data)} bytes; the cap for {path} is {cap // KB} KB")
        remaining = None
        if context is not None and hasattr(context, "get_remaining_time_in_millis"):
            try:
                remaining = int(context.get_remaining_time_in_millis())
            except Exception:
                remaining = None
        return cls(method, path, query, headers, data, rc.get("requestId") or getattr(context, "aws_request_id", None),
                   remaining)


@dataclass
class Response:
    status: int
    body: bytes | str
    content_type: str
    api: bool = True
    binary: bool = False
    extra_headers: dict = field(default_factory=dict)

    def to_event(self) -> dict:
        headers = {"content-type": self.content_type, "content-security-policy": CSP,
                   "x-content-type-options": "nosniff", "referrer-policy": "no-referrer"}
        headers["cache-control"] = "no-store" if self.api else "no-cache"
        headers.update(self.extra_headers)
        if self.binary:
            body = base64.b64encode(self.body if isinstance(self.body, bytes) else self.body.encode()).decode("ascii")
        else:
            body = self.body.decode("utf-8") if isinstance(self.body, bytes) else self.body
        return {"statusCode": int(self.status), "headers": headers, "body": body, "isBase64Encoded": bool(self.binary)}


@dataclass
class Ctx:
    """Per-request facts the handler logs."""

    cold: bool = False
    version: str = "dev"
    session: dict | None = None
    result: object = None
    timings: dict = field(default_factory=dict)


def json_response(status: int, payload, *, api: bool = True) -> Response:
    return Response(status, json.dumps(payload, separators=(",", ":"), allow_nan=False), "application/json", api=api)


def error_response(status: int, code: str, message: str) -> Response:
    return json_response(status, {"error": {"code": code, "message": message}})


# ---- request helpers -------------------------------------------------------------------------

def _json_body(req: Request) -> dict:
    if not req.body:
        raise HttpError(400, "BAD_REQUEST", "request body must be a JSON object")
    try:
        data = json.loads(req.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise HttpError(400, "BAD_REQUEST", f"body is not valid JSON ({type(e).__name__})") from None
    if not isinstance(data, dict):
        raise HttpError(400, "BAD_REQUEST", "request body must be a JSON object")
    return data


def _int(v, name: str, lo: int, hi: int, default=None):
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
        raise HttpError(400, "BAD_REQUEST", f"{name} must be an integer")
    v = int(v)
    if not lo <= v <= hi:
        raise HttpError(400, "BAD_REQUEST", f"{name} must be in [{lo}, {hi}]")
    return v


def _key() -> bytes:
    try:
        return state_key()
    except ValueError as e:  # misconfigured deployment: logged as INTERNAL by the handler
        raise RuntimeError(str(e)) from None


def _load_state(body: dict, ctx: Ctx):
    tok = body.get("state")
    if not isinstance(tok, str) or not tok:
        raise HttpError(400, "BAD_REQUEST", "state (the token from the previous response) is required")
    t = time.perf_counter()
    st = decode_token(tok, _key())
    ctx.timings["decode"] = 1000 * (time.perf_counter() - t)
    ctx.session = {"sid": st.sid, "mode": st.mode, "preset": st.preset, "actuator": st.actuator}
    return st


def _encode(st, ctx: Ctx) -> str:
    t = time.perf_counter()
    tok = encode_token(st, _key())
    ctx.timings["encode"] = 1000 * (time.perf_counter() - t)
    return tok


def _deadline(req: Request) -> float:
    budget = float(TH.step_deadline_s)
    if req.remaining_ms is not None:
        budget = max(2.0, min(budget, req.remaining_ms / 1000.0 - HANDLER_RESERVE_S))
    return time.perf_counter() + budget


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's SOF header, read before decoding (no pixel bombs); None if not a JPEG."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        m = data[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0x01, 0xD8) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m in (0xD9, 0xDA):
            return None
        seg = int.from_bytes(data[i + 2:i + 4], "big")
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            if i + 9 > n:
                return None
            h = int.from_bytes(data[i + 5:i + 7], "big")
            w = int.from_bytes(data[i + 7:i + 9], "big")
            return w, h
        if seg < 2:
            return None
        i += 2 + seg
    return None


def _frame_jpeg(image: np.ndarray | None, ctx: Ctx) -> dict | None:
    if image is None:
        return None
    t = time.perf_counter()
    h, w = image.shape[:2]
    small = cv2.resize(image, SIM_FRAME, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, SIM_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    ctx.timings["frame_jpeg"] = 1000 * (time.perf_counter() - t)
    return {"jpeg_b64": base64.b64encode(buf.tobytes()).decode("ascii"), "w": SIM_FRAME[0], "h": SIM_FRAME[1],
            "scale": round(SIM_FRAME[0] / w, 6)}


HISTORY_KEYS = ("step", "rule", "accepted", "views", "model", "U95", "p_exceed", "true_p95", "baseline_views",
                "baseline_true_p95", "realised", "status")


def _session_summary(st) -> dict:
    return {"session_id": st.sid, "mode": st.mode, "preset": st.preset, "actuator": st.actuator, "seed": st.seed,
            "budget": st.budget, "attempts": st.attempts, "max_attempts": st.max_attempts, "views": len(st.views),
            "model": st.model, "flags": list(st.flags), "tau_px": st.tau, "status": st.status,
            "image_size": list(st.image_size)}


def _cycle_response(st, res, ctx: Ctx, *, frame: bool) -> dict:
    out = res.to_response()
    out["state"] = _encode(st, ctx)
    out["session"] = _session_summary(st)
    out["history"] = [{k: h.get(k) for k in HISTORY_KEYS if h.get(k) is not None} for h in st.history]
    if frame:
        out["frame"] = _frame_jpeg(res.image, ctx)
    return out


# ---- routes ------------------------------------------------------------------------------------

def health(req: Request, ctx: Ctx) -> Response:
    return json_response(200, {"ok": True, "version": ctx.version, "opencv": cv2.__version__, "cold": ctx.cold})


def presets(req: Request, ctx: Ctx) -> Response:
    return json_response(200, {
        "version": ctx.version, "opencv": cv2.__version__,
        "default": PRESETS.default_id,
        "presets": [{"id": p.id, "label": p.label, "family": p.family, "actuator": p.actuator,
                     "nuisances": list(p.nuisances), "failure_demo": p.failure_demo} for p in PRESETS],
        "thresholds": {"tau_px": TH.tau_px, "budget_frames": TH.budget_frames, "max_attempts": TH.max_attempts,
                       "stop_p_exceed": TH.stop_p_exceed, "model_switch_min_views": TH.model_switch_min_views},
        "stop_calibration": {"calibrated": bool(STOPCAL.calibrated), "note": STOPCAL.note},
        "board": _board_info(),
        "caps": {"jpeg_bytes": MAX_JPEG_BYTES, "frame_max": list(MAX_FRAME), "frame_min": list(MIN_FRAME)},
    })


def _board_info() -> dict:
    return {"squares": list(board_mod.SQUARES), "square_mm": round(board_mod.SQUARE_M * 1000, 3),
            "marker_mm": round(board_mod.MARKER_M * 1000, 3), "dictionary": board_mod.DICT_NAME}


def session(req: Request, ctx: Ctx) -> Response:
    body = _json_body(req)
    mode = body.get("mode", "sim")
    if mode not in ("sim", "webcam"):
        raise HttpError(400, "BAD_REQUEST", "mode must be 'sim' or 'webcam'")
    seed = _int(body.get("seed"), "seed", 0, MAX_SEED, 0)
    budget = _int(body.get("budget"), "budget", 3, int(TH.budget_frames), int(TH.budget_frames))
    actuator = body.get("actuator")
    if actuator is not None and actuator not in ("robot", "human"):
        raise HttpError(400, "BAD_REQUEST", "actuator must be 'robot' or 'human'")
    preset = None
    size = (1280, 720)
    if mode == "sim":
        pid = body.get("preset") or PRESETS.default_id
        if not isinstance(pid, str) or pid not in PRESETS:
            raise HttpError(400, "BAD_REQUEST", f"unknown preset; known: {list(PRESETS.ids)}")
        preset = PRESETS.get(pid)
        actuator = actuator or preset.actuator
    else:
        actuator = actuator or "human"
        isz = body.get("image_size", [1280, 720])
        if not isinstance(isz, (list, tuple)) or len(isz) != 2:
            raise HttpError(400, "BAD_REQUEST", "image_size must be [width, height]")
        size = (_int(isz[0], "image_size[0]", MIN_FRAME[0], MAX_FRAME[0]),
                _int(isz[1], "image_size[1]", MIN_FRAME[1], MAX_FRAME[1]))
    t0 = time.perf_counter()
    st = new_session(mode=mode, preset=None if preset is None else preset.id, actuator=actuator, seed=seed,
                     budget=budget, th=TH, image_size=size)
    ctx.session = {"sid": st.sid, "mode": st.mode, "preset": st.preset, "actuator": st.actuator}
    if mode == "sim":  # build this camera's simulator and distortion maps now, so the first step does not pay for them
        t = time.perf_counter()
        src = make_source(st)
        src.sim.renderer(src.camera(0))
        ctx.timings["sim_warm"] = 1000 * (time.perf_counter() - t)
    out = {
        "session_id": st.sid, "state": _encode(st, ctx), "status": st.status, "mode": st.mode, "actuator": st.actuator,
        "seed": st.seed, "image_size": list(st.image_size), "board": _board_info(),
        "preset": None if preset is None else {"id": preset.id, "label": preset.label, "family": preset.family},
        "thresholds": {"tau_px": st.tau, "budget_frames": st.budget, "max_attempts": st.max_attempts,
                       "stop_p_exceed": TH.stop_p_exceed, "model_switch_min_views": TH.model_switch_min_views},
        "stop_calibration": {"calibrated": bool(STOPCAL.calibrated),
                             "label": "transferred from simulation" if mode == "webcam" else "fitted on the simulation DEV split"},
        "session": _session_summary(st),
    }
    out["latency_ms"] = {"total": round(1000 * (time.perf_counter() - t0), 1)}
    return json_response(200, out)


def _choose(body: dict):
    return _int(body.get("choose_candidate"), "choose_candidate", 1, 5, None)


def step(req: Request, ctx: Ctx) -> Response:
    body = _json_body(req)
    st = _load_state(body, ctx)
    if st.mode != "sim":
        raise HttpError(400, "BAD_REQUEST", "webcam sessions post frames to /api/frame")
    res = run_step(st, th=TH, stopcal=STOPCAL, deadline=_deadline(req), choose_candidate=_choose(body))
    ctx.result = res
    return json_response(200, _cycle_response(st, res, ctx, frame=True))


def _decode_frame(body: dict) -> np.ndarray:
    b64 = body.get("jpeg_b64")
    if not isinstance(b64, str) or not b64:
        raise HttpError(400, "BAD_REQUEST", "jpeg_b64 is required")
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    if len(b64) > 4 * (MAX_JPEG_BYTES // 3) + 8:
        raise HttpError(413, "PAYLOAD_TOO_LARGE", f"JPEG over {MAX_JPEG_BYTES // KB} KB")
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HttpError(400, "BAD_REQUEST", "jpeg_b64 is not valid base64") from None
    if len(data) > MAX_JPEG_BYTES:
        raise HttpError(413, "PAYLOAD_TOO_LARGE", f"JPEG is {len(data)} bytes; the cap is {MAX_JPEG_BYTES // KB} KB")
    wh = jpeg_size(data)
    if wh is None:
        raise HttpError(400, "BAD_REQUEST", "frame is not a baseline/progressive JPEG")
    allowed = f"allowed {MIN_FRAME[0]}x{MIN_FRAME[1]} to {MAX_FRAME[0]}x{MAX_FRAME[1]}"
    if wh[0] > MAX_FRAME[0] or wh[1] > MAX_FRAME[1]:
        raise HttpError(413, "PAYLOAD_TOO_LARGE", f"frame is {wh[0]}x{wh[1]}; {allowed}")
    if wh[0] < MIN_FRAME[0] or wh[1] < MIN_FRAME[1]:
        raise HttpError(400, "BAD_REQUEST", f"frame is {wh[0]}x{wh[1]}; {allowed}")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None or (img.shape[1], img.shape[0]) != wh:
        raise HttpError(400, "BAD_REQUEST", "JPEG could not be decoded")
    return img


def frame(req: Request, ctx: Ctx) -> Response:
    body = _json_body(req)
    st = _load_state(body, ctx)
    if st.mode != "webcam":
        raise HttpError(400, "BAD_REQUEST", "sim sessions use /api/step")
    t = time.perf_counter()
    img = _decode_frame(body)
    ctx.timings["frame_decode"] = 1000 * (time.perf_counter() - t)
    if (img.shape[1], img.shape[0]) != tuple(st.image_size):
        raise HttpError(400, "BAD_REQUEST", f"frame is {img.shape[1]}x{img.shape[0]}; this session was started for "
                                            f"{st.image_size[0]}x{st.image_size[1]}")
    res = run_step(st, th=TH, stopcal=STOPCAL, deadline=_deadline(req), choose_candidate=_choose(body), image=img)
    del img
    ctx.result = res
    out = _cycle_response(st, res, ctx, frame=False)
    out.pop("truth", None)
    out.pop("baseline", None)
    return json_response(200, out)


def approve(req: Request, ctx: Ctx) -> Response:
    body = _json_body(req)
    st = _load_state(body, ctx)
    aid = body.get("approval_id")
    decision = body.get("decision")
    reason = body.get("reason") or ""
    if not isinstance(aid, str) or not isinstance(decision, str):
        raise HttpError(400, "BAD_REQUEST", "approval_id and decision are required")
    if not isinstance(reason, str):
        raise HttpError(400, "BAD_REQUEST", "reason must be a string")
    res = apply_approval(st, aid, decision, reason[:REASON_MAX], th=TH, stopcal=STOPCAL, deadline=_deadline(req))
    ctx.result = res
    out = _cycle_response(st, res, ctx, frame=False)
    if st.mode != "sim":
        out.pop("truth", None)
        out.pop("baseline", None)
    if res.export_files:
        out["export"] = _store_export(st, aid, res, ctx)
    return json_response(200, out)


def _store_export(st, approval_id: str, res, ctx: Ctx) -> dict:
    record = {"sid": st.sid, "mode": st.mode, "preset": st.preset, "actuator": st.actuator, "seed": st.seed,
              "export_meta": res.export_meta, "approvals": st.approvals, "history": st.history, "trace": st.trace}
    store = s3io.get_store()
    t = time.perf_counter()
    try:
        if store is None:
            return s3io.inline_export(res.export_files)
        try:
            out = store.put_export(st.sid, approval_id, res.export_files, json.loads(json.dumps(record, default=str)))
        except s3io.ExportError as e:
            from . import obs

            obs.emit({"evt": "export_error", "sid": st.sid, "error": str(e)[:500]})
            out = s3io.inline_export(res.export_files)
            out["note"] = "storage unavailable; files returned inline"
        return out
    finally:
        ctx.timings["export_store"] = 1000 * (time.perf_counter() - t)


def board_png(req: Request, ctx: Ctx) -> Response:
    kind = req.query.get("kind", "screen")
    if kind not in ("screen", "print"):
        raise HttpError(400, "BAD_REQUEST", "kind must be 'screen' or 'print'")
    data = board_mod.board_png(kind)
    return Response(200, data, "image/png", api=True, binary=True,
                    extra_headers={"content-disposition": f'inline; filename="nextview_board_{kind}.png"'})


@lru_cache(maxsize=1)
def _evidence_payload() -> bytes:
    natural = lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]  # noqa: E731 - E2 before E10
    figs = sorted((p.name for p in EVIDENCE_DIR.glob("*.png")), key=natural) if EVIDENCE_DIR.is_dir() else []
    path = EVIDENCE_DIR / "results.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = {"available": False, "error": "results.json is not valid JSON"}
        if isinstance(data, dict):
            data.setdefault("available", True)
        else:
            data = {"available": True, "results": data}
    else:
        data = {"available": False}
    data["_figures"] = figs
    return json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8")


def evidence(req: Request, ctx: Ctx) -> Response:
    return Response(200, _evidence_payload(), "application/json", api=True)


def evidence_png(req: Request, ctx: Ctx, name: str) -> Response:
    p = EVIDENCE_DIR / name
    if not p.is_file():
        raise HttpError(404, "NOT_FOUND", f"no evidence figure {name}")
    return Response(200, p.read_bytes(), "image/png", api=False, binary=True,
                    extra_headers={"cache-control": "public, max-age=300"})


def export_file(req: Request, ctx: Ctx, key: str, name: str) -> Response:
    store = s3io.get_store()
    got = store.get(key, name) if isinstance(store, s3io.LocalStore) else None
    if got is None:
        raise HttpError(404, "NOT_FOUND", "no such export")
    data, ctype = got
    return Response(200, data, ctype, api=True, binary=True,
                    extra_headers={"content-disposition": f'attachment; filename="{name}"'})


@lru_cache(maxsize=8)
def _static_cached(name: str) -> bytes | None:
    p = WEB_DIR / name
    return p.read_bytes() if p.is_file() else None


def _static(name: str) -> bytes | None:
    if os.environ.get("NEXTVIEW_DEV_RELOAD") == "1":  # dev server: pick up edits without a restart
        p = WEB_DIR / name
        return p.read_bytes() if p.is_file() else None
    return _static_cached(name)


FALLBACK_HTML = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>NextView</title></head>"
                 "<body><h1>NextView</h1><p>The web UI files are missing from this build.</p></body></html>")


IMMUTABLE = "public, max-age=31536000, immutable"


def static(req: Request, ctx: Ctx) -> Response:
    """UI files. The page links its script and stylesheet with ``?v=<build>``, so those can be cached
    for good and a returning visitor spends no throttle tokens on them (the stage allows 2 req/s)."""
    name, ctype = STATIC[req.path]
    data = _static(name)
    if data is None:
        if name == "index.html":
            return Response(200, FALLBACK_HTML, ctype, api=False)
        raise HttpError(404, "NOT_FOUND", f"no {name} in this build")
    if name == "index.html":
        v = re.sub(r"[^A-Za-z0-9._-]", "", ctx.version)[:64] or "dev"
        html = data.decode("utf-8").replace('src="/app.js"', f'src="/app.js?v={v}"').replace(
            'href="/style.css"', f'href="/style.css?v={v}"')
        return Response(200, html, ctype, api=False)
    extra = {"cache-control": IMMUTABLE} if req.query.get("v") and ctx.version != "dev" else {}
    return Response(200, data, ctype, api=False, extra_headers=extra)


ROUTES = {
    "/api/health": ("GET", health),
    "/api/presets": ("GET", presets),
    "/api/session": ("POST", session),
    "/api/step": ("POST", step),
    "/api/frame": ("POST", frame),
    "/api/approve": ("POST", approve),
    "/api/board.png": ("GET", board_png),
    "/api/evidence": ("GET", evidence),
}


def route_name(path: str) -> str:
    """Path as logged (evidence and export file names collapsed)."""
    if EVIDENCE_RE.match(path):
        return "/evidence/*"
    if EXPORT_RE.match(path):
        return "/api/export/*"
    if path in ROUTES or path in STATIC:
        return path
    return "other"


def dispatch(req: Request, ctx: Ctx) -> Response:
    if req.path in ROUTES:
        method, fn = ROUTES[req.path]
        if req.method != method:
            raise HttpError(405, "BAD_REQUEST", f"use {method} for {req.path}")
        return fn(req, ctx)
    if req.path in STATIC:
        if req.method != "GET":
            raise HttpError(405, "BAD_REQUEST", "use GET")
        return static(req, ctx)
    m = EVIDENCE_RE.match(req.path)
    if m and req.method == "GET":
        return evidence_png(req, ctx, m.group(1))
    m = EXPORT_RE.match(req.path)
    if m and req.method == "GET":
        return export_file(req, ctx, m.group(1), m.group(2))
    raise HttpError(404, "NOT_FOUND", f"no route for {req.method} {req.path}")


def warm_start() -> dict:
    """Touch the perception code paths once at container start (Lambda loads image pages lazily, so
    the first detection in a fresh container is several times slower than later ones)."""
    from ..tools.detect import detect_board
    from ..tools.nbv import k_initial
    from ..tools.pose import estimate_pose
    from ..tools.quality import assess_frame

    out = {}
    t = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(board_mod.board_png("screen"), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (480, 336), interpolation=cv2.INTER_AREA)
    det = detect_board(img)
    out["detect"] = 1000 * (time.perf_counter() - t)
    if det.ok:
        t = time.perf_counter()
        estimate_pose(det, k_initial((480, 336), None), np.zeros(5))
        assess_frame(cv2.resize(img, (240, 168), interpolation=cv2.INTER_AREA), detect_board(cv2.resize(img, (240, 168))))
        out["pose_assess"] = 1000 * (time.perf_counter() - t)
    return {k: round(v, 1) for k, v in out.items()}


def map_exception(e: BaseException) -> tuple[int, str, str] | None:
    """(status, code, message) for the expected request errors; None for a 500."""
    if isinstance(e, HttpError):
        return e.status, e.code, e.message
    if isinstance(e, StateError):
        return e.http_status, e.code, str(e)
    if isinstance(e, (LoopError, ApprovalError)):
        return 400, getattr(e, "code", "BAD_REQUEST") or "BAD_REQUEST", str(e)
    return None
