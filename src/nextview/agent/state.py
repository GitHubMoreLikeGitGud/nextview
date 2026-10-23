"""Session state (spec section 2.1) and the signed, size-capped, expiring state token (section 7.1).

The whole session lives in the client-held token. Every step rebuilds the state from it; K and
dist are recomputed from the views (the stored calibration is only a warm start for the solver,
and it is covered by the HMAC like everything else in the token).

Token format: ``"v1." + b64url(zlib(json, 6)) + "." + b64url(HMAC_SHA256(key, "v1." + body))``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import secrets
import time
import zlib
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

from ..types import Quality, View

__all__ = [
    "TOKEN_VERSION",
    "MAX_TOKEN_BYTES",
    "SESSION_TTL_S",
    "TERMINAL_STATUSES",
    "FLAGS",
    "StateError",
    "SessionState",
    "new_session",
    "state_key",
    "encode_token",
    "decode_token",
    "round_corners",
    "jsonable",
]

TOKEN_VERSION = 1
TOKEN_PREFIX = "v1."
MAX_TOKEN_BYTES = 256 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024  # decompression guard
SESSION_TTL_S = 6 * 3600
TERMINAL_STATUSES = ("exported", "discarded", "ended")
FLAGS = ("UNDERFIT", "WARP", "DRIFT", "AMBIGUOUS", "OUT_OF_SCOPE")
MAX_TRACE_LINES = 400
ENV_KEY = "NEXTVIEW_STATE_KEY"

_ephemeral_key: bytes | None = None


class StateError(Exception):
    """A token that cannot be used: ``code`` is the API error code, ``http_status`` its status."""

    STATUS = {"STATE_TAMPERED": 400, "STATE_EXPIRED": 400, "STATE_TOO_LARGE": 413}

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.http_status = self.STATUS.get(code, 400)


def round_corners(corners) -> np.ndarray:
    """Corners as stored in the token: float64 rounded to 0.01 px (rounded at acceptance, so the
    token round trip is lossless and a local run equals a token-driven run)."""
    return np.round(np.asarray(corners, np.float64).reshape(-1, 2), 2)


def _f(x, nd: int | None = None):
    """JSON-safe float (None for NaN/inf)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, nd) if nd is not None else x


def _nan(x) -> float:
    return math.nan if x is None else float(x)


def jsonable(x):
    """Plain-JSON copy: numpy arrays/scalars to Python, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _f(x)
    return x


def _quality_to_dict(q: Quality | None) -> dict | None:
    if q is None:
        return None
    return {
        "rise": _f(q.edge_rise_px, 3),
        "rise_dir": [_f(v, 3) for v in q.rise_by_dir],
        "aniso": _f(q.anisotropy, 3),
        "blur_dir": _f(q.blur_dir_deg, 1),
        "sat": _f(q.saturation_frac, 5),
        "area": _f(q.board_area_frac, 5),
        "glare": None if q.glare_centroid is None else [_f(v, 1) for v in q.glare_centroid],
        "edges": int(q.n_edges),
    }


def _quality_from_dict(d: dict | None) -> Quality | None:
    if d is None:
        return None
    return Quality(
        edge_rise_px=_nan(d["rise"]),
        rise_by_dir=tuple(_nan(v) for v in d["rise_dir"]),
        anisotropy=_nan(d["aniso"]),
        blur_dir_deg=d["blur_dir"],
        saturation_frac=_nan(d["sat"]),
        board_area_frac=_nan(d["area"]),
        glare_centroid=None if d["glare"] is None else (float(d["glare"][0]), float(d["glare"][1])),
        n_edges=int(d["edges"]),
    )


def view_to_dict(v: View) -> dict:
    return {
        "id": int(v.id),
        "ids": np.asarray(v.corner_ids).astype(int).tolist(),
        "c": round_corners(v.corners).ravel().tolist(),
        "q": _quality_to_dict(v.quality),
        "r": [_f(x, 6) for x in np.asarray(v.rvec).ravel()],
        "t": [_f(x, 6) for x in np.asarray(v.tvec).ravel()],
        "pq": _f(v.prequential_rms, 4),
    }


def view_from_dict(d: dict) -> View:
    ids = np.asarray(d["ids"], np.int32)
    return View(
        id=int(d["id"]),
        corner_ids=ids,
        corners=np.asarray(d["c"], np.float64).reshape(-1, 2),
        quality=_quality_from_dict(d.get("q")),
        rvec=np.asarray(d["r"], np.float64),
        tvec=np.asarray(d["t"], np.float64),
        prequential_rms=d.get("pq"),
    )


@dataclass(eq=False)
class SessionState:
    """Everything a session needs between steps (spec 2.1 + 7.1).

    ``attempt`` is the capture index that seeds the simulator streams and never resets;
    ``attempts`` counts captures since the (re)start and is capped by ``max_attempts``.
    ``n_accepted_total`` counts accepted frames over the whole session (it drives autofocus
    drift in the simulator, which a restart does not undo).
    """

    sid: str
    mode: str  # "sim" | "webcam"
    preset: str | None
    actuator: str  # "robot" | "human"
    seed: int
    camera_seed: int
    budget: int
    max_attempts: int
    tau: float
    image_size: tuple[int, int]
    exp: int
    step: int = 0
    attempt: int = 0
    attempts: int = 0
    n_accepted_total: int = 0
    next_view_id: int = 0
    next_approval_id: int = 1
    status: str = "ready"
    model: dict = field(default_factory=lambda: {"name": "k5", "pinned": False})
    views: list = field(default_factory=list)  # list[View]
    rejected: list = field(default_factory=list)  # {step, attempt, rule, metric, value}
    last_target: dict | None = None  # {rvec, tvec, source, rank, pred_gain, pred_U95_after, instruction, desc}
    candidates: list = field(default_factory=list)  # top-5 of the last plan (same keys as last_target)
    last_realised: dict | None = None  # estimated pose of the last accepted view {rvec, tvec}
    sim_prev_pose: dict | None = None  # sim only: previous realised GT pose (hand-held blur direction)
    pending_approval: dict | None = None
    approvals: list = field(default_factory=list)  # {id, kind, decision, reason, ts, step}
    flags: list = field(default_factory=list)
    calib: dict | None = None  # warm start: {model, K, dist, rms}
    focal_obs: list = field(default_factory=list)  # opening phase: single-frame focal estimates, px
    preq_history: list = field(default_factory=list)  # prequential RMS of every accepted frame since (re)start
    seek_full_board: bool = False  # R9 pre-check: the NBV asks for whole-board views (warp suspected, < 6 full views)
    r9: dict | None = None  # last completed R9 cross-validation {n_full, step, gain, cv_ro, cv_k5}
    baseline: dict | None = None  # sim only: shadow random-policy track
    history: list = field(default_factory=list)
    trace: list = field(default_factory=list)

    # ---- helpers -------------------------------------------------------------------------
    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def has_flag(self, name: str) -> bool:
        return name in self.flags

    def set_flag(self, name: str) -> None:
        if name not in FLAGS:
            raise ValueError(f"unknown flag {name!r}")
        if name not in self.flags:
            self.flags.append(name)

    def clear_flag(self, name: str) -> None:
        if name in self.flags:
            self.flags.remove(name)

    def add_trace(self, line: str) -> None:
        self.trace.append(line)
        if len(self.trace) > MAX_TRACE_LINES:
            del self.trace[: len(self.trace) - MAX_TRACE_LINES]

    def last_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.last_realised:
            return None
        return np.asarray(self.last_realised["rvec"], float), np.asarray(self.last_realised["tvec"], float)

    # ---- serialisation ---------------------------------------------------------------------
    def to_dict(self) -> dict:
        d: dict[str, Any] = {"v": TOKEN_VERSION}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "views":
                v = [view_to_dict(x) for x in v]
            elif f.name == "image_size":
                v = [int(v[0]), int(v[1])]
            elif f.name == "baseline" and v is not None:
                v = dict(v, views=[view_to_dict(x) if isinstance(x, View) else x for x in v.get("views", [])])
            d[f.name] = jsonable(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        if d.get("v") != TOKEN_VERSION:
            raise StateError("STATE_TAMPERED", f"unsupported state version {d.get('v')!r}")
        kw = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if f.name == "views":
                v = [view_from_dict(x) for x in v]
            elif f.name == "image_size":
                v = (int(v[0]), int(v[1]))
            elif f.name == "baseline" and v is not None:
                v = dict(v, views=[view_from_dict(x) for x in v.get("views", [])])
            kw[f.name] = v
        return cls(**kw)


def new_session(
    *,
    mode: str = "sim",
    preset: str | None = None,
    actuator: str = "robot",
    seed: int = 0,
    camera_seed: int | None = None,
    budget: int | None = None,
    th=None,
    image_size: tuple[int, int] = (1280, 720),
    baseline: bool | None = None,
    now: float | None = None,
    sid: str | None = None,
) -> SessionState:
    """A fresh session. ``th`` is a ``config.Thresholds`` (budget, attempt cap and tau come from it).

    ``camera_seed`` defaults to ``seed``. ``baseline`` (sim only, default on in sim) keeps a shadow
    random-policy track for the race chart.
    """
    if mode not in ("sim", "webcam"):
        raise ValueError(f"unknown mode {mode!r}")
    if actuator not in ("robot", "human"):
        raise ValueError(f"unknown actuator {actuator!r}")
    if th is None:
        from ..config import load_thresholds

        th = load_thresholds()
    budget = int(th.budget_frames if budget is None else budget)
    if not 3 <= budget <= int(th.budget_frames):
        raise ValueError(f"budget must be in [3, {th.budget_frames}], got {budget}")
    now = time.time() if now is None else float(now)
    use_baseline = (mode == "sim") if baseline is None else bool(baseline and mode == "sim")
    return SessionState(
        sid=sid or "nv_" + secrets.token_hex(6),
        mode=mode,
        preset=preset,
        actuator=actuator,
        seed=int(seed),
        camera_seed=int(seed if camera_seed is None else camera_seed),
        budget=budget,
        max_attempts=int(th.max_attempts),
        tau=float(th.tau_px),
        image_size=(int(image_size[0]), int(image_size[1])),
        exp=int(now + SESSION_TTL_S),
        baseline={"attempt": 0, "views": [], "calib": None, "focal_obs": [], "truth": None, "next_view_id": 0,
                  "sim_prev_pose": None} if use_baseline else None,
    )


# ---- token -----------------------------------------------------------------------------------

def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def state_key(env: dict | None = None, allow_ephemeral: bool = False) -> bytes:
    """The HMAC key from ``NEXTVIEW_STATE_KEY`` (base64 of at least 32 random bytes).

    With ``allow_ephemeral`` (local dev server only) a missing key is replaced by a random
    per-process key, so tokens die with the process.
    """
    global _ephemeral_key
    env = os.environ if env is None else env
    raw = env.get(ENV_KEY)
    if raw:
        try:
            key = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            key = b""
        if len(key) < 32:
            raise ValueError(f"{ENV_KEY} must be base64 of at least 32 bytes")
        return key
    if not allow_ephemeral:
        raise ValueError(f"{ENV_KEY} is not set")
    if _ephemeral_key is None:
        _ephemeral_key = secrets.token_bytes(32)
    return _ephemeral_key


def _sign(key: bytes, msg: str) -> str:
    return _b64e(hmac.new(key, msg.encode("ascii"), hashlib.sha256).digest())


def encode_token(state: SessionState, key: bytes) -> str:
    """Signed token for ``state``; raises ``StateError("STATE_TOO_LARGE")`` above 256 KB."""
    raw = json.dumps(state.to_dict(), separators=(",", ":"), allow_nan=False).encode("utf-8")
    body = TOKEN_PREFIX + _b64e(zlib.compress(raw, 6))
    token = body + "." + _sign(key, body)
    if len(token) > MAX_TOKEN_BYTES:
        raise StateError("STATE_TOO_LARGE", f"state token is {len(token)} bytes (cap {MAX_TOKEN_BYTES})")
    return token


def decode_token(token: str, key: bytes, now: float | None = None) -> SessionState:
    """Verify and decode a token. Errors: STATE_TOO_LARGE (413), STATE_TAMPERED, STATE_EXPIRED (400)."""
    if not isinstance(token, str):
        raise StateError("STATE_TAMPERED", "state must be a string")
    if not token.isascii():
        raise StateError("STATE_TAMPERED", "state token has non-ASCII characters")
    if len(token) > MAX_TOKEN_BYTES:
        raise StateError("STATE_TOO_LARGE", f"state token is {len(token)} bytes (cap {MAX_TOKEN_BYTES})")
    body, sep, sig = token.rpartition(".")
    if not sep or not body.startswith(TOKEN_PREFIX):
        raise StateError("STATE_TAMPERED", "malformed state token")
    if not hmac.compare_digest(_sign(key, body), sig):
        raise StateError("STATE_TAMPERED", "state signature does not match")
    try:
        dec = zlib.decompressobj()
        raw = dec.decompress(_b64d(body[len(TOKEN_PREFIX):]), MAX_JSON_BYTES)
        if dec.unconsumed_tail:
            raise StateError("STATE_TOO_LARGE", "state payload too large")
        data = json.loads(raw)
    except StateError:
        raise
    except (ValueError, zlib.error, binascii.Error) as e:
        raise StateError("STATE_TAMPERED", f"undecodable state: {e}") from None
    now = time.time() if now is None else float(now)
    if not isinstance(data, dict) or int(data.get("exp", 0)) <= now:
        if isinstance(data, dict) and "exp" in data:
            raise StateError("STATE_EXPIRED", "session expired")
        raise StateError("STATE_TAMPERED", "state has no expiry")
    try:
        return SessionState.from_dict(data)
    except StateError:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise StateError("STATE_TAMPERED", f"invalid state contents: {e}") from None
