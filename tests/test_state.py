import base64
import json
import zlib

import numpy as np
import pytest

from nextview.agent.state import (
    MAX_TOKEN_BYTES,
    SessionState,
    StateError,
    decode_token,
    encode_token,
    new_session,
    round_corners,
    state_key,
)
from nextview.config import load_thresholds
from nextview.types import Quality, View

KEY = b"k" * 32
NOW = 1_800_000_000.0
TH = load_thresholds()


def _state() -> SessionState:
    st = new_session(mode="sim", preset="wide_robot_handheld", actuator="human", seed=42, th=TH, now=NOW, sid="nv_0123456789ab")
    rng = np.random.default_rng(0)
    q = Quality(1.42, (1.4, 1.5, float("nan"), 1.3), 1.08, None, 0.0, 0.18, None, 22)
    for i in range(3):
        ids = np.arange(40, dtype=np.int32)
        st.views.append(View(i, ids, round_corners(rng.uniform(0, 1280, (40, 2))), q, rng.normal(size=3), rng.normal(size=3),
                             None if i == 0 else 0.123))
    st.flags.append("AMBIGUOUS")
    st.approvals.append({"id": "ap_1", "kind": "switch_model", "decision": "deny", "reason": "x", "ts": NOW, "step": 2})
    st.history.append({"step": 1, "U95": float("nan"), "rule": "R13_NBV"})
    st.calib = {"model": "k5", "K": np.eye(3).tolist(), "dist": [0.1, 0, 0, 0, 0], "rms": 0.2}
    st.trace.append("#1 R13_NBV: x → y")
    return st


def test_round_trip_is_lossless():
    st = _state()
    tok = encode_token(st, KEY)
    assert tok.startswith("v1.") and tok.count(".") == 2
    back = decode_token(tok, KEY, now=NOW + 60)
    a, b = st.to_dict(), back.to_dict()
    assert a == b
    assert back.views[1].prequential_rms == pytest.approx(0.123)
    np.testing.assert_array_equal(back.views[2].corners, st.views[2].corners)
    assert back.views[0].quality.rise_by_dir[2] != back.views[0].quality.rise_by_dir[2]  # NaN survives as NaN
    assert back.history[0]["U95"] is None  # non-finite floats become null in the token
    assert back.image_size == (1280, 720) and back.flags == ["AMBIGUOUS"]


def test_corners_are_rounded_to_hundredths():
    c = round_corners([[1.23456, 2.98765]])
    assert c.tolist() == [[1.23, 2.99]]


@pytest.mark.parametrize("where", ["body", "signature", "prefix"])
def test_one_byte_tamper_is_rejected(where):
    tok = encode_token(_state(), KEY)
    body, _, sig = tok.rpartition(".")
    if where == "body":
        i = len(body) // 2
        tok = body[:i] + ("A" if body[i] != "A" else "B") + body[i + 1 :] + "." + sig
    elif where == "signature":
        tok = body + "." + ("A" if sig[0] != "A" else "B") + sig[1:]
    else:
        tok = "v2" + tok[2:]
    with pytest.raises(StateError) as e:
        decode_token(tok, KEY, now=NOW)
    assert e.value.code == "STATE_TAMPERED" and e.value.http_status == 400


def test_wrong_key_is_tampered_and_same_key_from_env_round_trips():
    env = {"NEXTVIEW_STATE_KEY": base64.b64encode(bytes(range(32))).decode()}
    k1, k2 = state_key(env), state_key(dict(env))
    assert k1 == k2 == bytes(range(32))
    tok = encode_token(_state(), k1)
    assert decode_token(tok, k2, now=NOW).sid == "nv_0123456789ab"
    with pytest.raises(StateError) as e:
        decode_token(tok, KEY, now=NOW)
    assert e.value.code == "STATE_TAMPERED"
    with pytest.raises(ValueError):
        state_key({})
    with pytest.raises(ValueError):
        state_key({"NEXTVIEW_STATE_KEY": base64.b64encode(b"short").decode()})
    eph = state_key({}, allow_ephemeral=True)
    assert len(eph) == 32 and eph == state_key({}, allow_ephemeral=True)


def test_expired_token():
    st = _state()
    tok = encode_token(st, KEY)
    assert decode_token(tok, KEY, now=st.exp - 1).sid == st.sid
    with pytest.raises(StateError) as e:
        decode_token(tok, KEY, now=st.exp + 1)
    assert e.value.code == "STATE_EXPIRED" and e.value.http_status == 400
    assert st.exp == int(NOW + 6 * 3600)


def test_oversize_token_gives_413():
    with pytest.raises(StateError) as e:
        decode_token("v1." + "A" * MAX_TOKEN_BYTES + ".sig", KEY, now=NOW)
    assert e.value.code == "STATE_TOO_LARGE" and e.value.http_status == 413
    st = _state()
    rng = np.random.default_rng(1)
    # incompressible random trace lines push the encoded token past the cap
    st.trace = ["".join(chr(33 + int(x)) for x in rng.integers(0, 90, 2000)) for _ in range(200)]
    with pytest.raises(StateError) as e:
        encode_token(st, KEY)
    assert e.value.code == "STATE_TOO_LARGE" and e.value.http_status == 413


def test_signed_but_wrong_version_or_garbage_is_tampered():
    import hashlib
    import hmac

    def forge(payload: bytes) -> str:
        body = "v1." + base64.urlsafe_b64encode(zlib.compress(payload)).rstrip(b"=").decode()
        sig = base64.urlsafe_b64encode(hmac.new(KEY, body.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
        return body + "." + sig

    d = _state().to_dict()
    d["v"] = 2
    with pytest.raises(StateError) as e:
        decode_token(forge(json.dumps(d).encode()), KEY, now=NOW)
    assert e.value.code == "STATE_TAMPERED"
    with pytest.raises(StateError) as e:
        decode_token(forge(b"not json"), KEY, now=NOW)
    assert e.value.code == "STATE_TAMPERED"


def test_new_session_validation_and_defaults():
    st = new_session(mode="webcam", actuator="human", seed=3, th=TH, now=NOW)
    assert st.baseline is None and st.camera_seed == 3 and st.budget == 30 and st.max_attempts == 60 and st.tau == TH.tau_px == 0.75
    assert st.sid.startswith("nv_") and len(st.sid) == 15
    sim = new_session(mode="sim", preset="inspection_gauge_cam", seed=1, camera_seed=7, budget=8, th=TH, now=NOW)
    assert sim.baseline is not None and sim.camera_seed == 7 and sim.budget == 8
    for bad in (dict(mode="x"), dict(actuator="cat"), dict(budget=2), dict(budget=31)):
        with pytest.raises(ValueError):
            new_session(th=TH, **bad)


@pytest.mark.parametrize("bad", ["v1.é.sig", "v1.abc.☃", "é" * 10])
def test_non_ascii_token_is_tampered_not_a_crash(bad):
    with pytest.raises(StateError) as e:
        decode_token(bad, KEY, now=NOW)
    assert e.value.code == "STATE_TAMPERED" and e.value.http_status == 400


def test_r9_state_round_trips():
    st = _state()
    st.seek_full_board = True
    st.r9 = {"n_full": 6, "ids": [0, 1, 2], "fold": 1, "sse_ro": 1.5, "n_ro": 80, "sse_k5": 2.5, "n_k5": 80,
             "fold_s": [0.3], "done": False}
    back = decode_token(encode_token(st, KEY), KEY, now=NOW)
    assert back.seek_full_board is True and back.r9 == st.r9
