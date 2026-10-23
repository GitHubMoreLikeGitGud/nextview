"""Lambda handler on API Gateway v2 events: UI files, the session -> step -> approve -> export flow,
caps, error shapes, webcam frames and what the logs may contain (spec 7, 10.5, 10.6, 13)."""

import base64
import json
import re
import secrets

import cv2
import numpy as np
import pytest

from nextview.lambda_app import routes, s3io
from nextview.lambda_app.devserver import make_event
from nextview.lambda_app.handler import handler
from nextview.sim import make_simulator
from nextview.sim.poses import pose_for_view

ONE_MB = 1024 * 1024


class FakeStore:
    kind = "fake"

    def __init__(self):
        self.calls = []

    def put_export(self, sid, approval_id, files, trace_record=None):
        self.calls.append((sid, approval_id, {k: len(v) for k, v in files.items()}, trace_record))
        return {f"{k}_url": f"https://example.invalid/{sid}/{k}" for k in files} | {"expires_s": 3600, "store": "fake"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("NEXTVIEW_STATE_KEY", base64.b64encode(secrets.token_bytes(32)).decode())
    store = FakeStore()
    s3io.set_store(store)
    yield store
    s3io.set_store(None)
    s3io._store_set = False


def call(method, path, body=None, raw=None, ctype="application/json"):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    out = handler(make_event(method, path, {"content-type": ctype}, data), None)
    assert len(out["body"]) < ONE_MB
    return out


def jbody(out):
    return json.loads(out["body"])


def log_lines(capsys):
    return [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]


def test_root_and_assets_are_served_with_csp():
    page = call("GET", "/")
    assert page["statusCode"] == 200 and page["headers"]["content-type"].startswith("text/html")
    assert "default-src 'self'" in page["headers"]["content-security-policy"]
    assert "NextView" in page["body"] and re.search(r'src="/app\.js\?v=[A-Za-z0-9._-]+"', page["body"])
    assert re.search(r'href="/style\.css\?v=[A-Za-z0-9._-]+"', page["body"])
    js = call("GET", "/app.js?v=abc")
    css = call("GET", "/style.css")
    assert js["statusCode"] == 200 and js["headers"]["content-type"].startswith("text/javascript")
    assert css["statusCode"] == 200 and css["headers"]["content-type"].startswith("text/css")
    # no external assets: scripts, styles and fetches stay on this origin
    assert not re.search(r'<script[^>]*\ssrc="(https?:)?//', page["body"])
    assert not re.search(r'<link[^>]*\shref="(https?:)?//', page["body"])
    assert "@import" not in css["body"] and "url(http" not in css["body"]
    assert "http://" not in js["body"] and "https://" not in js["body"]


def test_session_step_approve_export_flow(_env, capsys):
    s = call("POST", "/api/session", {"mode": "sim", "preset": "inspection_gauge_cam", "seed": 3, "budget": 3})
    assert s["statusCode"] == 200
    sb = jbody(s)
    assert sb["status"] == "ready" and sb["image_size"] == [1280, 720] and sb["thresholds"]["budget_frames"] == 3
    assert sb["board"] == {"squares": [9, 6], "square_mm": 30.0, "marker_mm": 22.0, "dictionary": "DICT_5X5_100"}
    assert sb["preset"]["family"] == "normal" and sb["actuator"] == "robot"
    state, card, steps = sb["state"], None, 0
    while card is None:
        steps += 1
        assert steps <= 12, "no approval card within 12 steps"
        r = call("POST", "/api/step", {"state": state, "choose_candidate": None})
        assert r["statusCode"] == 200, r["body"][:300]
        b = jbody(r)
        assert r["headers"]["cache-control"] == "no-store"
        state, card = b["state"], b["approval"]
        assert b["step"] == steps and b["decision"]["rule"]
        assert b["frame"]["w"] == 960 and b["frame"]["h"] == 540
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b["frame"]["jpeg_b64"]), np.uint8), cv2.IMREAD_UNCHANGED)
        assert img.shape[:2] == (540, 960)
        assert "baseline" in b and b["history"][-1]["step"] == steps
    assert card["kind"] == "accept_with_warnings" and b["status"] == "escalated"
    assert len(b["uncertainty"]["sigma_px"]) == 32 * 18 and len(b["coverage"]["counts"]) == 16 * 9

    wait = jbody(call("POST", "/api/step", {"state": state}))  # a step while the card is open captures nothing
    assert wait["decision"]["rule"] == card["rule"] and wait["approval"]["id"] == card["id"] and wait["frame"] is None

    a = call("POST", "/api/approve", {"state": state, "approval_id": card["id"], "decision": "approve",
                                      "reason": "test sign-off"})
    assert a["statusCode"] == 200, a["body"][:300]
    ab = jbody(a)
    assert ab["status"] == "exported"
    assert set(ab["export"]) >= {"yaml_url", "png_url", "json_url", "expires_s"}
    sid, aid, sizes, record = _env.calls[-1]
    assert sid == sb["session_id"] and aid == card["id"] and all(n > 100 for n in sizes.values())
    assert record["approvals"][-1]["decision"] == "approve" and record["approvals"][-1]["reason"] == "test sign-off"

    again = call("POST", "/api/step", {"state": ab["state"]})
    assert again["statusCode"] == 400 and jbody(again)["error"]["code"] == "BAD_REQUEST"

    lines = log_lines(capsys)
    steps_logged = [ln for ln in lines if ln["evt"] == "step"]
    assert len(steps_logged) == steps + 2
    emf = steps_logged[0]
    assert emf["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "NextView"
    assert emf["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [["Mode", "Route"]]
    names = {m["Name"] for m in emf["_aws"]["CloudWatchMetrics"][0]["Metrics"]}
    assert names == {"LatencyMs", "NbvMs", "Rejects", "ApprovalsRequested", "Stops", "ColdStart"}
    assert all(k in emf for k in names) and emf["Mode"] == "sim" and emf["Route"] == "/api/step"
    assert emf["latency_ms"]["total"] > 0 and "agent" in emf["latency_ms"]
    refused = [ln for ln in steps_logged if ln["status"] != 200]
    assert refused and all("_aws" not in ln and "Mode" not in ln for ln in refused)
    approve_line = next(ln for ln in lines if ln["evt"] == "approve")
    assert approve_line["status_agent"] == "exported" and "export_store" in approve_line["latency_ms"]


def test_body_caps_bad_json_and_methods(capsys):
    big = call("POST", "/api/step", raw=b"{" + b" " * (600 * 1024) + b"}")
    assert big["statusCode"] == 413 and jbody(big)["error"]["code"] == "PAYLOAD_TOO_LARGE"
    big_session = call("POST", "/api/session", raw=json.dumps({"seed": 1, "pad": "x" * 5000}).encode())
    assert big_session["statusCode"] == 413
    b64 = handler(dict(make_event("POST", "/api/step", {"content-type": "application/octet-stream"}, b"x" * (600 * 1024))), None)
    assert b64["statusCode"] == 413
    bad = call("POST", "/api/step", raw=b"{not json")
    assert bad["statusCode"] == 400 and jbody(bad)["error"]["code"] == "BAD_REQUEST"
    not_obj = call("POST", "/api/session", raw=b"[1,2]")
    assert not_obj["statusCode"] == 400
    no_state = call("POST", "/api/step", {"choose_candidate": 1})
    assert no_state["statusCode"] == 400 and "state" in jbody(no_state)["error"]["message"]
    assert call("GET", "/api/step")["statusCode"] == 405
    assert call("POST", "/api/health", {})["statusCode"] == 405
    nf = call("GET", "/api/nope")
    assert nf["statusCode"] == 404 and jbody(nf)["error"]["code"] == "NOT_FOUND"
    lines = log_lines(capsys)
    assert any(ln["status"] == 413 and ln["route"] == "/api/step" for ln in lines)
    assert not any("_aws" in ln for ln in lines)


@pytest.mark.parametrize("body,msg", [
    ({"mode": "lab"}, "mode"),
    ({"preset": "nope"}, "preset"),
    ({"seed": -1}, "seed"),
    ({"seed": True}, "seed"),
    ({"budget": 2}, "budget"),
    ({"budget": 31}, "budget"),
    ({"actuator": "drone"}, "actuator"),
    ({"mode": "webcam", "image_size": [2000, 720]}, "image_size"),
])
def test_session_validation(body, msg):
    r = call("POST", "/api/session", body)
    assert r["statusCode"] == 400 and msg in jbody(r)["error"]["message"]


def test_state_tampered_and_expired(monkeypatch):
    tok = jbody(call("POST", "/api/session", {"preset": "inspection_gauge_cam"}))["state"]
    body, sig = tok.rsplit(".", 1)
    flipped = body[:-3] + ("A" if body[-3] != "A" else "B") + body[-2:] + "." + sig
    r = call("POST", "/api/step", {"state": flipped})
    assert r["statusCode"] == 400 and jbody(r)["error"]["code"] == "STATE_TAMPERED"
    other_key = base64.b64encode(secrets.token_bytes(32)).decode()
    monkeypatch.setenv("NEXTVIEW_STATE_KEY", other_key)
    r = call("POST", "/api/step", {"state": tok})
    assert jbody(r)["error"]["code"] == "STATE_TAMPERED"
    monkeypatch.setattr("nextview.agent.state.SESSION_TTL_S", -10)
    expired = jbody(call("POST", "/api/session", {"preset": "inspection_gauge_cam"}))["state"]
    r = call("POST", "/api/step", {"state": expired})
    assert r["statusCode"] == 400 and jbody(r)["error"]["code"] == "STATE_EXPIRED"
    huge = call("POST", "/api/step", {"state": "v1." + "A" * (300 * 1024)})
    assert huge["statusCode"] == 413


def _board_jpeg(size=(1280, 720)):
    sim = make_simulator("normal", "robot", camera_seed=2, seed=0, occluder_p=0.0, corruption_p=0.0)
    rv, tv = pose_for_view(sim.camera, (560, 330), 0.5, 25, 45)
    img = sim.render_exact(rv, tv)
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def test_webcam_frame_logs_no_corners_or_jpeg(capsys):
    s = jbody(call("POST", "/api/session", {"mode": "webcam", "seed": 1, "image_size": [1280, 720]}))
    assert s["preset"] is None and s["stop_calibration"]["label"] == "transferred from simulation"
    jpg = _board_jpeg()
    b64 = base64.b64encode(jpg).decode()
    capsys.readouterr()
    r = call("POST", "/api/frame", {"state": s["state"], "jpeg_b64": b64})
    assert r["statusCode"] == 200, r["body"][:300]
    b = jbody(r)
    assert b["decision"]["accepted"] and b["perception"]["n_corners"] >= 30
    assert "frame" not in b and "truth" not in b and "baseline" not in b
    logs = capsys.readouterr().out
    assert logs.strip()
    assert "corners" not in logs and "jpeg" not in logs and b64[:64] not in logs and s["state"][:40] not in logs
    ok_line = next(json.loads(ln) for ln in logs.splitlines() if ln.startswith("{") and '"route":"/api/frame"' in ln)
    assert ok_line["Mode"] == "webcam" and ok_line["Route"] == "/api/frame" and "_aws" in ok_line
    sim_step = call("POST", "/api/step", {"state": b["state"]})
    assert sim_step["statusCode"] == 400  # webcam sessions take frames only
    wrong = call("POST", "/api/frame", {"state": b["state"], "jpeg_b64": base64.b64encode(_board_jpeg((640, 360))).decode()})
    assert wrong["statusCode"] == 400 and "1280x720" in jbody(wrong)["error"]["message"]
    refused = log_lines(capsys)  # refused requests are logged, but add no Mode x Route metric series
    assert [ln["status"] for ln in refused] == [400, 400]
    assert all("_aws" not in ln and "Mode" not in ln and "Route" not in ln for ln in refused)


def test_frame_caps_and_bad_jpeg():
    tok = jbody(call("POST", "/api/session", {"mode": "webcam", "image_size": [640, 480]}))["state"]
    big = base64.b64encode(secrets.token_bytes(310 * 1024)).decode()
    r = call("POST", "/api/frame", {"state": tok, "jpeg_b64": big})
    assert r["statusCode"] == 413 and jbody(r)["error"]["code"] == "PAYLOAD_TOO_LARGE"
    png = base64.b64encode(cv2.imencode(".png", np.zeros((480, 640), np.uint8))[1].tobytes()).decode()
    assert call("POST", "/api/frame", {"state": tok, "jpeg_b64": png})["statusCode"] == 400
    assert call("POST", "/api/frame", {"state": tok, "jpeg_b64": "@@@"})["statusCode"] == 400
    # a small file whose header claims 4000x3000 is refused before it is decoded
    small = bytearray(cv2.imencode(".jpg", np.zeros((16, 16), np.uint8))[1].tobytes())
    i = next(i for i in range(2, len(small) - 9) if small[i] == 0xFF and small[i + 1] in (0xC0, 0xC2))
    small[i + 5:i + 7] = (3000).to_bytes(2, "big")
    small[i + 7:i + 9] = (4000).to_bytes(2, "big")
    assert routes.jpeg_size(bytes(small)) == (4000, 3000)
    r = call("POST", "/api/frame", {"state": tok, "jpeg_b64": base64.b64encode(bytes(small)).decode()})
    assert r["statusCode"] == 413 and "4000x3000" in jbody(r)["error"]["message"]


def test_jpeg_size_reads_the_header():
    for w, h in ((640, 480), (1280, 720), (321, 241)):
        data = cv2.imencode(".jpg", np.zeros((h, w), np.uint8))[1].tobytes()
        assert routes.jpeg_size(data) == (w, h)
        prog = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8), [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])[1].tobytes()
        assert routes.jpeg_size(prog) == (w, h)
    assert routes.jpeg_size(b"\x89PNG\r\n") is None and routes.jpeg_size(b"") is None


def test_internal_errors_hide_the_trace(monkeypatch, capsys):
    tok = jbody(call("POST", "/api/session", {"preset": "inspection_gauge_cam"}))["state"]

    def boom(*a, **k):
        raise RuntimeError("secret detail 1234")

    monkeypatch.setattr(routes, "run_step", boom)
    capsys.readouterr()
    r = call("POST", "/api/step", {"state": tok})
    assert r["statusCode"] == 500 and jbody(r)["error"]["code"] == "INTERNAL"
    assert "secret detail" not in r["body"]
    logs = log_lines(capsys)
    err = next(ln for ln in logs if ln["evt"] == "error")
    assert "secret detail 1234" in err["traceback"]
    assert next(ln for ln in logs if ln["evt"] == "step")["status"] == 500


def test_missing_state_key_is_internal(monkeypatch):
    monkeypatch.delenv("NEXTVIEW_STATE_KEY")
    r = call("POST", "/api/session", {"preset": "inspection_gauge_cam"})
    assert r["statusCode"] == 500 and jbody(r)["error"]["code"] == "INTERNAL"
