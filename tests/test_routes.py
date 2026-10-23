"""Routes beyond the agent cycle: board images, presets, evidence, export stores, the dev server."""

import base64
import json
import secrets
import threading
import urllib.request

import pytest

from nextview.lambda_app import routes, s3io
from nextview.lambda_app.devserver import make_event, make_server
from nextview.lambda_app.handler import handler

PNG_SIG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("NEXTVIEW_STATE_KEY", base64.b64encode(secrets.token_bytes(32)).decode())
    yield
    s3io.set_store(None)
    s3io._store_set = False


def get(path):
    return handler(make_event("GET", path, {}, None), None)


def test_board_png_screen_and_print():
    for kind in ("screen", "print"):
        r = get(f"/api/board.png?kind={kind}")
        assert r["statusCode"] == 200 and r["isBase64Encoded"] is True
        assert r["headers"]["content-type"] == "image/png"
        data = base64.b64decode(r["body"])
        assert data.startswith(PNG_SIG)
        assert len(r["body"]) < 6 * 1024 * 1024  # Lambda response limit
    assert get("/api/board.png")["statusCode"] == 200  # screen by default
    bad = get("/api/board.png?kind=poster")
    assert bad["statusCode"] == 400 and json.loads(bad["body"])["error"]["code"] == "BAD_REQUEST"


def test_presets_route_lists_the_config():
    b = json.loads(get("/api/presets")["body"])
    assert b["default"] == "wide_robot_handheld"
    assert [p["id"] for p in b["presets"]][:2] == ["wide_robot_handheld", "inspection_gauge_cam"]
    assert b["thresholds"]["tau_px"] == routes.TH.tau_px and b["caps"]["jpeg_bytes"] == 300 * 1024


def test_evidence_json_and_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "EVIDENCE_DIR", tmp_path)
    routes._evidence_payload.cache_clear()
    b = json.loads(get("/api/evidence")["body"])
    assert b == {"available": False, "_figures": []}
    (tmp_path / "results.json").write_text(json.dumps({"headline": {"x": 1}}), encoding="utf-8")
    (tmp_path / "E1.png").write_bytes(PNG_SIG + b"fake")
    routes._evidence_payload.cache_clear()
    b = json.loads(get("/api/evidence")["body"])
    assert b["available"] is True and b["headline"] == {"x": 1} and b["_figures"] == ["E1.png"]
    fig = get("/evidence/E1.png")
    assert fig["statusCode"] == 200 and base64.b64decode(fig["body"]).startswith(PNG_SIG)
    assert "max-age" in fig["headers"]["cache-control"]
    assert get("/evidence/E2.png")["statusCode"] == 404
    assert get("/evidence/..%2Fhandler.py")["statusCode"] == 404
    for name in ("E10_b.png", "E2_a.png"):
        (tmp_path / name).write_bytes(PNG_SIG + b"fake")
    routes._evidence_payload.cache_clear()
    assert json.loads(get("/api/evidence")["body"])["_figures"] == ["E1.png", "E2_a.png", "E10_b.png"]
    routes._evidence_payload.cache_clear()


class _FakeS3:
    def __init__(self, fail=False):
        self.puts, self.fail = [], fail

    def put_object(self, **kw):
        if self.fail:
            raise ConnectionError("no network")
        self.puts.append(kw)

    def generate_presigned_url(self, op, ExpiresIn, Params):
        return f"https://{Params['Bucket']}.s3.ap-southeast-1.amazonaws.com/{Params['Key']}?X-Amz-Expires={ExpiresIn}"


def test_s3_store_keys_types_and_presign():
    fake = _FakeS3()
    st = s3io.S3Store("bucket-x", "ap-southeast-1", client=fake)
    out = st.put_export("nv_abc", "ap_2", {"yaml": b"%YAML", "png": PNG_SIG, "json": b"{}"}, {"trace": ["#1"]})
    keys = [p["Key"] for p in fake.puts]
    assert keys == ["reports/nv_abc/ap_2/nextview_calibration.yaml", "reports/nv_abc/ap_2/nextview_report.png",
                    "reports/nv_abc/ap_2/nextview_calibration.json", "traces/nv_abc/ap_2.json"]
    assert [p["ContentType"] for p in fake.puts][:3] == ["application/x-yaml", "image/png", "application/json"]
    assert out["expires_s"] == 3600 and "X-Amz-Expires=3600" in out["png_url"]
    with pytest.raises(s3io.ExportError):
        s3io.S3Store("b", client=_FakeS3(fail=True)).put_export("s", "a", {"yaml": b"x"})


def test_local_store_serves_exports():
    store = s3io.LocalStore()
    s3io.set_store(store)
    out = store.put_export("nv_1", "ap_1", {"yaml": b"%YAML:1.0\n", "json": b"{}"})
    r = get(out["yaml_url"])
    assert r["statusCode"] == 200 and base64.b64decode(r["body"]) == b"%YAML:1.0\n"
    assert "attachment" in r["headers"]["content-disposition"]
    assert get(out["yaml_url"].replace("calibration.yaml", "report.png"))["statusCode"] == 404
    assert get("/api/export/" + "0" * 24 + "/nextview_calibration.yaml")["statusCode"] == 404


def test_inline_export_when_storage_fails(monkeypatch):
    s3io.set_store(s3io.S3Store("b", client=_FakeS3(fail=True)))

    class Res:
        export_files = {"yaml": b"%YAML", "png": PNG_SIG}
        export_meta = {"warnings": True}

    class St:
        sid, mode, preset, actuator, seed, approvals, history, trace = "nv_x", "sim", "p", "robot", 0, [], [], []

    ctx = routes.Ctx()
    exp = routes._store_export(St, "ap_1", Res, ctx)
    assert exp["store"] == "inline" and base64.b64decode(exp["files"]["png"]["b64"]) == PNG_SIG
    assert "export_store" in ctx.timings


def test_devserver_round_trip(monkeypatch):
    monkeypatch.setenv("NEXTVIEW_LOCAL_EXPORTS", "1")
    srv = make_server("127.0.0.1", 0, quiet=True)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        with urllib.request.urlopen(base + "/api/health", timeout=10) as r:
            assert r.status == 200 and json.loads(r.read())["opencv"] == "5.0.0"
        req = urllib.request.Request(base + "/api/session", data=json.dumps({"preset": "drone_survey_cam"}).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            assert json.loads(r.read())["preset"]["family"] == "tele"
        with urllib.request.urlopen(base + "/api/board.png?kind=screen", timeout=10) as r:
            assert r.headers["content-type"] == "image/png" and r.read().startswith(PNG_SIG)
    finally:
        srv.shutdown()
        srv.server_close()


def test_event_files_parse():
    from pathlib import Path

    for p in sorted((Path(__file__).parent / "events").glob("*.json")):
        ev = json.loads(p.read_text(encoding="utf-8"))
        out = handler(ev, None)
        assert out["statusCode"] in (200, 400), (p.name, out["body"][:200])


def test_warm_start_touches_detection_and_session_builds_the_simulator(capsys):
    parts = routes.warm_start()
    assert parts["detect"] > 0 and "pose_assess" in parts
    capsys.readouterr()
    r = handler(make_event("POST", "/api/session", {"content-type": "application/json"},
                           json.dumps({"preset": "inspection_gauge_cam", "seed": 5}).encode()), None)
    assert r["statusCode"] == 200 and json.loads(r["body"])["latency_ms"]["total"] > 0
    line = next(json.loads(ln) for ln in capsys.readouterr().out.splitlines() if '"evt":"session"' in ln)
    assert line["latency_ms"]["sim_warm"] > 0
