import copy
import importlib
import json
from pathlib import Path

import cv2
import pytest

EVENTS = Path(__file__).parent / "events"


def _event(name: str) -> dict:
    return json.loads((EVENTS / name).read_text(encoding="utf-8"))


@pytest.fixture()
def handler_mod():
    import nextview.lambda_app.handler as mod

    return importlib.reload(mod)


def test_health_reports_opencv_and_cold_flag(handler_mod):
    first = handler_mod.handler(_event("get_health.json"), None)
    second = handler_mod.handler(_event("get_health.json"), None)

    assert first["statusCode"] == 200
    assert first["headers"]["content-type"] == "application/json"
    assert first["headers"]["cache-control"] == "no-store"
    body = json.loads(first["body"])
    assert body["ok"] is True
    assert body["opencv"] == cv2.__version__ == "5.0.0"
    assert isinstance(body["version"], str) and body["version"]
    assert body["cold"] is True
    assert json.loads(second["body"])["cold"] is False


def test_version_falls_back_to_env(handler_mod, monkeypatch):
    monkeypatch.setattr(handler_mod, "__version__", "dev")
    monkeypatch.setenv("NEXTVIEW_VERSION", "abc1234")
    body = json.loads(handler_mod.handler(_event("get_health.json"), None)["body"])
    assert body["version"] == "abc1234"


def test_root_serves_html_with_csp(handler_mod):
    resp = handler_mod.handler(_event("get_root.json"), None)
    assert resp["statusCode"] == 200
    assert resp["headers"]["content-type"].startswith("text/html")
    assert "default-src 'self'" in resp["headers"]["content-security-policy"]
    assert "NextView" in resp["body"]
    assert resp["isBase64Encoded"] is False


def test_unknown_route_is_json_404(handler_mod):
    ev = copy.deepcopy(_event("get_health.json"))
    ev["rawPath"] = ev["requestContext"]["http"]["path"] = "/api/nope"
    resp = handler_mod.handler(ev, None)
    assert resp["statusCode"] == 404
    assert json.loads(resp["body"])["error"]["code"] == "NOT_FOUND"


def test_health_rejects_post(handler_mod):
    ev = copy.deepcopy(_event("get_health.json"))
    ev["requestContext"]["http"]["method"] = "POST"
    resp = handler_mod.handler(ev, None)
    assert resp["statusCode"] == 405
    assert json.loads(resp["body"])["error"]["code"] == "BAD_REQUEST"
