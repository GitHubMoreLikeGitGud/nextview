import importlib.util
from pathlib import Path

import pytest

from nextview.data import metric, samples

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_metric.py"


@pytest.fixture()
def script():
    spec = importlib.util.spec_from_file_location("fetch_metric_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_samples_alone_uses_root_and_skips_zenodo(script, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(metric, "zenodo_record", lambda *a, **k: pytest.fail("Zenodo record requested"))
    monkeypatch.setattr(samples, "fetch_samples", lambda root=None, **k: seen.setdefault("fetch", root) and [])
    monkeypatch.setattr(samples, "load_samples", lambda root=None, **k: seen.setdefault("load", root) and [])
    assert script.main(["--samples", "--root", str(tmp_path)]) == 0
    assert seen == {"fetch": tmp_path / "samples", "load": tmp_path / "samples"}


def test_metric_actions_use_root_metric(script, tmp_path, monkeypatch):
    roots = []
    monkeypatch.setattr(metric, "zenodo_record", lambda root=None, **k: roots.append(root) or {"metadata": {"title": "t"}})
    monkeypatch.setattr(script, "cmd_attribution", lambda args, record: roots.append("attribution"))
    assert script.main(["--skip-fetch", "--attribution", "--root", str(tmp_path)]) == 0
    assert roots == [tmp_path / "metric", "attribution"]
