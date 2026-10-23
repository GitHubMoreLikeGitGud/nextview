"""Reproducing results from the committed files: repo-relative paths in result metadata, the summary from the
compact TEST tables, and the back-projection check of the mm table."""

import csv
import gzip
import json
import shutil
from pathlib import Path

import pytest

import nextview
from nextview.bench.consequence import inversion_check
from nextview.bench.run_synthetic import code_manifest, portable_path
from nextview.bench.summarize import summarize
from nextview.sim.lens import make_camera

REPO = Path(__file__).resolve().parents[1]


def test_result_metadata_paths_are_repo_relative(tmp_path, monkeypatch):
    pkg = Path(nextview.__file__).resolve().parent
    if not any((q / ".git").exists() for q in pkg.parents):
        pytest.skip("not a git checkout")
    assert portable_path(pkg / "config.py") == "src/nextview/config.py"
    assert code_manifest()["package_dir"] == "src/nextview"
    outside = tmp_path / "stop_calibration.json"
    outside.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert portable_path(outside) == "stop_calibration.json"
    for split in ("synth_dev", "synth_test", "synth_failure"):
        meta = json.loads((REPO / "results" / split / "meta.json").read_text(encoding="utf-8"))
        for value in (meta["stop_calibration"], meta["code"]["package_dir"]):
            assert not Path(value).is_absolute() and "\\" not in value and ":" not in value


def test_summarize_reads_the_committed_compact_tables(tmp_path):
    src = REPO / "results" / "synth_test"
    d = tmp_path / "synth_test"
    d.mkdir()
    for name in ("runs_core.csv.gz", "sessions.jsonl.gz", "meta.json"):
        shutil.copy(src / name, d / name)
    got = summarize(d, REPO / "config" / "stop_calibration.json")
    ref = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    assert got["runs"] == ref["runs"] and got["errors"] == ref["errors"]
    for a, b in zip(got["primary_endpoints"], ref["primary_endpoints"], strict=True):
        assert a["id"] == b["id"] and a["significant_holm_0.05"] == b["significant_holm_0.05"] and a["wins"] == b["wins"]
        assert a["median_diff"] == pytest.approx(b["median_diff"], abs=0.01)


def test_inversion_check_of_an_exact_estimate(tmp_path):
    cam = make_camera("normal", 1000)
    d = [float(x) for x in cam.dist] + [""] * (8 - len(cam.dist))
    row = {"camera": 1000, "family": "normal", "seed": 0, "actuator": "robot", "policy": "A", "n": 10, "model": cam.model,
           "fx": cam.K[0, 0], "fy": cam.K[1, 1], "cx": cam.K[0, 2], "cy": cam.K[1, 2], "rms": 0.1,
           **{f"d{i}": d[i] for i in range(8)}}
    p = tmp_path / "estimates.csv.gz"
    with gzip.open(p, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    e = inversion_check(p, ns=(10,))["table"]["A/robot/n10/normal"]
    assert e["runs"] == 1 and e["median_endpoint_fail_share"] == 0.0 and e["runs_with_failed_endpoints"] == 0
    assert e["median_of_run_p95_mm"] < 1e-3 and e["median_of_run_p95_mm_invertible_pairs"] < 1e-3
