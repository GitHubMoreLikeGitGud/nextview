"""Smoke bench (spec 13): 2 cameras x {B0, A} x budget 8 through the real runner; CSV schema valid."""

import csv
import json

import numpy as np
import pytest

from nextview.bench.common import ROW_FIELDS, split_jobs
from nextview.bench.policies import B2App, grid_b4, template_b1
from nextview.bench.run_synthetic import run_jobs
from nextview.config import load_stop_calibration, load_thresholds


def test_smoke_bench_runs_and_schema(tmp_path):
    jobs = split_jobs("smoke")
    assert len(jobs) == 4 and {j.policy for j in jobs} == {"A", "B0"} and all(j.budget == 8 for j in jobs)
    out = tmp_path / "smoke"
    res = run_jobs(jobs, out, workers=1, th=load_thresholds(), stopcal_path=None, log=lambda *a, **k: None)
    assert res["jobs"] == 4 and not res["errors"]
    with open(out / "runs.csv", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        assert tuple(r.fieldnames) == ROW_FIELDS
        rows = list(r)
    for j in jobs:
        mine = [x for x in rows if int(x["camera"]) == j.camera and x["policy"] == j.policy]
        assert [int(x["n"]) for x in mine] == list(range(1, 9))  # one row per accepted frame up to the budget
        assert all(x["true_p95"] == "" for x in mine[:2])  # no calibration before 3 views
        assert all(float(x["true_p95"]) > 0 and float(x["U95"]) > 0 for x in mine[2:])
        assert all(int(x["attempts"]) >= int(x["n"]) for x in mine)
    sessions = [json.loads(line) for line in open(out / "sessions.jsonl", encoding="utf-8")]
    assert len(sessions) == 4 and all("error" not in s for s in sessions)
    # deterministic: a second run in another directory gives byte-identical rows
    out2 = tmp_path / "smoke2"
    run_jobs(jobs[:2], out2, workers=1, th=load_thresholds(), stopcal_path=None, log=lambda *a, **k: None)
    a = [x for x in rows if (int(x["camera"]), x["policy"]) in {(j.camera, j.policy) for j in jobs[:2]}]
    with open(out2 / "runs.csv", newline="", encoding="utf-8") as f:
        b = list(csv.DictReader(f))
    assert a == b


def test_templates():
    t = template_b1()
    assert len(t) == 12 and t[0] == (0.5, 0.5, 0.5, 0.0, 0.0, 0.0)
    g = grid_b4()
    assert len(g) == 45 and len(set(g)) == 45
    assert g[0][:2] == (0.5, 0.5) and g[0][3] == 0.0  # starts at the centre frontal pose


def test_b2_angle_check_rejects_steep_frames():
    import cv2

    from nextview.board import object_points
    from nextview.geometry import rvec_from_tilt
    from nextview.sim.lens import make_camera

    cam = make_camera("normal", 0)
    app = B2App(cam.size)
    obj = object_points()
    for tilt, ok in ((20.0, True), (55.0, False)):
        rv = rvec_from_tilt(tilt, 0.0)
        R, _ = cv2.Rodrigues(np.asarray(rv, float).reshape(3, 1))
        t = np.array([0.0, 0.0, 0.6]) - R @ np.array([0.135, 0.09, 0.0])
        img = cam.project(obj, rv, t)
        assert app.angle_ok(obj, img) is ok


def test_corrupted_view_dropped_by_r7_in_its_own_step_is_counted(monkeypatch):
    """A corrupted view that R7 drops in the step that accepted it is a corrupted drop, not a clean one."""
    from nextview.agent.loop import Observation
    from nextview.bench import policies
    from nextview.bench.common import Job
    from nextview.types import Detection

    forced = []

    class Forced(policies.RecordingSource):
        def capture(self, rvec, tvec, **kw):
            o = super().capture(rvec, tvec, **kw)
            d = o.det
            if not forced and kw["n_accepted"] >= 6 and d.ok and len(d.ids) >= 12:
                c = np.asarray(d.corners, float).copy()
                c[::2] += 3.0  # every other corner moved by (3, 3) px: per-view RMS far above R7's cut
                forced.append(kw["attempt"])
                o = Observation(o.image, Detection(d.ids, c, d.obj, d.n_markers, d.ok),
                                np.asarray(d.ids)[::2].astype(int), o.truth, o.ms)
            else:
                o = Observation(o.image, d, np.zeros(0, int), o.truth, o.ms)
            self.last = o
            return o

    monkeypatch.setattr(policies, "RecordingSource", Forced)
    job = Job("smoke", 0, "normal", 0, "robot", "A", 10, ())
    rows, sess = policies.run_agent(job, load_thresholds(), load_stop_calibration())
    assert len(forced) == 1 and "error" not in sess
    r7 = (sess["rejected"] or {}).get("R7_OUTLIER", 0)
    assert sess["corrupted_accepted"] == 1 and sess["corrupted_dropped"] == 1
    assert sess["clean_dropped"] == r7 - 1 and sess["corrupted_cleared_by_restart"] == 0
