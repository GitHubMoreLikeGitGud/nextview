import dataclasses
import json
import math
import pickle

import numpy as np
import pytest

from nextview import config, types


def test_thresholds_load_spec_values():
    th = config.load_thresholds()
    assert th.min_corners == 12
    assert th.max_edge_rise_px == 3.0
    assert th.ippe_ambiguity_ratio == 0.85
    assert th.tau_px == 0.75 and th.stop_p_exceed == 0.10  # tau frozen on DEV (spec 2.6 rule)
    assert th.budget_frames == 30 and th.max_attempts == 60
    assert th.nbv_candidates == 300
    assert th.grid_uncertainty == (32, 18) and th.grid_coverage == (16, 9)
    assert th.off_target_deg == 15  # extra key, attribute access
    with pytest.raises(AttributeError):
        th.no_such_key
    with pytest.raises(dataclasses.FrozenInstanceError):
        th.tau_px = 0.75


def test_thresholds_roundtrip_pickle_and_replace():
    th = config.load_thresholds()
    th2 = pickle.loads(pickle.dumps(th))
    assert th2 == th and th2.nbv_margin_px == th.nbv_margin_px
    th3 = th.replace(tau_px=0.5)
    assert th3.tau_px == 0.5 and th.tau_px == 0.75
    json.dumps(th.to_dict())


def test_thresholds_missing_key_rejected(tmp_path):
    d = config.load_thresholds().to_dict()
    del d["tau_px"]
    p = tmp_path / "t.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="tau_px"):
        config.load_thresholds(p)


def test_presets():
    ps = config.load_presets()
    assert ps.default.id == "wide_robot_handheld"
    assert ps.default.family == "wide" and ps.default.actuator == "human"
    assert set(ps.ids) == {
        "wide_robot_handheld", "inspection_gauge_cam", "drone_survey_cam",
        "phone_crack_metrology", "fisheye_out_of_scope", "autofocus_drift",
    }
    assert "glare" in ps.get("phone_crack_metrology").nuisances
    assert ps.get("fisheye_out_of_scope").family == "fisheye130"
    with pytest.raises(KeyError):
        ps.get("nope")


def test_stop_calibration_placeholder_is_uncalibrated(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"calibrated": False, "quantiles": {"robot": None, "human": None}}))
    sc = config.load_stop_calibration(p)
    assert sc.calibrated is False
    assert math.isnan(sc.p_exceed(0.8, "robot", 0.5))


def test_stop_calibration_p_exceed(tmp_path):
    q = np.linspace(0.5, 1.5, 101).tolist()  # r uniform on [0.5, 1.5]
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"calibrated": True, "quantiles": {"robot": q, "human": None}}))
    sc = config.load_stop_calibration(p)
    assert sc.p_exceed(0.5, "robot", 0.5) == pytest.approx(0.5)  # tau/U95 = 1
    assert sc.p_exceed(0.25, "robot", 0.5) == pytest.approx(0.0)  # tau/U95 = 2 > max r
    assert sc.p_exceed(1.0, "robot", 0.5) == pytest.approx(1.0)  # tau/U95 = 0.5 = min r
    assert math.isnan(sc.p_exceed(0.5, "human", 0.5))
    pickle.loads(pickle.dumps(sc))


def test_types_are_frozen_contracts():
    det = types.Detection(ids=np.arange(3), corners=np.zeros((3, 2)), obj=np.zeros((3, 3)), n_markers=2, ok=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        det.ok = True
    q = types.Quality(1.0, (1.0, 1.0, 1.0, 1.0), 1.0, None, 0.0, 0.2, None, 12)
    v = types.View(id=0, corner_ids=det.ids, corners=det.corners, quality=q, rvec=np.zeros(3), tvec=np.zeros(3))
    assert v.prequential_rms is None
    names = {f.name for f in dataclasses.fields(types.CalibResult)}
    assert names == {"model", "K", "dist", "rms", "per_view", "rvecs", "tvecs", "std_intr", "sigma2", "F", "cov"}
    names = {f.name for f in dataclasses.fields(types.Candidate)}
    assert names == {"rank", "rvec", "tvec", "gain", "pred_U95_after", "effort", "score", "ghost_quad", "instruction"}
