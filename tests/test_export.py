import json

import cv2
import numpy as np

import _synth as S
from nextview.tools.calibrate import calibrate
from nextview.tools.export import DISCLAIMER, export_calibration
from nextview.tools.uncertainty import uncertainty_map
from nextview.types import View

SIZE = (1280, 720)


def _calib():
    K = S.make_K(900.0, SIZE)
    dist = np.array([-0.15, 0.05, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(12)
    views = []
    for i in range(8):
        rv, tv = S.random_pose(K, SIZE, rng, dist)
        p = S.project(K, dist, rv, tv)
        views.append(View(i, np.arange(40, dtype=np.int32), p + rng.normal(0, 0.2, p.shape), None, rv, tv))
    c = calibrate(views, "k5", SIZE)
    return c, uncertainty_map(c, views, SIZE, None)


def test_export_yaml_png_json():
    c, unc = _calib()
    trace = ["#1 R13_NBV: opening pose 1", "#2 R13_NBV: opening pose 2"]
    out = export_calibration(c, unc, trace, size=SIZE, meta={"session_id": "nv_test", "approved": True})
    assert set(out) == {"yaml", "png", "json"}

    fs = cv2.FileStorage(out["yaml"].decode("utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    np.testing.assert_allclose(fs.getNode("camera_matrix").mat(), c.K)
    np.testing.assert_allclose(fs.getNode("distortion_coefficients").mat().ravel(), c.dist)
    assert int(fs.getNode("image_width").real()) == 1280 and int(fs.getNode("image_height").real()) == 720
    assert fs.getNode("calibration_model").string() == "k5"
    assert fs.getNode("uncertainty_U95_px").real() == np.float64(unc.U95)
    assert fs.getNode("session_id").string() == "nv_test"
    fs.release()
    assert DISCLAIMER in out["yaml"].decode("utf-8")

    img = cv2.imdecode(np.frombuffer(out["png"], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (600, 960, 3)

    rec = json.loads(out["json"])
    assert rec["model"] == "k5" and rec["image_size"] == [1280, 720]
    np.testing.assert_allclose(np.array(rec["K"]), c.K)
    assert rec["uncertainty"]["U95_px"] == unc.U95
    assert rec["uncertainty"]["p_exceed"] is None  # NaN is written as null
    assert np.array(rec["uncertainty"]["sigma_px"]).shape == (18, 32)
    assert rec["trace"] == trace and rec["disclaimer"] == DISCLAIMER


def test_export_numpy_meta_and_covariance_std():
    c, unc = _calib()
    meta = {"approved": np.bool_(True), "n_attempts": np.int64(17), "tau_px": np.float32(0.5), "missing": float("nan")}
    out = export_calibration(c, unc, None, size=SIZE, meta=meta)
    rec = json.loads(out["json"])
    assert rec["meta"] == {"approved": True, "n_attempts": 17, "tau_px": 0.5, "missing": None}
    std = np.sqrt(np.diag(c.cov))
    np.testing.assert_allclose(rec["std_intrinsics"], std)
    np.testing.assert_allclose(rec["std_intrinsics_opencv"], c.std_intr[:9])
    fs = cv2.FileStorage(out["yaml"].decode("utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    assert int(fs.getNode("approved").real()) == 1
    assert int(fs.getNode("n_attempts").real()) == 17
    assert fs.getNode("tau_px").real() == 0.5
    assert fs.getNode("missing").empty()
    np.testing.assert_allclose(fs.getNode("std_intrinsics").mat().ravel(), std)
    fs.release()
    assert DISCLAIMER == "Decision support for camera calibration — not certified metrology."
