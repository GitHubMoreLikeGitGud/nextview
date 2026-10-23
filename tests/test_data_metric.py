import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from nextview.data import metric
from nextview.data.board_probe import BoardSpec, probe_board, small_marker_params
from nextview.data.remotezip import RemoteZip

REPO = Path(__file__).resolve().parents[1]
REAL_METRIC = REPO / "data_ext" / "metric"

D455_INTRINSICS = """fx: 633.708557129
fy: 632.945678711
cx: 627.227111816
cy: 372.594696045
has_dist_coeff: 1
dist_k0: -0.0537141859531
dist_k1: 0.0669208541512
dist_px: -0.000328784022713
dist_py: -0.000466717610834
dist_k2: -0.0210480950773
dist_k3: 0
dist_k4: 0
dist_k5: 0
img_width: 1280
img_height: 720
"""

GT_CAM = """0.84643978 -0.35280883 0.39895603 -0.96095061
-0.52565378 -0.43283212 0.73249394 -1.6566881
-0.085746966 -0.82965702 -0.55183232 1.2723601
0 0 0 1
"""

RECORD = {
    "id": 7976757,
    "doi": "10.5281/zenodo.7976757",
    "metadata": {
        "title": "METRIC - Multi-Eye To Robot Indoor Calibration Dataset",
        "publication_date": "2023-05-26",
        "version": "1.1",
        "license": {"id": "cc-by-4.0"},
        "creators": [
            {"name": "Davide Allegro", "affiliation": "University of Padua", "orcid": "0009-0008-1180-9290"},
            {"name": "Matteo Terreran", "affiliation": "University of Padua"},
            {"name": "Stefano Ghidoni"},
        ],
    },
    "files": [],
}

METRIC_BOARD = BoardSpec("DICT_6X6_50", (5, 4), 1.0, 0.79, True)


def render_board(spec: BoardSpec, size=(1280, 720), px_per_square=70, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sx, sy = spec.squares
    bw, bh = sx * px_per_square, sy * px_per_square
    img = spec.make().generateImage((bw, bh), marginSize=0, borderBits=1)
    W, H = size
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    off = np.float32([W / 2 - bw / 2, H / 2 - bh / 2]) + rng.uniform(-100, 100, 2).astype(np.float32)
    dst = src + off + rng.uniform(-25, 25, (4, 2)).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((H, W), 110, np.uint8)
    warped = cv2.warpPerspective(img, Hm, (W, H), borderValue=110)
    mask = cv2.warpPerspective(np.full((bh, bw), 255, np.uint8), Hm, (W, H)) > 0
    canvas[mask] = warped[mask]
    return cv2.GaussianBlur(canvas, (0, 0), 0.7)


# ----------------------------------------------------------------------------- readers


def test_read_intrinsics_maps_dataset_keys_to_opencv_order(tmp_path):
    p = tmp_path / "intrinsic_pars_file.yaml"
    p.write_text(D455_INTRINSICS)
    K, dist, size = metric.read_intrinsics(p)
    np.testing.assert_allclose(K, [[633.708557129, 0, 627.227111816], [0, 632.945678711, 372.594696045], [0, 0, 1]])
    np.testing.assert_allclose(dist, [-0.0537141859531, 0.0669208541512, -0.000328784022713, -0.000466717610834, -0.0210480950773])
    assert size == (1280, 720)


def test_read_intrinsics_rational_and_no_dist(tmp_path):
    p = tmp_path / "i.yaml"
    p.write_text(D455_INTRINSICS.replace("dist_k3: 0", "dist_k3: 0.5"))
    _, dist, _ = metric.read_intrinsics(p)
    assert dist.shape == (8,) and dist[5] == 0.5
    p.write_text(D455_INTRINSICS.replace("has_dist_coeff: 1", "has_dist_coeff: 0"))
    assert not np.any(metric.read_intrinsics(p)[1])
    p.write_text("fx: 1\nfy: 1\n")
    with pytest.raises(ValueError, match="missing"):
        metric.read_intrinsics(p)


def test_read_matrix4(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text(GT_CAM)
    m = metric.read_matrix4(p)
    assert m.shape == (4, 4) and m[3, 3] == 1
    np.testing.assert_allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-3)  # the shipped GT is orthonormal to ~2e-4
    p.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError):
        metric.read_matrix4(p)
    p.write_text(GT_CAM.replace("0.84643978", "nan"))
    with pytest.raises(ValueError):
        metric.read_matrix4(p)


def test_attribution_markdown_from_record():
    md = metric.attribution_markdown(RECORD)
    assert "Davide Allegro; Matteo Terreran; Stefano Ghidoni" in md
    assert "ORCID 0009-0008-1180-9290" in md
    assert "https://doi.org/10.5281/zenodo.7976757" in md
    assert "CC BY 4.0" in md and "creativecommons.org/licenses/by/4.0" in md
    assert "Version 1.1" in md and "(2023)" in md


def test_archive_url_rejects_foreign_host():
    rec = dict(RECORD, files=[{"key": metric.ARCHIVE_KEY, "size": 1, "links": {"self": "https://evil.example/x.zip"}}])
    with pytest.raises(ValueError):
        metric.archive_url(rec)
    rec = dict(RECORD, files=[{"key": "other.zip", "size": 1, "links": {"self": "https://zenodo.org/x"}}])
    with pytest.raises(ValueError):
        metric.archive_url(rec)


# ----------------------------------------------------------------------------- fetching (offline)


def _metric_zip() -> tuple[bytes, dict[str, bytes]]:
    pre = metric.SUBSETS["d455"].prefix
    ok, png = cv2.imencode(".png", render_board(METRIC_BOARD, seed=1))
    files = {
        pre + "GT/gt_cam1.csv": GT_CAM.encode(),
        pre + "GT/gt_cam2.csv": GT_CAM.encode(),
        pre + "camera1/intrinsic_pars_file.yaml": D455_INTRINSICS.encode(),
        pre + "camera1/image/0001.png": png.tobytes(),
        pre + "camera1/pose/0001.csv": GT_CAM.encode(),
        pre + "camera2/image/0001.png": png.tobytes(),
        # names outside the expected layout are ignored
        pre + "camera1/image/../../../evil.png": b"x",
        pre + "camera1/image/readme.txt": b"x",
        pre + "camera5/image/0001.png": b"x",
        "METRIC dataset/real_data/kinectV2/charuco/small_workcell/camera1/image/0001.png": b"x",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue(), files


class _RangeServer:
    def __init__(self, payload: bytes):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                a, b = self.headers["Range"][6:].split("-")
                a, b = int(a), min(int(b), len(payload) - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(payload)}")
                self.send_header("Content-Length", str(b - a + 1))
                self.end_headers()
                self.wfile.write(payload[a : b + 1])

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/m.zip"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_fetch_subset_offline(tmp_path):
    payload, files = _metric_zip()
    rec = dict(RECORD, files=[{"key": metric.ARCHIVE_KEY, "size": len(payload), "checksum": "md5:x", "links": {"self": "https://zenodo.org/fake"}}])
    (tmp_path / "record.json").write_text(json.dumps(rec))
    srv = _RangeServer(payload)
    try:
        with RemoteZip(srv.url) as rz:
            members = metric.subset_members(rz, "d455", cameras=[1])
            assert sorted(members) == [
                "d455/GT/gt_cam1.csv",
                "d455/camera1/image/0001.png",
                "d455/camera1/intrinsic_pars_file.yaml",
                "d455/camera1/pose/0001.csv",
            ]
            m = metric.fetch_subset("d455", [1, 2], tmp_path, rz=rz, workers=2)
            assert m["n_images"] == 2 and m["license"] == "cc-by-4.0" and m["fetch"]["written"] == 6
            assert m["fetch"]["bytes_fetched"] > 0 and m["fetch"]["requests"] >= 1
            again = metric.fetch_subset("d455", [1, 2], tmp_path, rz=rz)
            assert again["fetch"]["written"] == 0 and again["fetch"]["skipped"] == 6
            # the first run's transfer stats survive the re-run
            hist = json.loads((tmp_path / "d455" / "manifest.json").read_text())["fetch_history"]
            assert [h["written"] for h in hist] == [6, 0] and hist[0]["bytes_fetched"] == m["fetch"]["bytes_fetched"]
    finally:
        srv.close()
    assert not (tmp_path / "evil.png").exists() and not list(tmp_path.rglob("readme.txt"))
    ref = metric.load_reference("d455", 1, tmp_path)
    assert ref.kind == "factory" and ref.image_size == (1280, 720)
    assert metric.load_gt_extrinsics("d455", 1, tmp_path).shape == (4, 4)
    assert metric.load_ee_pose("d455", 1, "0001", tmp_path).shape == (4, 4)
    with pytest.raises(ValueError):
        metric.load_ee_pose("d455", 1, "../x", tmp_path)
    assert [p.name for p in metric.image_paths("d455", 2, tmp_path)] == ["0001.png"]
    manifest = json.loads((tmp_path / "d455" / "manifest.json").read_text())
    assert manifest["cameras"] == [1, 2] and manifest["n_files"] == 6


def test_kinect_principal_point_converted_to_opencv(tmp_path):
    for key, cx, cy in (("kinect_synth", 960.5, 540.5), ("d455", 627.227111816, 372.594696045)):
        d = tmp_path / key / "camera1"
        d.mkdir(parents=True)
        text = D455_INTRINSICS.replace("cx: 627.227111816", f"cx: {cx}").replace("cy: 372.594696045", f"cy: {cy}")
        (d / "intrinsic_pars_file.yaml").write_text(text)
    k = metric.load_reference("kinect_synth", 1, tmp_path)
    assert (k.K[0, 2], k.K[1, 2]) == (959.5, 539.5) and (k.K_file[0, 2], k.K_file[1, 2]) == (960.5, 540.5)
    assert k.K[0, 0] == k.K_file[0, 0] and k.principal_shift == -1.0
    d = metric.load_reference("d455", 1, tmp_path)
    np.testing.assert_array_equal(d.K, d.K_file)


def test_legacy_manifest_fetch_is_kept_in_history(tmp_path):
    payload, _ = _metric_zip()
    rec = dict(RECORD, files=[{"key": metric.ARCHIVE_KEY, "size": len(payload), "links": {"self": "https://zenodo.org/fake"}}])
    (tmp_path / "record.json").write_text(json.dumps(rec))
    (tmp_path / "d455").mkdir()
    old = {"fetch": {"written": 2016, "bytes_fetched": 123}, "files": {}}
    (tmp_path / "d455" / "manifest.json").write_text(json.dumps(old))
    srv = _RangeServer(payload)
    try:
        with RemoteZip(srv.url) as rz:
            m = metric.fetch_subset("d455", [1], tmp_path, rz=rz)
    finally:
        srv.close()
    assert m["fetch_history"][0] == old["fetch"] and m["fetch_history"][1]["written"] == 4


# ----------------------------------------------------------------------------- pool


def test_load_pool_detects_and_caches(tmp_path, monkeypatch):
    img_dir = tmp_path / "kinect_synth" / "camera3" / "image"
    img_dir.mkdir(parents=True)
    for i in range(1, 5):
        cv2.imwrite(str(img_dir / f"{i:04d}.png"), render_board(METRIC_BOARD, size=(1920, 1080), seed=i))
    cv2.imwrite(str(img_dir / "0005.png"), np.full((1080, 1920), 90, np.uint8))
    (tmp_path / "kinect_synth" / "board_spec.json").write_text(json.dumps(METRIC_BOARD.to_dict()))

    frames = metric.load_pool(3, "kinect_synth", tmp_path)
    assert [f.name for f in frames] == ["0001", "0002", "0003", "0004", "0005"]
    assert all(f.n_corners == METRIC_BOARD.n_corners and f.det.ok for f in frames[:4])
    assert not frames[4].det.ok and frames[4].n_corners == 0
    assert frames[0].image_size == (1920, 1080) and frames[0].source == "metric/kinect_synth/camera3"
    np.testing.assert_allclose(frames[0].det.obj, METRIC_BOARD.object_points()[frames[0].det.ids])
    assert all(f.board == METRIC_BOARD.planar() for f in frames) and not frames[0].board.is_nextview

    # second call is served from the cache without running the detector
    import nextview.data.board_probe as bp

    real_detect, real_params = bp.detect_charuco, bp.small_marker_params
    monkeypatch.setattr(bp, "detect_charuco", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-detected")))
    again = metric.load_pool(3, "kinect_synth", tmp_path)
    for a, b in zip(frames, again):
        np.testing.assert_array_equal(a.det.ids, b.det.ids)
        np.testing.assert_allclose(a.det.corners, b.det.corners)

    # a changed detector setting invalidates the cache
    calls = []

    def changed_params():
        p = real_params()
        p.adaptiveThreshWinSizeMax = 41
        return p

    def counting_detect(*a, **k):
        calls.append(1)
        return real_detect(*a, **k)

    monkeypatch.setattr(bp, "small_marker_params", changed_params)
    with pytest.raises(FileNotFoundError, match="detection cache"):
        metric.load_pool(3, "kinect_synth", tmp_path, detect=False)
    monkeypatch.setattr(bp, "detect_charuco", counting_detect)
    metric.load_pool(3, "kinect_synth", tmp_path)
    assert len(calls) == 5


def test_load_pool_needs_board_spec(tmp_path):
    (tmp_path / "d455" / "camera1" / "image").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="board_spec"):
        metric.load_pool(1, "d455", tmp_path)


def test_bad_camera_and_subset():
    with pytest.raises(ValueError):
        metric.image_paths("d455", 5)
    with pytest.raises(ValueError):
        metric.image_paths("nope", 1)


# ----------------------------------------------------------------------------- real data (skipped if absent)

needs_metric = pytest.mark.skipif(
    not (REAL_METRIC / "kinect_synth" / "camera2" / "image").is_dir(), reason="data_ext/metric not fetched"
)


def cached_pool(camera: int, subset: str = "kinect_synth"):
    """Real METRIC frames from the detection cache; skips instead of running the 2x-upscale detector."""
    if not metric.board_spec_path(subset, REAL_METRIC).exists():
        pytest.skip("board_spec.json not written yet (fetch_metric.py --probe)")
    try:
        return metric.load_pool(camera, subset, REAL_METRIC, detect=False)
    except FileNotFoundError:
        pytest.skip("no valid detection cache (fetch_metric.py --skip-fetch --report builds it)")


@needs_metric
def test_real_metric_probe_finds_dictionary_on_three_frames():
    # three full-board frames from the cache, cropped around the board so the dictionary search
    # over every predefined dictionary at 2x upscale stays fast
    frames = [f for f in cached_pool(2) if f.n_corners == 12]
    crops = []
    for f in frames[:: max(len(frames) // 3, 1)][:3]:
        x0, y0 = f.det.corners.min(axis=0)
        x1, y1 = f.det.corners.max(axis=0)
        m = 1.5 * max(x1 - x0, y1 - y0)
        img = f.image(cv2.IMREAD_GRAYSCALE)
        crops.append(img[max(int(y0 - m), 0) : int(y1 + m), max(int(x0 - m), 0) : int(x1 + m)])
    res = probe_board(crops, n_frames=3, layout_frames=3, upscale=metric.DETECT_UPSCALE, params=small_marker_params())
    assert len(res.frames) == 3
    assert res.spec.dictionary.startswith("DICT_6X6_")
    assert res.spec.squares == (5, 4) and res.spec.legacy


@pytest.mark.slow
@needs_metric
def test_real_metric_probe_on_full_frames():
    paths = metric.image_paths("kinect_synth", 2, REAL_METRIC)
    res = probe_board(paths, n_frames=3, layout_frames=3, upscale=metric.DETECT_UPSCALE, params=small_marker_params())
    assert res.spec.dictionary.startswith("DICT_6X6_")
    assert res.spec.squares == (5, 4) and res.spec.legacy


@needs_metric
def test_real_metric_pool_and_reference():
    frames = cached_pool(2)
    assert len(frames) == 251
    assert sum(f.n_corners == 12 for f in frames) >= 50
    assert all(f.det.ok == f.board.usable(f.det.ids) for f in frames)
    ref = metric.load_reference("kinect_synth", 2, REAL_METRIC)
    assert ref.kind == "gt" and ref.image_size == (1920, 1080) and abs(ref.K[0, 0] - 1371.02278154) < 1e-6
    assert (ref.K[0, 2], ref.K[1, 2]) == (959.5, 539.5)


@needs_metric
def test_real_metric_square_size_from_gt_rig():
    cached_pool(2), cached_pool(4)
    est = metric.estimate_square_size("kinect_synth", REAL_METRIC, cameras=[2, 4], detect=False)
    assert abs(est["square_m"] - 0.050) < 0.001 and est["residual_rms_m"] < 0.005


@needs_metric
def test_real_kinect_principal_point_agrees_with_gt_chain():
    for cam in metric.CAMERAS:
        cached_pool(cam)
    fixed = metric.gt_chain_fit("kinect_synth", REAL_METRIC, detect=False)
    # with the OpenCV-convention principal point the chain leaves no mean residual and no offset to fit
    for cam in metric.CAMERAS:
        assert np.all(np.abs(fixed["mean_residual_px"][cam]) < 0.15)
        assert np.all(np.abs(fixed["principal_offset_px"][cam]) < 0.3)
    assert fixed["rms_px"] < 1.1 * fixed["rms_px_free_principal"]
    assert abs(fixed["square_m"] - 0.050) < 0.0005
    # the file values sit about 1 px off in both axes
    raw = metric.gt_chain_fit("kinect_synth", REAL_METRIC, detect=False, file_intrinsics=True)
    for cam in metric.CAMERAS:
        assert np.all(np.array(raw["principal_offset_px"][cam]) < -0.7)
    assert raw["rms_px"] > 2.0 * raw["rms_px_free_principal"]
