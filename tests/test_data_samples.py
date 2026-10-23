import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from nextview.data import samples

REPO = Path(__file__).resolve().parents[1]
REAL_SAMPLES = REPO / "data_ext" / "samples"


def chessboard_frame(pattern=(9, 6), sq=40, size=(640, 480), seed=0) -> np.ndarray:
    cols, rows = pattern
    board = np.zeros(((rows + 1) * sq, (cols + 1) * sq), np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq] = 255
    board = cv2.copyMakeBorder(board, sq, sq, sq, sq, cv2.BORDER_CONSTANT, value=255)
    rng = np.random.default_rng(seed)
    h, w = board.shape
    W, H = size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    s = 0.95 * min(W / w, H / h)
    dst = src * s + np.float32([(W - w * s) / 2, (H - h * s) / 2]) + rng.uniform(-15, 15, (4, 2)).astype(np.float32)
    img = cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (W, H), borderValue=100)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def test_object_points_row_major():
    obj = samples.chessboard_object_points((9, 6), 2.0)
    assert obj.shape == (54, 3)
    np.testing.assert_allclose(obj[:2], [[0, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(obj[9], [0, 2, 0])
    # the whole board extends one square beyond the outer inner corners
    board = samples.chessboard_board((9, 6), 2.0)
    np.testing.assert_allclose(board.outline()[[0, 2], :2], [[-2, -2], [18, 12]])
    assert board.n_cols == 9 and board.n_corners == 54


def test_load_samples_on_synthetic_chessboards(tmp_path):
    for i in (1, 2):
        cv2.imwrite(str(tmp_path / f"left0{i}.jpg"), chessboard_frame(seed=i))
    cv2.imwrite(str(tmp_path / "left03.jpg"), np.full((480, 640), 128, np.uint8))
    frames = samples.load_samples(tmp_path)
    assert [f.name for f in frames] == ["left01", "left02", "left03"]
    for f in frames[:2]:
        assert f.det.ok and f.n_corners == 54 and f.image_size == (640, 480)
        assert f.board == samples.chessboard_board((9, 6)) and f.board.squares == (10, 7) and f.board.usable(f.det.ids)
        assert np.isfinite(f.meta["sharpness_px"]) and f.meta["sharpness_px"] > 0
        # corners follow the object-point order: one homography maps the grid onto them
        H, _ = cv2.findHomography(f.det.obj[:, :2], f.det.corners, 0)
        proj = cv2.perspectiveTransform(f.det.obj[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2)
        assert np.sqrt(np.mean(np.sum((proj - f.det.corners) ** 2, 1))) < 0.5  # JPEG + blur; a wrong order is tens of px
    assert not frames[2].det.ok and frames[2].meta["sharpness_px"] is None


def test_fetch_rejects_digest_mismatch(tmp_path, monkeypatch):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self, n=-1):
            return b"not the pinned image"

    monkeypatch.setattr(samples.urllib.request, "urlopen", lambda *a, **k: Resp())
    with pytest.raises(ValueError, match="SHA-256"):
        samples.fetch_samples(tmp_path)
    assert not list(tmp_path.glob("left*.jpg"))


def test_fetch_skips_intact_files(tmp_path, monkeypatch):
    if not (REAL_SAMPLES / "left01.jpg").exists():
        pytest.skip("data_ext/samples not fetched")
    for name in samples.SAMPLE_SHA256:
        (tmp_path / name).write_bytes((REAL_SAMPLES / name).read_bytes())
    monkeypatch.setattr(samples.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    assert len(samples.fetch_samples(tmp_path)) == 13


@pytest.mark.skipif(not (REAL_SAMPLES / "left14.jpg").exists(), reason="data_ext/samples not fetched")
def test_real_samples_detect_and_calibrate():
    for name, digest in samples.SAMPLE_SHA256.items():
        assert hashlib.sha256((REAL_SAMPLES / name).read_bytes()).hexdigest() == digest
    frames = samples.load_samples(REAL_SAMPLES)
    assert len(frames) == 13 and all(f.det.ok and f.n_corners == 54 for f in frames)
    obj = [f.det.obj.astype(np.float32) for f in frames]
    img = [f.det.corners.astype(np.float32) for f in frames]
    rms, K, _, _, _ = cv2.calibrateCamera(obj, img, (640, 480), None, None)
    assert rms < 0.5 and abs(K[0, 0] - 532) < 10
