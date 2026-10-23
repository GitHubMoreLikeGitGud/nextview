import json

import cv2
import numpy as np
import pytest

from nextview.board import BOARD, N_CORNERS
from nextview.data.frame import NEXTVIEW_BOARD
from nextview.data.owner_capture import extract_frames, load_owner_pool
from nextview.tools.detect import detect_board


def write_video(path, n=23, size=(960, 540)) -> None:
    W, H = size
    board = BOARD.generateImage((900, 600), marginSize=0, borderBits=1)
    bh, bw = board.shape
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
    assert vw.isOpened()
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    for i in range(n):
        t = i / max(n - 1, 1)
        dst = np.float32([[180 + 40 * t, 70], [780, 90 + 30 * t], [760, 470], [200, 450 - 20 * t]])
        frame = cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), size, borderValue=200)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    vw.release()


def test_extract_every_fifth_frame_and_detect(tmp_path):
    video = tmp_path / "my capture.avi"
    write_video(video)
    out = tmp_path / "frames"
    paths = extract_frames(video, out, every=5)
    assert [p.name for p in paths] == [f"frame_{i:06d}.png" for i in (0, 5, 10, 15, 20)]
    meta = json.loads((out / "capture.json").read_text())
    assert meta["frames_read"] == 23 and meta["frames_written"] == 5 and meta["every"] == 5

    frames = load_owner_pool(out)
    assert len(frames) == 5 and frames[0].source == "owner/frames"
    assert all(f.det.ok and f.n_corners == N_CORNERS for f in frames)
    assert frames[0].det.obj[:, :2].max() <= 0.27 + 1e-9  # metres on the 9x6 x 30 mm board
    assert all(f.board is NEXTVIEW_BOARD for f in frames)
    # the same detector as the live webcam path
    live = detect_board(cv2.imread(str(paths[2]), cv2.IMREAD_GRAYSCALE))
    np.testing.assert_array_equal(frames[2].det.ids, live.ids)
    np.testing.assert_array_equal(frames[2].det.corners, live.corners)


def test_reextraction_drops_stale_frames(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video, n=12)
    out = tmp_path / "frames"
    assert len(extract_frames(video, out, every=1)) == 12
    (out / "notes.txt").write_text("kept")
    paths = extract_frames(video, out, every=5)
    assert sorted(p.name for p in out.glob("frame_*.png")) == [p.name for p in paths] == [
        "frame_000000.png",
        "frame_000005.png",
        "frame_000010.png",
    ]
    assert (out / "notes.txt").exists()
    assert [f.name for f in load_owner_pool(out)] == ["frame_000000", "frame_000005", "frame_000010"]


def test_extract_max_frames(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video, n=12)
    assert len(extract_frames(video, tmp_path / "f", every=2, max_frames=3)) == 3


def test_extract_rejects_bad_input(tmp_path):
    with pytest.raises(OSError):
        extract_frames(tmp_path / "missing.mp4", tmp_path / "o")
    with pytest.raises(ValueError):
        extract_frames(tmp_path / "missing.mp4", tmp_path / "o", every=0)
