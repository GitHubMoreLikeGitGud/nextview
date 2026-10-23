import cv2
import numpy as np
import pytest

from nextview.data.board_probe import BoardSpec, detect_charuco, detector_signature, probe_board, small_marker_params

PX_PER_SQUARE = 45


def render_frames(spec: BoardSpec, n: int = 3, size=(800, 450), seed: int = 0) -> list[np.ndarray]:
    """Board image warped into frames by random mild perspective homographies."""
    rng = np.random.default_rng(seed)
    sx, sy = spec.squares
    bw, bh = sx * PX_PER_SQUARE, sy * PX_PER_SQUARE
    board_img = spec.make().generateImage((bw, bh), marginSize=0, borderBits=1)
    W, H = size
    frames = []
    for _ in range(n):
        src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
        scale = rng.uniform(0.9, 1.15)
        cx, cy = W / 2 + rng.uniform(-60, 60), H / 2 + rng.uniform(-20, 20)
        dst = np.float32(
            [[cx + (x - bw / 2) * scale, cy + (y - bh / 2) * scale] for x, y in src]
        ) + rng.uniform(-25, 25, (4, 2)).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src, dst)
        canvas = np.full((H, W), 128, np.uint8)
        warped = cv2.warpPerspective(board_img, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=128)
        mask = cv2.warpPerspective(np.full((bh, bw), 255, np.uint8), Hm, (W, H)) > 0
        canvas[mask] = warped[mask]
        canvas = cv2.GaussianBlur(canvas, (0, 0), 0.7)
        frames.append(canvas)
    return frames


@pytest.mark.parametrize(
    "spec",
    [
        BoardSpec("DICT_4X4_50", (7, 5), 1.0, 0.7, False),
        BoardSpec("DICT_6X6_250", (6, 4), 1.0, 0.78, True),
        BoardSpec("DICT_5X5_100", (8, 6), 1.0, 0.75, False),
    ],
    ids=["4x4_7x5", "6x6_6x4_legacy", "5x5_8x6"],
)
def test_probe_recovers_known_board(spec):
    frames = render_frames(spec)
    res = probe_board(frames, n_frames=3)
    got = res.spec
    # Dictionaries of one family share their first ids, so the probe can only name the family and
    # picks its smallest member that holds every id on the board.
    assert got.dictionary.rsplit("_", 1)[0] == spec.dictionary.rsplit("_", 1)[0]
    assert int(got.dictionary.rsplit("_", 1)[1]) >= spec.n_markers
    assert res.max_id == spec.n_markers - 1
    assert got.squares == spec.squares
    assert got.legacy == spec.legacy
    assert abs(got.marker_len - spec.marker_len) < 0.03
    assert res.layout_scores[0]["median_err_px"] < 1.0
    # the identified board detects every inner corner on the clean renders
    board = got.make()
    for f in frames:
        assert len(detect_charuco(f, board).ids) == spec.n_corners


def test_detect_charuco_returns_sorted_ids_and_board_units():
    spec = BoardSpec("DICT_4X4_50", (5, 4), 1.0, 0.7, False)
    frame = render_frames(spec, n=1, seed=3)[0]
    det = detect_charuco(frame, spec.make())
    assert det.ok and np.all(np.diff(det.ids) > 0)
    np.testing.assert_allclose(det.obj, spec.object_points()[det.ids])
    assert det.obj[:, :2].max() <= max(spec.squares) and det.corners.shape == (len(det.ids), 2)


def test_detect_charuco_on_blank_image():
    det = detect_charuco(np.full((480, 640), 128, np.uint8), BoardSpec("DICT_4X4_50", (5, 4)).make())
    assert not det.ok and len(det.ids) == 0 and det.obj.shape == (0, 3)


def test_spec_roundtrip_and_validation():
    spec = BoardSpec("DICT_6X6_50", (5, 4), 1.0, 0.78, True)
    assert BoardSpec.from_dict(spec.to_dict()) == spec
    assert spec.n_corners == 12 and spec.n_markers == 10
    with pytest.raises(ValueError):
        BoardSpec.from_dict({"dictionary": "DICT_NOPE", "squares": [5, 4]})
    with pytest.raises(ValueError):
        BoardSpec.from_dict({"dictionary": "DICT_4X4_50", "squares": [1, 400]})


def test_probe_without_markers_raises():
    with pytest.raises(ValueError, match="no ArUco markers"):
        probe_board([np.full((240, 320), 200, np.uint8)] * 2)


def test_collinear_corners_are_not_ok():
    spec = BoardSpec("DICT_6X6_50", (5, 4), 1.0, 0.79, True)
    P = 80
    tex = spec.make().generateImage((5 * P, 4 * P), marginSize=0, borderBits=1)
    img = np.full((600, 800), 120, np.uint8)
    img[100 : 100 + 4 * P, 150 : 150 + 5 * P] = tex
    full = detect_charuco(img, spec.make(), spec)
    assert full.ok and len(full.ids) == spec.n_corners
    img[100 + int(2.5 * P) :, :] = 120  # only the first row of inner corners keeps both neighbouring markers
    row = detect_charuco(img, spec.make(), spec)
    assert row.ids.tolist() == [0, 1, 2, 3] and not row.ok


def test_planar_board_span_rule():
    b = BoardSpec("DICT_6X6_50", (5, 4), 1.0, 0.79, True).planar()
    assert b.n_cols == 4 and b.n_rows == 3 and b.n_corners == 12
    assert not b.usable([0, 1, 2, 3])  # one row
    assert not b.usable([0, 4, 8])  # three corners
    assert not b.usable([1, 5, 9, 1])  # one column
    assert b.usable([0, 1, 4, 5])
    np.testing.assert_allclose(b.outline()[:, :2], [[0, 0], [5, 0], [5, 4], [0, 4]])


def test_detector_signature_tracks_settings():
    p = small_marker_params()
    a = detector_signature(p, 2)
    assert a == detector_signature(small_marker_params(), 2)
    p.adaptiveThreshWinSizeMax = 41
    assert detector_signature(p, 2) != a
    assert detector_signature(small_marker_params(), 1) != a
    assert a["charuco"]["tryRefineMarkers"] is True and "minMarkerPerimeterRate" in a["detector"]
