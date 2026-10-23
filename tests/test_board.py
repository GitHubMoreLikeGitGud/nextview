import struct

import cv2
import numpy as np
import pytest

from nextview import board


def _decode(png: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    return img


def _detect(img: np.ndarray):
    det = cv2.aruco.CharucoDetector(board.make_board())
    corners, ids, _, _ = det.detectBoard(img)
    return corners, ids


def test_board_dims():
    b = board.make_board()
    assert tuple(b.getChessboardSize()) == (9, 6)
    assert b.getSquareLength() == pytest.approx(0.030)
    assert b.getMarkerLength() == pytest.approx(0.022)
    ref = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    assert np.array_equal(b.getDictionary().bytesList, ref.bytesList)
    obj = board.object_points()
    assert obj.shape == (40, 3) and board.N_CORNERS == 40
    # row-major; id 7 is the top-right inner corner
    np.testing.assert_allclose(obj[0], [0.03, 0.03, 0.0], atol=1e-9)
    np.testing.assert_allclose(obj[7], [0.24, 0.03, 0.0], atol=1e-9)
    assert board.BOARD_SIZE_M == pytest.approx((0.27, 0.18))


def test_screen_png_detects_all_corners():
    img = _decode(board.board_png("screen"))
    corners, ids = _detect(img)
    assert ids is not None and len(ids) == 40
    assert sorted(ids.ravel().tolist()) == list(range(40))
    # corners sit on the expected 160 px lattice offset by the 80 px margin
    order = np.argsort(ids.ravel())
    obj = board.object_points()[:, :2]
    expected = obj / board.SQUARE_M * board.SCREEN_SQUARE_PX + board.SCREEN_MARGIN_PX
    err = np.abs(corners.reshape(-1, 2)[order] - expected)
    assert err.max() < 1.0


def test_print_png_scale_and_detection():
    png = board.board_png("print")
    img = _decode(png)
    assert img.shape == (2480, 3508)  # A4 landscape at 300 dpi
    corners, ids = _detect(img)
    assert ids is not None and len(ids) == 40
    order = np.argsort(ids.ravel())
    c = corners.reshape(-1, 2)[order].reshape(5, 8, 2)
    px_per_square = 30.0 / 25.4 * 300
    assert np.median(np.diff(c[:, :, 0], axis=1)) == pytest.approx(px_per_square, abs=1.0)
    assert np.median(np.diff(c[:, :, 1], axis=0)) == pytest.approx(px_per_square, abs=1.0)


def test_print_png_has_80mm_check_bar():
    img = _decode(board.board_png("print"))
    _, bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    bars = [
        s for s in stats[1:]
        if s[cv2.CC_STAT_WIDTH] > 500
        and 20 <= s[cv2.CC_STAT_HEIGHT] <= 60
        and s[cv2.CC_STAT_AREA] >= 0.95 * s[cv2.CC_STAT_WIDTH] * s[cv2.CC_STAT_HEIGHT]
    ]
    assert len(bars) == 1
    assert abs(bars[0][cv2.CC_STAT_WIDTH] - 945) <= 2  # 80 mm at 300 dpi


def test_print_png_declares_300_dpi():
    png = board.board_png("print")
    i = png.index(b"pHYs")
    ppx, ppy, unit = struct.unpack(">IIB", png[i + 4 : i + 13])
    assert unit == 1 and ppx == ppy == 11811


def test_board_png_rejects_unknown_kind():
    with pytest.raises(ValueError):
        board.board_png("poster")
