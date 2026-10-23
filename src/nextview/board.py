"""The ChArUco calibration board and its printable / on-screen images."""

from __future__ import annotations

import struct
import zlib
from functools import lru_cache

import cv2
import numpy as np

__all__ = [
    "SQUARES",
    "SQUARE_M",
    "MARKER_M",
    "DICT",
    "DICT_NAME",
    "N_CORNERS",
    "BOARD_SIZE_M",
    "BOARD",
    "make_board",
    "object_points",
    "board_png",
    "PRINT_DPI",
    "CHECK_BAR_MM",
]

SQUARES = (9, 6)  # (squaresX, squaresY)
SQUARE_M = 0.030
MARKER_M = 0.022
DICT = cv2.aruco.DICT_5X5_100
DICT_NAME = "DICT_5X5_100"
N_CORNERS = (SQUARES[0] - 1) * (SQUARES[1] - 1)  # 40 inner corners, row-major
BOARD_SIZE_M = (SQUARES[0] * SQUARE_M, SQUARES[1] * SQUARE_M)

PRINT_DPI = 300
A4_LANDSCAPE_MM = (297.0, 210.0)
CHECK_BAR_MM = 80.0

SCREEN_SQUARE_PX = 160
SCREEN_MARGIN_PX = 80


def make_board() -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.CharucoBoard(SQUARES, SQUARE_M, MARKER_M, dictionary)


BOARD = make_board()


def object_points() -> np.ndarray:
    """(40, 3) float64 board-frame corner coordinates in metres; index = ChArUco corner id."""
    return BOARD.getChessboardCorners().astype(np.float64).reshape(-1, 3)


def _mm_to_px(mm: float, dpi: int = PRINT_DPI) -> int:
    return int(round(mm * dpi / 25.4))


def _encode_png(img: np.ndarray, dpi: int | None = None) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    data = buf.tobytes()
    if dpi is None:
        return data
    # Insert a pHYs chunk right after IHDR so printers honour the physical size.
    ppm = int(round(dpi / 0.0254))
    payload = struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", len(payload)) + b"pHYs" + payload
    chunk += struct.pack(">I", zlib.crc32(b"pHYs" + payload) & 0xFFFFFFFF)
    ihdr_end = 8 + 4 + 4 + 13 + 4
    return data[:ihdr_end] + chunk + data[ihdr_end:]


def _put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, thickness: int) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thickness, cv2.LINE_AA)


def _render_screen() -> np.ndarray:
    sx, sy = SQUARES
    m = SCREEN_MARGIN_PX
    size = (sx * SCREEN_SQUARE_PX + 2 * m, sy * SCREEN_SQUARE_PX + 2 * m)
    return BOARD.generateImage(size, marginSize=m, borderBits=1)


def _render_print() -> np.ndarray:
    page_w, page_h = (_mm_to_px(v) for v in A4_LANDSCAPE_MM)
    page = np.full((page_h, page_w), 255, np.uint8)

    bw, bh = _mm_to_px(BOARD_SIZE_M[0] * 1000), _mm_to_px(BOARD_SIZE_M[1] * 1000)
    board_img = BOARD.generateImage((bw, bh), marginSize=0, borderBits=1)
    x0 = (page_w - bw) // 2
    y0 = _mm_to_px(6.0)
    page[y0 : y0 + bh, x0 : x0 + bw] = board_img

    # Bottom strip: 80 mm check bar (left), caption (centre), 30 mm ruler box (right).
    strip_top = y0 + bh + _mm_to_px(3.0)
    bar_len = _mm_to_px(CHECK_BAR_MM)
    bar_h = _mm_to_px(3.0)
    cv2.rectangle(page, (x0, strip_top), (x0 + bar_len - 1, strip_top + bar_h - 1), 0, thickness=-1)
    _put_text(page, "this bar must measure 80 mm", (x0, strip_top + bar_h + _mm_to_px(4.5)), 1.5, 3)
    _put_text(page, "print at 100 % (actual size)", (x0, strip_top + bar_h + _mm_to_px(9.0)), 1.5, 3)

    sq = _mm_to_px(SQUARE_M * 1000)
    box_x1 = x0 + bw - 1
    box_x0 = box_x1 - sq + 1
    box_y0 = strip_top
    box_y1 = strip_top + _mm_to_px(8.0)
    cv2.rectangle(page, (box_x0, box_y0), (box_x1, box_y1), 0, thickness=3)
    for mm in range(0, 31):
        x = box_x0 + int(round(mm * PRINT_DPI / 25.4))
        x = min(x, box_x1)
        tick = _mm_to_px(3.5 if mm % 10 == 0 else 2.5 if mm % 5 == 0 else 1.5)
        cv2.line(page, (x, box_y1), (x, box_y1 - tick), 0, 2)
    _put_text(page, "square = 30 mm", (box_x0 + _mm_to_px(2.0), box_y0 + _mm_to_px(3.2)), 1.0, 2)

    caption = f"NextView ChArUco {SQUARES[0]}x{SQUARES[1]}  {DICT_NAME}  square 30 mm  marker 22 mm"
    (tw, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cx = (x0 + bar_len + box_x0) // 2 - tw // 2
    _put_text(page, caption, (max(cx, x0 + bar_len + _mm_to_px(3.0)), strip_top + _mm_to_px(12.0)), 1.1, 2)
    return page


@lru_cache(maxsize=2)
def board_png(kind: str = "screen") -> bytes:
    """PNG bytes of the board.

    ``"screen"``: the board with a white margin, for display on a monitor.
    ``"print"``: A4 landscape at 300 dpi (true 30 mm squares) with an 80 mm print-check bar
    and a ``square = 30 mm`` ruler tick box.
    """
    if kind == "screen":
        return _encode_png(_render_screen())
    if kind == "print":
        return _encode_png(_render_print(), dpi=PRINT_DPI)
    raise ValueError(f"kind must be 'screen' or 'print', got {kind!r}")
