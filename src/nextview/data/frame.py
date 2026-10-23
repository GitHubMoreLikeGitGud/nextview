"""A frame from an external image pool, with its board detection and the board it belongs to.

Pool frames do not all show the NextView board: METRIC uses a 5x4 legacy ChArUco board and the
OpenCV samples a 10x7-square chessboard, both in square units. Every ``Frame`` therefore carries
its ``PlanarBoard``. The tools in ``nextview.tools`` (``quality.assess_frame``,
``esf.measure_edges``, ``detect.corner_span``, ``pose.estimate_pose``) assume the NextView board
in metres and give wrong board areas, edge positions and distances on any other board; use
``nextview.data.pool_tools`` for pool frames unless ``frame.board.is_nextview``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nextview.board import DICT_NAME, MARKER_M, SQUARE_M, SQUARES
from nextview.types import Detection

__all__ = ["Frame", "PlanarBoard", "NEXTVIEW_BOARD", "empty_detection", "data_root"]


def data_root() -> Path:
    """``$NEXTVIEW_DATA_DIR`` when set, otherwise ``<repo>/data_ext`` (gitignored)."""
    env = os.environ.get("NEXTVIEW_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data_ext"


def empty_detection(n_markers: int = 0) -> Detection:
    return Detection(
        ids=np.zeros((0,), np.int32),
        corners=np.zeros((0, 2), np.float64),
        obj=np.zeros((0, 3), np.float64),
        n_markers=int(n_markers),
        ok=False,
    )


@dataclass(frozen=True)
class PlanarBoard:
    """Geometry of the planar target whose inner corners a ``Frame`` holds.

    Corner ids are row-major over ``n_cols`` inner corners per row. Object points are in
    ``units``: ``"m"`` for the NextView board, ``"square"`` (square side = ``square_len``) for
    boards whose physical size is not documented. ``origin`` is the object-plane (x, y) of the
    outer top-left board corner. ``square_m`` is the physical square side when it is known.
    """

    kind: str  # "charuco" | "chessboard"
    squares: tuple[int, int]  # (squaresX, squaresY) of the whole pattern
    square_len: float  # square side in object units
    units: str  # "m" | "square"
    origin: tuple[float, float] = (0.0, 0.0)
    marker_len: float | None = None  # ChArUco marker side in object units
    dictionary: str | None = None
    legacy: bool = False
    square_m: float | None = None

    @property
    def n_cols(self) -> int:
        return self.squares[0] - 1

    @property
    def n_rows(self) -> int:
        return self.squares[1] - 1

    @property
    def n_corners(self) -> int:
        return self.n_cols * self.n_rows

    @property
    def metres_per_unit(self) -> float | None:
        if self.units == "m":
            return 1.0
        return None if self.square_m is None else self.square_m / self.square_len

    @property
    def is_nextview(self) -> bool:
        return self == NEXTVIEW_BOARD

    def outline(self) -> np.ndarray:
        """(4, 3) outer board outline in object units: TL, TR, BR, BL."""
        x0, y0 = self.origin
        w, h = self.squares[0] * self.square_len, self.squares[1] * self.square_len
        return np.array([[x0, y0, 0.0], [x0 + w, y0, 0.0], [x0 + w, y0 + h, 0.0], [x0, y0 + h, 0.0]])

    def centre(self) -> np.ndarray:
        return self.outline().mean(axis=0)

    def corner_span(self, ids: np.ndarray) -> tuple[int, int]:
        """Number of distinct inner-corner rows and columns covered by ``ids``."""
        ids = np.asarray(ids).ravel()
        if ids.size == 0:
            return 0, 0
        return int(np.unique(ids // self.n_cols).size), int(np.unique(ids % self.n_cols).size)

    def usable(self, ids: np.ndarray) -> bool:
        """At least 4 corners spanning 2 rows and 2 columns (a non-degenerate homography / PnP input)."""
        rows, cols = self.corner_span(ids)
        return bool(len(np.asarray(ids).ravel()) >= 4 and rows >= 2 and cols >= 2)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "squares": list(self.squares),
            "square_len": self.square_len,
            "units": self.units,
            "origin": list(self.origin),
            "marker_len": self.marker_len,
            "dictionary": self.dictionary,
            "legacy": self.legacy,
            "square_m": self.square_m,
        }


NEXTVIEW_BOARD = PlanarBoard(
    kind="charuco",
    squares=tuple(SQUARES),
    square_len=SQUARE_M,
    units="m",
    marker_len=MARKER_M,
    dictionary=DICT_NAME,
    legacy=False,
    square_m=SQUARE_M,
)


@dataclass(frozen=True, eq=False)
class Frame:
    """One pool image, its detected board corners and the board they belong to.

    ``det.ids`` index ``board``'s inner-corner list (row-major); ``det.obj`` is in ``board.units``.
    """

    name: str
    path: Path
    source: str  # e.g. "metric/d455/camera1", "samples/left", "owner/<video stem>"
    image_size: tuple[int, int]  # (width, height)
    det: Detection
    board: PlanarBoard
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_corners(self) -> int:
        return int(len(self.det.ids))

    def image(self, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
        img = cv2.imread(str(self.path), flags)
        if img is None:
            raise FileNotFoundError(f"cannot read image {self.path}")
        return img
