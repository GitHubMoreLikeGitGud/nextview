"""OpenCV's ``samples/data/left*.jpg`` chessboard set (Apache-2.0), the fallback real pool.

Thirteen 640x480 views of a chessboard with 9x6 inner corners, fetched from the opencv GitHub
repository at a pinned tag and checked against pinned SHA-256 digests. Corners come from
``findChessboardCornersSB``; ``estimateChessboardSharpness`` is stored per frame as an
independent cross-check of the ESF blur gate. The square size is not documented, so object
points are in square units.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from nextview.data.frame import Frame, PlanarBoard, data_root, empty_detection
from nextview.types import Detection

__all__ = [
    "OPENCV_TAG",
    "PATTERN",
    "SAMPLE_SHA256",
    "samples_root",
    "fetch_samples",
    "chessboard_object_points",
    "chessboard_board",
    "detect_chessboard",
    "load_samples",
    "attribution_markdown",
]

OPENCV_TAG = "5.0.0"
BASE_URL = f"https://raw.githubusercontent.com/opencv/opencv/{OPENCV_TAG}/samples/data/"
PATTERN = (9, 6)  # inner corners (columns, rows)
# Tried in order; the first that finds the board wins (left05 needs NORMALIZE_IMAGE and is
# missed by EXHAUSTIVE | ACCURACY alone).
SB_FLAG_SETS = (
    cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY,
    cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
)
LICENSE = "Apache-2.0"

SAMPLE_SHA256 = {
    "left01.jpg": "9621899098adffa1440c8264606bb7be535ccfd9ad126ae8e0bd9b0c5a5b8676",
    "left02.jpg": "052e700bbcc865c112958ec2ea105cb1bce4deca50ac8d6efae13adca9df7a2c",
    "left03.jpg": "c85b9426c3d8fcf161e1e727356105091009c3fd0b1b1a65bc2da91fea2724d2",
    "left04.jpg": "b490599dccfbfe5eb56d33363415b02227a9548ab8c9b8e7b05d3580e97266d9",
    "left05.jpg": "40bab8934fa76f82ec614a7fd0e6ebb4d9f74391d5e8b0c6f2f0ed6080c6c4de",
    "left06.jpg": "1cb164db61b3bd52f58a7e9340bb27c5e0278895bc80b6f292fba5cf9bccc4a1",
    "left07.jpg": "d2211fe9f971646b6ff88ed93a4f0d3f443741eb2894095cba15caade7c8c6e4",
    "left08.jpg": "f7a0737894812a28243af5b409c5689f4fadfe7610958b8ca5fc79aec2f2cd1b",
    "left09.jpg": "753939bc62d093c001d9118e989bef2b887f1b2f9ac03603df915a8b85aa81d3",
    "left11.jpg": "e736ace3d515c94c547009c3c10d818c81506ed7059cb31123749b9cc3f94ee8",
    "left12.jpg": "04bdd78b3b507fe0551aa7062a3ffd94846a147ee2bea7fb38082c720bdbe231",
    "left13.jpg": "2637fb6dd5e557f4fadb06b353a453eb0b9223f4671f6f572f815302bc4a70b3",
    "left14.jpg": "d4c032971dd7b859f9d6aed5090c9888bd8039b82e0604bee5bfe05444fce834",
}


def samples_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else data_root() / "samples"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_samples(root: Path | str | None = None, *, timeout: float = 30.0) -> list[Path]:
    """Download the pinned ``left*.jpg`` files (skipping ones already present and intact)."""
    base = samples_root(root)
    base.mkdir(parents=True, exist_ok=True)
    out = []
    for name, digest in SAMPLE_SHA256.items():
        path = base / name
        if not (path.exists() and _sha256(path) == digest):
            req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "nextview-data/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read(1 << 22)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"{name}: SHA-256 mismatch against the pinned digest")
            tmp = path.with_name(name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        out.append(path)
    (base / "manifest.json").write_text(
        json.dumps({"source": BASE_URL, "tag": OPENCV_TAG, "license": LICENSE, "sha256": SAMPLE_SHA256}, indent=1),
        encoding="utf-8",
    )
    return out


def chessboard_object_points(pattern: tuple[int, int] = PATTERN, square: float = 1.0) -> np.ndarray:
    """(cols*rows, 3) float64 grid, row-major, z = 0."""
    cols, rows = pattern
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(cols * rows)], axis=1).astype(np.float64) * square


def chessboard_board(pattern: tuple[int, int] = PATTERN, square: float = 1.0) -> PlanarBoard:
    """Board geometry for ``chessboard_object_points``: (cols+1) x (rows+1) squares, origin one square outside."""
    cols, rows = pattern
    return PlanarBoard(kind="chessboard", squares=(cols + 1, rows + 1), square_len=square, units="square", origin=(-square, -square))


def detect_chessboard(gray: np.ndarray, pattern: tuple[int, int] = PATTERN) -> tuple[Detection, float | None]:
    """``findChessboardCornersSB`` corners as a ``Detection`` plus the mean edge sharpness (px)."""
    found, corners = False, None
    for flags in SB_FLAG_SETS:
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if found and corners is not None:
            break
    if not found or corners is None:
        return empty_detection(), None
    corners = corners.reshape(-1, 2).astype(np.float64)
    retval, _ = cv2.estimateChessboardSharpness(gray, pattern, corners.reshape(-1, 1, 2).astype(np.float32))
    n = len(corners)
    det = Detection(
        ids=np.arange(n, dtype=np.int32),
        corners=corners,
        obj=chessboard_object_points(pattern),
        n_markers=0,
        ok=True,
    )
    return det, float(retval[0])


def load_samples(root: Path | str | None = None, pattern: tuple[int, int] = PATTERN) -> list[Frame]:
    """Frames for every ``left*.jpg`` in the samples folder, sorted by name.

    ``det.obj`` is in square units with the first inner corner at the origin; ``frame.board`` is
    the whole chessboard (``pattern`` + 1 squares each way).
    """
    base = samples_root(root)
    paths = sorted(base.glob("left*.jpg"))
    board = chessboard_board(pattern)
    frames = []
    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        det, sharp = detect_chessboard(gray, pattern)
        frames.append(
            Frame(
                name=p.stem,
                path=p,
                source="samples/left",
                image_size=(gray.shape[1], gray.shape[0]),
                det=det,
                board=board,
                meta={"sharpness_px": sharp, "pattern": list(pattern), "license": LICENSE},
            )
        )
    return frames


def attribution_markdown() -> str:
    """ATTRIBUTION.md section for the sample images."""
    names = ", ".join(n.removesuffix(".jpg") for n in SAMPLE_SHA256)
    return "\n".join(
        [
            "## OpenCV sample images (fallback real pool)",
            "",
            f"- **Files:** `samples/data/{{{names}}}.jpg` from the OpenCV repository at tag `{OPENCV_TAG}`",
            f"- **Source:** <https://github.com/opencv/opencv/tree/{OPENCV_TAG}/samples/data>",
            "- **Copyright:** OpenCV contributors (see the repository's COPYRIGHT and LICENSE files)",
            "- **Licence:** Apache License 2.0, <https://www.apache.org/licenses/LICENSE-2.0>",
            "- **Used:** chessboard corners (`findChessboardCornersSB`) for the fallback pool and as a cross-check "
            "of the blur gate (`estimateChessboardSharpness`); fetched unmodified by `scripts/fetch_metric.py "
            "--samples` and checked against pinned SHA-256 digests; not redistributed in this repository.",
        ]
    ) + "\n"
