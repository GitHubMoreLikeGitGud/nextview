"""Owner capture: a hand-held phone video of the NextView board shown on a monitor.

Every 5th frame is written as PNG under ``data_ext/owner/<video stem>/`` and detected with
``tools.detect.detect_board``, the same detector as the live webcam path. The pool protocol then
treats these frames like a METRIC camera (held-out RMS only, since there is no reference).
Released under CC BY 4.0; the video must contain no people.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2

from nextview.board import DICT_NAME, MARKER_M, SQUARE_M, SQUARES
from nextview.data.board_probe import BoardSpec, detect_charuco
from nextview.data.frame import NEXTVIEW_BOARD, Frame, data_root
from nextview.tools.detect import detect_board

__all__ = ["NEXTVIEW_SPEC", "owner_root", "extract_frames", "load_owner_pool"]

NEXTVIEW_SPEC = BoardSpec(dictionary=DICT_NAME, squares=SQUARES, square_len=SQUARE_M, marker_len=MARKER_M, legacy=False)
_FRAME_RE = re.compile(r"^frame_\d{6}\.png$")


def owner_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else data_root() / "owner"


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")
    return stem or "capture"


def extract_frames(
    video: Path | str, out_dir: Path | str | None = None, *, every: int = 5, max_frames: int | None = None
) -> list[Path]:
    """Write every ``every``-th frame of ``video`` as ``frame_NNNNNN.png`` (NNNNNN = source frame index).

    Frames left in ``out_dir`` by an earlier extraction are deleted first, so the folder only
    holds this extraction.
    """
    video = Path(video)
    if every < 1:
        raise ValueError("every must be >= 1")
    out = Path(out_dir) if out_dir is not None else owner_root() / _safe_stem(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"cannot open video {video}")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        if old.is_file() and (_FRAME_RE.match(old.name) or old.name == "capture.json"):
            old.unlink()
    written: list[Path] = []
    meta = {
        "video": video.name,
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "every": every,
    }
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % every == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                p = out / f"frame_{idx:06d}.png"
                if not cv2.imwrite(str(p), frame):
                    raise OSError(f"cannot write {p}")
                written.append(p)
                if max_frames is not None and len(written) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    meta["frames_read"] = idx
    meta["frames_written"] = len(written)
    meta["frames"] = [p.name for p in written]
    (out / "capture.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return written


def load_owner_pool(frames_dir: Path | str, spec: BoardSpec | None = None, *, units: str = "m") -> list[Frame]:
    """Frames extracted by ``extract_frames`` with their detections, sorted by frame index.

    The NextView board (default) is detected with ``tools.detect.detect_board``; another
    ChArUco ``spec`` uses ``board_probe.detect_charuco``, with its lengths in ``units``.
    When ``capture.json`` lists the extracted frames, only those are loaded.
    """
    d = Path(frames_dir)
    paths = sorted(p for p in d.glob("frame_*.png") if _FRAME_RE.match(p.name))
    meta_path = d / "capture.json"
    if meta_path.exists():
        listed = json.loads(meta_path.read_text(encoding="utf-8")).get("frames")
        if isinstance(listed, list):
            keep = {str(n) for n in listed}
            paths = [p for p in paths if p.name in keep]
    if spec is None or spec == NEXTVIEW_SPEC:
        board, cv_board = NEXTVIEW_BOARD, None
    else:
        board, cv_board = spec.planar(units=units, square_m=spec.square_len if units == "m" else None), spec.make()
    frames = []
    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        det = detect_board(gray) if cv_board is None else detect_charuco(gray, cv_board, spec)
        frames.append(
            Frame(
                name=p.stem,
                path=p,
                source=f"owner/{d.name}",
                image_size=(gray.shape[1], gray.shape[0]),
                det=det,
                board=board,
                meta={"license": "CC-BY-4.0"},
            )
        )
    return frames
