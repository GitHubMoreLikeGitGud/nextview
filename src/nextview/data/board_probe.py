"""Identify an unknown ChArUco board from images: dictionary, layout, marker size and legacy pattern.

1. Run ``ArucoDetector`` with every predefined dictionary and keep the family that finds the
   most markers; within a family pick the smallest dictionary that holds the largest id seen.
2. For every ``(squaresX, squaresY)`` whose marker count fits the largest id, and both legacy
   patterns, map each marker id to its square centre and fit a homography to the detected
   marker centres. The true layout is the one whose centres fit (median error in px).
3. Estimate the marker/square length ratio by mapping detected marker corners back onto the
   board plane through that homography.
4. Run ``CharucoDetector`` with ``setLegacyPattern(False/True)`` and keep the setting that
   yields more ChArUco corners.

Frames are scanned in a spread-out order until enough of them show markers, so sequences where
the board is often turned away or tiny still probe. For small markers (about 15 px), pass
``upscale=2`` and ``small_marker_params()``; corner coordinates are mapped back to the original
pixel grid.

Square length is 1.0 (unitless) unless the dataset documents it; intrinsics are scale-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from nextview.types import Detection

from .frame import PlanarBoard

__all__ = [
    "BoardSpec",
    "ProbeResult",
    "DICTIONARY_NAMES",
    "probe_board",
    "detect_charuco",
    "charuco_parameters",
    "detector_signature",
    "small_marker_params",
    "score_dictionaries",
    "fit_layouts",
]

# Distinct predefined dictionaries (the lower-case AprilTag / MIP aliases are the same objects).
DICTIONARY_NAMES: tuple[str, ...] = tuple(
    n
    for n in sorted(d for d in dir(cv2.aruco) if d.startswith("DICT_"))
    if not (n.startswith("DICT_APRILTAG_") and n != n.upper())
    and not (n.startswith("DICT_ARUCO_MIP_") and n != n.upper())
)

MIN_LAYOUT_MARKERS = 6


def _family(name: str) -> str:
    # "DICT_6X6_250" -> "DICT_6X6"; AprilTag / original / MIP dictionaries are their own family.
    parts = name.split("_")
    if len(parts) == 3 and parts[2].isdigit():
        return "_".join(parts[:2])
    return name


def _dict_size(name: str) -> int:
    return int(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)).bytesList.shape[0])


@dataclass(frozen=True)
class BoardSpec:
    dictionary: str
    squares: tuple[int, int]  # (squaresX, squaresY)
    square_len: float = 1.0
    marker_len: float = 0.75
    legacy: bool = False

    def make(self) -> cv2.aruco.CharucoBoard:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        b = cv2.aruco.CharucoBoard(tuple(self.squares), float(self.square_len), float(self.marker_len), d)
        b.setLegacyPattern(bool(self.legacy))
        return b

    @property
    def n_corners(self) -> int:
        return (self.squares[0] - 1) * (self.squares[1] - 1)

    @property
    def n_markers(self) -> int:
        return (self.squares[0] * self.squares[1]) // 2

    def object_points(self) -> np.ndarray:
        """(n_corners, 3) float64 board-frame corner coordinates; index = ChArUco corner id."""
        return self.make().getChessboardCorners().astype(np.float64).reshape(-1, 3)

    def planar(self, units: str = "square", square_m: float | None = None) -> PlanarBoard:
        """The board geometry carried on pool ``Frame``s detected with this spec."""
        return PlanarBoard(
            kind="charuco",
            squares=tuple(self.squares),
            square_len=float(self.square_len),
            units=units,
            marker_len=float(self.marker_len),
            dictionary=self.dictionary,
            legacy=bool(self.legacy),
            square_m=square_m,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["squares"] = list(self.squares)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BoardSpec":
        name = str(d["dictionary"])
        if name not in DICTIONARY_NAMES:
            raise ValueError(f"unknown dictionary {name!r}")
        sx, sy = (int(v) for v in d["squares"])
        if not (2 <= sx <= 64 and 2 <= sy <= 64):
            raise ValueError(f"implausible board size {(sx, sy)}")
        return cls(
            dictionary=name,
            squares=(sx, sy),
            square_len=float(d.get("square_len", 1.0)),
            marker_len=float(d.get("marker_len", 0.75)),
            legacy=bool(d.get("legacy", False)),
        )


@dataclass
class ProbeResult:
    spec: BoardSpec
    frames: list[str]
    dictionary_scores: dict[str, int]  # markers found per dictionary over the probe frames
    max_id: int
    layout_scores: list[dict]  # best few layouts: squares, legacy, median centre error px, frames fitted
    marker_ratio: float  # estimated marker side / square side
    corners_by_legacy: dict[str, int]  # ChArUco corners found with legacy False / True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "frames": self.frames,
            "dictionary_scores": self.dictionary_scores,
            "max_id": self.max_id,
            "layout_scores": self.layout_scores,
            "marker_ratio": self.marker_ratio,
            "corners_by_legacy": self.corners_by_legacy,
            "notes": self.notes,
        }


def _gray(img_or_path) -> tuple[np.ndarray, str]:
    if isinstance(img_or_path, (str, Path)):
        img = cv2.imread(str(img_or_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"cannot read {img_or_path}")
        p = Path(img_or_path)
        return img, (f"{p.parent.parent.name}/{p.name}" if p.parent.name == "image" else p.name)
    img = np.asarray(img_or_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, "<array>"


def small_marker_params() -> cv2.aruco.DetectorParameters:
    """Detector settings for markers of roughly 10-20 px (distant boards)."""
    p = cv2.aruco.DetectorParameters()
    p.minMarkerPerimeterRate = 0.01
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 33
    p.adaptiveThreshWinSizeStep = 4
    p.perspectiveRemovePixelPerCell = 8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.15
    p.polygonalApproxAccuracyRate = 0.05
    return p


def charuco_parameters() -> cv2.aruco.CharucoParameters:
    """ChArUco settings used by ``detect_charuco``."""
    cparams = cv2.aruco.CharucoParameters()
    cparams.tryRefineMarkers = True
    return cparams


def _scalar_attrs(obj) -> dict:
    out = {}
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        val = getattr(obj, name)
        if isinstance(val, (bool, int, float, str)) or val is None:
            out[name] = val
    return out


def detector_signature(params: cv2.aruco.DetectorParameters | None = None, upscale: int = 1) -> dict:
    """Every setting that changes ``detect_charuco``'s output, for cache keys."""
    return {
        "detector": _scalar_attrs(params if params is not None else cv2.aruco.DetectorParameters()),
        "charuco": _scalar_attrs(charuco_parameters()),
        "refine": _scalar_attrs(cv2.aruco.RefineParameters()),
        "upscale": int(upscale),
        "opencv": cv2.__version__,
    }


def _scaled(gray: np.ndarray, upscale: int) -> np.ndarray:
    if upscale == 1:
        return gray
    return cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)


def _unscale(pts: np.ndarray, upscale: int) -> np.ndarray:
    # pixel centres: x_up = (x + 0.5) * s - 0.5
    return pts if upscale == 1 else (pts + 0.5) / upscale - 0.5


def _aruco(gray: np.ndarray, name: str, upscale: int = 1, params=None) -> tuple[list[np.ndarray], np.ndarray]:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    det = cv2.aruco.ArucoDetector(d, params if params is not None else cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(_scaled(gray, upscale))
    if ids is None:
        return [], np.zeros(0, np.int32)
    corners = [_unscale(np.asarray(c, np.float64).reshape(4, 2), upscale) for c in corners]
    return corners, ids.reshape(-1).astype(np.int32)


def score_dictionaries(
    grays: Sequence[np.ndarray], names: Iterable[str] = DICTIONARY_NAMES, *, upscale: int = 1, params=None
) -> dict[str, int]:
    """Number of markers each dictionary detects over the frames."""
    return {n: sum(len(_aruco(g, n, upscale, params)[1]) for g in grays) for n in list(names)}


def _pick_dictionary(scores: dict[str, int], max_id_by_dict: dict[str, int]) -> tuple[str, int]:
    best_family = max(
        {_family(n) for n in scores}, key=lambda f: max(v for n, v in scores.items() if _family(n) == f)
    )
    members = sorted((n for n in scores if _family(n) == best_family), key=_dict_size)
    # Members of a family share their first ids; take the smallest one that holds every id seen.
    max_id = max(max_id_by_dict.get(n, -1) for n in members)
    for n in members:
        if _dict_size(n) > max_id:
            return n, max_id
    return members[-1], max_id


def _marker_centres_board(sx: int, sy: int, legacy: bool, n: int) -> np.ndarray:
    """(n, 2) board-plane centres (square units) of marker ids 0..n-1 for a layout."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    b = cv2.aruco.CharucoBoard((sx, sy), 1.0, 0.5, d)
    b.setLegacyPattern(legacy)
    objs = b.getObjPoints()
    return np.array([np.asarray(o, np.float64).reshape(4, 3)[:, :2].mean(0) for o in objs[:n]])


def fit_layouts(
    dets: Sequence[tuple[list[np.ndarray], np.ndarray]], max_id: int, *, max_side: int = 24, slack: int = 40
) -> list[dict]:
    """Score every layout that can hold ``max_id`` by how well marker centres fit one homography."""
    obs = []
    for corners, ids in dets:
        if len(ids) >= MIN_LAYOUT_MARKERS:
            centres = np.array([np.asarray(c).reshape(4, 2).mean(0) for c in corners], np.float64)
            obs.append((ids, centres))
    if not obs:
        return []
    out = []
    for sx in range(3, max_side + 1):
        for sy in range(3, max_side + 1):
            n = (sx * sy) // 2
            if n <= max_id or n > max_id + 1 + slack:
                continue
            for legacy in (False, True):
                if legacy and sy % 2 == 1:
                    continue  # the legacy flag only changes boards with an even number of rows
                centres_b = _marker_centres_board(sx, sy, legacy, n)
                errs, fitted = [], 0
                for ids, centres in obs:
                    src = centres_b[ids]
                    # least squares over every marker: a robust fit would ignore exactly the
                    # markers that tell layouts apart when only 6-8 are visible
                    H, _ = cv2.findHomography(src, centres, 0)
                    if H is None:
                        errs.append(np.inf)
                        continue
                    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
                    errs.append(float(np.sqrt(np.mean(np.sum((proj - centres) ** 2, axis=1)))))
                    fitted += 1
                out.append(
                    {"squares": [sx, sy], "legacy": legacy, "median_err_px": float(np.median(errs)), "frames": fitted}
                )
    out.sort(key=lambda r: (r["median_err_px"], r["squares"][0] * r["squares"][1]))
    if out:
        # Larger boards whose extra markers were never seen fit exactly as well; prefer the
        # smallest board among the layouts that fit as well as the best one.
        tol = 1.05 * out[0]["median_err_px"] + 1e-6
        tied = [r for r in out if r["median_err_px"] <= tol]
        pick = min(tied, key=lambda r: (r["squares"][0] * r["squares"][1], r["median_err_px"]))
        out.remove(pick)
        out.insert(0, pick)
    return out


def _marker_ratio(dets, sx: int, sy: int, legacy: bool) -> float:
    n = (sx * sy) // 2
    centres_b = _marker_centres_board(sx, sy, legacy, n)
    sides = []
    for corners, ids in dets:
        if len(ids) < MIN_LAYOUT_MARKERS:
            continue
        centres = np.array([np.asarray(c).reshape(4, 2).mean(0) for c in corners], np.float64)
        H, _ = cv2.findHomography(centres_b[ids], centres, cv2.LMEDS)
        if H is None:
            continue
        Hi = np.linalg.inv(H)
        for c in corners:
            q = cv2.perspectiveTransform(np.asarray(c, np.float64).reshape(-1, 1, 2), Hi).reshape(4, 2)
            sides.append(np.mean(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)))
    return float(np.median(sides)) if sides else float("nan")


def detect_charuco(
    gray: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    spec: BoardSpec | None = None,
    *,
    upscale: int = 1,
    params: cv2.aruco.DetectorParameters | None = None,
) -> Detection:
    """ChArUco corners of ``board`` in a grayscale image, as a ``Detection`` in board units.

    With ``upscale > 1`` the image is enlarged (bicubic) before detection and the corners are
    mapped back to the original pixel grid. ``ok`` means the corners can feed a homography or
    PnP: at least 4 of them, spanning at least 2 rows and 2 columns (the rule of
    ``tools.detect.detect_board``).
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.CharucoDetector(
        board,
        charuco_parameters(),
        params if params is not None else cv2.aruco.DetectorParameters(),
        cv2.aruco.RefineParameters(),
    )
    ch_corners, ch_ids, _, m_ids = det.detectBoard(_scaled(gray, upscale))
    n_markers = 0 if m_ids is None else int(len(m_ids))
    if ch_ids is None or len(ch_ids) == 0:
        return Detection(
            ids=np.zeros(0, np.int32), corners=np.zeros((0, 2)), obj=np.zeros((0, 3)), n_markers=n_markers, ok=False
        )
    ids = ch_ids.reshape(-1).astype(np.int32)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    corners = _unscale(ch_corners.reshape(-1, 2).astype(np.float64)[order], upscale)
    obj = board.getChessboardCorners().astype(np.float64).reshape(-1, 3)[ids]
    sx, sy = (int(v) for v in board.getChessboardSize())
    ok = PlanarBoard("charuco", (sx, sy), 1.0, "square").usable(ids)
    return Detection(ids=ids, corners=corners, obj=obj, n_markers=n_markers, ok=ok)


def _spread_order(n: int) -> list[int]:
    """Indices 0..n-1 ordered so that every prefix is spread over the whole range."""
    order: list[int] = []
    seen: set[int] = set()
    step = 1
    while step < n:
        step *= 2
    while step >= 1:
        for i in range(0, n, step):
            if i not in seen:
                seen.add(i)
                order.append(i)
        step //= 2
    return order


def probe_board(
    images: Sequence,
    n_frames: int = 10,
    *,
    layout_frames: int = 10,
    max_scan: int = 400,
    upscale: int = 1,
    params: cv2.aruco.DetectorParameters | None = None,
    dictionaries: Iterable[str] = DICTIONARY_NAMES,
) -> ProbeResult:
    """Identify the ChArUco board seen in ``images`` (paths or arrays).

    Dictionaries are scored on the first ``n_frames`` frames (in spread order) where any
    dictionary finds a marker; the layout is then fitted on up to ``layout_frames`` frames with
    at least 6 markers of the chosen dictionary. At most ``max_scan`` frames are read.
    """
    if len(images) == 0:
        raise ValueError("no images to probe")
    dictionaries = list(dictionaries)
    order = _spread_order(len(images))[:max_scan]
    notes: list[str] = []
    cache: dict[int, tuple[np.ndarray, str]] = {}

    def load(i: int) -> tuple[np.ndarray, str]:
        if i not in cache:
            cache[i] = _gray(images[i])
        return cache[i]

    # 1. dictionaries, on frames where any dictionary finds something
    scores = {n: 0 for n in dictionaries}
    max_ids: dict[str, int] = {}
    hit_frames: list[int] = []
    scanned = 0
    for i in order:
        if len(hit_frames) >= n_frames:
            break
        g, _ = load(i)
        scanned += 1
        found = False
        for n in dictionaries:
            _, ids = _aruco(g, n, upscale, params)
            if len(ids):
                found = True
                scores[n] += len(ids)
                max_ids[n] = max(max_ids.get(n, -1), int(ids.max()))
        if found:
            hit_frames.append(i)
        else:
            cache.pop(i, None)
    if not hit_frames:
        raise ValueError("no ArUco markers found with any predefined dictionary")
    dictionary, max_id = _pick_dictionary(scores, max_ids)

    # 2. layout, on frames with enough markers of the chosen dictionary
    layout_idx: list[int] = []
    dets: list[tuple[list[np.ndarray], np.ndarray]] = []
    for i in order:
        if len(layout_idx) >= layout_frames:
            break
        g, _ = load(i)
        corners, ids = _aruco(g, dictionary, upscale, params)
        if len(ids):
            max_id = max(max_id, int(ids.max()))
        if len(ids) >= MIN_LAYOUT_MARKERS:
            layout_idx.append(i)
            dets.append((corners, ids))
        elif i not in hit_frames:
            cache.pop(i, None)
    if _dict_size(dictionary) <= max_id:
        raise ValueError(f"id {max_id} exceeds {dictionary}; mixed markers in the scene?")
    same = sorted((n for n in scores if _family(n) == _family(dictionary) and _dict_size(n) > max_id), key=_dict_size)
    if len(same) > 1:
        notes.append(f"ids 0..{max_id} are identical in {same}; any of them detects this board")
    layouts = fit_layouts(dets, max_id)
    if not layouts:
        raise ValueError(f"no frame with {MIN_LAYOUT_MARKERS}+ markers of {dictionary}; cannot fit a layout")
    best = layouts[0]
    sx, sy = best["squares"]
    tied = [r for r in layouts[1:] if r["median_err_px"] <= 1.05 * best["median_err_px"] + 1e-6]
    if tied:
        notes.append(
            f"{len(tied)} larger layout(s) fit equally (their extra ids {best['squares'][0] * best['squares'][1] // 2}+ "
            f"never appear); chose the smallest board holding ids 0..{max_id}"
        )
    distinct = [r for r in layouts[1:] if r not in tied]
    if distinct:
        r = distinct[0]
        msg = f"best different layout {r['squares']} legacy={r['legacy']}: {r['median_err_px']:.3f} px vs {best['median_err_px']:.3f} px"
        close = r["median_err_px"] < 2.0 * best["median_err_px"] + 0.5
        notes.append(("runner-up layout close: " if close else "") + msg)
    if best["frames"] < 3:
        notes.append(f"layout fitted on only {best['frames']} frame(s)")
    ratio = _marker_ratio(dets, sx, sy, best["legacy"])
    marker_len = round(ratio, 3) if np.isfinite(ratio) and 0.2 < ratio < 0.98 else 0.75
    if marker_len != round(ratio, 3):
        notes.append(f"marker ratio estimate {ratio:.3f} out of range; using 0.75")

    # 3. legacy pattern, by ChArUco corner count on the layout frames
    grays = [load(i)[0] for i in layout_idx]
    corners_by_legacy = {}
    for legacy in (False, True):
        board = BoardSpec(dictionary, (sx, sy), 1.0, marker_len, legacy).make()
        corners_by_legacy[str(legacy)] = int(
            sum(len(detect_charuco(g, board, upscale=upscale, params=params).ids) for g in grays)
        )
    if corners_by_legacy["True"] > corners_by_legacy["False"]:
        legacy = True
    elif corners_by_legacy["True"] < corners_by_legacy["False"]:
        legacy = False
    else:
        legacy = bool(best["legacy"])
        if sy % 2 == 1:
            notes.append("odd squaresY: legacy and current patterns are identical")
    if legacy != best["legacy"] and sy % 2 == 0:
        notes.append(f"legacy from corner count ({legacy}) disagrees with the layout fit ({best['legacy']})")
    notes.append(f"scanned {scanned} frames for dictionaries ({len(hit_frames)} with markers)")

    used = sorted(set(hit_frames) | set(layout_idx))
    return ProbeResult(
        spec=BoardSpec(dictionary, (sx, sy), 1.0, marker_len, legacy),
        frames=[load(i)[1] for i in used],
        dictionary_scores={k: v for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v},
        max_id=max_id,
        layout_scores=layouts[:5],
        marker_ratio=float(ratio),
        corners_by_legacy=corners_by_legacy,
        notes=notes,
    )
