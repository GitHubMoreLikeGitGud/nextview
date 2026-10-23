"""The METRIC calibration dataset (Zenodo record 7976757, CC BY 4.0).

Allegro, Terreran and Ghidoni, "METRIC - Multi-Eye To Robot Indoor Calibration Dataset",
doi:10.5281/zenodo.7976757. The record is a single 25.9 GB ZIP; only the two ChArUco subsets
used by the pool protocol are fetched, with HTTP range reads (``remotezip``):

* ``d455``: ``real_data/DepthD455/charuco/small_workcell/camera1..4``, 251 PNG frames at 1280x720
  per camera. The intrinsics shipped with it are the sensor's factory values: a reference, not GT.
* ``kinect_synth``: ``synthetic_data/KinectV2/charuco/small_workcell/camera1..4``, 251 rendered PNG
  frames at 1920x1080 per camera, with GT intrinsics (about zero distortion) and GT rig poses.
  Its intrinsics files put the principal point at ``((W+1)/2, (H+1)/2) = (960.5, 540.5)``, the
  renderer's 1-based pixel convention; ``load_reference`` converts it to OpenCV's 0-based
  pixel-centre convention, ``(959.5, 539.5)`` (see ``gt_chain_fit`` for the check).

Detections are in square units of the METRIC board (5x4 legacy ChArUco); each ``Frame`` carries
that board, so use ``nextview.data.pool_tools`` rather than ``nextview.tools`` on these frames.

Layout under ``data_ext/metric/``::

    record.json                       Zenodo API metadata (licence, creators, checksum)
    <subset>/manifest.json            files, sizes and CRCs as fetched
    <subset>/camera<k>/image/NNNN.png
    <subset>/camera<k>/pose/NNNN.csv  4x4 robot end-effector pose for that frame
    <subset>/camera<k>/intrinsic_pars_file.yaml
    <subset>/GT/gt_cam<k>.csv         4x4 camera pose in the robot base frame
    <subset>/board_spec.json          board identified by board_probe on this subset
    <subset>/camera<k>/detections.npz cached ChArUco detections for load_pool

Everything inside the archive is treated as untrusted: images are decoded with OpenCV, the
intrinsics file is parsed with a regex, matrices with numpy, and output paths are rebuilt
from validated relative names.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from nextview.data.frame import Frame, data_root
from nextview.data.remotezip import RemoteZip

__all__ = [
    "RECORD_ID",
    "DOI",
    "LICENSE_ID",
    "Subset",
    "SUBSETS",
    "Reference",
    "metric_root",
    "zenodo_record",
    "archive_url",
    "subset_members",
    "fetch_subset",
    "read_intrinsics",
    "read_matrix4",
    "load_reference",
    "load_gt_extrinsics",
    "load_ee_pose",
    "board_spec_path",
    "load_board_spec",
    "image_paths",
    "load_pool",
    "estimate_square_size",
    "gt_chain_fit",
    "attribution_markdown",
]

RECORD_ID = 7976757
DOI = "10.5281/zenodo.7976757"
LICENSE_ID = "cc-by-4.0"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
ARCHIVE_KEY = "METRIC dataset.zip"
CAMERAS = (1, 2, 3, 4)
# The boards are 10-20 px markers at this distance: detect on a 2x bicubic upscale with
# small-marker settings (more frames detect, corner-fit residual is lower than at 1x).
DETECT_UPSCALE = 2


@dataclass(frozen=True)
class Subset:
    key: str
    prefix: str  # path prefix inside the archive
    kind: str  # "real" | "synthetic"
    reference: str  # "factory" (reference only) | "gt"
    image_size: tuple[int, int]
    # added to the file's cx, cy to reach OpenCV's convention (pixel (0, 0) centred on (0, 0))
    principal_shift: float = 0.0


SUBSETS: dict[str, Subset] = {
    "d455": Subset(
        key="d455",
        prefix="METRIC dataset/real_data/DepthD455/charuco/small_workcell/",
        kind="real",
        reference="factory",
        image_size=(1280, 720),
    ),
    "kinect_synth": Subset(
        key="kinect_synth",
        prefix="METRIC dataset/synthetic_data/KinectV2/charuco/small_workcell/",
        kind="synthetic",
        reference="gt",
        image_size=(1920, 1080),
        principal_shift=-1.0,
    ),
}


def metric_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else data_root() / "metric"


def _subset(subset: str | Subset) -> Subset:
    if isinstance(subset, Subset):
        return subset
    try:
        return SUBSETS[subset]
    except KeyError:
        raise ValueError(f"unknown METRIC subset {subset!r}; known: {sorted(SUBSETS)}") from None


def _camera_index(camera: int | str) -> int:
    k = int(str(camera).removeprefix("camera"))
    if k not in CAMERAS:
        raise ValueError(f"camera must be one of {CAMERAS}, got {camera!r}")
    return k


# ----------------------------------------------------------------------------- Zenodo record


def zenodo_record(root: Path | str | None = None, *, refresh: bool = False, timeout: float = 30.0) -> dict:
    """Zenodo API metadata for the record, cached in ``metric/record.json``."""
    path = metric_root(root) / "record.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    req = urllib.request.Request(API_URL, headers={"Accept": "application/json", "User-Agent": "nextview-data/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        record = json.loads(resp.read().decode("utf-8"))
    if int(record.get("id", -1)) != RECORD_ID:
        raise ValueError(f"Zenodo returned record {record.get('id')!r}, expected {RECORD_ID}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


def archive_url(record: dict) -> tuple[str, int, str]:
    """``(url, size, checksum)`` of the dataset ZIP listed in the record."""
    for f in record.get("files", []):
        if f.get("key") == ARCHIVE_KEY:
            url = f["links"]["self"]
            if not url.startswith("https://zenodo.org/"):
                raise ValueError(f"unexpected archive URL {url!r}")
            return url, int(f["size"]), str(f.get("checksum", ""))
    raise ValueError(f"{ARCHIVE_KEY!r} not listed in Zenodo record {RECORD_ID}")


def record_license(record: dict) -> str:
    lic = record.get("metadata", {}).get("license", {})
    return str(lic.get("id", lic) if isinstance(lic, dict) else lic)


# ----------------------------------------------------------------------------- fetching

_REL_OK = re.compile(r"^(GT/gt_cam[1-4]\.csv|camera[1-4]/(intrinsic_pars_file\.yaml|image/\d{4}\.png|pose/\d{4}\.csv))$")


def subset_members(
    rz: RemoteZip, subset: str | Subset, cameras: Iterable[int] = CAMERAS, *, include_images: bool = True
) -> dict[str, zipfile.ZipInfo]:
    """``{relative output path: ZipInfo}`` for the chosen cameras of a subset.

    Only names matching the expected layout are accepted; anything else in the prefix is ignored.
    """
    sub = _subset(subset)
    cams = {f"camera{_camera_index(c)}" for c in cameras}
    out: dict[str, zipfile.ZipInfo] = {}
    for info in rz.members(sub.prefix):
        rel = info.filename[len(sub.prefix) :]
        if not _REL_OK.match(rel):
            continue
        head = rel.split("/", 1)[0]
        if head == "GT":
            if f"camera{rel[len('GT/gt_cam')]}" not in cams:
                continue
        elif head not in cams:
            continue
        if not include_images and rel.endswith(".png"):
            continue
        out[f"{sub.key}/{rel}"] = info
    return out


def fetch_subset(
    subset: str | Subset,
    cameras: Iterable[int] = CAMERAS,
    root: Path | str | None = None,
    *,
    include_images: bool = True,
    workers: int = 4,
    rz: RemoteZip | None = None,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict:
    """Download one subset into ``data_ext/metric/<subset>/`` and write its manifest.

    Already-present files with a matching size and CRC are skipped, so the call is resumable.
    The manifest's ``fetch`` holds this run's transfer counts and ``fetch_history`` every run's.
    Returns the manifest dict.
    """
    sub = _subset(subset)
    cams = tuple(_camera_index(c) for c in cameras)
    base = metric_root(root)
    record = zenodo_record(base)
    url, size, checksum = archive_url(record)
    own = rz is None
    t0 = time.time()
    rz = rz or RemoteZip(url)
    try:
        fetched0, requests0 = rz.reader.bytes_fetched, rz.reader.requests
        if rz.size != size:
            raise ValueError(f"archive size {rz.size} differs from the record's {size}")
        targets = subset_members(rz, sub, cams, include_images=include_images)
        if not targets:
            raise ValueError(f"no members found under {sub.prefix!r}")
        counts = rz.extract(targets, base, workers=workers, progress=progress)
        fetched, requests = rz.reader.bytes_fetched - fetched0, rz.reader.requests - requests0
    finally:
        if own:
            rz.close()
    files = sorted(targets.items())
    run = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "cameras": list(cams),
        "include_images": include_images,
        **counts,
        "bytes_fetched": fetched,
        "requests": requests,
        "seconds": round(time.time() - t0, 1),
    }
    manifest = {
        "record": RECORD_ID,
        "doi": DOI,
        "license": record_license(record),
        "archive": {"key": ARCHIVE_KEY, "size": size, "checksum": checksum},
        "subset": sub.key,
        "prefix": sub.prefix,
        "cameras": list(cams),
        "n_files": len(files),
        "n_images": sum(1 for rel, _ in files if rel.endswith(".png")),
        "bytes": sum(i.file_size for _, i in files),
        "fetch": run,  # this run
        "fetch_history": [run],  # every run, oldest first
        "files": {rel: {"size": i.file_size, "crc32": f"{i.CRC:08x}"} for rel, i in files},
    }
    mpath = base / sub.key / "manifest.json"
    old = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
    if old:
        history = old.get("fetch_history")
        if not isinstance(history, list):
            history = [old["fetch"]] if isinstance(old.get("fetch"), dict) else []
        manifest["fetch_history"] = history + [run]
    if old and old.get("files"):
        merged = dict(old["files"])
        merged.update(manifest["files"])
        manifest["files"] = dict(sorted(merged.items()))
        manifest["cameras"] = sorted(set(old.get("cameras", [])) | set(cams))
        manifest["n_files"] = len(manifest["files"])
        manifest["n_images"] = sum(1 for rel in manifest["files"] if rel.endswith(".png"))
        manifest["bytes"] = sum(v["size"] for v in manifest["files"].values())
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


# ----------------------------------------------------------------------------- readers

_KV = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([-+0-9.eE]+)\s*$")


def read_intrinsics(path: Path | str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Parse ``intrinsic_pars_file.yaml`` -> ``(K, dist, (width, height))``.

    The file is plain ``key: value`` lines. Its ``dist_k0, dist_k1, dist_px, dist_py, dist_k2``
    are OpenCV's ``k1, k2, p1, p2, k3``; ``dist_k3..k5`` (``k4..k6``) are zero in this dataset
    and are dropped unless non-zero, in which case the 8-coefficient rational order is returned.
    """
    vals: dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _KV.match(line)
        if m:
            vals[m.group(1)] = float(m.group(2))
    need = ("fx", "fy", "cx", "cy", "img_width", "img_height")
    missing = [k for k in need if k not in vals]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    K = np.array([[vals["fx"], 0.0, vals["cx"]], [0.0, vals["fy"], vals["cy"]], [0.0, 0.0, 1.0]])
    d5 = [vals.get(k, 0.0) for k in ("dist_k0", "dist_k1", "dist_px", "dist_py", "dist_k2")]
    d_hi = [vals.get(k, 0.0) for k in ("dist_k3", "dist_k4", "dist_k5")]
    dist = np.array(d5 + d_hi if any(d_hi) else d5, dtype=np.float64)
    if not vals.get("has_dist_coeff", 1.0):
        dist[:] = 0.0
    return K, dist, (int(vals["img_width"]), int(vals["img_height"]))


def read_matrix4(path: Path | str) -> np.ndarray:
    m = np.loadtxt(Path(path), dtype=np.float64, ndmin=2)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError(f"{path}: expected a finite 4x4 matrix, got shape {m.shape}")
    return m


@dataclass(frozen=True, eq=False)
class Reference:
    K: np.ndarray  # OpenCV pixel-centre convention
    dist: np.ndarray
    image_size: tuple[int, int]
    kind: str  # "factory" (reference only, not ground truth) | "gt"
    K_file: np.ndarray | None = None  # K exactly as written in intrinsic_pars_file.yaml
    principal_shift: float = 0.0  # added to the file's cx and cy to give K


def load_reference(subset: str | Subset, camera: int | str, root: Path | str | None = None) -> Reference:
    """Intrinsics shipped with a METRIC camera, in OpenCV's pixel convention.

    ``kinect_synth`` files give ``cx, cy = (W+1)/2, (H+1)/2`` (1-based pixel centres, as the
    renderer writes them); 1 px is subtracted from both so that pixel (0, 0) is centred on
    (0, 0) as in OpenCV. Fitting the principal point on the GT rig chain (``gt_chain_fit``)
    gives offsets of about -1 px against the file values and about 0 against the converted ones.
    ``d455`` factory values are used as written.
    """
    sub = _subset(subset)
    k = _camera_index(camera)
    K_file, dist, size = read_intrinsics(metric_root(root) / sub.key / f"camera{k}" / "intrinsic_pars_file.yaml")
    K = K_file.copy()
    K[0, 2] += sub.principal_shift
    K[1, 2] += sub.principal_shift
    return Reference(K=K, dist=dist, image_size=size, kind=sub.reference, K_file=K_file, principal_shift=sub.principal_shift)


def load_gt_extrinsics(subset: str | Subset, camera: int | str, root: Path | str | None = None) -> np.ndarray:
    """4x4 camera pose in the robot base frame (``GT/gt_cam<k>.csv``)."""
    sub = _subset(subset)
    return read_matrix4(metric_root(root) / sub.key / "GT" / f"gt_cam{_camera_index(camera)}.csv")


def load_ee_pose(subset: str | Subset, camera: int | str, frame: str, root: Path | str | None = None) -> np.ndarray:
    """4x4 robot end-effector pose recorded with image ``frame`` (e.g. ``"0001"``)."""
    sub = _subset(subset)
    if not re.fullmatch(r"\d{4}", frame):
        raise ValueError(f"bad frame name {frame!r}")
    return read_matrix4(metric_root(root) / sub.key / f"camera{_camera_index(camera)}" / "pose" / f"{frame}.csv")


def image_paths(subset: str | Subset, camera: int | str, root: Path | str | None = None) -> list[Path]:
    sub = _subset(subset)
    d = metric_root(root) / sub.key / f"camera{_camera_index(camera)}" / "image"
    return sorted(p for p in d.glob("*.png") if re.fullmatch(r"\d{4}\.png", p.name))


# ----------------------------------------------------------------------------- pool


def board_spec_path(subset: str | Subset, root: Path | str | None = None) -> Path:
    return metric_root(root) / _subset(subset).key / "board_spec.json"


def load_board_spec(subset: str | Subset, root: Path | str | None = None):
    """The board identified by ``board_probe`` for this subset (``<subset>/board_spec.json``)."""
    from nextview.data.board_probe import BoardSpec

    path = board_spec_path(subset, root)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/fetch_metric.py --skip-fetch --probe first")
    return BoardSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _spec_for(sub: Subset, base: Path, board_spec):
    from nextview.data.board_probe import BoardSpec

    if board_spec is not None:
        return board_spec if isinstance(board_spec, BoardSpec) else BoardSpec.from_dict(board_spec)
    return load_board_spec(sub, base)


def load_pool(
    camera: int | str,
    subset: str | Subset = "d455",
    root: Path | str | None = None,
    *,
    board_spec=None,
    use_cache: bool = True,
    detect: bool = True,
) -> list[Frame]:
    """Every frame of one METRIC camera with its ChArUco detection, sorted by filename.

    Detections use the board identified by ``board_probe`` (``board_spec.json``) and are cached
    in ``<subset>/camera<k>/detections.npz``; the cache is keyed on the board spec, every
    detector setting (``board_probe.detector_signature``), the OpenCV version and the file list.
    With ``detect=False`` a missing or stale cache raises ``FileNotFoundError`` instead of
    running the detector.

    ``det.ok`` means at least 4 corners spanning 2 rows and 2 columns; pool policies apply the
    R0 corner gate themselves. Frames where the board was not found are kept with
    ``det.ok = False``. ``det.obj`` is in square units of ``frame.board`` (the METRIC board,
    not the NextView one).
    """
    from nextview.data import board_probe

    sub = _subset(subset)
    k = _camera_index(camera)
    base = metric_root(root)
    spec = _spec_for(sub, base, board_spec)
    paths = image_paths(sub, k, base)
    if not paths:
        raise FileNotFoundError(f"no images for {sub.key}/camera{k} under {base}; run scripts/fetch_metric.py")
    source = f"metric/{sub.key}/camera{k}"
    cache = base / sub.key / f"camera{k}" / "detections.npz"
    params = board_probe.small_marker_params()
    key = json.dumps(
        {
            "spec": spec.to_dict(),
            "detector": board_probe.detector_signature(params, DETECT_UPSCALE),
            "names": [p.name for p in paths],
        },
        sort_keys=True,
    )

    if use_cache and cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if str(z["key"]) == key:
                return _frames_from_cache(z, paths, source, sub, spec)
    if not detect:
        raise FileNotFoundError(f"no valid detection cache at {cache}")

    board = spec.make()
    rows = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rows.append((np.zeros(0, np.int32), np.zeros((0, 2)), 0, (0, 0)))
            continue
        det = board_probe.detect_charuco(img, board, spec, upscale=DETECT_UPSCALE, params=params)
        rows.append((det.ids, det.corners, det.n_markers, (img.shape[1], img.shape[0])))
    counts = np.array([len(r[0]) for r in rows], np.int64)
    np.savez_compressed(
        cache,
        key=np.array(key),
        counts=counts,
        ids=np.concatenate([r[0] for r in rows]).astype(np.int32) if counts.sum() else np.zeros(0, np.int32),
        corners=np.concatenate([r[1] for r in rows]).astype(np.float64) if counts.sum() else np.zeros((0, 2)),
        n_markers=np.array([r[2] for r in rows], np.int64),
        sizes=np.array([r[3] for r in rows], np.int64),
    )
    with np.load(cache, allow_pickle=False) as z:
        return _frames_from_cache(z, paths, source, sub, spec)


def _frames_from_cache(z, paths, source, sub, spec) -> list[Frame]:
    from nextview.types import Detection

    obj_all = spec.object_points()
    board = spec.planar(units="square")
    counts = z["counts"]
    offs = np.concatenate([[0], np.cumsum(counts)])
    ids_all, corners_all = z["ids"], z["corners"]
    frames = []
    for i, p in enumerate(paths):
        ids = ids_all[offs[i] : offs[i + 1]].astype(np.int32)
        corners = corners_all[offs[i] : offs[i + 1]].astype(np.float64)
        n_markers = int(z["n_markers"][i])
        w, h = (int(v) for v in z["sizes"][i])
        det = Detection(ids=ids, corners=corners, obj=obj_all[ids], n_markers=n_markers, ok=board.usable(ids))
        frames.append(
            Frame(
                name=p.stem,
                path=p,
                source=source,
                image_size=(w, h) if w else sub.image_size,
                det=det,
                board=board,
                meta={"subset": sub.key, "reference": sub.reference},
            )
        )
    return frames


# ----------------------------------------------------------------------------- attribution


def estimate_square_size(
    subset: str | Subset, root: Path | str | None = None, cameras: Iterable[int] = CAMERAS, *, detect: bool = True
) -> dict:
    """Physical square length (metres) from the rig GT and the recorded robot poses.

    For every full-board view, the board origin from PnP (square units, shipped intrinsics) must
    land on the same end-effector point in the robot base frame::

        s * R_cam @ t_board + t_cam = R_ee @ x + t_ee

    with ``GT/gt_cam<k>.csv`` = camera-to-base and ``pose/NNNN.csv`` = end-effector-to-base. This is
    linear in the scale ``s`` and the board offset ``x``; it is solved by least squares over all
    cameras. On ``kinect_synth`` the intrinsics are GT; on ``d455`` they are factory values, so the
    estimate there inherits their error.
    """
    sub = _subset(subset)
    base = metric_root(root)
    spec = load_board_spec(sub, base)
    A, b, per_cam = [], [], {}
    for cam in cameras:
        ref = load_reference(sub, cam, base)
        T_cam = load_gt_extrinsics(sub, cam, base)
        n = 0
        for f in load_pool(cam, sub, base, detect=detect):
            if f.n_corners < spec.n_corners:
                continue
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(f.det.obj, f.det.corners, ref.K, ref.dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            rv, tv = cv2.solvePnPRefineLM(f.det.obj, f.det.corners, ref.K, ref.dist, rvecs[0], tvecs[0])
            T_ee = load_ee_pose(sub, cam, f.name, base)
            A.append(np.hstack([(T_cam[:3, :3] @ tv.reshape(3))[:, None], -T_ee[:3, :3]]))
            b.append(T_ee[:3, 3] - T_cam[:3, 3])
            n += 1
        per_cam[cam] = n
    if len(A) < 3:
        raise ValueError("fewer than 3 full-board views; cannot estimate the square size")
    A_, b_ = np.vstack(A), np.concatenate(b)
    sol, *_ = np.linalg.lstsq(A_, b_, rcond=None)
    res = (A_ @ sol - b_).reshape(-1, 3)
    return {
        "subset": sub.key,
        "square_m": float(sol[0] * spec.square_len),
        "board_offset_ee_m": [float(v) for v in sol[1:]],
        "residual_rms_m": float(np.sqrt(np.mean(np.sum(res**2, axis=1)))),
        "views": len(A),
        "views_per_camera": per_cam,
        "intrinsics": sub.reference,
    }


def _lm(fun: Callable[[np.ndarray], np.ndarray], p0: np.ndarray, iters: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Levenberg-Marquardt with a central-difference Jacobian; returns ``(p, residuals, J)``."""
    p = np.asarray(p0, np.float64).copy()
    r = fun(p)
    cost = float(r @ r)
    lam = 1e-3

    def jac(p):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            h = 1e-6 * max(1.0, abs(p[k]))
            dp = np.zeros_like(p)
            dp[k] = h
            J[:, k] = (fun(p + dp) - fun(p - dp)) / (2.0 * h)
        return J

    J = jac(p)
    for _ in range(iters):
        A, g = J.T @ J, J.T @ r
        improved = False
        while lam < 1e10:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r_new = fun(p + step)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                rel = (cost - cost_new) / max(cost, 1e-300)
                p, r, cost, lam, improved = p + step, r_new, cost_new, max(lam / 3.0, 1e-12), True
                break
            lam *= 4.0
        if not improved or rel < 1e-12:
            break
        J = jac(p)
    return p, r, J


def gt_chain_fit(
    subset: str | Subset = "kinect_synth",
    root: Path | str | None = None,
    cameras: Iterable[int] = CAMERAS,
    *,
    file_intrinsics: bool = False,
    detect: bool = True,
) -> dict:
    """Check the shipped intrinsics against the GT rig chain.

    Every full-board corner is predicted from the recorded robot pose and the GT camera pose::

        x = project(K, inv(T_cam) @ T_ee @ T_board_ee @ (s * obj))

    with ``T_board_ee`` (board in end-effector frame, 6 dof) and the square size ``s`` fitted by
    least squares over all cameras. A second fit also frees a principal-point offset per camera;
    offsets near 0 mean the intrinsics agree with the chain. ``file_intrinsics=True`` uses the
    file values without the convention conversion of ``load_reference``.
    RMS values are ``sqrt(mean |e|^2)`` over corners, in px.
    """
    sub = _subset(subset)
    base = metric_root(root)
    spec = load_board_spec(sub, base)
    cams = [_camera_index(c) for c in cameras]
    refs, T_cams = {}, {}
    obj_l, img_l, Ree_l, tee_l, cam_l, per_cam = [], [], [], [], [], {}
    for j, cam in enumerate(cams):
        ref = load_reference(sub, cam, base)
        refs[cam] = (ref.K_file if file_intrinsics and ref.K_file is not None else ref.K, ref.dist)
        T_cams[cam] = load_gt_extrinsics(sub, cam, base)
        n = 0
        for f in load_pool(cam, sub, base, detect=detect):
            if f.n_corners < spec.n_corners:
                continue
            T_ee = load_ee_pose(sub, cam, f.name, base)
            m = len(f.det.ids)
            obj_l.append(f.det.obj)
            img_l.append(f.det.corners)
            Ree_l.append(np.repeat(T_ee[None, :3, :3], m, axis=0))
            tee_l.append(np.repeat(T_ee[None, :3, 3], m, axis=0))
            cam_l.append(np.full(m, j))
            n += 1
        per_cam[cam] = n
    if sum(per_cam.values()) < 3:
        raise ValueError("fewer than 3 full-board views; cannot fit the GT chain")
    obj, img = np.vstack(obj_l), np.vstack(img_l)
    Ree, tee, cam_idx = np.concatenate(Ree_l), np.concatenate(tee_l), np.concatenate(cam_l)
    Rc_inv = np.stack([T_cams[c][:3, :3].T for c in cams])
    tc = np.stack([T_cams[c][:3, 3] for c in cams])

    def predict(p: np.ndarray, free_pp: bool) -> np.ndarray:
        R_be, _ = cv2.Rodrigues(p[:3].reshape(3, 1))
        X_ee = (p[6] * obj) @ R_be.T + p[3:6]
        X_base = np.einsum("nij,nj->ni", Ree, X_ee) + tee
        X_cam = np.einsum("nij,nj->ni", Rc_inv[cam_idx], X_base - tc[cam_idx])
        out = np.empty((len(obj), 2))
        for j, cam in enumerate(cams):
            sel = cam_idx == j
            K, dist = refs[cam]
            K = K.copy()
            if free_pp:
                K[0, 2] += p[7 + 2 * j]
                K[1, 2] += p[8 + 2 * j]
            q, _ = cv2.projectPoints(X_cam[sel].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
            out[sel] = q.reshape(-1, 2)
        return out

    # start: square size and board offset from the linear solve, rotation from per-view PnP
    lin = estimate_square_size(sub, base, cams, detect=detect)
    rots = []
    for o, im, R_ee, j in zip(obj_l, img_l, Ree_l, cam_l):
        K, dist = refs[cams[j[0]]]
        ok, rv, _ = cv2.solvePnP(o, im, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            rots.append(R_ee[0].T @ T_cams[cams[j[0]]][:3, :3] @ cv2.Rodrigues(rv)[0])
    U, _, Vt = np.linalg.svd(np.sum(rots, axis=0))
    R0 = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    p0 = np.concatenate([cv2.Rodrigues(R0)[0].ravel(), lin["board_offset_ee_m"], [lin["square_m"] / spec.square_len]])

    def rms(r: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum(r.reshape(-1, 2) ** 2, axis=1))))

    p_fix, r_fix, _ = _lm(lambda p: (predict(p, False) - img).ravel(), p0)
    p_free, r_free, J = _lm(lambda p: (predict(p, True) - img).ravel(), np.concatenate([p_fix, np.zeros(2 * len(cams))]))
    dof = max(r_free.size - p_free.size, 1)
    cov = float(r_free @ r_free) / dof * np.linalg.pinv(J.T @ J)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    res_fix = r_fix.reshape(-1, 2)
    return {
        "subset": sub.key,
        "intrinsics": "file" if file_intrinsics else "opencv",
        "principal_shift": 0.0 if file_intrinsics else sub.principal_shift,
        "views": int(sum(per_cam.values())),
        "views_per_camera": per_cam,
        "square_m": float(p_fix[6] * spec.square_len),
        "rms_px": rms(r_fix),
        "rms_px_free_principal": rms(r_free),
        # predicted minus observed, per camera, with the principal point fixed
        "mean_residual_px": {cam: [float(v) for v in res_fix[cam_idx == j].mean(axis=0)] for j, cam in enumerate(cams)},
        # fitted correction to add to (cx, cy), and its standard error
        "principal_offset_px": {cam: [float(p_free[7 + 2 * j]), float(p_free[8 + 2 * j])] for j, cam in enumerate(cams)},
        "principal_offset_se_px": {cam: [float(se[7 + 2 * j]), float(se[8 + 2 * j])] for j, cam in enumerate(cams)},
    }


def attribution_markdown(record: dict) -> str:
    """ATTRIBUTION.md section built from the Zenodo API metadata."""
    md = record.get("metadata", {})
    creators = md.get("creators", [])

    def who(c: dict) -> str:
        s = c.get("name", "").strip()
        if c.get("affiliation"):
            s += f" ({c['affiliation'].strip()})"
        if c.get("orcid"):
            s += f", ORCID {c['orcid'].strip()}"
        return s

    names = "; ".join(c.get("name", "").strip() for c in creators)
    doi = record.get("doi", DOI)
    lic = record_license(record)
    lic_text = {"cc-by-4.0": "Creative Commons Attribution 4.0 International (CC BY 4.0)"}.get(lic, lic)
    lic_url = {"cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/"}.get(lic, "")
    lines = [
        "## METRIC dataset",
        "",
        f"- **Title:** {md.get('title', '').strip()}",
        f"- **Creators:** {names}",
    ]
    lines += [f"  - {who(c)}" for c in creators]
    lines += [
        f"- **Version:** {md.get('version', '')} (published {md.get('publication_date', '')})",
        f"- **DOI:** [{doi}](https://doi.org/{doi})",
        f"- **Licence:** {lic_text}" + (f", <{lic_url}>" if lic_url else ""),
        f"- **Source:** <https://zenodo.org/records/{RECORD_ID}>",
        "- **Used:** `real_data/DepthD455/charuco/small_workcell/camera1..4` and "
        "`synthetic_data/KinectV2/charuco/small_workcell/camera1..4` (images, intrinsics files, GT poses), "
        "fetched unmodified by `scripts/fetch_metric.py` into `data_ext/metric/` and not redistributed in this "
        "repository. Benchmark results derived from these images are released under CC BY 4.0 with this attribution.",
        "- **Changes:** none to the files. When loading, NextView converts the principal point of the "
        "KinectV2 synthetic intrinsics from the files' 1-based convention, ((W+1)/2, (H+1)/2) = (960.5, 540.5), "
        "to OpenCV's 0-based pixel-centre convention by subtracting 1 px from cx and cy; fitted on the GT rig "
        "chain, the file values are off by about 1 px in both axes. The board's physical square size is not "
        "documented; detections use square units.",
        "- **Citation:** "
        + f"{names} ({str(md.get('publication_date', ''))[:4]}). *{md.get('title', '').strip()}* "
        + f"(Version {md.get('version', '')}) [Data set]. Zenodo. https://doi.org/{doi}",
    ]
    return "\n".join(lines) + "\n"
