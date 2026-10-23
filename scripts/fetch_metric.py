"""Fetch the METRIC subsets used by the pool protocol, identify their board and summarise detections.

    python scripts/fetch_metric.py                       # fetch d455 + kinect_synth, cameras 1-4
    python scripts/fetch_metric.py --subset kinect_synth --cameras 1 2
    python scripts/fetch_metric.py --list                # sizes only, nothing written
    python scripts/fetch_metric.py --skip-fetch --probe --report
    python scripts/fetch_metric.py --skip-fetch --attribution
    python scripts/fetch_metric.py --samples             # OpenCV left*.jpg fallback set only

Files land in data_ext/metric/ and data_ext/samples/ (gitignored), or under --root. Re-running
resumes: files already present with a matching CRC are skipped. The Zenodo record is only
requested for --list, fetching and --attribution (and is cached in metric/record.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from nextview.data import metric, pool_tools  # noqa: E402
from nextview.data.remotezip import RemoteZip  # noqa: E402


def _mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


def _metric_root(args) -> Path:
    return metric.metric_root(args.root / "metric" if args.root is not None else None)


def _samples_root(args) -> Path | None:
    return args.root / "samples" if args.root is not None else None


def cmd_list(args, record) -> None:
    url, size, _ = metric.archive_url(record)
    with RemoteZip(url) as rz:
        for key in args.subset:
            members = metric.subset_members(rz, key, args.cameras)
            imgs = [i for r, i in members.items() if r.endswith(".png")]
            print(
                f"{key}: {len(members)} files, {len(imgs)} images, "
                f"{_mb(sum(i.compress_size for i in members.values()))} compressed, "
                f"{_mb(sum(i.file_size for i in members.values()))} on disk"
            )
        print(f"archive {_mb(size)}, listing used {rz.reader.requests} requests / {_mb(rz.reader.bytes_fetched)}")


def cmd_fetch(args, record) -> None:
    url, _, _ = metric.archive_url(record)
    base = _metric_root(args)
    with RemoteZip(url) as rz:
        for key in args.subset:
            t0 = time.time()
            f0 = rz.reader.bytes_fetched
            last = [0.0]

            def progress(done, total, fetched, key=key, t0=t0, f0=f0):
                now = time.time()
                if now - last[0] >= 10 or done == total:
                    last[0] = now
                    got = fetched - f0
                    rate = got / max(now - t0, 1e-6) / 1e6
                    print(f"  {key}: {done}/{total} files, {_mb(got)} fetched ({rate:.1f} MB/s)", flush=True)

            m = metric.fetch_subset(key, args.cameras, base, rz=rz, workers=args.workers, progress=progress)
            f = m["fetch"]
            print(
                f"{key}: {m['n_files']} files ({m['n_images']} images), {_mb(m['bytes'])} on disk; "
                f"written {f['written']}, skipped {f['skipped']}, {f['spans']} spans, {f['seconds']} s",
                flush=True,
            )
        print(f"total fetched {_mb(rz.reader.bytes_fetched)} in {rz.reader.requests} requests")


def cmd_probe(args) -> dict:
    from nextview.data import board_probe

    base = _metric_root(args)
    results = {}
    for key in args.subset:
        # interleave the cameras so any prefix of the probe's spread order covers all of them
        per_cam = [metric.image_paths(key, cam, base) for cam in args.cameras]
        paths = [p for group in zip_longest(*per_cam) for p in group if p is not None]
        if not paths:
            print(f"{key}: no images to probe; fetch first")
            continue
        t0 = time.time()
        res = board_probe.probe_board(
            paths, n_frames=args.probe_frames, upscale=metric.DETECT_UPSCALE, params=board_probe.small_marker_params()
        )
        out = res.to_dict()
        out["seconds"] = round(time.time() - t0, 1)
        metric.board_spec_path(key, base).write_text(json.dumps(res.spec.to_dict(), indent=1), encoding="utf-8")
        (base / key / "board_probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"{key}: {json.dumps(out, indent=1)}", flush=True)
        results[key] = out
    return results


def _plane_fit_rms(frame) -> float:
    """RMS (px) of a plane homography fitted to the detected corners."""
    H, _ = cv2.findHomography(frame.det.obj[:, :2], frame.det.corners, 0)
    proj = cv2.perspectiveTransform(frame.det.obj[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((proj - frame.det.corners) ** 2, axis=1))))


def _sanity_calibration(frames, size, ref) -> dict:
    """Plain k5 calibrateCamera on every full-board frame, against the shipped intrinsics."""
    obj = [f.det.obj.astype(np.float32) for f in frames]
    img = [f.det.corners.astype(np.float32) for f in frames]
    K0 = ref.K.copy()
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, K0, np.zeros(5), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    return {
        "views": len(frames),
        "rms_px": round(float(rms), 3),
        "fx": round(float(K[0, 0]), 2),
        "fy": round(float(K[1, 1]), 2),
        "cx": round(float(K[0, 2]), 2),
        "cy": round(float(K[1, 2]), 2),
        "k1": round(float(dist.ravel()[0]), 4),
        "ref_kind": ref.kind,
        "ref_fx": round(float(ref.K[0, 0]), 2),
        "ref_fy": round(float(ref.K[1, 1]), 2),
        "ref_cx": round(float(ref.K[0, 2]), 2),
        "ref_cy": round(float(ref.K[1, 2]), 2),
        "fx_rel_err_pct": round(100.0 * float(K[0, 0] / ref.K[0, 0] - 1.0), 2),
        "fy_rel_err_pct": round(100.0 * float(K[1, 1] / ref.K[1, 1] - 1.0), 2),
        "cx_err_px": round(float(K[0, 2] - ref.K[0, 2]), 2),
        "cy_err_px": round(float(K[1, 2] - ref.K[1, 2]), 2),
    }


def _q(values, fn) -> float | None:
    return round(float(fn(values)), 4) if len(values) else None


def cmd_report(args) -> dict:
    from nextview.config import load_thresholds

    th = load_thresholds()
    base = _metric_root(args)
    report: dict = {"min_corners": th.min_corners, "min_board_area_frac": th.min_board_area_frac, "cameras": {}}
    for key in args.subset:
        spec = metric.load_board_spec(key, base)
        report.setdefault("board", {})[key] = spec.to_dict()
        for cam in args.cameras:
            t0 = time.time()
            frames = metric.load_pool(cam, key, base)
            usable = [f for f in frames if f.det.ok]
            full = [f for f in frames if f.n_corners == spec.n_corners]
            area_full = [pool_tools.board_area_frac(f) for f in full]
            area_usable = [pool_tools.board_area_frac(f) for f in usable]
            hres = [_plane_fit_rms(f) for f in full]
            entry = {
                "frames": len(frames),
                "usable": len(usable),  # >= 4 corners spanning 2 rows and 2 columns
                "any_corners": sum(1 for f in frames if f.n_corners > 0),
                "clean_ge_min_corners": sum(1 for f in frames if f.n_corners >= th.min_corners),
                "full_board": len(full),
                "board_corners": spec.n_corners,
                # whole-board outline (homography-projected), as quality.board_area_frac measures it
                "median_area_frac_full": _q(area_full, np.median),
                "max_area_frac_full": _q(area_full, np.max),
                "max_area_frac_usable": _q(area_usable, np.max),
                "above_min_area_usable": sum(1 for a in area_usable if a >= th.min_board_area_frac),
                "median_plane_fit_rms_px": _q(hres, np.median),
                "p90_plane_fit_rms_px": _q(hres, lambda v: np.percentile(v, 90)),
            }
            if len(full) >= 4:
                ref = metric.load_reference(key, cam, base)
                entry["sanity_calibration"] = _sanity_calibration(full, ref.image_size, ref)
            entry["seconds"] = round(time.time() - t0, 1)
            report["cameras"][f"{key}/camera{cam}"] = entry
            print(f"{key}/camera{cam}: {json.dumps(entry)}", flush=True)
        try:
            sq = metric.estimate_square_size(key, base, args.cameras)
            report.setdefault("square_size", {})[key] = sq
            print(f"{key}: square size {json.dumps(sq)}", flush=True)
        except (FileNotFoundError, ValueError) as exc:
            print(f"{key}: square size not estimated ({exc})", flush=True)
        if metric.SUBSETS[key].reference == "gt":
            chain = {}
            for label, file_k in (("file", True), ("opencv", False)):
                chain[label] = metric.gt_chain_fit(key, base, args.cameras, file_intrinsics=file_k)
                print(f"{key}: GT chain ({label} intrinsics) {json.dumps(chain[label])}", flush=True)
            report.setdefault("gt_chain", {})[key] = chain
    (base / "detection_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


ATTR_HEADER = """# Attribution

NextView's code is MIT-licensed (see LICENSE). Benchmark outputs released with it are CC BY 4.0.
It uses the third-party data below, which is downloaded by the scripts and not stored in this
repository. The generated sections are rewritten by `scripts/fetch_metric.py --attribution`.
"""


def _put_block(text: str, tag: str, section: str) -> str:
    begin = f"<!-- {tag}:begin (generated by scripts/fetch_metric.py --attribution) -->"
    end = f"<!-- {tag}:end -->"
    block = f"{begin}\n{section}{end}"
    if begin in text and end in text:
        return text[: text.index(begin)] + block + text[text.index(end) + len(end) :]
    return text.rstrip("\n") + "\n\n" + block + "\n"


def cmd_attribution(args, record) -> None:
    from nextview.data import samples

    path = REPO / "ATTRIBUTION.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ATTR_HEADER
    text = _put_block(text, "metric", metric.attribution_markdown(record))
    text = _put_block(text, "opencv-samples", samples.attribution_markdown())
    path.write_text(text, encoding="utf-8")
    print(f"updated {path}")


def cmd_samples(args) -> None:
    from nextview.data import samples

    root = _samples_root(args)
    paths = samples.fetch_samples(root)
    frames = samples.load_samples(root)
    ok = sum(1 for f in frames if f.det.ok)
    print(f"samples: {len(paths)} files in {samples.samples_root(root)}, {ok}/{len(frames)} chessboards found")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", nargs="+", default=list(metric.SUBSETS), choices=list(metric.SUBSETS))
    ap.add_argument("--cameras", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    ap.add_argument("--root", type=Path, default=None, help="data root (default data_ext): uses ROOT/metric and ROOT/samples")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="print subset sizes and exit")
    ap.add_argument("--skip-fetch", action="store_true", help="do not fetch METRIC files")
    ap.add_argument("--probe", action="store_true", help="identify the board and write board_spec.json")
    ap.add_argument("--probe-frames", type=int, default=10)
    ap.add_argument("--report", action="store_true", help="detect on every frame and count clean frames")
    ap.add_argument("--attribution", action="store_true", help="write the METRIC entry in ATTRIBUTION.md")
    ap.add_argument("--samples", action="store_true", help="fetch the OpenCV left*.jpg fallback set")
    ap.add_argument("--refresh-record", action="store_true")
    args = ap.parse_args(argv)

    # --samples on its own touches only the OpenCV samples
    do_fetch = not args.skip_fetch and (args.list or args.probe or args.report or args.attribution or not args.samples)
    record = None
    if args.list or do_fetch or args.attribution or args.refresh_record:
        record = metric.zenodo_record(_metric_root(args), refresh=args.refresh_record)
        print(
            f"Zenodo {metric.RECORD_ID}: {record['metadata']['title']} | licence {metric.record_license(record)} "
            f"| doi {record.get('doi')}"
        )
    if args.list:
        cmd_list(args, record)
        return 0
    if do_fetch:
        cmd_fetch(args, record)
    if args.probe:
        cmd_probe(args)
    if args.report:
        cmd_report(args)
    if args.attribution:
        cmd_attribution(args, record)
    if args.samples:
        cmd_samples(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
