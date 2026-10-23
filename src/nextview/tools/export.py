"""Calibration export: OpenCV YAML, a PNG report card and a JSON record."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ..types import CalibResult, UncMap

__all__ = ["DISCLAIMER", "export_calibration", "calibration_yaml", "report_card_png", "calibration_json", "intrinsic_std"]

DISCLAIMER = "Decision support for camera calibration — not certified metrology."


def _num(x: Any) -> Any:
    """JSON-safe copy: arrays to lists, numpy scalars to Python scalars, non-finite floats to None."""
    if isinstance(x, np.ndarray):
        return _num(x.tolist())
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    if isinstance(x, Mapping):
        return {str(k): _num(v) for k, v in x.items()}
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _scalar(v: Any) -> bool | int | float | str | None:
    """A meta value as a plain Python scalar, or None when it is not a finite scalar."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, str):
        return v
    return None


def intrinsic_std(calib: CalibResult) -> np.ndarray:
    """sqrt(diag(cov)) of (fx, fy, cx, cy, d1..dn); OpenCV's ``stdDeviationsIntrinsics`` when cov is unusable."""
    p = 4 + int(np.asarray(calib.dist).size)
    cov = None if calib.cov is None else np.asarray(calib.cov, np.float64)
    if cov is not None and cov.shape == (p, p) and np.all(np.isfinite(cov)):
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return np.asarray(calib.std_intr, np.float64).ravel()[:p]


def _trace_lines(trace) -> list[str]:
    if trace is None:
        return []
    if isinstance(trace, str):
        return trace.splitlines()
    return [str(t) for t in trace]


def calibration_yaml(calib: CalibResult, unc: UncMap | None, size: tuple[int, int] | None = None, meta: Mapping | None = None) -> bytes:
    fs = cv2.FileStorage(".yaml", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
    try:
        fs.writeComment(DISCLAIMER)
        if size is not None:
            fs.write("image_width", int(size[0]))
            fs.write("image_height", int(size[1]))
        fs.write("calibration_model", calib.model)
        fs.write("camera_matrix", np.asarray(calib.K, np.float64))
        fs.write("distortion_coefficients", np.asarray(calib.dist, np.float64).reshape(1, -1))
        fs.write("avg_reprojection_error", float(calib.rms))
        fs.write("n_views", int(len(calib.per_view)))
        p = 4 + int(np.asarray(calib.dist).size)
        fs.write("std_intrinsics", intrinsic_std(calib).reshape(1, -1))
        fs.write("std_intrinsics_opencv", np.asarray(calib.std_intr, np.float64).ravel()[:p].reshape(1, -1))
        if calib.cov is not None and np.all(np.isfinite(calib.cov)):
            fs.write("intrinsic_covariance", np.asarray(calib.cov, np.float64))
        if unc is not None:
            fs.write("uncertainty_U95_px", float(unc.U95))
            if math.isfinite(unc.p_exceed):
                fs.write("p_exceed", float(unc.p_exceed))
        new_obj = getattr(calib, "new_obj", None)
        if new_obj is not None:
            fs.write("refined_board_points", np.asarray(new_obj, np.float64))
        for k, v in (meta or {}).items():
            s = _scalar(v)
            if s is not None:
                fs.write(str(k), int(s) if isinstance(s, bool) else s)
        text = fs.releaseAndGetString()
    finally:
        if fs.isOpened():
            fs.release()
    return text.encode("utf-8")


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.55, color=(30, 30, 30), thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def report_card_png(calib: CalibResult, unc: UncMap | None, trace=None, meta: Mapping | None = None) -> bytes:
    """960 x 600 card: uncertainty heat map with a colour bar, intrinsics and the stop numbers."""
    Wc, Hc = 960, 600
    card = np.full((Hc, Wc, 3), 255, np.uint8)
    _put(card, "NextView calibration report", (20, 34), 0.8, (0, 0, 0), 2)
    map_w, map_h = 576, 324
    x0, y0 = 20, 60
    if unc is not None:
        s = np.asarray(unc.sigma_px, np.float64)
        vmax = max(float(np.nanmax(s)) if s.size else 0.0, 1e-6)
        norm = np.clip(np.nan_to_num(s / vmax), 0, 1)
        heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        heat = cv2.resize(heat, (map_w, map_h), interpolation=cv2.INTER_NEAREST)
        card[y0 : y0 + map_h, x0 : x0 + map_w] = heat
        cv2.rectangle(card, (x0 - 1, y0 - 1), (x0 + map_w, y0 + map_h), (0, 0, 0), 1)
        bar = cv2.applyColorMap(np.linspace(0, 255, map_w).astype(np.uint8).reshape(1, -1), cv2.COLORMAP_VIRIDIS)
        card[y0 + map_h + 8 : y0 + map_h + 22, x0 : x0 + map_w] = cv2.resize(bar, (map_w, 14), interpolation=cv2.INTER_NEAREST)
        _put(card, "0 px", (x0, y0 + map_h + 40), 0.45)
        _put(card, f"{vmax:.2f} px", (x0 + map_w - 70, y0 + map_h + 40), 0.45)
        _put(card, "predicted reprojection std per image cell", (x0 + 130, y0 + map_h + 40), 0.45)
    xt, yt = 620, 80
    K = np.asarray(calib.K)
    lines = [
        f"model: {calib.model}",
        f"views: {len(calib.per_view)}",
        f"fx {K[0, 0]:.2f}  fy {K[1, 1]:.2f}",
        f"cx {K[0, 2]:.2f}  cy {K[1, 2]:.2f}",
        f"RMS reprojection: {calib.rms:.3f} px",
    ]
    d = np.asarray(calib.dist).ravel()
    for i in range(0, d.size, 3):
        lines.append("dist " + " ".join(f"{x:+.4f}" for x in d[i : i + 3]))
    if unc is not None:
        lines.append(f"U95: {unc.U95:.3f} px")
        pe = "n/a" if not math.isfinite(unc.p_exceed) else f"{unc.p_exceed:.2f}"
        lines.append(f"P(field p95 > tau): {pe}")
    for k, v in (meta or {}).items():
        s = _scalar(v)
        if s is not None:
            lines.append(f"{k}: {s}")
    for i, ln in enumerate(lines[:16]):
        _put(card, ln, (xt, yt + 26 * i))
    tl = _trace_lines(trace)[-4:]
    for i, ln in enumerate(tl):
        _put(card, ln[:110], (20, 470 + 22 * i), 0.42, (60, 60, 60))
    _put(card, DISCLAIMER, (20, Hc - 18), 0.5, (0, 0, 160))
    ok, buf = cv2.imencode(".png", card)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def calibration_json(calib: CalibResult, unc: UncMap | None, trace=None, size=None, meta: Mapping | None = None) -> bytes:
    p = 4 + int(np.asarray(calib.dist).size)
    rec: dict[str, Any] = {
        "model": calib.model,
        "image_size": None if size is None else [int(size[0]), int(size[1])],
        "K": calib.K,
        "dist": calib.dist,
        "rms_px": calib.rms,
        "per_view_rms_px": calib.per_view,
        "std_intrinsics": intrinsic_std(calib),
        "std_intrinsics_opencv": np.asarray(calib.std_intr).ravel()[:p],
        "sigma2": calib.sigma2,
        "cov": calib.cov,
    }
    if getattr(calib, "new_obj", None) is not None:
        rec["refined_board_points"] = calib.new_obj
    if unc is not None:
        rec["uncertainty"] = {
            "grid": list(unc.grid),
            "sigma_px": np.round(np.asarray(unc.sigma_px, np.float64), 3),
            "U95_px": unc.U95,
            "p_exceed": unc.p_exceed,
        }
    rec["trace"] = _trace_lines(trace)
    rec["meta"] = dict(meta or {})
    rec["disclaimer"] = DISCLAIMER
    return json.dumps(_num(rec), separators=(",", ":")).encode("utf-8")


def export_calibration(
    calib: CalibResult,
    unc: UncMap | None,
    trace: Sequence[str] | str | None = None,
    *,
    size: tuple[int, int] | None = None,
    meta: Mapping | None = None,
) -> dict[str, bytes]:
    """``{"yaml", "png", "json"}`` bytes for one calibration."""
    return {
        "yaml": calibration_yaml(calib, unc, size, meta),
        "png": report_card_png(calib, unc, trace, meta),
        "json": calibration_json(calib, unc, trace, size, meta),
    }
