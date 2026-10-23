"""Loaders for config/thresholds.json, config/presets.json and config/stop_calibration.json.

The config directory is ``$NEXTVIEW_CONFIG_DIR`` when set, otherwise ``<repo>/config``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

__all__ = [
    "config_dir",
    "Thresholds",
    "Preset",
    "Presets",
    "StopCalibration",
    "load_thresholds",
    "load_presets",
    "load_stop_calibration",
    "LENS_FAMILY_IDS",
    "ACTUATOR_IDS",
]

LENS_FAMILY_IDS = ("normal", "wide", "tele", "webcam", "fisheye130")
ACTUATOR_IDS = ("robot", "human")
STOP_QUANTILE_LEVELS = 101


def config_dir() -> Path:
    env = os.environ.get("NEXTVIEW_CONFIG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config"


def _read_json(path: Path | str | None, name: str) -> dict:
    p = Path(path) if path is not None else config_dir() / name
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Thresholds:
    min_corners: int
    min_board_area_frac: float
    max_edge_rise_px: float
    motion_anisotropy: float
    max_saturation_frac: float
    ippe_ambiguity_ratio: float
    outlier_mad_k: float
    outlier_min_abs_px: float
    underfit_p: float
    model_switch_min_views: int
    model_switch_cv_gain: float
    warp_min_full_views: int
    warp_explained_var: float
    warp_cv_gain: float
    drift_factor: float
    drift_consecutive: int
    tau_px: float
    stop_p_exceed: float
    min_views_stop: int
    prequential_ratio: float
    budget_frames: int
    max_attempts: int
    nbv_candidates: int
    nbv_time_budget_s: float
    effort_lambda_human: float
    max_tilt_deg: float
    grid_uncertainty: tuple[int, int]
    grid_coverage: tuple[int, int]
    out_of_scope_hfov_deg: float
    # Keys present in the JSON but not declared above; readable as attributes too.
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __getattr__(self, name: str) -> Any:
        if name == "extra" or name.startswith("__"):
            raise AttributeError(name)
        try:
            return self.extra[name]
        except KeyError:
            raise AttributeError(f"Thresholds has no key {name!r}") from None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Thresholds":
        declared = {f.name: f for f in fields(cls) if f.name != "extra"}
        values = {k: v for k, v in data.items() if not k.startswith("_")}
        missing = sorted(set(declared) - set(values))
        if missing:
            raise ValueError(f"thresholds.json is missing keys: {missing}")
        kwargs: dict[str, Any] = {}
        for name in declared:
            v = values[name]
            if name.startswith("grid_"):
                v = tuple(int(x) for x in v)
                if len(v) != 2:
                    raise ValueError(f"{name} must have 2 entries, got {v}")
            kwargs[name] = v
        extra = {k: v for k, v in values.items() if k not in declared}
        return cls(**kwargs, extra=MappingProxyType(extra))

    def to_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        for k in ("grid_uncertainty", "grid_coverage"):
            out[k] = list(out[k])
        out.update(self.extra)
        return out

    def replace(self, **changes: Any) -> "Thresholds":
        d = self.to_dict()
        d.update(changes)
        return Thresholds.from_dict(d)

    def __reduce__(self):
        return (Thresholds.from_dict, (self.to_dict(),))


def load_thresholds(path: Path | str | None = None) -> Thresholds:
    return Thresholds.from_dict(_read_json(path, "thresholds.json"))


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    family: str
    actuator: str
    nuisances: tuple[str, ...] = ()
    failure_demo: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "family": self.family,
            "actuator": self.actuator,
            "nuisances": list(self.nuisances),
            "failure_demo": self.failure_demo,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Presets:
    items: tuple[Preset, ...]
    default_id: str

    def __post_init__(self) -> None:
        ids = [p.id for p in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate preset ids: {ids}")
        if self.default_id not in ids:
            raise ValueError(f"default preset {self.default_id!r} not in {ids}")
        for p in self.items:
            if p.family not in LENS_FAMILY_IDS:
                raise ValueError(f"preset {p.id}: unknown family {p.family!r}")
            if p.actuator not in ACTUATOR_IDS:
                raise ValueError(f"preset {p.id}: unknown actuator {p.actuator!r}")

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, preset_id: object) -> bool:
        return any(p.id == preset_id for p in self.items)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.items)

    @property
    def default(self) -> Preset:
        return self.get(self.default_id)

    def get(self, preset_id: str) -> Preset:
        for p in self.items:
            if p.id == preset_id:
                return p
        raise KeyError(f"unknown preset {preset_id!r}; known: {list(self.ids)}")


def load_presets(path: Path | str | None = None) -> Presets:
    data = _read_json(path, "presets.json")
    items = tuple(
        Preset(
            id=p["id"],
            label=p["label"],
            family=p["family"],
            actuator=p["actuator"],
            nuisances=tuple(p.get("nuisances", ())),
            failure_demo=bool(p.get("failure_demo", False)),
            notes=p.get("notes", ""),
        )
        for p in data["presets"]
    )
    return Presets(items=items, default_id=data["default"])


@dataclass(frozen=True)
class StopCalibration:
    """Empirical quantiles of r = true_p95 / U95 per actuator (spec section 2.6).

    ``quantiles[actuator]`` holds 101 non-decreasing values at levels 0.00, 0.01, ..., 1.00.
    P_exceed(U95) = 1 - F_r(tau / U95).
    """

    calibrated: bool
    quantiles: Mapping[str, tuple[float, ...]]
    note: str = ""

    def p_exceed(self, u95: float, actuator: str, tau_px: float) -> float:
        """Probability that the true field p95 exceeds ``tau_px``.

        Returns NaN while uncalibrated, so a ``p_exceed < threshold`` stop test never passes.
        """
        q = self.quantiles.get(actuator)
        if not self.calibrated or q is None:
            return math.nan
        if not math.isfinite(u95):
            return 1.0
        if u95 <= 0.0:
            return 0.0
        levels = np.linspace(0.0, 1.0, len(q))
        cdf = float(np.interp(tau_px / u95, np.asarray(q, dtype=float), levels, left=0.0, right=1.0))
        return 1.0 - cdf


def load_stop_calibration(path: Path | str | None = None) -> StopCalibration:
    data = _read_json(path, "stop_calibration.json")
    calibrated = bool(data.get("calibrated", False))
    quantiles: dict[str, tuple[float, ...]] = {}
    for actuator, q in (data.get("quantiles") or {}).items():
        if q is None:
            continue
        arr = np.asarray(q, dtype=float)
        if arr.shape != (STOP_QUANTILE_LEVELS,):
            raise ValueError(f"stop calibration for {actuator!r}: expected {STOP_QUANTILE_LEVELS} quantiles, got {arr.shape}")
        if np.any(np.diff(arr) < 0) or not np.all(np.isfinite(arr)):
            raise ValueError(f"stop calibration for {actuator!r}: quantiles must be finite and non-decreasing")
        quantiles[actuator] = tuple(float(x) for x in arr)
    if calibrated and not quantiles:
        raise ValueError("stop_calibration.json says calibrated but holds no quantiles")
    return StopCalibration(calibrated=calibrated, quantiles=quantiles, note=data.get("note", ""))
