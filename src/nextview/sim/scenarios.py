"""UI / MCP presets -> (lens family, actuator, nuisances) -> a ready Simulator (spec 6.5)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..config import Preset, Presets, load_presets
from .lens import GTCamera, make_camera
from .nuisance import NuisanceConfig
from .simulator import Simulator

__all__ = ["Scenario", "make_simulator", "scenario_from_preset", "scenario_table"]


@dataclass(frozen=True, eq=False)
class Scenario:
    preset_id: str | None
    label: str
    family: str
    actuator: str
    nuisances: tuple[str, ...]
    camera: GTCamera
    simulator: Simulator

    def to_dict(self) -> dict:
        return {
            "preset_id": self.preset_id,
            "label": self.label,
            "family": self.family,
            "actuator": self.actuator,
            "nuisances": list(self.nuisances),
            "camera": self.camera.to_dict(),
        }


def make_simulator(family: str, actuator: str, camera_seed: int, seed: int = 0, nuisances=(),
                   clean: bool = False, **overrides) -> Simulator:
    """Simulator for camera ``camera_seed`` of ``family``.

    ``nuisances`` names preset extras ("glare", "autofocus_drift"); ``clean=True`` turns off noise,
    JPEG, occluders and corruption (geometry checks); ``overrides`` set NuisanceConfig fields.
    """
    cam = make_camera(family, camera_seed)
    cfg = NuisanceConfig.for_camera(cam, nuisances, **overrides)
    if clean:
        cfg = replace(NuisanceConfig.clean(), glare=cfg.glare, autofocus_drift=cfg.autofocus_drift,
                      names=cfg.names, **overrides)
    return Simulator(cam, actuator, cfg, seed=seed, camera_seed=camera_seed)


def scenario_from_preset(preset: str | Preset, camera_seed: int = 0, seed: int = 0,
                         presets: Presets | None = None, actuator: str | None = None) -> Scenario:
    """Scenario for a preset; ``actuator`` ("robot" | "human") overrides the preset's own."""
    if isinstance(preset, str):
        preset = (presets or load_presets()).get(preset)
    act = preset.actuator if actuator is None else actuator
    sim = make_simulator(preset.family, act, camera_seed, seed, preset.nuisances)
    return Scenario(preset.id, preset.label, preset.family, act, tuple(preset.nuisances), sim.camera, sim)


def scenario_table(presets: Presets | None = None) -> list[dict]:
    """Preset id, label, family, actuator and nuisances for every preset (no cameras built)."""
    ps = presets or load_presets()
    return [{"id": p.id, "label": p.label, "family": p.family, "actuator": p.actuator,
             "nuisances": list(p.nuisances), "failure_demo": p.failure_demo, "default": p.id == ps.default_id}
            for p in ps]
