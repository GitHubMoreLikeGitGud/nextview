"""Calibration-board simulator: GT lens families, renderer, nuisances, actuators and truth metrics."""

from .actuator import Actuation, HumanActuator, RobotActuator, make_actuator
from .lens import IMAGE_SIZE, LENS_FAMILIES, GTCamera, make_camera, sample_camera
from .nuisance import NuisanceConfig, apply_corruption
from .poses import pose_for_view, random_pose, visible_fraction
from .render import Renderer, distortion_maps
from .scenarios import Scenario, make_simulator, scenario_from_preset, scenario_table
from .simulator import FrameTruth, SimFrame, Simulator
from .truth import field_error, field_error_vectors, fx_rel_err, grid_centres, intrinsics_error, planar_mm_error
from .warp import sag_object_points, warped_observation

__all__ = [
    "Actuation",
    "HumanActuator",
    "RobotActuator",
    "make_actuator",
    "IMAGE_SIZE",
    "LENS_FAMILIES",
    "GTCamera",
    "make_camera",
    "sample_camera",
    "NuisanceConfig",
    "apply_corruption",
    "pose_for_view",
    "random_pose",
    "visible_fraction",
    "Renderer",
    "distortion_maps",
    "Scenario",
    "make_simulator",
    "scenario_from_preset",
    "scenario_table",
    "FrameTruth",
    "SimFrame",
    "Simulator",
    "field_error",
    "field_error_vectors",
    "fx_rel_err",
    "grid_centres",
    "intrinsics_error",
    "planar_mm_error",
    "sag_object_points",
    "warped_observation",
]
