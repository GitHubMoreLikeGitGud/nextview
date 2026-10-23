"""Actuators that turn a target board pose into the pose actually realised (spec 6.4).

Robot: sigma_rot = 1 deg per axis, sigma_trans = 5 mm, no motion blur.
Human: sigma_rot = 6 deg per axis, sigma_trans = 3 cm x (Z / 0.5 m). With p = 0.5 the frame is
motion-blurred along the board-centre image displacement from the previous realised pose,
L = min(10, 1 + 0.04 |dc|) px; otherwise L ~ U(0, 1) px.

Rotation noise is applied about the board centre, so it does not move the centre; translation
noise moves the centre. Each ``execute`` call draws a fixed count of random numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .lens import GTCamera
from ..geometry import BOARD_CENTRE_M

__all__ = ["Actuation", "RobotActuator", "HumanActuator", "make_actuator", "ACTUATORS"]


@dataclass(frozen=True, eq=False)
class Actuation:
    rvec: np.ndarray  # realised pose
    tvec: np.ndarray
    target_rvec: np.ndarray
    target_tvec: np.ndarray
    blur_len_px: float
    blur_angle_deg: float
    rot_err_deg: float  # angle of the realised rotation relative to the target
    centre_err_m: float  # distance between realised and target board centres


def _perturb(target_rvec, target_tvec, rot_noise_rad: np.ndarray, centre_noise_m: np.ndarray):
    R_t, _ = cv2.Rodrigues(np.asarray(target_rvec, dtype=np.float64).reshape(3, 1))
    t_t = np.asarray(target_tvec, dtype=np.float64).reshape(3)
    c_t = R_t @ BOARD_CENTRE_M + t_t
    dR, _ = cv2.Rodrigues(rot_noise_rad.reshape(3, 1))
    R = dR @ R_t
    c = c_t + centre_noise_m
    t = c - R @ BOARD_CENTRE_M
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3), t, c_t, c


def _centre_px(cam: GTCamera, rvec, tvec) -> np.ndarray:
    return cam.project(BOARD_CENTRE_M.reshape(1, 3), rvec, tvec)[0]


class RobotActuator:
    kind = "robot"

    def __init__(self, sigma_rot_deg: float = 1.0, sigma_trans_m: float = 0.005):
        self.sigma_rot_deg = sigma_rot_deg
        self.sigma_trans_m = sigma_trans_m

    def execute(self, cam: GTCamera, target_rvec, target_tvec, rng: np.random.Generator,
                prev_rvec=None, prev_tvec=None) -> Actuation:
        z = rng.standard_normal(6)
        rvec, tvec, c_t, c = _perturb(target_rvec, target_tvec, z[:3] * math.radians(self.sigma_rot_deg),
                                      z[3:] * self.sigma_trans_m)
        return Actuation(rvec, tvec, np.asarray(target_rvec, float).reshape(3),
                         np.asarray(target_tvec, float).reshape(3), 0.0, 0.0,
                         float(np.degrees(np.linalg.norm(z[:3] * math.radians(self.sigma_rot_deg)))),
                         float(np.linalg.norm(c - c_t)))


class HumanActuator:
    kind = "human"

    def __init__(self, sigma_rot_deg: float = 6.0, sigma_trans_m_at_half_m: float = 0.03,
                 p_motion: float = 0.5, max_len_px: float = 10.0, base_len_px: float = 1.0,
                 len_per_px: float = 0.04):
        self.sigma_rot_deg = sigma_rot_deg
        self.sigma_trans_m_at_half_m = sigma_trans_m_at_half_m
        self.p_motion = p_motion
        self.max_len_px = max_len_px
        self.base_len_px = base_len_px
        self.len_per_px = len_per_px

    def execute(self, cam: GTCamera, target_rvec, target_tvec, rng: np.random.Generator,
                prev_rvec=None, prev_tvec=None) -> Actuation:
        z = rng.standard_normal(6)
        u = rng.uniform(0.0, 1.0, 3)  # motion-blur coin, small-blur length, fallback angle
        R_t, _ = cv2.Rodrigues(np.asarray(target_rvec, dtype=np.float64).reshape(3, 1))
        Z = float((R_t @ BOARD_CENTRE_M + np.asarray(target_tvec, float).reshape(3))[2])
        sig_t = self.sigma_trans_m_at_half_m * max(Z, 0.0) / 0.5
        rot = z[:3] * math.radians(self.sigma_rot_deg)
        rvec, tvec, c_t, c = _perturb(target_rvec, target_tvec, rot, z[3:] * sig_t)
        angle = 360.0 * float(u[2])
        if u[0] < self.p_motion:
            dc = 0.0
            if prev_rvec is not None and prev_tvec is not None:
                d = _centre_px(cam, rvec, tvec) - _centre_px(cam, prev_rvec, prev_tvec)
                if np.all(np.isfinite(d)):
                    dc = float(np.hypot(d[0], d[1]))
                    if dc > 1e-6:
                        angle = math.degrees(math.atan2(d[1], d[0])) % 360.0
            length = min(self.max_len_px, self.base_len_px + self.len_per_px * dc)
        else:
            length = float(u[1])
        return Actuation(rvec, tvec, np.asarray(target_rvec, float).reshape(3),
                         np.asarray(target_tvec, float).reshape(3), float(length), float(angle),
                         float(np.degrees(np.linalg.norm(rot))), float(np.linalg.norm(c - c_t)))


ACTUATORS = {"robot": RobotActuator, "human": HumanActuator}


def make_actuator(kind: str):
    try:
        return ACTUATORS[kind]()
    except KeyError:
        raise ValueError(f"unknown actuator {kind!r}; known: {list(ACTUATORS)}") from None
