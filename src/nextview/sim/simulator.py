"""Simulated capture: target pose -> actuator -> render -> nuisances -> 8-bit frame.

Captures are stateless: every random draw comes from a generator seeded by
(camera_seed, seed, attempt, stream), so any step can be rebuilt from the session state alone,
and two policies on the same (camera, seed) see the same actuator error and nuisances at the
same attempt index (common random numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from ..board import BOARD_SIZE_M
from ..geometry import BOARD_CENTRE_M
from .actuator import Actuation, make_actuator
from .lens import GTCamera
from .nuisance import (
    CorruptionSpec,
    GlareSpec,
    NuisanceConfig,
    OccluderSpec,
    add_glare,
    add_noise,
    apply_corruption,
    apply_psf,
    combine_kernels,
    defocus_kernel,
    defocus_radius,
    draw_occluder,
    glare_hotspot,
    jpeg_roundtrip,
    motion_kernel,
    sample_corruption,
    sample_glare,
    sample_occluder,
    to_uint8,
)
from .poses import visible_fraction
from .render import BACKGROUND, Renderer, board_texture

__all__ = ["FrameTruth", "SimFrame", "Simulator"]

STREAM_ACTUATOR = 1
STREAM_NUISANCE = 2


def _r(x, n=6):
    return None if x is None else np.round(np.asarray(x, dtype=float), n).tolist()


@dataclass(frozen=True, eq=False)
class FrameTruth:
    attempt: int
    camera: GTCamera  # GT camera for this frame (after any drift)
    drifted: bool
    actuation: Actuation
    defocus_px: float
    occluder: OccluderSpec
    glare: GlareSpec | None  # set only when a hotspot was drawn
    glare_centre_m: np.ndarray | None
    glare_px: np.ndarray | None
    corruption: CorruptionSpec
    board_depth_m: float
    visible_frac: float
    rendered: bool  # False when the realised board was not entirely in front of the camera

    @property
    def rvec(self) -> np.ndarray:
        return self.actuation.rvec

    @property
    def tvec(self) -> np.ndarray:
        return self.actuation.tvec

    @property
    def corrupted(self) -> bool:
        """The corruption draw fired for this frame. The view is only corrupted once the caller
        passes its detection through ``SimFrame.observe`` (or ``SimFrame.corrupt``)."""
        return self.corruption.active

    def to_dict(self) -> dict:
        a = self.actuation
        return {
            "attempt": self.attempt,
            "drifted": self.drifted,
            "K": _r(self.camera.K),
            "dist": _r(self.camera.dist, 8),
            "rvec": _r(a.rvec),
            "tvec": _r(a.tvec),
            "target_rvec": _r(a.target_rvec),
            "target_tvec": _r(a.target_tvec),
            "rot_err_deg": round(a.rot_err_deg, 4),
            "centre_err_m": round(a.centre_err_m, 5),
            "blur_len_px": round(a.blur_len_px, 4),
            "blur_angle_deg": round(a.blur_angle_deg, 3),
            "defocus_px": round(self.defocus_px, 4),
            "occluder": self.occluder.to_dict(),
            "glare": None if self.glare is None else {
                "amplitude": round(self.glare.amplitude, 2), "sigma_frac": round(self.glare.sigma_frac, 4),
                "centre_m": _r(self.glare_centre_m), "centre_px": _r(self.glare_px, 2)},
            "corrupted": self.corruption.active,
            "corruption": self.corruption.to_dict() if self.corruption.active else None,
            "board_depth_m": round(self.board_depth_m, 5),
            "visible_frac": round(self.visible_frac, 4),
            "rendered": self.rendered,
        }


@dataclass(frozen=True, eq=False)
class SimFrame:
    image: np.ndarray  # (H, W) uint8
    truth: FrameTruth

    def corrupt(self, ids, corners):
        """Apply this frame's post-detection corruption: returns (corners, moved_ids)."""
        return apply_corruption(ids, corners, self.truth.corruption)

    def observe(self, detect=None):
        """Detect the board and apply this frame's corruption: returns (Detection, moved_ids).

        This is how the agent loop and the benchmark turn a simulated frame into an observation,
        so that ``truth.corrupted`` (used for outlier recall) matches what the calibration saw.
        ``detect`` defaults to ``nextview.tools.detect.detect_board``.
        """
        if detect is None:
            from ..tools.detect import detect_board as detect
        det = detect(self.image)
        corners, moved = self.corrupt(det.ids, det.corners)
        if len(moved) == 0:
            return det, moved
        return replace(det, corners=corners), moved


class Simulator:
    """One simulated camera in one scenario.

    ``capture(target_rvec, target_tvec, attempt, n_accepted, prev_rvec, prev_tvec) -> SimFrame``:
    ``attempt`` indexes the random streams, ``n_accepted`` drives autofocus drift, and the
    previous realised pose sets the hand-held motion-blur direction. A session must therefore keep
    the attempt index, the accepted count and the previous realised pose in its state.

    Corruption (spec 6.3) acts on detected corners, not on the image: callers must detect through
    ``SimFrame.observe`` (or apply ``SimFrame.corrupt`` to their own detection) before using a frame.
    """

    def __init__(self, camera: GTCamera, actuator="robot", nuisances: NuisanceConfig | None = None,
                 seed: int = 0, camera_seed: int | None = None):
        self.camera = camera
        self.actuator = make_actuator(actuator) if isinstance(actuator, str) else actuator
        self.nuisances = nuisances if nuisances is not None else NuisanceConfig.for_camera(camera)
        self.seed = int(seed)
        cs = camera_seed if camera_seed is not None else camera.seed
        self.camera_seed = int(cs) if cs is not None else 0
        self._drifted = camera.drifted(self.nuisances.drift_focal_scale, self.nuisances.drift_dk1)
        self._renderers: dict[tuple, Renderer] = {}

    @property
    def actuator_kind(self) -> str:
        return getattr(self.actuator, "kind", type(self.actuator).__name__)

    def camera_at(self, n_accepted: int = 0) -> GTCamera:
        nz = self.nuisances
        if nz.autofocus_drift and n_accepted >= nz.drift_after_accepted:
            return self._drifted
        return self.camera

    def renderer(self, cam: GTCamera | None = None) -> Renderer:
        cam = cam or self.camera
        key = cam.key()
        r = self._renderers.get(key)
        if r is None:
            r = self._renderers[key] = Renderer(cam)
        return r

    def rng(self, attempt: int, stream: int) -> np.random.Generator:
        return np.random.default_rng([self.camera_seed, self.seed, int(attempt), int(stream)])

    def render_exact(self, rvec, tvec, n_accepted: int = 0) -> np.ndarray:
        """The board at exactly (rvec, tvec): no actuator error, no nuisances, 8-bit."""
        return to_uint8(self.renderer(self.camera_at(n_accepted)).render(rvec, tvec))

    def capture(self, target_rvec, target_tvec, attempt: int, n_accepted: int = 0,
                prev_rvec=None, prev_tvec=None) -> SimFrame:
        nz = self.nuisances
        cam = self.camera_at(n_accepted)
        act = self.actuator.execute(cam, target_rvec, target_tvec, self.rng(attempt, STREAM_ACTUATOR),
                                    prev_rvec, prev_tvec)
        rng = self.rng(attempt, STREAM_NUISANCE)
        occ = sample_occluder(rng, nz.occluder_p)
        glare_spec = sample_glare(rng)
        corr = sample_corruption(rng, nz.corruption_p)

        R, _ = cv2.Rodrigues(act.rvec.reshape(3, 1))
        depth = float((R @ BOARD_CENTRE_M + act.tvec)[2])
        outline = np.array([[-0.02, -0.02, 0], [BOARD_SIZE_M[0] + 0.02, -0.02, 0],
                            [BOARD_SIZE_M[0] + 0.02, BOARD_SIZE_M[1] + 0.02, 0], [-0.02, BOARD_SIZE_M[1] + 0.02, 0]])
        in_front = bool(np.min((outline @ R.T + act.tvec)[:, 2]) > 0.01)

        tex = board_texture()
        glare_used, glare_c, glare_px = None, None, None
        if nz.glare and in_front:
            glare_c = glare_hotspot(act.rvec, act.tvec)
            if glare_c is not None:
                tex = add_glare(tex, glare_c, glare_spec)
                glare_used = glare_spec
                glare_px = cam.project(np.array([[glare_c[0], glare_c[1], 0.0]]), act.rvec, act.tvec)[0]
        tex = draw_occluder(tex, occ)

        W, H = cam.size
        if in_front:
            img = self.renderer(cam).render(act.rvec, act.tvec, tex)
        else:
            img = np.full((H, W), BACKGROUND, np.float32)

        r_def = defocus_radius(depth, nz.z_near)
        kernel = combine_kernels(motion_kernel(act.blur_len_px, act.blur_angle_deg), defocus_kernel(r_def))
        img = apply_psf(img, kernel)
        img = add_noise(img, nz.noise_sigma, rng)
        img = jpeg_roundtrip(to_uint8(img), nz.jpeg_quality)

        vis = visible_fraction(cam, act.rvec, act.tvec, margin_px=0.0) if in_front else 0.0
        truth = FrameTruth(
            attempt=int(attempt), camera=cam, drifted=cam is self._drifted, actuation=act,
            defocus_px=r_def, occluder=occ, glare=glare_used, glare_centre_m=glare_c, glare_px=glare_px,
            corruption=corr, board_depth_m=depth, visible_frac=vis, rendered=in_front,
        )
        return SimFrame(img, truth)
