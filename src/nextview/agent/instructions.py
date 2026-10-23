"""Operator-facing wording: where the board should go next, and why a frame was rejected.

``describe_target`` turns a target pose into a short numeric instruction (spec 2.8, graft G2):
position word (one of 9 image cells), "board ≈ {s} % of frame width", "tilt ~{θ}° ({edge} away)",
the roll when it exceeds 15°, and in simulation the distance to the camera.
"""

from __future__ import annotations

import math

import numpy as np

from ..board import BOARD_SIZE_M
from ..geometry import BOARD_CENTRE_M, as_rotation, board_centre_px, roll_of, tilt_azimuth_of, tilt_of

__all__ = [
    "position_word",
    "edge_away",
    "size_fraction",
    "describe_target",
    "short_desc",
    "describe_reject",
    "glare_tilt",
    "REJECT_TEMPLATES",
]

_POS = (("top-left", "top", "top-right"), ("left", "centre", "right"), ("bottom-left", "bottom", "bottom-right"))
_EDGES = ("right edge", "bottom-right corner", "bottom edge", "bottom-left corner",
          "left edge", "top-left corner", "top edge", "top-right corner")

REJECT_TEMPLATES = {
    "R0_NO_BOARD": "Board not found — bring the whole board into view{closer}",
    "R1_SMALL": "Move closer: board should fill ≥ 1/4 of the frame width",
    "R2_BLUR_MOTION": "Hold still — motion blur along ~{dir}° (edge rise {rise} px)",
    "R2_BLUR_SOFT": "Image is soft ({rise} px) — hold still and check focus / move back slightly",
    "R3_GLARE": "Reflection on the board — tilt ~15° about the {axis} axis ({edge} away) or dim the screen/light",
    "R4_AMBIGUOUS": "Pose ambiguous (IPPE ratio {ratio}) — next view needs a steeper tilt",
    "R5_OFF_TARGET": "Off target by {delta}; re-planning from the new view",
}


def position_word(px, size) -> str:
    """One of 9 cells (3 x 3) of the image."""
    W, H = size
    x, y = float(px[0]), float(px[1])
    col = 0 if x < W / 3 else (2 if x >= 2 * W / 3 else 1)
    row = 0 if y < H / 3 else (2 if y >= 2 * H / 3 else 1)
    return _POS[row][col]


def edge_away(azimuth_deg: float) -> str:
    """Name of the board edge (or corner) that points away from the camera."""
    return _EDGES[int(round(float(azimuth_deg) / 45.0)) % 8]


def size_fraction(rvec, tvec, K, size) -> float:
    """Board width as a fraction of the image width at the board-centre depth: fx * bw / (Z * W)."""
    R = as_rotation(rvec)
    Z = float((R @ BOARD_CENTRE_M + np.asarray(tvec, float).ravel())[2])
    if Z <= 0:
        return math.nan
    return float(np.asarray(K)[0, 0]) * BOARD_SIZE_M[0] / (Z * float(size[0]))


def _round_to(x: float, step: float) -> int:
    return int(step * round(x / step))


def describe_target(rvec, tvec, K, dist, size, *, sim: bool = False) -> str:
    """E.g. "Board to bottom-left, ≈45% of frame width, tilt ~35° (left edge away), ≈38 cm"."""
    c = board_centre_px(rvec, tvec, K, dist)
    pos = position_word(c, size) if np.all(np.isfinite(c)) else "centre"
    head = "Board at the centre" if pos == "centre" else f"Board to {pos}"
    parts = [head]
    s = size_fraction(rvec, tvec, K, size)
    if math.isfinite(s):
        parts.append(f"≈{max(5, _round_to(100 * s, 5))}% of frame width")
    tilt = tilt_of(rvec)
    if tilt < 7.5:
        parts.append("facing the camera")
    else:
        parts.append(f"tilt ~{_round_to(tilt, 5)}° ({edge_away(tilt_azimuth_of(rvec))} away)")
    roll = roll_of(rvec)
    if abs(roll) > 15:
        parts.append(f"rotated ~{_round_to(abs(roll), 5)}° {'clockwise' if roll > 0 else 'counter-clockwise'}")
    if sim:  # depth Z of the board centre (spec 2.8)
        R = as_rotation(rvec)
        Z = float((R @ BOARD_CENTRE_M + np.asarray(tvec, float).ravel())[2])
        parts.append(f"≈{int(round(100 * Z))} cm")
    return ", ".join(parts)


def short_desc(rvec, tvec, K, dist, size) -> str:
    """Candidate-table label, e.g. "bottom-left, tilt 35° az 180°"."""
    c = board_centre_px(rvec, tvec, K, dist)
    pos = position_word(c, size) if np.all(np.isfinite(c)) else "centre"
    tilt = tilt_of(rvec)
    if tilt < 7.5:
        return f"{pos}, frontal"
    return f"{pos}, tilt {_round_to(tilt, 5)}° az {_round_to(tilt_azimuth_of(rvec), 45) % 360}°"


def glare_tilt(offset) -> tuple[str, float]:
    """(axis, azimuth of the edge to send away) for a glare hotspot at image ``offset`` from the board centre.

    The tilt axis is perpendicular to the offset (a sideways offset needs a tilt about the vertical
    axis). The edge opposite the hotspot goes away, so the hotspot's side comes towards the camera
    and the specular point slides further out along the offset, off the squares. Checked against
    the simulator's lamp model (validation only, the agent never uses it): this direction clears
    the hotspot in 326 of 373 webcam poses, the opposite direction in 29.
    """
    dx, dy = float(offset[0]), float(offset[1])
    if abs(dx) >= abs(dy):
        return "vertical", (180.0 if dx > 0 else 0.0)
    return "horizontal", (270.0 if dy > 0 else 90.0)


def describe_reject(rule: str, **m) -> str:
    """Instruction for a gate rule. Keyword arguments carry the measurements:

    R0: ``closer`` (bool); R2: ``rise``, ``anisotropy``, ``blur_dir``, ``motion_anisotropy``;
    R3: ``glare_offset`` (dx, dy) of the hotspot from the board centre in px; R4: ``ratio``;
    R5: ``deg``, ``frac``.
    """
    if rule == "R0_NO_BOARD":
        return REJECT_TEMPLATES[rule].format(closer=", closer" if m.get("closer") else "")
    if rule == "R1_SMALL":
        return REJECT_TEMPLATES[rule]
    if rule == "R2_BLUR":
        rise = f"{float(m['rise']):.1f}"
        aniso = m.get("anisotropy")
        bdir = m.get("blur_dir")
        if aniso is not None and math.isfinite(aniso) and aniso >= m.get("motion_anisotropy", 1.5) and bdir is not None:
            return REJECT_TEMPLATES["R2_BLUR_MOTION"].format(dir=int(round(float(bdir))), rise=rise)
        return REJECT_TEMPLATES["R2_BLUR_SOFT"].format(rise=rise)
    if rule == "R3_GLARE":
        axis, az = glare_tilt(m.get("glare_offset") or (0.0, 0.0))
        return REJECT_TEMPLATES[rule].format(axis=axis, edge=edge_away(az))
    if rule == "R4_AMBIGUOUS":
        return REJECT_TEMPLATES[rule].format(ratio=f"{float(m['ratio']):.2f}")
    if rule == "R5_OFF_TARGET":
        return REJECT_TEMPLATES[rule].format(delta=f"{float(m['deg']):.0f}° / {100 * float(m['frac']):.0f}% of the width")
    raise ValueError(f"no instruction template for {rule!r}")
