"""Board renderer: planar homography onto a pinhole canvas, then a distortion remap (spec 6.2).

The board texture is ``board.generateImage`` at 2500 px/m with a 40 px white margin. Texture
pixel centres sit at integer coordinates, so the board edge lies half a pixel inside the first
board pixel: board x = (u - M + 0.5) / PPM. Without the +0.5 the detected corners are biased by
about -0.27 px in both axes.

The undistorted intermediate image is drawn with a virtual camera K_v on a canvas that covers
every output pixel's ray (wide lenses need more than the output frame). The output image is then
``remap``-ed from that canvas with per-camera maps computed once and cached.

Anti-aliasing: when the board is minified, the texture is first blurred with a Gaussian of
``AA_SIGMA_OUT_PX`` output pixels (measured at the board centre). Point-sampling the hard-edged
texture otherwise leaves staircase edges that roughly double the corner RMS and, under heavy
JPEG, produce multi-pixel corner outliers.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from ..board import BOARD, BOARD_SIZE_M, SQUARE_M, SQUARES
from .lens import FISHEYE_MAX_THETA_DEG, GTCamera

__all__ = [
    "PPM",
    "TEX_MARGIN_PX",
    "INK",
    "PAPER",
    "BACKGROUND",
    "AA_SIGMA_OUT_PX",
    "MAX_CANVAS_SCALE",
    "board_texture",
    "texture_to_board",
    "board_to_texture",
    "DistortionMaps",
    "distortion_maps",
    "Renderer",
]

PPM = 2500.0  # texture px per board metre
TEX_MARGIN_PX = 40
INK = 20.0  # rendered level of black squares/markers
PAPER = 224.0  # rendered level of white paper
BACKGROUND = 90.0
AA_SIGMA_OUT_PX = 0.5
MAX_CANVAS_SCALE = 3  # canvas <= 3W x 3H
CANVAS_PAD_PX = 4
_MAP_CACHE_SIZE = 8


@lru_cache(maxsize=1)
def _texture_u8() -> np.ndarray:
    sx, sy = SQUARES
    w = int(round(sx * SQUARE_M * PPM))
    h = int(round(sy * SQUARE_M * PPM))
    if not (math.isclose(w, sx * SQUARE_M * PPM) and math.isclose(h, sy * SQUARE_M * PPM)):
        raise ValueError("PPM must give an integer texture size for the board")
    img = BOARD.generateImage((w, h), marginSize=0, borderBits=1)
    m = TEX_MARGIN_PX
    return cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=255)


def board_texture() -> np.ndarray:
    """A fresh float32 copy of the board texture at the INK / PAPER levels."""
    t = _texture_u8().astype(np.float32)
    return INK + t * np.float32((PAPER - INK) / 255.0)


def texture_to_board() -> np.ndarray:
    """3x3 matrix mapping texture pixel (u, v, 1) to board-plane metres (x, y, 1)."""
    off = (-TEX_MARGIN_PX + 0.5) / PPM
    return np.array([[1.0 / PPM, 0.0, off], [0.0, 1.0 / PPM, off], [0.0, 0.0, 1.0]])


def board_to_texture(xy_m: np.ndarray) -> np.ndarray:
    """Board-plane metres (N, 2) -> texture pixel coordinates (N, 2)."""
    xy = np.asarray(xy_m, dtype=np.float64).reshape(-1, 2)
    return xy * PPM + (TEX_MARGIN_PX - 0.5)


@dataclass(frozen=True, eq=False)
class DistortionMaps:
    """Output pixel -> canvas pixel maps for one camera."""

    map_x: np.ndarray  # (H, W) float32
    map_y: np.ndarray  # (H, W) float32
    K_v: np.ndarray  # (3, 3) virtual pinhole camera of the canvas
    canvas_size: tuple[int, int]  # (Wc, Hc)
    valid: np.ndarray  # (H, W) bool, False where the ray is not rendered (fisheye beyond 75 deg)
    scale: float  # K_v focal / K focal; 1.0 unless the canvas cap forced a shrink


_maps_cache: "OrderedDict[tuple, DistortionMaps]" = OrderedDict()


def _build_maps(cam: GTCamera) -> DistortionMaps:
    W, H = cam.size
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    uv = np.stack([xs.ravel(), ys.ravel()], 1)
    n = cam.pixel_to_normalized(uv)
    valid = np.all(np.isfinite(n), axis=1)
    if cam.model == "fisheye":
        valid &= np.degrees(np.arctan(np.hypot(n[:, 0], n[:, 1]))) <= FISHEYE_MAX_THETA_DEG + 1e-9
    nv = n[valid]
    xmin, ymin = nv.min(0)
    xmax, ymax = nv.max(0)
    pad = CANVAS_PAD_PX
    s = min(
        1.0,
        (MAX_CANVAS_SCALE * W - 2 * pad - 1) / ((xmax - xmin) * cam.fx),
        (MAX_CANVAS_SCALE * H - 2 * pad - 1) / ((ymax - ymin) * cam.fy),
    )
    fxv, fyv = s * cam.fx, s * cam.fy
    cxv, cyv = pad - xmin * fxv, pad - ymin * fyv
    Wc = int(math.ceil((xmax - xmin) * fxv + 2 * pad)) + 1
    Hc = int(math.ceil((ymax - ymin) * fyv + 2 * pad)) + 1
    K_v = np.array([[fxv, 0.0, cxv], [0.0, fyv, cyv], [0.0, 0.0, 1.0]])
    mx = np.full(len(uv), -1.0e4, np.float64)
    my = np.full(len(uv), -1.0e4, np.float64)
    mx[valid] = fxv * nv[:, 0] + cxv
    my[valid] = fyv * nv[:, 1] + cyv
    return DistortionMaps(
        map_x=mx.reshape(H, W).astype(np.float32),
        map_y=my.reshape(H, W).astype(np.float32),
        K_v=K_v,
        canvas_size=(Wc, Hc),
        valid=valid.reshape(H, W),
        scale=float(s),
    )


def distortion_maps(cam: GTCamera) -> DistortionMaps:
    """Cached maps for ``cam`` (about 0.7 s to build at 1280x720)."""
    key = cam.key()
    maps = _maps_cache.get(key)
    if maps is None:
        maps = _build_maps(cam)
        _maps_cache[key] = maps
        while len(_maps_cache) > _MAP_CACHE_SIZE:
            _maps_cache.popitem(last=False)
    else:
        _maps_cache.move_to_end(key)
    return maps


class Renderer:
    """Renders the board texture for a GT camera at a board pose (rvec, tvec: board -> camera)."""

    def __init__(self, cam: GTCamera):
        self.cam = cam
        self.maps = distortion_maps(cam)

    def homography(self, rvec, tvec) -> np.ndarray:
        """Texture pixel -> canvas pixel homography for the pose."""
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        return self.maps.K_v @ np.column_stack([R[:, 0], R[:, 1], t]) @ texture_to_board()

    def minification(self, rvec, tvec) -> float:
        """Texture pixels per output pixel at the board centre (geometric mean of the two axes)."""
        c = np.array([BOARD_SIZE_M[0] / 2.0, BOARD_SIZE_M[1] / 2.0, 0.0])
        h = 1e-3
        p = self.cam.project(np.array([c, c + [h, 0.0, 0.0], c + [0.0, h, 0.0]]), rvec, tvec)
        J = np.column_stack([(p[1] - p[0]) / h, (p[2] - p[0]) / h])  # output px per board metre
        det = abs(float(np.linalg.det(J)))
        if not np.isfinite(det) or det <= 0.0:
            return 1.0
        return PPM / math.sqrt(det)

    def render(self, rvec, tvec, texture: np.ndarray | None = None,
               background: float = BACKGROUND, antialias: bool = True) -> np.ndarray:
        """float32 (H, W) image of the board; pixels off the board get ``background``.

        The board must lie entirely in front of the camera (all corners at z > 0).
        """
        tex = board_texture() if texture is None else texture
        if tex.dtype != np.float32:
            tex = tex.astype(np.float32)
        if antialias:
            m = self.minification(rvec, tvec)
            if m > 1.0:
                tex = cv2.GaussianBlur(tex, (0, 0), AA_SIGMA_OUT_PX * m)
        Hm = self.homography(rvec, tvec)
        Wc, Hc = self.maps.canvas_size
        canvas = self._warp_roi(tex, Hm, Wc, Hc, background)
        return cv2.remap(canvas, self.maps.map_x, self.maps.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=float(background))

    @staticmethod
    def _warp_roi(tex: np.ndarray, Hm: np.ndarray, Wc: int, Hc: int, background: float) -> np.ndarray:
        """warpPerspective restricted to the board's bounding box on the canvas."""
        th, tw = tex.shape[:2]
        quad = np.array([[-0.5, -0.5, 1.0], [tw - 0.5, -0.5, 1.0], [tw - 0.5, th - 0.5, 1.0], [-0.5, th - 0.5, 1.0]])
        q = quad @ Hm.T
        canvas = np.full((Hc, Wc), background, np.float32)
        if np.any(q[:, 2] <= 0):
            raise ValueError("board is not entirely in front of the camera")
        q = q[:, :2] / q[:, 2:]
        x0 = max(int(math.floor(q[:, 0].min())) - 2, 0)
        y0 = max(int(math.floor(q[:, 1].min())) - 2, 0)
        x1 = min(int(math.ceil(q[:, 0].max())) + 3, Wc)
        y1 = min(int(math.ceil(q[:, 1].max())) + 3, Hc)
        if x1 <= x0 or y1 <= y0:
            return canvas
        T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
        roi = cv2.warpPerspective(tex, T @ Hm, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=float(background))
        canvas[y0:y1, x0:x1] = roi
        return canvas
