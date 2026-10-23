"""Slanted-edge blur measurement on the board's own square edges.

Method (one estimate per inner chessboard edge):

1. A local homography board -> image is fitted to the detected corners near the edge. It gives
   the across-edge scale (pixels per square measured along the image normal of the edge).
2. Every inner edge separates a black square from a white square, and the white square holds
   a marker. Only the central 60 % of each edge is used, between its two detected corners.
3. Pixels within 0.12 square widths of the edge line on the white side, and 0.4 square widths
   on the black side (the black square is uniform there), are binned by signed distance at
   0.25 px. Because the edges are slanted with respect to the pixel grid, this builds a
   super-resolved edge spread function (ESF) without interpolation.
4. A Gaussian edge is fitted by least squares: dark plateau, contrast, position and sigma.
   On the white side the marker's black border starts 0.133 squares from the edge; the model
   includes that second (falling) edge at its known distance, so heavy blur that reaches the
   marker border does not bias sigma. Rise (10-90 %) = 2.563 sigma.
5. Edges shorter than 8 px are skipped. Fewer than 8 valid edges gives ``edge_rise_px = nan``.
6. Edges are binned by image-normal angle (0/45/90/135 deg). Motion blur along a direction
   widens the edges whose normal points along it, and anisotropy = max / min bin rise.
   ``edge_rise_px`` is the largest per-direction median (bins with at least 3 edges), i.e. the
   blur in the worst direction. When anisotropy reaches the threshold, ``blur_dir_deg`` is the
   mean measured normal angle of the edges in the bin with the largest rise (not the bin
   centre, which is up to 22.5 deg off on a rolled board). The board has only two edge
   families, so a blur at angle a from the dominant family's normal cannot be told from one at
   -a; the reported direction is that family's normal, which can still be up to about 35 deg
   from the blur direction when anisotropy is just above 1.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..board import MARKER_M, SQUARE_M, SQUARES
from ..types import Detection

__all__ = [
    "RISE_PER_SIGMA",
    "GAP_FRAC",
    "BAND_FRAC",
    "DARK_BAND_FRAC",
    "DIR_BINS_DEG",
    "EdgeFit",
    "EsfResult",
    "erf",
    "norm_cdf",
    "inner_edges",
    "edge_profile",
    "fit_edge_profile",
    "measure_edge",
    "measure_edges",
    "axial_mean_deg",
]

RISE_PER_SIGMA = 2.0 * 1.2815515655446004  # 10-90 % rise of a Gaussian-blurred step
GAP_FRAC = (SQUARE_M - MARKER_M) / 2.0 / SQUARE_M  # white gap to the marker border, in squares
BAND_FRAC = 0.12  # white side of the edge, in squares
DARK_BAND_FRAC = 0.4  # black side of the edge, in squares
T_RANGE = (0.2, 0.8)
BIN_PX = 0.25
MIN_EDGE_PX = 8.0
MIN_EDGES = 8
MIN_EDGES_PER_DIR = 3
MIN_CONTRAST = 12.0  # grey levels
DIR_BINS_DEG = (0.0, 45.0, 90.0, 135.0)

_N_COLS = SQUARES[0] - 1
_N_ROWS = SQUARES[1] - 1


@dataclass(frozen=True)
class EdgeFit:
    ok: bool
    sigma: float
    rise: float
    mu: float  # edge offset from the corner-to-corner line, px (towards the light side)
    contrast: float  # fitted light minus dark level
    normal_deg: float  # image angle of the edge normal, [0, 180)
    length_px: float
    n_px: int


@dataclass(frozen=True)
class EsfResult:
    edge_rise_px: float
    rise_by_dir: tuple[float, float, float, float]
    anisotropy: float
    blur_dir_deg: float | None
    n_edges: int
    edges: tuple[EdgeFit, ...]


# --- special functions (numpy only; the Lambda image has no scipy) -------------------------

_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26, absolute error < 1.5e-7."""
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + _ERF_P * a)
    a1, a2, a3, a4, a5 = _ERF_A
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    return s * (1.0 - poly * np.exp(-a * a))


def norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(np.asarray(z, dtype=np.float64) / _SQRT2))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * z * z)


# --- board edges ------------------------------------------------------------------------------


def inner_edges() -> list[tuple[int, int]]:
    """ChArUco corner-id pairs bounding each inner square edge (both ends are inner corners)."""
    pairs = []
    for r in range(_N_ROWS):
        for c in range(_N_COLS - 1):
            pairs.append((r * _N_COLS + c, r * _N_COLS + c + 1))
    for r in range(_N_ROWS - 1):
        for c in range(_N_COLS):
            pairs.append((r * _N_COLS + c, (r + 1) * _N_COLS + c))
    return pairs


_EDGES = inner_edges()


def edge_profile(
    gray: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    band_px: float,
    t_range: tuple[float, float] = T_RANGE,
    bin_px: float = BIN_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binned ESF across the segment p0 -> p1.

    Signed distance is positive to the left of p0 -> p1 in image coordinates (normal
    ``(-u_y, u_x)``). Returns ``(bin_centres, mean_values, counts)`` for populated bins.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v = p1 - p0
    L = float(np.hypot(v[0], v[1]))
    u = v / L
    n = np.array([-u[1], u[0]])
    a, b = t_range
    quad = np.array([p0 + a * v + band_px * n, p0 + b * v + band_px * n, p0 + a * v - band_px * n, p0 + b * v - band_px * n])
    h, w = gray.shape[:2]
    x0 = max(int(math.floor(quad[:, 0].min())), 0)
    x1 = min(int(math.ceil(quad[:, 0].max())), w - 1)
    y0 = max(int(math.floor(quad[:, 1].min())), 0)
    y1 = min(int(math.ceil(quad[:, 1].max())), h - 1)
    if x1 < x0 or y1 < y0:
        empty = np.zeros(0)
        return empty, empty, empty
    ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    rx = xs - p0[0]
    ry = ys - p0[1]
    t = (rx * u[0] + ry * u[1]) / L
    d = rx * n[0] + ry * n[1]
    m = (t >= a) & (t <= b) & (np.abs(d) <= band_px)
    if not np.any(m):
        empty = np.zeros(0)
        return empty, empty, empty
    dv = d[m]
    vals = gray[y0 : y1 + 1, x0 : x1 + 1][m].astype(np.float64)
    nb = int(math.ceil(2 * band_px / bin_px)) + 1
    idx = np.clip(np.floor((dv + band_px) / bin_px).astype(np.int64), 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(np.float64)
    sm = np.bincount(idx, weights=vals, minlength=nb)
    keep = cnt > 0
    centres = -band_px + (np.arange(nb) + 0.5) * bin_px
    return centres[keep], sm[keep] / cnt[keep], cnt[keep]


def _model(params: np.ndarray, d: np.ndarray, gap: float | None):
    lo, amp, mu, s = params
    sigma = math.exp(s)
    z1 = (d - mu) / sigma
    shape = norm_cdf(z1)
    pdf1 = _norm_pdf(z1)
    d_mu = -pdf1 / sigma
    d_s = -z1 * pdf1
    if gap is not None:
        z2 = (d - mu - gap) / sigma
        pdf2 = _norm_pdf(z2)
        shape = shape - norm_cdf(z2)
        d_mu = d_mu + pdf2 / sigma
        d_s = d_s + z2 * pdf2
    f = lo + amp * shape
    J = np.column_stack([np.ones_like(d), shape, amp * d_mu, amp * d_s])
    return f, J


def fit_edge_profile(
    d: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    gap_px: float | None = None,
    max_iter: int = 40,
) -> tuple[float, float, float, float, bool]:
    """Weighted least-squares fit of a Gaussian edge (dark side at negative ``d``).

    Model: ``lo + amp * (Phi((d - mu) / sigma) - Phi((d - mu - gap) / sigma))``; the second
    term is dropped when ``gap_px`` is None. Returns ``(lo, amp, mu, sigma, converged)``.
    """
    d = np.asarray(d, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(d) if w is None else np.asarray(w, dtype=np.float64)
    if d.size < 6:
        return math.nan, math.nan, math.nan, math.nan, False
    span = float(d.max() - d.min())
    lo0 = float(np.average(y[d <= np.quantile(d, 0.2)]))
    hi0 = float(y[d >= 0].max()) if np.any(d >= 0) else float(y.max())
    amp0 = max(hi0 - lo0, 1e-3)
    level = (y - lo0) / amp0
    above = np.nonzero(level >= 0.5)[0]
    mu0 = float(d[above[0]]) if above.size else 0.0
    i25 = np.nonzero(level >= 0.25)[0]
    i75 = np.nonzero(level >= 0.75)[0]
    if i25.size and i75.size and d[i75[0]] > d[i25[0]]:
        sig0 = (d[i75[0]] - d[i25[0]]) / 1.349
    else:
        sig0 = 0.5
    sig0 = float(np.clip(sig0, 0.2, max(span / 3.0, 0.3)))
    params = np.array([lo0, amp0, mu0, math.log(sig0)])
    sw = np.sqrt(w)
    f, J = _model(params, d, gap_px)
    r = (y - f) * sw
    cost = float(r @ r)
    lam = 1e-3
    converged = False
    rel = math.inf
    for _ in range(max_iter):
        Jw = J * sw[:, None]
        A = Jw.T @ Jw
        g = Jw.T @ r
        step_ok = False
        for _ in range(8):
            Ad = A + lam * np.diag(np.diag(A) + 1e-12)
            try:
                delta = np.linalg.solve(Ad, g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            cand = params + delta
            cand[3] = float(np.clip(cand[3], math.log(0.02), math.log(max(span, 0.1))))
            f_c, J_c = _model(cand, d, gap_px)
            r_c = (y - f_c) * sw
            cost_c = float(r_c @ r_c)
            if cost_c <= cost:
                step_ok = True
                rel = (cost - cost_c) / max(cost, 1e-12)
                params, f, J, r, cost = cand, f_c, J_c, r_c, cost_c
                lam = max(lam / 3.0, 1e-9)
                break
            lam *= 4.0
        if not step_ok:
            converged = True
            break
        if rel < 1e-9 and np.max(np.abs(delta[2:])) < 1e-6:
            converged = True
            break
    else:
        converged = rel < 1e-4
    lo, amp, mu, s = params
    return float(lo), float(amp), float(mu), float(math.exp(s)), converged


def _bad_edge(normal_deg: float, length: float, n_px: int = 0) -> EdgeFit:
    return EdgeFit(False, math.nan, math.nan, math.nan, math.nan, normal_deg, length, n_px)


def measure_edge(
    gray: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    across_px_per_square: float,
    band_frac: float = BAND_FRAC,
    dark_band_frac: float = DARK_BAND_FRAC,
    gap_frac: float | None = GAP_FRAC,
) -> EdgeFit:
    """ESF fit for one square edge between image points p0 and p1.

    ``across_px_per_square`` is the board-square size in pixels measured along the edge's
    image normal; it sets the sampling band and the distance to the marker border.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v = p1 - p0
    L = float(np.hypot(v[0], v[1]))
    if L < 1e-9:
        return _bad_edge(math.nan, L)
    n = np.array([-v[1], v[0]]) / L
    normal_deg = math.degrees(math.atan2(n[1], n[0])) % 180.0
    if L < MIN_EDGE_PX or not np.isfinite(across_px_per_square) or across_px_per_square <= 0:
        return _bad_edge(normal_deg, L)
    band = band_frac * across_px_per_square
    dark_band = max(dark_band_frac, band_frac) * across_px_per_square
    d, y, cnt = edge_profile(gray, p0, p1, dark_band)
    if d.size < 8:
        return _bad_edge(normal_deg, L, int(cnt.sum()) if cnt.size else 0)
    neg = (d < 0) & (d >= -band)
    pos = (d > 0) & (d <= band)
    if not (np.any(neg) and np.any(pos)):
        return _bad_edge(normal_deg, L, int(cnt.sum()))
    if np.average(y[neg], weights=cnt[neg]) > np.average(y[pos], weights=cnt[pos]):
        d, y, cnt = -d[::-1], y[::-1], cnt[::-1]
    keep = d <= band
    d, y, cnt = d[keep], y[keep], cnt[keep]
    gap = None if gap_frac is None else gap_frac * across_px_per_square
    lo, amp, mu, sigma, conv = fit_edge_profile(d, y, cnt, gap)
    ok = bool(
        conv
        and np.isfinite(sigma)
        and amp >= MIN_CONTRAST
        and 0.02 < sigma <= 0.5 * dark_band
        and abs(mu) <= 0.5 * band
    )
    return EdgeFit(ok, sigma, RISE_PER_SIGMA * sigma, mu, amp, normal_deg, L, int(cnt.sum()))


def _across_scale(H: np.ndarray, mid_m: np.ndarray, normal_m: np.ndarray, n_img: np.ndarray) -> float:
    pts = np.array([mid_m - 0.5 * SQUARE_M * normal_m, mid_m + 0.5 * SQUARE_M * normal_m], dtype=np.float64)
    q = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    return abs(float(np.dot(q[1] - q[0], n_img)))


def _homography(obj_xy: np.ndarray, img: np.ndarray) -> np.ndarray | None:
    if len(obj_xy) < 4:
        return None
    H, _ = cv2.findHomography(obj_xy.astype(np.float64), img.astype(np.float64), 0)
    return H


def _dir_bin(normal_deg: float) -> int:
    return int(round(normal_deg / 45.0)) % 4


def measure_edges(gray: np.ndarray, det: Detection, anisotropy_threshold: float = 1.5) -> EsfResult:
    """Board-level ESF summary for one frame (see the module docstring)."""
    nan4 = (math.nan, math.nan, math.nan, math.nan)
    if not det.ok or len(det.ids) < 4:
        return EsfResult(math.nan, nan4, math.nan, None, 0, ())
    pos = {int(i): k for k, i in enumerate(det.ids)}
    obj_xy = det.obj[:, :2]
    H_global = _homography(obj_xy, det.corners)
    fits: list[EdgeFit] = []
    for a, b in _EDGES:
        ka, kb = pos.get(a), pos.get(b)
        if ka is None or kb is None:
            continue
        p0, p1 = det.corners[ka], det.corners[kb]
        v = p1 - p0
        L = float(np.hypot(v[0], v[1]))
        if L < MIN_EDGE_PX:
            continue
        n_img = np.array([-v[1], v[0]]) / L
        ma, mb = obj_xy[ka], obj_xy[kb]
        mid = 0.5 * (ma + mb)
        e = (mb - ma) / np.linalg.norm(mb - ma)
        normal_m = np.array([-e[1], e[0]])
        near = np.max(np.abs(obj_xy - mid), axis=1) <= 1.6 * SQUARE_M
        H = None
        if np.count_nonzero(near) >= 6:
            ids_near = det.ids[near]
            rows = np.unique(ids_near // _N_COLS).size
            cols = np.unique(ids_near % _N_COLS).size
            if rows >= 2 and cols >= 2:
                H = _homography(obj_xy[near], det.corners[near])
        if H is None:
            H = H_global
        if H is None:
            continue
        s_across = _across_scale(H, mid, normal_m, n_img)
        fits.append(measure_edge(gray, p0, p1, s_across))
    good = [f for f in fits if f.ok]
    if len(good) < MIN_EDGES:
        return EsfResult(math.nan, nan4, math.nan, None, len(good), tuple(fits))
    by_bin: list[list[float]] = [[], [], [], []]
    for f in good:
        by_bin[_dir_bin(f.normal_deg)].append(f.rise)
    rise_by_dir = tuple(float(np.median(r)) if r else math.nan for r in by_bin)
    qualified = [k for k in range(4) if len(by_bin[k]) >= MIN_EDGES_PER_DIR]
    if qualified:
        meds = np.array([rise_by_dir[k] for k in qualified])
        edge_rise = float(meds.max())
        anisotropy = float(meds.max() / meds.min()) if len(qualified) >= 2 else 1.0
        kmax = qualified[int(np.argmax(meds))]
    else:
        edge_rise = float(np.median([f.rise for f in good]))
        anisotropy = 1.0
        kmax = None
    blur_dir = None
    if kmax is not None and anisotropy >= anisotropy_threshold:
        blur_dir = axial_mean_deg([f.normal_deg for f in good if _dir_bin(f.normal_deg) == kmax])
    return EsfResult(edge_rise, rise_by_dir, anisotropy, blur_dir, len(good), tuple(fits))


def axial_mean_deg(angles_deg) -> float:
    """Mean of axial angles (theta and theta + 180 are the same direction), in [0, 180)."""
    a = np.radians(2.0 * np.asarray(angles_deg, dtype=np.float64))
    return float((math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) / 2.0) % 180.0)
