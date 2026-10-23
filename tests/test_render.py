import cv2
import numpy as np
import pytest

from nextview.board import BOARD, BOARD_SIZE_M, SQUARE_M, object_points
from nextview.sim.lens import make_camera
from nextview.sim.poses import pose_for_view, random_pose
from nextview.sim.render import (
    INK,
    PAPER,
    TEX_MARGIN_PX,
    Renderer,
    board_to_texture,
    distortion_maps,
    texture_to_board,
    _texture_u8,
)
from nextview.sim.scenarios import make_simulator

FAMILIES = ["normal", "wide", "tele", "webcam", "fisheye130"]
OBJ = object_points()
CAMERA_SEEDS = (0, 1)
FRAMES_PER_CAMERA = 10  # 20 robot frames per family


def _detect(img):
    det = cv2.aruco.CharucoDetector(BOARD)
    corners, ids, _, _ = det.detectBoard(img)
    if ids is None:
        return np.zeros(0, int), np.zeros((0, 2))
    return ids.ravel().astype(int), corners.reshape(-1, 2).astype(np.float64)


OUTLIER_PX = 1.0  # corners farther than this from the projected truth are detector outliers
MAX_OUTLIER_FRAC = 0.01


def _bias_stats(family, clean):
    diffs, radial, frames_ok = [], [], 0
    for cs in CAMERA_SEEDS:
        sim = make_simulator(family, "robot", camera_seed=cs, seed=0, clean=clean,
                             occluder_p=0.0, corruption_p=0.0)
        rng = np.random.default_rng(1000 + cs)
        for a in range(FRAMES_PER_CAMERA):
            rv, tv = random_pose(sim.camera, rng)
            fr = sim.capture(rv, tv, attempt=a)
            ids, corners = _detect(fr.image)
            if len(ids) < 12:
                continue
            frames_ok += 1
            cam = fr.truth.camera
            gt = cam.project(OBJ[ids], fr.truth.rvec, fr.truth.tvec)
            d = corners - gt
            out = gt - [cam.cx, cam.cy]
            r = np.linalg.norm(out, axis=1, keepdims=True)
            diffs.append(d)
            radial.append(np.sum(d * out / np.maximum(r, 1e-9), axis=1))
    d = np.concatenate(diffs)
    e = np.linalg.norm(d, axis=1)
    inl = e <= OUTLIER_PX
    return {
        "frames": frames_ok,
        "n": len(d),
        "bias": d.mean(0),
        "radial": float(np.concatenate(radial).mean()),
        "rms": float(np.sqrt((e ** 2).mean())),
        "rms_inlier": float(np.sqrt((e[inl] ** 2).mean())),
        "outlier_frac": float(1.0 - inl.mean()),
    }


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("clean", [True, False], ids=["clean", "family_noise_jpeg"])
def test_render_corner_bias(family, clean, record_property):
    """Detected ChArUco corners against the projected GT corners over 20 robot frames.

    Gates: |mean error| < 0.05 px, |mean radial error| < 0.05 px (a scale or canvas error would
    cancel in the mean vector but not here), and RMS < 0.15 px over corners within 1 px of the
    truth, with at most 1 % of corners beyond 1 px. Those few outliers (1.0-2.8 px) come from the
    ChArUco detector's corner refinement on boards tilted 38-50 degrees with small squares: they
    appear in noise-free renders too, and cornerSubPix started at the GT position with a 5x5
    window stays within 0.16 px of it, while a 7x7 window can jump 2-3 px to a neighbouring
    saddle. They are not render error.
    """
    st = _bias_stats(family, clean)
    bias = st["bias"]
    record_property("bias_px", bias.tolist())
    record_property("radial_bias_px", st["radial"])
    record_property("rms_px", st["rms"])
    record_property("rms_inlier_px", st["rms_inlier"])
    print(f"{family} {'clean' if clean else 'noisy'}: frames {st['frames']}/20 corners {st['n']} "
          f"bias ({bias[0]:+.4f}, {bias[1]:+.4f}) |bias| {np.linalg.norm(bias):.4f} "
          f"radial {st['radial']:+.4f} rms {st['rms']:.4f} rms<=1px {st['rms_inlier']:.4f} "
          f"outliers {100 * st['outlier_frac']:.2f}%")
    # ChArUco interpolation gives up on strongly curved fisheye boards; that is the detector, not the render
    assert st["frames"] >= (15 if family == "fisheye130" else 18)
    assert np.linalg.norm(bias) < 0.05
    assert abs(st["radial"]) < 0.05
    assert st["outlier_frac"] <= MAX_OUTLIER_FRAC
    assert st["rms_inlier"] < 0.15


@pytest.mark.parametrize("family", FAMILIES)
def test_distortion_map_roundtrip(family):
    cam = make_camera(family, 0)
    m = distortion_maps(cam)
    W, H = cam.size
    ys, xs = np.mgrid[0:H, 0:W]
    ok = m.valid.ravel()
    cu = m.map_x.ravel()[ok].astype(np.float64)
    cv_ = m.map_y.ravel()[ok].astype(np.float64)
    Kv = m.K_v
    rays = np.column_stack([(cu - Kv[0, 2]) / Kv[0, 0], (cv_ - Kv[1, 2]) / Kv[1, 1], np.ones(len(cu))])
    back = cam.project_camera_points(rays)
    target = np.column_stack([xs.ravel()[ok], ys.ravel()[ok]]).astype(np.float64)
    err = np.linalg.norm(back - target, axis=1)
    print(f"{family}: max round-trip {err.max():.2e} px over {ok.sum()} px, canvas {m.canvas_size}, scale {m.scale:.3f}")
    assert err.max() < 1e-3
    if cam.model != "fisheye":
        assert ok.all()
    else:
        assert ok.mean() > 0.99


@pytest.mark.parametrize("family", FAMILIES)
def test_canvas_covers_every_ray(family):
    cam = make_camera(family, 0)
    m = distortion_maps(cam)
    Wc, Hc = m.canvas_size
    W, H = cam.size
    assert Wc <= 3 * W and Hc <= 3 * H
    mx, my = m.map_x[m.valid], m.map_y[m.valid]
    assert mx.min() >= 0 and my.min() >= 0
    assert mx.max() <= Wc - 1 and my.max() <= Hc - 1


@pytest.mark.parametrize("camera_seed", [0, 1, 2])
def test_wide_canvas_no_background_inside_board(camera_seed):
    """Every output pixel whose ray hits the board's square area samples the board, not background,
    including at the image corners where barrel distortion pulls rays far outside the output frame."""
    cam = make_camera("wide", camera_seed)
    r = Renderer(cam)
    m = r.maps
    Kv = m.K_v
    W, H = cam.size
    d = np.stack([(m.map_x - Kv[0, 2]) / Kv[0, 0], (m.map_y - Kv[1, 2]) / Kv[1, 1], np.ones((H, W))], -1)
    corner_zone = np.zeros((H, W), bool)
    corner_zone[:60, :60] = corner_zone[:60, -60:] = corner_zone[-60:, :60] = corner_zone[-60:, -60:] = True
    sentinel = -1000.0
    hits = corner_hits = 0
    for c in [(60, 40), (W - 60, 40), (60, H - 40), (W - 60, H - 40), (W / 2, H / 2), (W / 2, 30)]:
        for tilt, az in [(0, 0), (35, 45), (50, 200)]:
            rv, tv = pose_for_view(cam, c, 0.5, tilt, az, 10.0)
            R, _ = cv2.Rodrigues(rv.reshape(3, 1))
            nrm = R[:, 2]
            lam = (tv @ nrm) / (d @ nrm)
            X = lam[..., None] * d
            b = (X - tv) @ R
            on_board = (lam > 0) & (b[..., 0] >= 0) & (b[..., 0] <= BOARD_SIZE_M[0])                 & (b[..., 1] >= 0) & (b[..., 1] <= BOARD_SIZE_M[1]) & m.valid
            img = r.render(rv, tv, background=sentinel)
            vals = img[on_board]
            hits += int(on_board.sum())
            corner_hits += int((on_board & corner_zone).sum())
            assert vals.min() > sentinel / 2, f"background inside the board at centre {c}, tilt {tilt}"
    assert hits > 1_000_000 and corner_hits > 5_000


def test_half_pixel_texture_convention():
    tex = _texture_u8()
    S = texture_to_board()
    # board origin sits half a texel inside the first board texel
    np.testing.assert_allclose(S @ [TEX_MARGIN_PX - 0.5, TEX_MARGIN_PX - 0.5, 1.0], [0.0, 0.0, 1.0], atol=1e-12)
    # the first inner corner (0.03, 0.03) m: the texel edge between 114 and 115 is the square edge
    u = board_to_texture(np.array([[SQUARE_M, SQUARE_M]]))[0]
    assert u.tolist() == pytest.approx([114.5, 114.5])
    row = int(TEX_MARGIN_PX + 37)  # middle of the first square row
    assert tex[row, 114] != tex[row, 115]
    assert tex[row, 113] == tex[row, 114] and tex[row, 115] == tex[row, 116]


def test_texture_levels_and_render_range():
    cam = make_camera("normal", 0)
    r = Renderer(cam)
    rv, tv = pose_for_view(cam, (640, 360), 0.6)
    img = r.render(rv, tv, antialias=False)
    assert img.dtype == np.float32 and img.shape == (720, 1280)
    assert img.min() >= INK - 1e-3 and img.max() <= PAPER + 1e-3


def test_bias_check_detects_missing_half_pixel(monkeypatch):
    """Without the +0.5 texel offset the corners shift by about -0.27 px; the bias gate must see that."""
    import nextview.sim.render as render_mod

    def no_offset():
        off = -TEX_MARGIN_PX / render_mod.PPM
        return np.array([[1.0 / render_mod.PPM, 0.0, off], [0.0, 1.0 / render_mod.PPM, off], [0.0, 0.0, 1.0]])

    monkeypatch.setattr(render_mod, "texture_to_board", no_offset)
    cam = make_camera("normal", 0)
    r = Renderer(cam)
    rng = np.random.default_rng(1000)
    diffs = []
    for _ in range(6):
        rv, tv = random_pose(cam, rng, max_tilt_deg=20, max_roll_deg=5)
        ids, corners = _detect(np.clip(np.rint(r.render(rv, tv)), 0, 255).astype(np.uint8))
        diffs.append(corners - cam.project(OBJ[ids], rv, tv))
    bias = np.concatenate(diffs).mean(0)
    assert np.all(bias < -0.2), bias


def test_radial_check_detects_canvas_scale_error():
    """A 0.05 % focal error on the canvas moves every corner outwards. On boards centred on the
    principal point the mean error vector cancels, so only the radial gate can see it."""
    from dataclasses import replace

    cam = make_camera("normal", 0)
    r = Renderer(cam)
    Kv = r.maps.K_v.copy()
    Kv[0, 0] *= 1.0005
    Kv[1, 1] *= 1.0005
    r.maps = replace(r.maps, K_v=Kv)
    diffs, radial = [], []
    for roll in (-20.0, 0.0, 20.0, 45.0):
        rv, tv = pose_for_view(cam, (cam.cx, cam.cy), 0.7, 10.0, 30.0, roll)
        ids, corners = _detect(np.clip(np.rint(r.render(rv, tv)), 0, 255).astype(np.uint8))
        gt = cam.project(OBJ[ids], rv, tv)
        out = gt - [cam.cx, cam.cy]
        diffs.append(corners - gt)
        radial.append(np.sum((corners - gt) * out / np.linalg.norm(out, axis=1, keepdims=True), axis=1))
    bias = np.concatenate(diffs).mean(0)
    rad = np.concatenate(radial).mean()
    print(f"scaled canvas: |bias| {np.linalg.norm(bias):.4f} radial {rad:+.4f}")
    assert np.linalg.norm(bias) < 0.05 and rad > 0.05
