"""Pool-protocol helpers (run_pool) on noise-free synthetic frames of the NextView board."""

from pathlib import Path

import numpy as np

from nextview.bench.run_pool import fit_k5, heldout_rms, p_agent, p_cov
from nextview.board import object_points
from nextview.data.frame import NEXTVIEW_BOARD, Frame
from nextview.sim.lens import make_camera
from nextview.sim.poses import random_pose
from nextview.types import Detection


def _frames(n, seed=0):
    cam = make_camera("normal", 3)
    rng = np.random.default_rng(seed)
    obj = object_points()
    out = []
    for i in range(n):
        rv, tv = random_pose(cam, rng)
        px, ok = cam.project_checked(obj, rv, tv)
        ids = np.nonzero(ok)[0].astype(np.int32)
        det = Detection(ids=ids, corners=px[ok], obj=obj[ok], n_markers=0, ok=True)
        out.append(Frame(f"{i:04d}", Path(f"{i:04d}.png"), "test", cam.size, det, NEXTVIEW_BOARD))
    return cam, out


def test_fit_k5_recovers_noise_free_camera_and_heldout_is_zero():
    cam, frames = _frames(14)
    rms, K, d, *_ = fit_k5(frames[:10], cam.size)
    assert rms < 1e-3
    assert abs(K[0, 0] - cam.fx) / cam.fx < 1e-4
    assert heldout_rms(frames[10:], K, d) < 1e-3


def test_p_cov_and_p_agent_select_distinct_frames():
    cam, frames = _frames(16, seed=1)
    cov = p_cov(frames, cam.size, 8)
    assert len(cov) == 8 and len({f.name for f in cov}) == 8
    chosen, used, dropped, steps = p_agent(frames, [True] * len(frames), cam.size, 8)
    assert len(chosen) == 8 and len({f.name for f in chosen}) == 8
    assert [f.name for f in chosen[:3]] == ["0000", "0001", "0002"]  # opening: first 3 gate-passing frames
    assert all(s["pred_gain"] >= 0 for s in steps)
