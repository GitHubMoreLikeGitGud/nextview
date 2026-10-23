"""Small helpers for reading views (``View`` or ``Detection``) in the tools."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..board import N_CORNERS, object_points

__all__ = ["view_arrays", "full_board_views", "views_used", "board_points"]


def view_arrays(view) -> tuple[np.ndarray, np.ndarray]:
    """(ids (N,), corners (N, 2)) from a ``View`` or a ``Detection``."""
    ids = getattr(view, "corner_ids", None)
    if ids is None:
        ids = view.ids
    return np.asarray(ids).ravel().astype(np.int64), np.asarray(view.corners, np.float64).reshape(-1, 2)


def full_board_views(views: Sequence) -> list:
    """Views in which every board corner was detected (required by the RO model)."""
    out = []
    for v in views:
        ids, _ = view_arrays(v)
        if ids.size == N_CORNERS and np.array_equal(np.sort(ids), np.arange(N_CORNERS)):
            out.append(v)
    return out


def views_used(model: str, views: Sequence) -> list:
    """The views a calibration of ``model`` was fitted on, in the order of its poses."""
    return full_board_views(views) if model == "ro" else list(views)


def board_points(calib) -> np.ndarray:
    """Board corner coordinates the calibration used: the refined board for RO, else nominal."""
    new_obj = getattr(calib, "new_obj", None)
    return object_points() if new_obj is None else np.asarray(new_obj, np.float64).reshape(-1, 3)
