"""Locating and reading benchmark outputs, plain or gzip, committed or local.

The full ``runs.csv`` of the TEST split (46 MB, every row with its K and distortion) is larger
than the committed-results budget, so the repository keeps ``runs_core.csv.gz`` (every row and
metric, 3-4 significant digits, no K / distortion) and ``estimates.csv.gz`` (the full estimate at
n = 10, 20, 30); the full files stay in ``results/tmp/full/<split>/`` (gitignored) and are
regenerated exactly by ``run_synthetic`` (deterministic).
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
import tempfile
from pathlib import Path

from . import analysis

__all__ = ["runs_file", "sessions_file", "load_runs", "load_sessions", "compact_runs", "CORE_DROP", "CORE_SIG"]

CORE_DROP = ("split", "fx", "fy", "cx", "cy", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "rms")
CORE_SIG = {"U95": 4, "p_exceed_loop": 3, "true_p95": 4, "true_mean": 4, "fx_rel_pct": 3, "fy_rel_pct": 3, "cx_err_px": 3,
            "cy_err_px": 3, "heldout_rms": 3, "cov50": 3, "cov80": 3, "cov90": 3, "cov95": 3}
ESTIMATE_FIELDS = ("camera", "family", "seed", "actuator", "policy", "n", "model", "fx", "fy", "cx", "cy",
                   "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "rms")


def _open_text(path: Path):
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, newline="", encoding="utf-8")


def _candidates(d: Path):
    return (d, d.parent / "tmp" / "full" / d.name)


def runs_file(d: Path | str, need_estimates: bool = False) -> Path:
    """``runs.csv`` / ``runs.csv.gz`` (full) or, unless estimates are needed, ``runs_core.csv.gz``,
    in ``d`` or in ``results/tmp/full/<d.name>``."""
    d = Path(d)
    names = ("runs.csv", "runs.csv.gz") + (() if need_estimates else ("runs_core.csv.gz",))
    for base in _candidates(d):
        for n in names:
            if (base / n).exists():
                return base / n
    raise FileNotFoundError(f"no runs table for {d}" + (" with per-row estimates" if need_estimates else ""))


def sessions_file(d: Path | str) -> Path:
    d = Path(d)
    for base in _candidates(d):
        for n in ("sessions.jsonl", "sessions.jsonl.gz"):
            if (base / n).exists():
                return base / n
    raise FileNotFoundError(f"no sessions file for {d}")


def load_runs(path: Path | str, need_estimates: bool = False) -> list[dict]:
    """``analysis.load_runs`` on a directory (resolved with ``runs_file``) or a plain/gzip file."""
    p = Path(path)
    if p.is_dir():
        p = runs_file(p, need_estimates)
    if p.suffix != ".gz":
        return analysis.load_runs(p)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "runs.csv"
        with gzip.open(p, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return analysis.load_runs(tmp)


def load_sessions(path: Path | str) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        p = sessions_file(p)
    with _open_text(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def compact_runs(src: Path | str, out_dir: Path | str, estimate_ns=(10, 20, 30)) -> dict:
    """Write ``runs_core.csv.gz`` and ``estimates.csv.gz`` (see the module docstring) to ``out_dir``."""
    out_dir = Path(out_dir)

    def sig(v, n):
        if v == "":
            return ""
        x = float(v)
        return "0" if x == 0 else f"{x:.{n}g}"

    core, est = io.StringIO(), io.StringIO()
    with _open_text(Path(src)) as f:
        r = csv.DictReader(f)
        wc = csv.DictWriter(core, fieldnames=[k for k in r.fieldnames if k not in CORE_DROP], lineterminator="\n",
                            extrasaction="ignore")
        we = csv.DictWriter(est, fieldnames=list(ESTIMATE_FIELDS), lineterminator="\n", extrasaction="ignore")
        wc.writeheader()
        we.writeheader()
        for row in r:
            if int(row["n"]) in estimate_ns:
                we.writerow(row)
            for k, n in CORE_SIG.items():
                row[k] = sig(row.get(k, ""), n)
            wc.writerow(row)
    sizes = {}
    for name, buf in (("runs_core.csv.gz", core), ("estimates.csv.gz", est)):
        p = out_dir / name
        with gzip.open(p, "wb", compresslevel=9) as g:
            g.write(buf.getvalue().encode("utf-8"))
        sizes[name] = p.stat().st_size
    return sizes
