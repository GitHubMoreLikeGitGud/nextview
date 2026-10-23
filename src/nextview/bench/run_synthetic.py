"""Synthetic benchmark runner (spec 11.2): every job of a split, in parallel, deterministic.

    python -m nextview.bench.run_synthetic --split test --workers 14 --out results/synth_test

Each job is one policy on one (camera, policy seed, actuator); all randomness comes from those
numbers, so the output does not depend on the worker count or the completion order (files are
sorted before they are written). Completed jobs are appended to ``<out>/../tmp/<name>.partial.jsonl`` as they
finish; a re-run with the same arguments skips them (resume after an interruption).

Outputs in ``--out``:
  runs.csv          long format: one row per (job, accepted frame n), columns ``common.ROW_FIELDS``
  sessions.jsonl    one record per job (stop point, end rule, approvals, rejections, outliers, timing)
  events.jsonl.gz   the agent's trace lines per A-family job
  meta.json         split, job counts, errors, thresholds, stop-calibration hash, code manifest, timing
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

__all__ = ["code_manifest", "file_sha256", "portable_path", "run_jobs", "main"]


def portable_path(path: Path | str) -> str:
    """``path`` as a POSIX path relative to its git repository (else the working directory), so that
    result files carry no machine-specific prefix."""
    p = Path(path).resolve()
    bases = [q for q in (p, *p.parents) if (q / ".git").exists()][:1] + [Path.cwd().resolve()]
    for base in bases:
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            continue
    return p.name


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_manifest() -> dict:
    """sha256 of every source file of the ``nextview`` package that is imported, plus git state."""
    import nextview

    pkg = Path(nextview.__file__).resolve().parent
    files = {str(p.relative_to(pkg.parent)).replace("\\", "/"): file_sha256(p) for p in sorted(pkg.rglob("*.py"))}
    tree = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in files.items()).encode()).hexdigest()
    git = {}
    repo = pkg.parents[1]
    try:
        git["head"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                                     timeout=10).stdout.strip() or None
        st = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout
        git["dirty_files"] = len([ln for ln in st.splitlines() if ln.strip()])
    except (OSError, subprocess.SubprocessError):
        pass
    return {"package_dir": portable_path(pkg), "source_tree_sha256": tree, "files": files, "git": git}


def _init_worker():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import cv2

    cv2.setNumThreads(1)


_TH = None
_SC = None


def _work(args):
    global _TH, _SC
    job, th_dict, sc_path = args
    if _TH is None:
        from nextview.config import Thresholds, load_stop_calibration

        _TH = Thresholds.from_dict(th_dict)
        _SC = load_stop_calibration(sc_path)
    from nextview.bench.policies import run_job

    rows, sess = run_job(job, _TH, _SC)
    return job, rows, sess


def _job_from(d: dict):
    from nextview.bench.common import Job

    d = dict(d)
    d["nuisances"] = tuple(d.get("nuisances", ()))
    return Job(**d)


def _key(j) -> str:
    return "|".join(str(x) for x in j.key)


def run_jobs(jobs, out: Path, *, workers: int, th, stopcal_path: Path | str | None, progress_every: int = 50,
             log=print) -> dict:
    """Run ``jobs`` (resuming from ``out/partial.jsonl``) and write the sorted outputs."""
    from dataclasses import asdict

    from nextview.bench.common import A_FAMILY, ROW_FIELDS

    out.mkdir(parents=True, exist_ok=True)
    partial = out.parent / "tmp" / f"{out.name}.partial.jsonl"  # results/tmp is gitignored
    partial.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if partial.exists():
        with open(partial, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["key"]] = rec
    todo = [j for j in jobs if _key(j) not in done]
    # slow jobs first for load balance
    todo.sort(key=lambda j: (0 if j.policy in A_FAMILY else 1, j.camera, j.seed, j.actuator, j.policy))
    log(f"{len(jobs)} jobs, {len(done)} already done, {len(todo)} to run on {workers} workers", flush=True)
    t0 = time.time()
    th_dict = th.to_dict()
    sc = str(stopcal_path) if stopcal_path else None
    n_err = 0
    with open(partial, "a", encoding="utf-8") as pf:
        if todo:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker, maxtasksperchild=200) as pool:
                for k, (job, rows, sess) in enumerate(pool.imap_unordered(_work, [(j, th_dict, sc) for j in todo], chunksize=1), 1):
                    rec = {"key": _key(job), "job": asdict(job), "rows": rows, "session": sess}
                    pf.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    pf.flush()
                    done[rec["key"]] = rec
                    if "error" in sess:
                        n_err += 1
                        log(f"ERROR {rec['key']}: {sess['error']}", flush=True)
                    if k % progress_every == 0 or k == len(todo):
                        el = time.time() - t0
                        log(f"progress {k}/{len(todo)} jobs, {el / 60:.1f} min elapsed, "
                            f"~{el / k * (len(todo) - k) / 60:.1f} min left, errors {n_err}", flush=True)
    wall = time.time() - t0
    keys = sorted(done, key=lambda s: tuple(int(x) if x.lstrip("-").isdigit() else x for x in s.split("|")))
    want = {_key(j) for j in jobs}
    keys = [k for k in keys if k in want]
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ROW_FIELDS), extrasaction="ignore")
        w.writeheader()
        for k in keys:
            for r in done[k]["rows"]:
                w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in ROW_FIELDS})
    errors = []
    with open(out / "sessions.jsonl", "w", encoding="utf-8") as f, gzip.open(out / "events.jsonl.gz", "wt", encoding="utf-8") as g:
        for k in keys:
            s = dict(done[k]["session"])
            trace = s.pop("trace", None)
            if "error" in s:
                errors.append({"key": k, "error": s["error"]})
            f.write(json.dumps(s, separators=(",", ":")) + "\n")
            if trace:
                g.write(json.dumps({"key": k, "trace": trace}, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"jobs": len(keys), "errors": errors, "wall_s_this_invocation": round(wall, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("dev", "test", "smoke", "failure"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", required=True)
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="first N jobs only (debugging)")
    ap.add_argument("--stopcal", default=None, help="stop calibration JSON (default config/stop_calibration.json)")
    a = ap.parse_args(argv)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import cv2
    import numpy as np

    from nextview.bench.common import split_jobs
    from nextview.config import config_dir, load_thresholds

    th = load_thresholds()
    sc_path = Path(a.stopcal) if a.stopcal else config_dir() / "stop_calibration.json"
    jobs = split_jobs(a.split, a.policies, budget=a.budget)
    if a.limit:
        jobs = jobs[: a.limit]
    out = Path(a.out)
    t0 = time.time()
    res = run_jobs(jobs, out, workers=a.workers, th=th, stopcal_path=sc_path)
    meta = {
        "split": a.split, "policies": sorted({j.policy for j in jobs}), "n_jobs": res["jobs"], "errors": res["errors"],
        "budget": a.budget, "workers": a.workers, "wall_s_last_invocation": res["wall_s_this_invocation"],
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": th.to_dict(), "thresholds_sha256": file_sha256(config_dir() / "thresholds.json"),
        "stop_calibration": portable_path(sc_path), "stop_calibration_sha256": file_sha256(sc_path) if sc_path.exists() else None,
        "bench_overrides": {"nbv_time_budget_s": 1e5, "step_deadline": "none (1e5 s)",
                            "after_first_stop": "export denied, stop_p_exceed = -1 for the rest of the run"},
        "versions": {"python": sys.version.split()[0], "opencv": cv2.__version__, "numpy": np.__version__,
                     "platform": platform.platform()},
        "code": code_manifest(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"done: {res['jobs']} jobs, {len(res['errors'])} errors, {(time.time() - t0) / 60:.1f} min; wrote {out}", flush=True)
    return 0 if not res["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
