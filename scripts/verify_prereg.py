"""Check the frozen benchmark snapshots against the recorded hashes, and list what changed since.

The DEV, TEST and failure-set runs were launched from byte-for-byte copies of the source tree and
config (``results/tmp/frozen_*``, not committed). Those copies are kept in
``results/prereg/frozen_snapshots.tar.xz``. This script reads the archive with the standard library
only and, for each snapshot:

* recomputes the raw-byte source-tree hash exactly as ``run_synthetic.code_manifest`` does
  (sha256 over ``nextview/<path>:<sha256 of the file>`` lines) and the config hashes, and compares
  them with the run's ``results/<split>/meta.json`` and with ``docs/preregistration.md``;
* prints the line-ending-independent tree hash (CRLF -> LF before hashing), which is the one a
  fresh checkout on any platform can reproduce;
* compares every snapshot file with the same file in this checkout's ``src/nextview`` (line endings
  ignored) and lists the files that differ, were added or were removed since.

    python scripts/verify_prereg.py            # verify; exit 0 when every recorded hash matches
    python scripts/verify_prereg.py --pack     # maintainer: rebuild the archive from results/tmp/frozen_*
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "results" / "prereg" / "frozen_snapshots.tar.xz"
# snapshot directory -> the split it ran (its meta.json holds the hashes recorded at run time)
SNAPSHOTS = {"frozen_dev": "synth_dev", "frozen_test": "synth_test", "frozen_r7fix": "synth_failure"}
CONFIGS = ("thresholds.json", "stop_calibration.json", "presets.json")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def tree_hash(files: dict[str, bytes], normalise: bool = False) -> str:
    """sha256 over sorted ``path:sha256(content)`` lines (``code_manifest``); optionally CRLF -> LF first."""
    return sha("".join(f"{k}:{sha(_lf(v) if normalise else v)}\n" for k, v in sorted(files.items())).encode())


def pack(out: Path) -> int:
    src = REPO / "results" / "tmp"
    members = []
    for snap in SNAPSHOTS:
        root = src / snap
        if not root.is_dir():
            print(f"missing {root}", file=sys.stderr)
            return 2
        for p in sorted(root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                members.append((f"{snap}/{p.relative_to(root).as_posix()}", p.read_bytes()))
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz", format=tarfile.PAX_FORMAT,
                      preset=9) as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(data))
    out.write_bytes(buf.getvalue())
    print(f"wrote {out.relative_to(REPO).as_posix()}: {len(members)} files, {out.stat().st_size:,} bytes")
    return 0


def read_archive(path: Path) -> dict[str, dict[str, bytes]]:
    snaps: dict[str, dict[str, bytes]] = {}
    with tarfile.open(path, mode="r:xz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            snap, _, rel = m.name.partition("/")
            f = tf.extractfile(m)
            snaps.setdefault(snap, {})[rel] = f.read() if f else b""
    return snaps


def checkout_files() -> dict[str, bytes]:
    pkg = REPO / "src" / "nextview"
    return {f"nextview/{p.relative_to(pkg).as_posix()}": p.read_bytes() for p in sorted(pkg.rglob("*.py"))
            if "__pycache__" not in p.parts and p.name != "_build.py"}


def verify(path: Path) -> int:
    if not path.exists():
        print(f"no archive at {path}", file=sys.stderr)
        return 2
    snaps = read_archive(path)
    prereg = (REPO / "docs" / "preregistration.md").read_text(encoding="utf-8")
    here = checkout_files()
    ok = True
    for snap, split in SNAPSHOTS.items():
        files = snaps.get(snap, {})
        py = {k: v for k, v in files.items() if k.startswith("nextview/") and k.endswith(".py")}
        meta = json.loads((REPO / "results" / split / "meta.json").read_text(encoding="utf-8"))
        rec = {"source tree": meta["code"]["source_tree_sha256"], "thresholds.json": meta["thresholds_sha256"],
               "stop_calibration.json": meta["stop_calibration_sha256"]}
        got = {"source tree": tree_hash(py)}
        for c in CONFIGS:
            got[c] = sha(files[f"config/{c}"]) if f"config/{c}" in files else None
        print(f"{snap} (ran results/{split}; {len(py)} source files)")
        for k, v in got.items():
            want = rec.get(k)
            if want is None:
                in_doc = v is not None and v in prereg
                print(f"  {k:22s} {v}  {'listed in docs/preregistration.md' if in_doc else ''}".rstrip())
                continue
            match = v == want
            ok &= match
            print(f"  {k:22s} {v}  {'= meta.json' if match else 'MISMATCH, meta.json has ' + want}"
                  f"{', listed in docs/preregistration.md' if match and v in prereg else ''}")
        lf = tree_hash(py, normalise=True)
        print(f"  {'tree, CRLF -> LF':22s} {lf}{'  listed in docs/preregistration.md' if lf in prereg else ''}")
        same = sorted(k for k in py if k in here and _lf(py[k]) == _lf(here[k]))
        diff = sorted(k for k in py if k in here and _lf(py[k]) != _lf(here[k]))
        gone = sorted(k for k in py if k not in here)
        new = sorted(k for k in here if k not in py)
        print(f"  vs this checkout (line endings ignored): {len(same)} identical, {len(diff)} changed, "
              f"{len(new)} added, {len(gone)} removed")
        for label, names in (("changed", diff), ("added", new), ("removed", gone)):
            if names:
                print(f"    {label}: " + ", ".join(n[len('nextview/'):] for n in names))
    print("all recorded hashes reproduce" if ok else "SOME RECORDED HASHES DO NOT REPRODUCE")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--pack", action="store_true", help="rebuild the archive from results/tmp/frozen_*")
    a = ap.parse_args(argv)
    return pack(Path(a.archive)) if a.pack else verify(Path(a.archive))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
