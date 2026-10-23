"""Read members of a large remote ZIP archive with HTTP range requests.

Only the central directory and the bytes of the requested members are downloaded.
Members that sit next to each other in the archive are fetched in one request, which
keeps the request count low on rate-limited hosts such as Zenodo.

The archive is untrusted: member names are never used as output paths directly, every
member is CRC-checked after decompression, and the server must answer with 206 so a
misbehaving host cannot start streaming the whole archive.
"""

from __future__ import annotations

import io
import os
import struct
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

__all__ = [
    "RangeError",
    "RangeReader",
    "RemoteZip",
    "Span",
    "plan_spans",
    "safe_join",
]

USER_AGENT = "nextview-data/0.1 (+range-reader)"
_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")
_LOCAL_SIG = b"PK\x03\x04"


class RangeError(RuntimeError):
    """The server did not honour a byte-range request."""


class RangeReader:
    """Byte-range GETs against one URL with retries, backoff and a byte counter."""

    def __init__(self, url: str, *, timeout: float = 60.0, retries: int = 6, chunk: int = 1 << 20) -> None:
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.chunk = chunk
        self.bytes_fetched = 0
        self.requests = 0
        self._lock = threading.Lock()
        self._size: int | None = None

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = self._probe_size()
        return self._size

    def _probe_size(self) -> int:
        with self._open(0, 0) as resp:
            cr = resp.headers.get("Content-Range", "")
            resp.read()
        total = cr.rsplit("/", 1)[-1]
        if not total.isdigit():
            raise RangeError(f"no total size in Content-Range {cr!r}")
        return int(total)

    def _open(self, start: int, end: int):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=self.timeout)
        if resp.status != 206:
            resp.close()
            raise RangeError(f"expected 206 for bytes={start}-{end}, got {resp.status}")
        cr = resp.headers.get("Content-Range", "")
        if not cr.startswith(f"bytes {start}-"):
            resp.close()
            raise RangeError(f"unexpected Content-Range {cr!r} for bytes={start}-{end}")
        with self._lock:
            self.requests += 1
        return resp

    def iter_range(self, start: int, end: int) -> Iterator[bytes]:
        """Yield the bytes ``start..end`` (inclusive) in chunks, resuming after dropped connections."""
        pos = start
        attempt = 0
        while pos <= end:
            try:
                with self._open(pos, end) as resp:
                    while pos <= end:
                        buf = resp.read(min(self.chunk, end - pos + 1))
                        if not buf:
                            break
                        pos += len(buf)
                        with self._lock:
                            self.bytes_fetched += len(buf)
                        attempt = 0
                        yield buf
                if pos <= end:
                    raise ConnectionError(f"short read at {pos} of {start}-{end}")
            except RangeError:
                raise
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                attempt += 1
                if attempt > self.retries:
                    raise
                time.sleep(_retry_delay(exc, attempt))

    def read(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        return b"".join(self.iter_range(start, start + length - 1))


def _retry_delay(exc: BaseException, attempt: int) -> float:
    if isinstance(exc, urllib.error.HTTPError):
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after and retry_after.isdigit():
            return min(120.0, float(retry_after) + 1.0)
        if exc.code == 429:
            return min(120.0, 15.0 * attempt)
    return min(60.0, 2.0**attempt)


class _RawRemote(io.RawIOBase):
    """Seekable raw stream over a RangeReader, for ``zipfile`` to parse the central directory."""

    def __init__(self, reader: RangeReader) -> None:
        self._r = reader
        self._n = reader.size
        self._p = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._p

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._p = offset
        elif whence == io.SEEK_CUR:
            self._p += offset
        else:
            self._p = self._n + offset
        return self._p

    def readinto(self, b) -> int:
        if self._p >= self._n:
            return 0
        n = min(len(b), self._n - self._p)
        data = self._r.read(self._p, n)
        b[: len(data)] = data
        self._p += len(data)
        return len(data)


@dataclass(frozen=True)
class Span:
    """A contiguous byte range of the archive holding one or more members."""

    start: int
    end: int  # inclusive
    members: tuple[zipfile.ZipInfo, ...]

    @property
    def nbytes(self) -> int:
        return self.end - self.start + 1


# Upper bound on the local header: 30 fixed bytes + name + extra field (Zip64 extra is 20 bytes,
# others are rare; the real length is read from the header and checked).
def _local_header_bound(info: zipfile.ZipInfo) -> int:
    return 30 + len(info.orig_filename.encode("utf-8")) + 1024


def plan_spans(
    members: Sequence[zipfile.ZipInfo], *, max_gap: int = 4 << 20, max_span: int = 64 << 20, archive_size: int | None = None
) -> list[Span]:
    """Group members into ranges so neighbouring members share one request."""
    spans: list[Span] = []
    cur: list[zipfile.ZipInfo] = []
    s = e = 0
    for info in sorted(members, key=lambda i: i.header_offset):
        a = info.header_offset
        b = a + _local_header_bound(info) + info.compress_size - 1
        if archive_size is not None:
            b = min(b, archive_size - 1)
        if cur and a - e - 1 <= max_gap and b - s + 1 <= max_span:
            cur.append(info)
            e = max(e, b)
        else:
            if cur:
                spans.append(Span(s, e, tuple(cur)))
            cur, s, e = [info], a, b
    if cur:
        spans.append(Span(s, e, tuple(cur)))
    return spans


def _decompress(info: zipfile.ZipInfo, raw: bytes) -> bytes:
    """Member bytes from its raw (stored or deflated) data, checked against size and CRC.

    Inflation stops one byte past the declared size, so a member that inflates to more than
    it declares is rejected without being expanded in memory.
    """
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        d = zlib.decompressobj(-15)
        data = d.decompress(raw, info.file_size + 1)
        if len(data) > info.file_size:
            raise ValueError(f"{info.filename}: inflates past its declared size {info.file_size}")
        if not d.eof:
            raise ValueError(f"{info.filename}: deflate stream is truncated")
    else:
        raise ValueError(f"{info.filename}: unsupported compression {info.compress_type}")
    if len(data) != info.file_size:
        raise ValueError(f"{info.filename}: size {len(data)} != {info.file_size}")
    if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise ValueError(f"{info.filename}: CRC mismatch")
    return data


def _members_from_buffer(span: Span, buf: bytes) -> Iterator[tuple[zipfile.ZipInfo, bytes]]:
    for info in span.members:
        off = info.header_offset - span.start
        hdr = buf[off : off + _LOCAL_HEADER.size]
        if len(hdr) < _LOCAL_HEADER.size:
            raise ValueError(f"{info.filename}: truncated local header")
        fields = _LOCAL_HEADER.unpack(hdr)
        if fields[0] != _LOCAL_SIG:
            raise ValueError(f"{info.filename}: bad local header signature")
        name_len, extra_len = fields[9], fields[10]
        data_off = off + _LOCAL_HEADER.size + name_len + extra_len
        raw = buf[data_off : data_off + info.compress_size]
        if len(raw) != info.compress_size:
            raise ValueError(f"{info.filename}: local extra field longer than the planned bound")
        yield info, _decompress(info, raw)


def safe_join(root: Path, rel: str) -> Path:
    """Join ``rel`` under ``root`` and refuse anything that would escape it."""
    rel = rel.replace("\\", "/")
    if rel.startswith("/"):
        raise ValueError(f"absolute path {rel!r}")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or ":" in p for p in parts):
        raise ValueError(f"unsafe relative path {rel!r}")
    out = root.joinpath(*parts)
    root_r = root.resolve()
    if root_r != out.resolve() and root_r not in out.resolve().parents:
        raise ValueError(f"path {rel!r} escapes {root}")
    return out


def _crc_of_file(path: Path) -> int:
    crc = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF


class RemoteZip:
    """A remote ZIP archive: list its members and extract chosen ones over HTTP ranges."""

    def __init__(self, url: str, *, timeout: float = 60.0, retries: int = 6) -> None:
        self.reader = RangeReader(url, timeout=timeout, retries=retries)
        raw = _RawRemote(self.reader)
        self._zip = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))

    @property
    def size(self) -> int:
        return self.reader.size

    def infolist(self) -> list[zipfile.ZipInfo]:
        return self._zip.infolist()

    def members(self, prefix: str = "", predicate: Callable[[zipfile.ZipInfo], bool] | None = None) -> list[zipfile.ZipInfo]:
        out = [i for i in self._zip.infolist() if i.filename.startswith(prefix) and not i.is_dir()]
        if predicate is not None:
            out = [i for i in out if predicate(i)]
        return out

    def read(self, info: zipfile.ZipInfo) -> bytes:
        span = plan_spans([info], archive_size=self.size)[0]
        buf = self.reader.read(span.start, span.nbytes)
        return next(_members_from_buffer(span, buf))[1]

    def iter_members(self, members: Iterable[zipfile.ZipInfo], **plan_kw) -> Iterator[tuple[zipfile.ZipInfo, bytes]]:
        """Yield ``(info, data)`` for each member, one HTTP request per span."""
        for span in plan_spans(list(members), archive_size=self.size, **plan_kw):
            buf = self.reader.read(span.start, span.nbytes)
            yield from _members_from_buffer(span, buf)

    def extract(
        self,
        targets: dict[str, zipfile.ZipInfo] | Iterable[tuple[str, zipfile.ZipInfo]],
        root: Path,
        *,
        verify_existing: bool = True,
        workers: int = 1,
        progress: Callable[[int, int, int], None] | None = None,
        **plan_kw,
    ) -> dict[str, int]:
        """Write members to ``root/<rel>`` for each ``(rel, info)``; skip files already present and intact.

        Spans are fetched by ``workers`` threads, one request each. Returns counts: ``written``,
        ``skipped``, ``bytes`` (uncompressed bytes written), ``spans``.
        ``progress(done, total, bytes_fetched)`` is called after each member.
        """
        pairs = list(targets.items()) if isinstance(targets, dict) else list(targets)
        root = Path(root)
        todo: list[zipfile.ZipInfo] = []
        dest: dict[int, Path] = {}
        skipped = 0
        for rel, info in pairs:
            path = safe_join(root, rel)
            if path.exists() and path.stat().st_size == info.file_size:
                if not verify_existing or _crc_of_file(path) == info.CRC:
                    skipped += 1
                    continue
            dest[id(info)] = path
            todo.append(info)
        spans = plan_spans(todo, archive_size=self.size, **plan_kw)
        lock = threading.Lock()
        counts = {"written": 0, "skipped": skipped, "bytes": 0, "spans": len(spans)}

        def run(span: Span) -> None:
            buf = self.reader.read(span.start, span.nbytes)
            for info, data in _members_from_buffer(span, buf):
                path = dest[id(info)]
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".part")
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
                with lock:
                    counts["written"] += 1
                    counts["bytes"] += len(data)
                    done = counts["written"]
                if progress is not None:
                    progress(done, len(todo), self.reader.bytes_fetched)

        if workers <= 1:
            for span in spans:
                run(span)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for fut in [pool.submit(run, sp) for sp in spans]:
                    fut.result()
        return counts

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "RemoteZip":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
