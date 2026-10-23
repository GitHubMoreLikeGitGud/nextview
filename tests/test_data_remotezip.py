import io
import threading
import zipfile
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from nextview.data import remotezip
from nextview.data.remotezip import RangeError, RemoteZip, _decompress, plan_spans, safe_join


def _make_zip() -> tuple[bytes, dict[str, bytes]]:
    rng = np.random.default_rng(0)
    files = {
        "top/a.txt": b"hello range reads\n" * 50,
        "top/sub/b.bin": rng.integers(0, 256, 70_000, dtype=np.uint8).tobytes(),
        "top/sub/c.txt": b"c" * 10,
        "other/d.txt": b"far away" * 1000,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, (name, data) in enumerate(files.items()):
            z.writestr(name, data, compress_type=zipfile.ZIP_STORED if i % 2 else zipfile.ZIP_DEFLATED)
    return buf.getvalue(), files


class _Server:
    def __init__(self, payload: bytes, honour_range: bool = True):
        payload_ = payload
        honour = honour_range
        self.requests = []

        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                outer.requests.append(rng)
                if honour and rng and rng.startswith("bytes="):
                    a, b = rng[6:].split("-")
                    a, b = int(a), min(int(b), len(payload_) - 1)
                    body = payload_[a : b + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {a}-{b}/{len(payload_)}")
                else:
                    body = payload_
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/archive.zip"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_list_read_and_extract(tmp_path):
    payload, files = _make_zip()
    with _Server(payload) as srv, RemoteZip(srv.url) as rz:
        assert rz.size == len(payload)
        names = {i.filename for i in rz.infolist()}
        assert names == set(files)
        sub = rz.members("top/sub/")
        assert sorted(i.filename for i in sub) == ["top/sub/b.bin", "top/sub/c.txt"]
        for info in rz.infolist():
            assert rz.read(info) == files[info.filename]

        targets = {f"out/{i.filename.replace('/', '_')}": i for i in rz.members("top/")}
        n_before = len(srv.requests)
        counts = rz.extract(targets, tmp_path)
        # three neighbouring members fit in one span, so one request
        assert counts["written"] == 3 and counts["skipped"] == 0 and counts["spans"] == 1
        assert len(srv.requests) - n_before == 1
        for rel, info in targets.items():
            assert (tmp_path / rel).read_bytes() == files[info.filename]

        again = rz.extract(targets, tmp_path, workers=2)
        assert again["written"] == 0 and again["skipped"] == 3

        # a corrupted file on disk is detected by CRC and re-fetched
        victim = tmp_path / "out" / "top_a.txt"
        victim.write_bytes(b"x" * victim.stat().st_size)
        fixed = rz.extract(targets, tmp_path)
        assert fixed["written"] == 1 and victim.read_bytes() == files["top/a.txt"]


def test_server_without_range_support_is_refused():
    payload, _ = _make_zip()
    with _Server(payload, honour_range=False) as srv:
        with pytest.raises(RangeError):
            RemoteZip(srv.url)


def test_crc_mismatch_raises():
    payload, files = _make_zip()
    with _Server(payload) as srv, RemoteZip(srv.url) as rz:
        info = next(i for i in rz.infolist() if i.filename == "top/sub/c.txt")
        info.CRC ^= 1
        with pytest.raises(ValueError, match="CRC"):
            rz.read(info)


def test_plan_spans_groups_neighbours_and_splits_gaps():
    infos = []
    for name, off, size in [("a", 0, 100), ("b", 200, 100), ("c", 10_000_000, 50)]:
        zi = zipfile.ZipInfo(name)
        zi.header_offset, zi.compress_size = off, size
        infos.append(zi)
    spans = plan_spans(infos, max_gap=1 << 20)
    assert [len(s.members) for s in spans] == [2, 1]
    assert plan_spans(infos, max_gap=1 << 20, max_span=150)[0].members == (infos[0],)


@pytest.mark.parametrize("rel", ["../evil.txt", "a/../../evil", "/abs/path", "C:/x", "a\\..\\..\\b", ""])
def test_safe_join_rejects_escapes(tmp_path, rel):
    with pytest.raises(ValueError):
        safe_join(tmp_path, rel)


def test_safe_join_accepts_nested(tmp_path):
    assert safe_join(tmp_path, "a/b/c.png") == tmp_path / "a" / "b" / "c.png"


def _member(name: str, data: bytes, *, deflate: bool = True, declared: int | None = None):
    if deflate:
        co = zlib.compressobj(9, zlib.DEFLATED, -15)
        raw = co.compress(data) + co.flush()
    else:
        raw = data
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED
    info.file_size = len(data) if declared is None else declared
    info.compress_size = len(raw)
    info.CRC = zlib.crc32(data) & 0xFFFFFFFF
    return info, raw


def test_decompress_checks_size_crc_and_stream_end():
    data = np.random.default_rng(1).integers(0, 4, 50_000, dtype=np.uint8).tobytes()
    info, raw = _member("a.bin", data)
    assert _decompress(info, raw) == data
    s_info, s_raw = _member("s.bin", data, deflate=False)
    assert _decompress(s_info, s_raw) == data
    with pytest.raises(ValueError, match="truncated"):
        _decompress(info, raw[: len(raw) // 2])
    info.CRC ^= 1
    with pytest.raises(ValueError, match="CRC"):
        _decompress(info, raw)
    short, raw_short = _member("short.bin", data, declared=len(data) + 10)
    with pytest.raises(ValueError, match="size"):
        _decompress(short, raw_short)
    empty, raw_empty = _member("empty", b"")
    assert _decompress(empty, raw_empty) == b""


def test_decompress_stops_at_declared_size(monkeypatch):
    # 10 MB of zeros deflates to about 10 KB; the member claims 1000 bytes
    info, raw = _member("bomb.bin", bytes(10_000_000), declared=1000)
    assert len(raw) < 20_000
    produced = []
    real = zlib.decompressobj

    class Spy:
        def __init__(self, *a):
            self._d = real(*a)

        def decompress(self, data, max_length=0):
            out = self._d.decompress(data, max_length)
            produced.append(len(out))
            return out

        def flush(self, *a):
            out = self._d.flush(*a)
            produced.append(len(out))
            return out

        @property
        def eof(self):
            return self._d.eof

    monkeypatch.setattr(remotezip.zlib, "decompressobj", Spy)
    with pytest.raises(ValueError, match="declared size"):
        _decompress(info, raw)
    assert produced and sum(produced) <= 1001
