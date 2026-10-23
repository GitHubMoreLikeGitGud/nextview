"""Local dev server: stdlib ``http.server`` in front of the Lambda handler.

Each HTTP request becomes an API Gateway v2 (payload 2.0) event, so the local run exercises the
same code path as the deployed function. Requests are served one at a time, like one Lambda
container. Without ``NEXTVIEW_STATE_KEY`` a random per-process key is used (tokens die with the
process), and approved exports are kept in memory and served from ``/api/export/...``.

    python -m nextview.lambda_app.devserver --port 8765
"""

from __future__ import annotations

import argparse
import base64
import os
import secrets
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

TEXT_TYPES = ("application/json", "text/", "application/x-www-form-urlencoded")
MAX_BODY = 10 * 1024 * 1024  # API Gateway's own payload limit


def make_event(method: str, target: str, headers: dict, body: bytes | None, source_ip: str = "127.0.0.1") -> dict:
    """An API Gateway HTTP API (payload 2.0) event for one request."""
    u = urlsplit(target)
    hdrs = {str(k).lower(): str(v) for k, v in headers.items()}
    ev = {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": u.path or "/",
        "rawQueryString": u.query or "",
        "headers": hdrs,
        "requestContext": {
            "http": {"method": method.upper(), "path": u.path or "/", "protocol": "HTTP/1.1", "sourceIp": source_ip,
                     "userAgent": hdrs.get("user-agent", "")},
            "requestId": uuid.uuid4().hex,
            "routeKey": "$default",
            "stage": "$default",
            "timeEpoch": int(time.time() * 1000),
        },
        "isBase64Encoded": False,
    }
    if body:
        ctype = hdrs.get("content-type", "")
        if any(ctype.startswith(t) for t in TEXT_TYPES):
            try:
                ev["body"] = body.decode("utf-8")
            except UnicodeDecodeError:
                ev["body"], ev["isBase64Encoded"] = base64.b64encode(body).decode("ascii"), True
        else:
            ev["body"], ev["isBase64Encoded"] = base64.b64encode(body).decode("ascii"), True
    return ev


class _Context:
    """The part of the Lambda context object the handler reads."""

    def __init__(self, timeout_s: float):
        self._end = time.monotonic() + timeout_s
        self.aws_request_id = uuid.uuid4().hex

    def get_remaining_time_in_millis(self) -> int:
        return max(0, int(1000 * (self._end - time.monotonic())))


def make_server(host: str = "127.0.0.1", port: int = 8765, timeout_s: float = 30.0, quiet: bool = False):
    os.environ.setdefault("NEXTVIEW_LOCAL_EXPORTS", "1")
    os.environ.setdefault("NEXTVIEW_DEV_RELOAD", "1")
    if not os.environ.get("NEXTVIEW_STATE_KEY"):
        os.environ["NEXTVIEW_STATE_KEY"] = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
    from .handler import handler

    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "NextViewDev/1"
        protocol_version = "HTTP/1.1"

        def _serve(self, method: str) -> None:
            n = int(self.headers.get("content-length") or 0)
            if n > MAX_BODY:
                self.send_error(413)
                return
            body = self.rfile.read(n) if n else None
            ev = make_event(method, self.path, dict(self.headers.items()), body, self.client_address[0])
            with lock:
                out = handler(ev, _Context(timeout_s))
            data = out.get("body") or ""
            payload = base64.b64decode(data) if out.get("isBase64Encoded") else data.encode("utf-8")
            self.send_response(int(out.get("statusCode", 500)))
            for k, v in (out.get("headers") or {}).items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            self._serve("GET")

        def do_POST(self):  # noqa: N802
            self._serve("POST")

        def do_HEAD(self):  # noqa: N802
            self._serve("HEAD")

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write("dev %s - %s\n" % (self.address_string(), fmt % args))

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NextView local dev server (the Lambda handler behind http.server)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--timeout", type=float, default=30.0, help="simulated Lambda timeout per request, s")
    ap.add_argument("--quiet", action="store_true", help="no access log (the handler's JSON log lines still print)")
    a = ap.parse_args(argv)
    srv = make_server(a.host, a.port, a.timeout, a.quiet)
    print(f"NextView dev server on http://{a.host}:{srv.server_address[1]}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
