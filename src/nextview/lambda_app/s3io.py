"""Where approved exports go (spec 2.7 ``export``, 10.5).

- ``S3Store``: on Lambda (``NEXTVIEW_BUCKET`` set). The YAML / PNG / JSON go to
  ``reports/<sid>/<approval_id>/``, the trace record to ``traces/<sid>/<approval_id>.json``, and
  presigned GET links (TTL 3600 s, SigV4, regional endpoint) are returned. The bucket is private.
- ``LocalStore``: the local dev server (``NEXTVIEW_LOCAL_EXPORTS=1``). Files are kept in this
  process and served by ``GET /api/export/<key>/<name>``.
- Neither, or a failed upload: the files are returned inline (base64) so the client can still
  save them.

boto3 comes from the Lambda base image; it is imported only when an export is stored.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import threading
import time
from collections import OrderedDict

__all__ = ["EXPORT_FILES", "PRESIGN_TTL_S", "ExportError", "S3Store", "LocalStore", "get_store", "set_store",
           "inline_export", "LOCAL_KEY_RE"]

PRESIGN_TTL_S = 3600
# kind -> (file name, content type)
EXPORT_FILES = {
    "yaml": ("nextview_calibration.yaml", "application/x-yaml"),
    "png": ("nextview_report.png", "image/png"),
    "json": ("nextview_calibration.json", "application/json"),
}
LOCAL_KEY_RE = re.compile(r"^[0-9a-f]{24}$")


class ExportError(RuntimeError):
    pass


def _safe(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(part))[:64] or "x"


class S3Store:
    kind = "s3"

    def __init__(self, bucket: str, region: str | None = None, client=None):
        self.bucket = bucket
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-1"
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3", region_name=self.region, endpoint_url=f"https://s3.{self.region}.amazonaws.com",
                config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"},
                              retries={"max_attempts": 2, "mode": "standard"}, connect_timeout=3, read_timeout=5))
        return self._client

    def put_export(self, sid: str, approval_id: str, files: dict, trace_record: dict | None = None) -> dict:
        prefix = f"reports/{_safe(sid)}/{_safe(approval_id)}/"
        urls = {}
        try:
            for kind, data in files.items():
                name, ctype = EXPORT_FILES[kind]
                key = prefix + name
                self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=ctype)
                urls[f"{kind}_url"] = self.client.generate_presigned_url(
                    "get_object", ExpiresIn=PRESIGN_TTL_S,
                    Params={"Bucket": self.bucket, "Key": key,
                            "ResponseContentDisposition": f'attachment; filename="{name}"'})
            if trace_record is not None:
                self.client.put_object(Bucket=self.bucket, Key=f"traces/{_safe(sid)}/{_safe(approval_id)}.json",
                                       Body=json.dumps(trace_record, separators=(",", ":")).encode("utf-8"),
                                       ContentType="application/json")
        except Exception as e:  # botocore raises many types; the caller falls back to inline files
            raise ExportError(f"S3 export failed: {type(e).__name__}: {e}") from e
        return {**urls, "expires_s": PRESIGN_TTL_S, "store": "s3", "prefix": prefix}


class LocalStore:
    """In-process export store for the dev server (bounded, newest 64 exports)."""

    kind = "local"
    MAX = 64

    def __init__(self):
        self._items: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def put_export(self, sid: str, approval_id: str, files: dict, trace_record: dict | None = None) -> dict:
        key = secrets.token_hex(12)
        with self._lock:
            self._items[key] = {"files": dict(files), "trace": trace_record, "t": time.time()}
            while len(self._items) > self.MAX:
                self._items.popitem(last=False)
        urls = {f"{kind}_url": f"/api/export/{key}/{EXPORT_FILES[kind][0]}" for kind in files}
        return {**urls, "expires_s": None, "store": "local"}

    def get(self, key: str, name: str) -> tuple[bytes, str] | None:
        with self._lock:
            item = self._items.get(key)
        if item is None:
            return None
        for kind, (fname, ctype) in EXPORT_FILES.items():
            if fname == name and kind in item["files"]:
                return item["files"][kind], ctype
        return None


def inline_export(files: dict) -> dict:
    """Files as base64 in the response (no store, or the upload failed)."""
    return {"store": "inline", "expires_s": None,
            "files": {kind: {"name": EXPORT_FILES[kind][0], "content_type": EXPORT_FILES[kind][1],
                             "b64": base64.b64encode(data).decode("ascii")} for kind, data in files.items()}}


_store = None
_store_set = False


def set_store(store) -> None:
    """Override the store (tests, dev server). ``None`` means inline only."""
    global _store, _store_set
    _store, _store_set = store, True


def get_store():
    global _store, _store_set
    if not _store_set:
        bucket = os.environ.get("NEXTVIEW_BUCKET")
        if bucket:
            _store = S3Store(bucket)
        elif os.environ.get("NEXTVIEW_LOCAL_EXPORTS") == "1":
            _store = LocalStore()
        else:
            _store = None
        _store_set = True
    return _store
