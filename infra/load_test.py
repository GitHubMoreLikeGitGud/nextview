"""Load and latency test against a deployed NextView endpoint (spec 10.6).

N sessions run in parallel (``--concurrency``), each for up to ``--steps`` agent calls
(``/api/step``, or ``/api/approve`` when a card is open: every card is approved, so sessions also
exercise the approval and export routes). Requests are retried exactly like the web UI does
(spec 8.4): 429 and 5xx (except the service's own ``INTERNAL`` error) and network errors, backoff
0.5, 1, 2, 4 s (each scaled by a random 0.5-1.5 jitter), at most 5 tries.

Reports, per route: logical requests, how many succeeded, HTTP 429 / 503 counts, how many needed a
retry and whether the retry succeeded, client latency of the successful try (p50/p95) and the
handler's own ``latency_ms.total`` (p50/p95). Per session: how many failed, and whether at the session
start or later.

``--busy-waits N`` also models the web UI's patience: when a request still fails after its 5 tries,
the session pauses 5 s and sends it again, up to N times at the start and N times in a row during the
run (the UI's session start and auto-run both use N = 6). The default 0 is the bare retry policy.

    python infra/load_test.py --sessions 10 --steps 10 --concurrency 10 --out results/ops/load_test.json
    python infra/load_test.py --sessions 3 --steps 20 --concurrency 1   # sequential latency profile
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

BACKOFF_S = (0.5, 1.0, 2.0, 4.0)
BUSY_WAIT_S = 5.0
BUSY = object()  # request() result: still 429/5xx/network error after every retry
PRESETS = ("wide_robot_handheld", "inspection_gauge_cam", "drone_survey_cam", "phone_crack_metrology")


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


class Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls: list[dict] = []

    def add(self, rec: dict):
        with self.lock:
            self.calls.append(rec)


def request(client: httpx.Client, url: str, route: str, body: dict | None, rec: Recorder, meta: dict):
    statuses, t_first = [], time.perf_counter()
    for attempt in range(len(BACKOFF_S) + 1):
        t = time.perf_counter()
        data, status, err = None, 0, None
        try:
            r = client.post(url + route, json=body) if body is not None else client.get(url + route)
            status = r.status_code
            try:
                data = r.json()
            except ValueError:
                data = None
        except httpx.HTTPError as e:
            err = type(e).__name__
        ms = 1000 * (time.perf_counter() - t)
        statuses.append(status or err)
        own = isinstance(data, dict) and isinstance(data.get("error"), dict) and data["error"].get("code")
        retryable = err is not None or status == 429 or (status >= 500 and own != "INTERNAL")
        if not retryable or attempt == len(BACKOFF_S):
            ok = 200 <= status < 300
            server = (data or {}).get("latency_ms", {}).get("total") if ok and isinstance(data, dict) else None
            rec.add({"route": route, "ok": ok, "status": status, "statuses": statuses, "tries": attempt + 1,
                     "client_ms": round(ms, 1), "total_ms": round(1000 * (time.perf_counter() - t_first), 1),
                     "server_ms": server, "rule": ((data or {}).get("decision") or {}).get("rule") if ok else None,
                     "error": None if ok else (own or err or f"HTTP {status}"), **meta})
            return data if ok else (BUSY if retryable else None)
        time.sleep(BACKOFF_S[attempt] * (0.5 + random.random()))  # same jitter as the web UI
    return None


class FrameMaker:
    """Webcam mode: renders the board locally (simulated 'normal' camera, no actuator error) at the
    pose the agent asked for, as a 1280x720 JPEG, and posts it to /api/frame."""

    def __init__(self, camera_seed: int):
        import cv2  # noqa: F401  (local import: only webcam mode needs the package)
        from nextview.sim import make_simulator
        from nextview.sim.poses import pose_for_view

        self.sim = make_simulator("normal", "robot", camera_seed=camera_seed, seed=0, occluder_p=0.0, corruption_p=0.0)
        self.first = [pose_for_view(self.sim.camera, (500 + 140 * k, 300 + 40 * k), 0.5, 25 + 5 * k, 70 * k) for k in range(3)]

    def jpeg_b64(self, k: int, target: dict | None) -> str:
        import base64

        import cv2
        import numpy as np

        if target and target.get("rvec"):
            rv, tv = np.asarray(target["rvec"], float), np.asarray(target["tvec"], float)
        else:
            rv, tv = self.first[k % 3]
        img = self.sim.render_exact(rv, tv)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("ascii")


def run_session(i: int, url: str, steps: int, preset: str, rec: Recorder, timeout: float, mode: str = "sim",
                busy_waits: int = 0) -> dict:
    out = {"session": i, "preset": preset if mode == "sim" else "webcam", "calls": 0, "status": None, "error": None,
           "busy_waits": 0}
    frames = FrameMaker(100 + i) if mode == "webcam" else None
    with httpx.Client(timeout=timeout, headers={"user-agent": "nextview-load-test/1"}) as c:
        body = ({"mode": "sim", "preset": preset, "seed": 1000 + i} if mode == "sim" else
                {"mode": "webcam", "seed": 1000 + i, "image_size": [1280, 720]})
        for w in range(busy_waits + 1):
            s = request(c, url, "/api/session", body, rec, {"session": i, "wait": w})
            if s is not BUSY or w == busy_waits:
                break
            out["busy_waits"] += 1
            time.sleep(BUSY_WAIT_S)
        if s is None or s is BUSY:
            out["error"] = "session start failed"
            return out
        state, card, target = s["state"], None, None
        k = run_waits = 0
        while k < steps:
            meta = {"session": i, "k": k}
            if card:
                opt = [o["decision"] for o in card.get("options", [])]
                d = "approve" if "approve" in opt else (opt[0] if opt else "approve")
                r = request(c, url, "/api/approve", {"state": state, "approval_id": card["id"], "decision": d,
                                                     "reason": "load test"}, rec, meta)
            elif frames is not None:
                r = request(c, url, "/api/frame", {"state": state, "jpeg_b64": frames.jpeg_b64(k, target)}, rec, meta)
            else:
                r = request(c, url, "/api/step", {"state": state}, rec, meta)
            out["calls"] += 1
            if r is BUSY and run_waits < busy_waits:
                run_waits += 1
                out["busy_waits"] += 1
                time.sleep(BUSY_WAIT_S)
                continue
            if r is None or r is BUSY:
                out["error"] = f"call {k} failed"
                break
            k, run_waits = k + 1, 0
            state, card, out["status"] = r["state"], r.get("approval"), r.get("status")
            target = (r.get("plan") or {}).get("target") or target
            if out["status"] in ("exported", "discarded", "ended"):
                break
    return out


def summarise(rec: Recorder, sessions: list, wall_s: float, args) -> dict:
    by = defaultdict(list)
    for c in rec.calls:
        by[c["route"]].append(c)
    routes = {}
    all_statuses = [s for c in rec.calls for s in c["statuses"]]
    for route, cs in sorted(by.items()):
        ok = [c for c in cs if c["ok"]]
        retried = [c for c in cs if c["tries"] > 1]
        st = [s for c in cs for s in c["statuses"]]
        routes[route] = {
            "requests": len(cs), "ok": len(ok), "failed": len(cs) - len(ok),
            "http_responses": len(st), "http_429": st.count(429), "http_503": st.count(503),
            "http_5xx_other": sum(1 for s in st if isinstance(s, int) and s >= 500 and s != 503),
            "network_errors": sum(1 for s in st if not isinstance(s, int)),
            "retried": len(retried), "retried_then_ok": sum(1 for c in retried if c["ok"]),
            "client_ms_p50": pct([c["client_ms"] for c in ok], 50), "client_ms_p95": pct([c["client_ms"] for c in ok], 95),
            "client_ms_max": max((c["client_ms"] for c in ok), default=None),
            "with_retries_ms_p95": pct([c["total_ms"] for c in ok], 95),
            "server_ms_p50": pct([c["server_ms"] for c in ok], 50), "server_ms_p95": pct([c["server_ms"] for c in ok], 95),
            "server_ms_max": max((c["server_ms"] for c in ok if c["server_ms"] is not None), default=None),
        }
    n429 = all_statuses.count(429)
    return {
        "url": args.url, "mode": args.mode, "sessions": args.sessions, "steps_per_session": args.steps, "concurrency": args.concurrency,
        "presets": sorted({s["preset"] for s in sessions}), "wall_s": round(wall_s, 1),
        "http_responses": len(all_statuses), "rate_429": round(n429 / max(1, len(all_statuses)), 4),
        "rate_503": round(all_statuses.count(503) / max(1, len(all_statuses)), 4),
        "logical_requests": len(rec.calls), "logical_failures": sum(1 for c in rec.calls if not c["ok"]),
        "busy_waits": args.busy_waits,
        "sessions_failed": sum(1 for s in sessions if s["error"]),
        "sessions_failed_at_start": sum(1 for s in sessions if s["error"] == "session start failed"),
        "session_failure_rate": round(sum(1 for s in sessions if s["error"]) / max(1, len(sessions)), 4),
        "routes": routes, "session_results": sessions,
        "rules": dict(sorted(Counter(c["rule"] for c in rec.calls if c.get("rule")).items())),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="API base URL (default: api_url in infra/deployed.json)")
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--preset", default=None, help="one preset for every session (default: rotate four presets)")
    ap.add_argument("--mode", choices=("sim", "webcam"), default="sim",
                    help="webcam: post locally rendered JPEG frames to /api/frame instead of /api/step")
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--calls", action="store_true", help="keep every call in the output")
    ap.add_argument("--busy-waits", type=int, default=0,
                    help="pause 5 s and resend after a request gives up, up to N times (the web UI uses 6)")
    a = ap.parse_args(argv)
    if not a.url:
        a.url = json.loads((Path(__file__).parent / "deployed.json").read_text(encoding="utf-8"))["api_url"]
    a.url = a.url.rstrip("/")
    rec = Recorder()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(run_session, i, a.url, a.steps, a.preset or PRESETS[i % len(PRESETS)], rec, a.timeout, a.mode,
                          a.busy_waits)
                for i in range(a.sessions)]
        sessions = [f.result() for f in futs]
    out = summarise(rec, sessions, time.perf_counter() - t0, a)
    if a.calls:
        out["calls"] = rec.calls
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    brief = {k: out[k] for k in ("wall_s", "http_responses", "rate_429", "rate_503", "logical_requests", "logical_failures",
                                 "sessions_failed", "sessions_failed_at_start", "session_failure_rate")}
    print(json.dumps(brief))
    for route, r in out["routes"].items():
        print(f"{route:14s} n={r['requests']:3d} ok={r['ok']:3d} 429={r['http_429']:3d} 503={r['http_503']:3d} "
              f"retried={r['retried']:3d} (ok {r['retried_then_ok']:3d})  client p50/p95 {r['client_ms_p50']}/{r['client_ms_p95']} ms"
              f"  server p50/p95 {r['server_ms_p50']}/{r['server_ms_p95']} ms")
    return 0 if out["sessions_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
