"""Cost and abuse guard for the deployed NextView endpoint (spec 10.4), plus the CloudWatch Logs
Insights numbers for the report's operations table (spec 10.6).

Check (the default; a dry run unless ``--apply``)
    Reads the last 24 h of CloudWatch metrics for the function (``AWS/Lambda`` Duration, Invocations,
    Errors, Throttles, ConcurrentExecutions) and its HTTP API (``AWS/ApiGateway`` Count, 4xx, 5xx),
    converts the Duration sum to GB-s with the function's configured memory, and picks a tier:

      ok        below --warn-gbs
      warn      >= --warn-gbs       reported only
      throttle  >= --throttle-gbs   stage ``$default`` lowered to --low-rate / --low-burst
      kill      >= --kill-gbs       the hard daily ceiling: reserved concurrency 0 (every request then
                                    gets HTTP 503), the stage is throttled too, and GUARD-TRIPPED.txt
                                    is written at the repo root

    Only ``--apply`` changes anything. The kill switch is never undone automatically. A throttle at
    exactly --low-rate / --low-burst is relaxed back to --rate / --burst once 24 h usage is below the
    warn tier again (hysteresis; ``--no-relax`` keeps it). Each run appends one JSON line to --log.

Restore (``--restore``; ``--dry-run`` only shows the plan)
    Removes a reserved concurrency of 0, sets the stage back to --rate / --burst and moves
    GUARD-TRIPPED.txt into results/tmp/.

Insights (``--insights OUT``; read-only)
    Runs the Logs Insights queries (latency per route, status and error counts, platform failures,
    cold starts, NBV time, GB-s) over two windows, from the first retained log event and from the
    function's last update, adds the CloudWatch metric totals for the same windows, and writes OUT.

Default thresholds (GB-s per rolling 24 h) assume the deployed 512 MB function. A typical step is
about 1.37 s x 0.5 GB = 0.68 GB-s. The 2 req/s stage throttle bounds a stream of typical steps to
about 118k GB-s/day; the account concurrency cap of 5 bounds anything to 5 x 0.5 GB x 86 400 s =
216k GB-s/day. The Lambda free tier (400k GB-s/month) is about 13.3k GB-s/day. Defaults: warn 15k,
throttle 40k, kill 100k (about US$1.67/day at the x86 list price of US$0.0000166667 per GB-s).
The Duration metric excludes init time, so cold-start-heavy traffic reads slightly low.

Credentials come from the standard AWS environment, or from ``--profile`` and ``--aws-dir`` (a
directory holding ``credentials`` and ``config``). The script calls the AWS CLI v2 and never reads
or prints keys.

Exit codes: 0 ok, 3 warn, 4 throttle, 5 kill or kill switch still engaged, 1 failure (restore and
insights: 0 or 1).

Hourly scheduled task (Windows, one line; dates in d/M/yyyy, the en-SG short date format):

    schtasks /Create /F /TN "NextView guard" /SC HOURLY /MO 1 /SD 26/10/2026 /ST 00:05 /ED 10/11/2026
        /TR "<repo>\\.venv\\Scripts\\pythonw.exe <repo>\\infra\\guard.py --apply --profile <profile> --aws-dir <dir>"

Examples:

    python infra/guard.py                         # dry run: usage, tier, what --apply would do
    python infra/guard.py --apply                 # act on the tier
    python infra/guard.py --restore               # undo the kill switch and the throttle
    python infra/guard.py --insights results/ops_insights.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO / "results" / "tmp" / "guard.jsonl"
DEFAULT_TRIPPED = REPO / "GUARD-TRIPPED.txt"
USD_PER_GBS = 0.0000166667  # Lambda x86 list price
USD_PER_REQUEST = 0.20 / 1e6  # Lambda request list price
EXIT = {"ok": 0, "warn": 3, "throttle": 4, "kill": 5}


class AwsError(RuntimeError):
    pass


class Aws:
    """Thin wrapper over the AWS CLI v2 that returns parsed JSON."""

    def __init__(self, region: str, cli: str | None = None, env: dict | None = None, runner=subprocess.run):
        self.region = region
        self.cli = cli or shutil.which("aws") or "aws"
        self.env = env
        self.runner = runner

    def __call__(self, *args: str):
        cmd = [self.cli, *args, "--region", self.region, "--output", "json"]
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = self.runner(cmd, capture_output=True, env=self.env, timeout=180, **kw)
        out = p.stdout.decode("utf-8", "replace") if isinstance(p.stdout, bytes) else (p.stdout or "")
        err = p.stderr.decode("utf-8", "replace") if isinstance(p.stderr, bytes) else (p.stderr or "")
        if p.returncode != 0:
            raise AwsError(f"aws {' '.join(args[:2])} failed: {err.strip()[:600]}")
        out = out.strip()
        return json.loads(out) if out else None


def aws_env(profile: str | None, aws_dir: str | None) -> dict:
    env = dict(os.environ)
    if aws_dir:
        env["AWS_SHARED_CREDENTIALS_FILE"] = str(Path(aws_dir) / "credentials")
        env["AWS_CONFIG_FILE"] = str(Path(aws_dir) / "config")
    if profile:
        env["AWS_PROFILE"] = profile
    return env


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    m = re.match(r"^(.*[+-]\d\d)(\d\d)$", s)  # Lambda LastModified: 2026-10-23T03:30:23.000+0000
    if m:
        s = f"{m.group(1)}:{m.group(2)}"
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- AWS reads


def function_info(aws: Aws, fn: str) -> dict:
    cfg = aws("lambda", "get-function-configuration", "--function-name", fn)
    conc = aws("lambda", "get-function-concurrency", "--function-name", fn) or {}
    reserved = conc.get("ReservedConcurrentExecutions")
    return {
        "memory_mb": int(cfg["MemorySize"]),
        "last_modified": iso(parse_time(cfg["LastModified"])),
        "version": ((cfg.get("Environment") or {}).get("Variables") or {}).get("NEXTVIEW_VERSION"),
        "reserved": None if reserved is None else int(reserved),
    }


def find_api(aws: Aws, name: str) -> dict:
    items = (aws("apigatewayv2", "get-apis") or {}).get("Items", [])
    api = next((a for a in items if a.get("Name") == name), None)
    if api is None:
        raise AwsError(f"no HTTP API named {name}")
    return api


def stage_throttle(aws: Aws, api_id: str) -> tuple[float | None, int | None]:
    st = aws("apigatewayv2", "get-stage", "--api-id", api_id, "--stage-name", "$default")
    rs = (st or {}).get("DefaultRouteSettings") or {}
    rate, burst = rs.get("ThrottlingRateLimit"), rs.get("ThrottlingBurstLimit")
    return (None if rate is None else float(rate), None if burst is None else int(burst))


def metric_totals(aws: Aws, fn: str, api_id: str | None, start: datetime, end: datetime) -> dict:
    """Sums (and the concurrency maximum) of the guard's metrics over [start, end), 1-minute periods."""
    specs = [
        ("duration_ms", "AWS/Lambda", "Duration", "Sum", {"FunctionName": fn}),
        ("invocations", "AWS/Lambda", "Invocations", "Sum", {"FunctionName": fn}),
        ("errors", "AWS/Lambda", "Errors", "Sum", {"FunctionName": fn}),
        ("throttles", "AWS/Lambda", "Throttles", "Sum", {"FunctionName": fn}),
        ("concurrency_max", "AWS/Lambda", "ConcurrentExecutions", "Maximum", {"FunctionName": fn}),
    ]
    if api_id:
        specs += [
            ("api_requests", "AWS/ApiGateway", "Count", "Sum", {"ApiId": api_id}),
            ("api_4xx", "AWS/ApiGateway", "4xx", "Sum", {"ApiId": api_id}),
            ("api_5xx", "AWS/ApiGateway", "5xx", "Sum", {"ApiId": api_id}),
        ]
    queries = [{
        "Id": key.replace("_", "").lower(),
        "MetricStat": {
            "Metric": {"Namespace": ns, "MetricName": name,
                       "Dimensions": [{"Name": k, "Value": v} for k, v in dims.items()]},
            "Period": 60, "Stat": stat,
        },
        "ReturnData": True,
    } for key, ns, name, stat, dims in specs]
    ids = {q["Id"]: spec[0] for q, spec in zip(queries, specs)}
    values: dict[str, list[float]] = {key: [] for key, *_ in specs}
    fd, qpath = tempfile.mkstemp(prefix="nv-guard-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(queries, f)
        token = None
        while True:
            args = ["cloudwatch", "get-metric-data", "--metric-data-queries", "file://" + Path(qpath).as_posix(),
                    "--start-time", iso(start), "--end-time", iso(end)]
            if token:
                args += ["--next-token", token]
            r = aws(*args) or {}
            for res in r.get("MetricDataResults", []):
                values[ids[res["Id"]]].extend(float(v) for v in res.get("Values", []))
            token = r.get("NextToken")
            if not token:
                break
    finally:
        os.unlink(qpath)
    out = {}
    for key, _ns, _name, stat, _dims in specs:
        vs = values[key]
        out[key] = (max(vs) if vs else 0.0) if stat == "Maximum" else sum(vs)
    for key in out:
        if key != "duration_ms":
            out[key] = int(round(out[key]))
    return out


# ---------------------------------------------------------------- decision (pure)


def tier_for(gbs: float, warn: float, throttle: float, kill: float) -> str:
    if gbs >= kill:
        return "kill"
    if gbs >= throttle:
        return "throttle"
    if gbs >= warn:
        return "warn"
    return "ok"


def decide(gbs: float, *, stage: tuple, reserved: int | None, tripped_file_exists: bool, warn: float,
           throttle: float, kill: float, low_rate: float, low_burst: int, relax: bool = True) -> tuple[str, list[str]]:
    """Tier for ``gbs`` and the actions that bring the endpoint to it.

    Actions: ``kill`` (reserved concurrency 0), ``throttle`` (stage to the low limits; only ever lowers),
    ``tripped`` (write GUARD-TRIPPED.txt), ``relax`` (stage back to the normal limits).
    """
    tier = tier_for(gbs, warn, throttle, kill)
    rate, burst = stage
    killed = reserved == 0
    above_low = rate is None or burst is None or rate > low_rate or burst > low_burst
    at_low = rate is not None and burst is not None and math.isclose(rate, low_rate) and burst == low_burst
    actions: list[str] = []
    if tier == "kill":
        if not killed:
            actions.append("kill")
        if above_low:
            actions.append("throttle")
        if not tripped_file_exists:
            actions.append("tripped")
    elif tier == "throttle":
        if above_low:
            actions.append("throttle")
    elif tier == "ok" and relax and at_low and not killed:
        actions.append("relax")
    return tier, actions


# ---------------------------------------------------------------- actions


def set_stage(aws: Aws, api_id: str, rate: float, burst: int) -> None:
    rate_s = f"{rate:g}"
    aws("apigatewayv2", "update-stage", "--api-id", api_id, "--stage-name", "$default",
        "--default-route-settings", f"ThrottlingBurstLimit={burst},ThrottlingRateLimit={rate_s}")


def write_tripped(path: Path, rec: dict) -> None:
    lines = [
        "NextView guard: KILL SWITCH ENGAGED",
        "",
        f"time (UTC):           {rec['time']}",
        f"function:             {rec['function']}",
        f"24 h usage:           {rec['gbs_24h']:.0f} GB-s (hard ceiling {rec['thresholds']['kill_gbs']:.0f})",
        f"24 h invocations:     {rec['metrics_24h'].get('invocations')}",
        f"24 h API requests:    {rec['metrics_24h'].get('api_requests')}",
        "",
        "Reserved concurrency is 0, so every request now gets HTTP 503, and the stage is throttled.",
        "Look at the CloudWatch logs before restoring. To restore:",
        "",
        "    python infra/guard.py --restore",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_log(path: Path | None, rec: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except OSError as e:  # logging must never stop the guard
        print(f"warning: could not write {path}: {e}", file=sys.stderr)


def run_check(aws: Aws, a, now: datetime | None = None) -> tuple[dict, int]:
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(hours=a.hours)
    info = function_info(aws, a.function)
    api = find_api(aws, a.api_name)
    stage = stage_throttle(aws, api["ApiId"])
    m = metric_totals(aws, a.function, api["ApiId"], start, now)
    gbs = m["duration_ms"] / 1000.0 * info["memory_mb"] / 1024.0
    tripped = Path(a.tripped_file)
    tier, actions = decide(gbs, stage=stage, reserved=info["reserved"], tripped_file_exists=tripped.exists(),
                           warn=a.warn_gbs, throttle=a.throttle_gbs, kill=a.kill_gbs,
                           low_rate=a.low_rate, low_burst=a.low_burst, relax=not a.no_relax)
    rec = {
        "time": iso(now), "mode": "check", "dry_run": not a.apply, "function": a.function,
        "version": info["version"], "memory_mb": info["memory_mb"], "window_h": a.hours,
        "gbs_24h": round(gbs, 1),
        "usd_24h_list_price": round(gbs * USD_PER_GBS + m["invocations"] * USD_PER_REQUEST, 4),
        "metrics_24h": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in m.items()},
        "stage": {"rate": stage[0], "burst": stage[1]}, "reserved": info["reserved"],
        "thresholds": {"warn_gbs": a.warn_gbs, "throttle_gbs": a.throttle_gbs, "kill_gbs": a.kill_gbs},
        "tier": tier, "actions": actions, "done": [],
    }
    if a.apply:
        for act in actions:
            if act == "kill":
                aws("lambda", "put-function-concurrency", "--function-name", a.function,
                    "--reserved-concurrent-executions", "0")
            elif act == "throttle":
                set_stage(aws, api["ApiId"], a.low_rate, a.low_burst)
            elif act == "relax":
                set_stage(aws, api["ApiId"], a.rate, a.burst)
            elif act == "tripped":
                write_tripped(tripped, rec)
            rec["done"].append(act)
    code = EXIT[tier]
    if info["reserved"] == 0 and "kill" not in actions:
        rec["note"] = "kill switch already engaged; run --restore once the cause is understood"
        code = EXIT["kill"]
    return rec, code


def run_restore(aws: Aws, a, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    info = function_info(aws, a.function)
    api = find_api(aws, a.api_name)
    stage = stage_throttle(aws, api["ApiId"])
    tripped = Path(a.tripped_file)
    plan = []
    if info["reserved"] == 0:
        plan.append("unkill")
    if stage[0] is None or stage[1] is None or not math.isclose(stage[0], a.rate) or stage[1] != a.burst:
        plan.append("relax")
    if tripped.exists():
        plan.append("archive_tripped")
    rec = {"time": iso(now), "mode": "restore", "dry_run": a.dry_run, "function": a.function,
           "reserved": info["reserved"], "stage": {"rate": stage[0], "burst": stage[1]},
           "target_stage": {"rate": a.rate, "burst": a.burst}, "actions": plan, "done": []}
    if not a.dry_run:
        for act in plan:
            if act == "unkill":
                aws("lambda", "delete-function-concurrency", "--function-name", a.function)
            elif act == "relax":
                set_stage(aws, api["ApiId"], a.rate, a.burst)
            elif act == "archive_tripped":
                dest = REPO / "results" / "tmp" / f"GUARD-TRIPPED-{now.strftime('%Y%m%dT%H%M%SZ')}.txt"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tripped), dest)
                rec["archived_to"] = str(dest)
            rec["done"].append(act)
    return rec


# ---------------------------------------------------------------- Logs Insights


INSIGHTS_QUERIES = {
    "lambda_report": (
        'filter @type = "REPORT"\n'
        "| stats count(*) as invocations, pct(@duration, 50) as p50_ms, pct(@duration, 95) as p95_ms,"
        " max(@duration) as max_ms, count(@initDuration) as cold_starts, avg(@initDuration) as init_avg_ms,"
        " max(@initDuration) as init_max_ms, sum(@billedDuration * @memorySize) / 1000 / 1073741824 as gb_s,"
        " max(@maxMemoryUsed) / 1048576 as max_memory_mb"
    ),
    "routes_200": (
        "filter ispresent(route) and ispresent(latency_ms.total) and status = 200\n"
        "| stats count(*) as n, pct(latency_ms.total, 50) as p50_ms, pct(latency_ms.total, 95) as p95_ms,"
        " max(latency_ms.total) as max_ms by route, mode\n"
        "| sort n desc"
    ),
    "status_counts": (
        "filter ispresent(route) and ispresent(status)\n"
        "| stats count(*) as n by status\n"
        "| sort status asc"
    ),
    "non_200": (
        "filter ispresent(route) and status != 200\n"
        "| stats count(*) as n by status, error, route\n"
        "| sort n desc\n"
        "| limit 40"
    ),
    "handler_exceptions": (
        'filter evt = "error"\n'
        "| stats count(*) as n by error, route\n"
        "| sort n desc"
    ),
    "platform_failures": (
        r"filter @message like /Task timed out|Runtime\.[A-Za-z]+Error|Runtime\.OutOfMemory|Process exited before completing/"
        "\n"
        r"| parse @message /(?<failure>Task timed out|Runtime\.[A-Za-z]+|Process exited before completing)/"
        "\n| stats count(*) as n by failure"
    ),
    "cold_by_route": (
        "filter ispresent(route) and ispresent(cold)\n"
        "| stats count(*) as n by route, cold\n"
        "| sort n desc"
    ),
    "init_warm": (
        'filter evt = "init_warm"\n'
        "| stats count(*) as n, pct(ms, 50) as p50_ms, max(ms) as max_ms, count(error) as failed"
    ),
    "nbv_sim": (
        'filter evt = "step" and mode = "sim" and status = 200 and ispresent(latency_ms.nbv)\n'
        "| stats count(*) as n, pct(latency_ms.nbv, 50) as p50_ms, pct(latency_ms.nbv, 95) as p95_ms,"
        " max(latency_ms.nbv) as max_ms, pct(n_candidates, 50) as candidates_p50, min(n_candidates) as candidates_min"
    ),
    "rules": (
        'filter evt = "step" and status = 200\n'
        "| stats count(*) as n by rule\n"
        "| sort n desc"
    ),
}


def _num(v: str):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return None
    return int(f) if f.is_integer() and abs(f) < 1e15 else round(f, 1)


def insights_query(aws: Aws, log_group: str, query: str, start: datetime, end: datetime,
                   timeout_s: float = 120) -> tuple[dict, list[dict]]:
    """Rows of one Logs Insights query (numbers parsed), and the scan statistics."""
    r = aws("logs", "start-query", "--log-group-name", log_group, "--start-time", str(int(start.timestamp())),
            "--end-time", str(int(end.timestamp())), "--query-string", query, "--limit", "1000")
    qid = r["queryId"]
    t0 = time.monotonic()
    while True:
        time.sleep(1.0)
        res = aws("logs", "get-query-results", "--query-id", qid)
        status = res.get("status")
        if status == "Complete":
            break
        if status in ("Failed", "Cancelled", "Timeout", "Unknown"):
            raise AwsError(f"Logs Insights query {status}: {query.splitlines()[0]}")
        if time.monotonic() - t0 > timeout_s:
            raise AwsError(f"Logs Insights query still {status} after {timeout_s:.0f} s")
    rows = []
    for row in res.get("results", []):
        rows.append({c["field"]: _num(c.get("value")) for c in row if not c["field"].startswith("@ptr")})
    stats = res.get("statistics") or {}
    meta = {"records_matched": _num(stats.get("recordsMatched")), "records_scanned": _num(stats.get("recordsScanned"))}
    return meta, rows


def run_insights(aws: Aws, a, now: datetime | None = None) -> dict:
    generated = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    # End the windows a few minutes back so that every log line in them has been ingested.
    now = generated - timedelta(minutes=3)
    info = function_info(aws, a.function)
    log_group = f"/aws/lambda/{a.function}"
    groups = (aws("logs", "describe-log-groups", "--log-group-name-prefix", log_group) or {}).get("logGroups", [])
    lg = next((g for g in groups if g.get("logGroupName") == log_group), None)
    if lg is None:
        raise AwsError(f"log group {log_group} not found")
    created = datetime.fromtimestamp(lg["creationTime"] / 1000, tz=timezone.utc)
    retention = lg.get("retentionInDays")
    first = max(created, now - timedelta(days=retention)) if retention else created
    try:
        api_id = find_api(aws, a.api_name)["ApiId"]
    except AwsError:
        api_id = None
    windows = {
        "all": {"start": first, "label": "from the first retained log event (log group created "
                                          f"{iso(created)}, retention {retention} days)"},
        "build": {"start": min(parse_time(info["last_modified"]), now),
                  "label": f"from the last function update (image tag {info['version']})"},
    }
    out = {
        "generated_at": iso(generated),
        "function": a.function, "memory_mb": info["memory_mb"], "region": aws.region,
        "live_version": info["version"], "live_since": info["last_modified"],
        "traffic_note": ("Before the entry was published (27 Oct 2026), all traffic was the author's own testing: "
                         "deploy smoke tests, load tests at up to 10 concurrent clients against a 2 req/s stage "
                         "throttle and a concurrency cap of 5, sequential latency runs and UI walkthroughs. The 429 "
                         "and 503 responses come from the load tests."),
        "notes": [
            "Latency fields are the handler's own latency_ms.total (server side); Lambda REPORT durations include "
            "the runtime overhead around the handler.",
            "gb_s is sum(billedDuration x memorySize); for container images the billed duration includes init.",
            "API Gateway 4xx includes the stage throttle's 429s, which never reach Lambda; Lambda Throttles are the "
            "concurrency cap's 503s.",
        ],
        "queries": INSIGHTS_QUERIES,
        "windows": {},
    }
    for key, w in windows.items():
        res = {"start": iso(w["start"]), "end": iso(now), "label": w["label"]}
        for name, q in INSIGHTS_QUERIES.items():
            meta, rows = insights_query(aws, log_group, q, w["start"], now)
            res[name] = rows
            if name == "lambda_report":
                res["records_scanned"] = meta.get("records_scanned")
        cw = metric_totals(aws, a.function, api_id, w["start"], now)
        cw["duration_ms"] = round(cw["duration_ms"], 1)
        res["cloudwatch_metrics"] = cw
        handler_4xx = sum(r.get("n", 0) for r in res["status_counts"]
                          if isinstance(r.get("status"), int) and 400 <= r["status"] < 500)
        handler_5xx = sum(r.get("n", 0) for r in res["status_counts"]
                          if isinstance(r.get("status"), int) and r["status"] >= 500)
        res["derived"] = {
            "handler_4xx": handler_4xx,
            "handler_5xx": handler_5xx,
            "stage_throttle_429_est": (cw["api_4xx"] - handler_4xx) if "api_4xx" in cw else None,
            "concurrency_cap_503": cw["throttles"],
            "note": "stage_throttle_429_est = API Gateway 4xx minus the handler's own 4xx; "
                    "concurrency_cap_503 = Lambda Throttles (returned to the client as 503).",
        }
        out["windows"][key] = res
    return out


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="take the tier's action (default: dry run)")
    mode.add_argument("--restore", action="store_true", help="undo the kill switch and the throttle")
    mode.add_argument("--insights", metavar="OUT", help="write the Logs Insights numbers to OUT (JSON)")
    p.add_argument("--dry-run", action="store_true", help="with --restore: show the plan only")
    p.add_argument("--function", default="opencv-hackathon-nextview")
    p.add_argument("--api-name", default=None, help="HTTP API name (default: the function name)")
    p.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-1")
    p.add_argument("--profile", default=None, help="AWS profile (default: the environment's)")
    p.add_argument("--aws-dir", default=None, help="directory with the AWS 'credentials' and 'config' files")
    p.add_argument("--aws-cli", default=None, help="path to the aws executable (default: on PATH)")
    p.add_argument("--hours", type=float, default=24.0, help="usage window (default 24)")
    p.add_argument("--warn-gbs", type=float, default=15_000.0)
    p.add_argument("--throttle-gbs", type=float, default=40_000.0)
    p.add_argument("--kill-gbs", type=float, default=100_000.0, help="hard daily ceiling")
    p.add_argument("--rate", type=float, default=2.0, help="normal stage rate limit (req/s)")
    p.add_argument("--burst", type=int, default=5, help="normal stage burst limit")
    p.add_argument("--low-rate", type=float, default=1.0, help="throttle tier rate limit")
    p.add_argument("--low-burst", type=int, default=2, help="throttle tier burst limit")
    p.add_argument("--no-relax", action="store_true", help="never relax a throttle automatically")
    p.add_argument("--log", default=str(DEFAULT_LOG), help="JSON-lines log ('' to disable)")
    p.add_argument("--tripped-file", default=str(DEFAULT_TRIPPED))
    p.add_argument("--json", action="store_true", help="print the full JSON record")
    return p


def summary(rec: dict) -> str:
    m = rec["metrics_24h"]
    st = rec["stage"]
    th = rec["thresholds"]
    acts = ", ".join(rec["actions"]) or "none"
    how = "dry run" if rec["dry_run"] else ("done: " + (", ".join(rec["done"]) or "none"))
    return (f"{rec['time']}  {rec['window_h']:g} h: {rec['gbs_24h']:.1f} GB-s at {rec['memory_mb']} MB "
            f"(warn {th['warn_gbs']:g} / throttle {th['throttle_gbs']:g} / kill {th['kill_gbs']:g}), "
            f"~US${rec['usd_24h_list_price']:.4f} list; {m['invocations']} invocations, {m['errors']} errors, "
            f"{m['throttles']} throttles, max concurrency {m['concurrency_max']}; API {m.get('api_requests')} requests "
            f"(4xx {m.get('api_4xx')}, 5xx {m.get('api_5xx')}); stage {st['rate']:g}/{st['burst']}, "
            f"reserved {rec['reserved'] if rec['reserved'] is not None else 'none'} -> tier {rec['tier'].upper()}, "
            f"actions: {acts} ({how})")


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    a.api_name = a.api_name or a.function
    if a.dry_run and not a.restore:
        a.apply = False
    aws = Aws(a.region, cli=a.aws_cli, env=aws_env(a.profile, a.aws_dir))
    log = Path(a.log) if a.log else None
    try:
        if a.insights:
            out = run_insights(aws, a)
            path = Path(a.insights)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
            for key, w in out["windows"].items():
                rep = (w["lambda_report"] or [{}])[0]
                print(f"{key}: {w['start']} -> {w['end']}: {rep.get('invocations')} invocations, "
                      f"p50 {rep.get('p50_ms')} ms, p95 {rep.get('p95_ms')} ms, {rep.get('cold_starts')} cold starts, "
                      f"{rep.get('gb_s')} GB-s")
            print(f"wrote {path}")
            return 0
        if a.restore:
            rec = run_restore(aws, a)
            append_log(log, rec)
            print(json.dumps(rec, indent=1) if a.json else
                  f"{rec['time']}  restore ({'dry run' if rec['dry_run'] else 'applied'}): "
                  f"reserved {rec['reserved'] if rec['reserved'] is not None else 'none'}, stage "
                  f"{rec['stage']['rate']}/{rec['stage']['burst']} -> actions: {', '.join(rec['actions']) or 'none'}")
            return 0
        rec, code = run_check(aws, a)
        append_log(log, rec)
        print(json.dumps(rec, indent=1) if a.json else summary(rec))
        if rec.get("note"):
            print("note: " + rec["note"])
        return code
    except (AwsError, subprocess.TimeoutExpired, KeyError, ValueError) as e:
        rec = {"time": iso(datetime.now(timezone.utc)), "mode": "error", "error": f"{type(e).__name__}: {e}"[:800]}
        append_log(log, rec)
        print(f"guard error: {rec['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
