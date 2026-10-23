"""Pre-registered analysis of the synthetic TEST split -> ``summary.json`` (spec 11.5, 11.6).

    python -m nextview.bench.summarize --runs results/synth_test --stopcal config/stop_calibration.json

Primary endpoints (Holm-corrected across the 8 tests), all A minus baseline:
  PE1/PE2  true field p95 after 10 accepted frames, A vs B0, robot / human
  PE3/PE4  accepted frames until the true field p95 is first < 1 px, A vs B0, robot / human
           (never reached within 30 frames = 31)
  PE5/PE6  true field p95 at n = 10, A vs B2, robot / human
  PE7/PE8  true field p95 at n = 10, A vs B1, robot / human
Per camera the median over the 3 policy seeds is taken (100 paired values); two-sided Wilcoxon
signed-rank test; 95 % percentile-bootstrap CI (10,000 resamples of cameras, seed 0) of the
median paired difference. A run that ends before n frames contributes its last estimate.

Everything else (per family, ablations, stop quality, risk-coverage, uncertainty honesty,
outlier recall, held-out RMS, effort, predicted vs realised gain) is descriptive.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analysis import p_exceed, replay_stop, sessions_of, value_at, frames_to
from .common import COV_LEVELS, FAMILIES
from .files import load_runs, load_sessions
from .stats import holm, km_curve, km_median, paired_endpoint

__all__ = ["per_camera", "primary_endpoints", "summarize", "main"]

NEVER = 31
ACTUATORS = ("robot", "human")
PE = (
    ("PE1", "p95@10", "B0", "robot"), ("PE2", "p95@10", "B0", "human"),
    ("PE3", "frames<1px", "B0", "robot"), ("PE4", "frames<1px", "B0", "human"),
    ("PE5", "p95@10", "B2", "robot"), ("PE6", "p95@10", "B2", "human"),
    ("PE7", "p95@10", "B1", "robot"), ("PE8", "p95@10", "B1", "human"),
)
N_GRID = (3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30)
SWEEP = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def _metric(srows, metric: str) -> float:
    if metric == "p95@10":
        return value_at(srows, 10, "true_p95")
    if metric.startswith("p95@"):
        return value_at(srows, int(metric[4:]), "true_p95")
    if metric == "frames<1px":
        n, ev = frames_to(srows, 1.0, 30)
        return float(n if ev else NEVER)
    if metric.startswith("heldout@"):
        return value_at(srows, int(metric[8:]), "heldout_rms")
    raise ValueError(metric)


def per_camera(S, policy: str, actuator: str, metric: str, family: str | None = None) -> dict[int, float]:
    """{camera: median over seeds of the metric} for one policy and actuator."""
    vals: dict[int, list[float]] = defaultdict(list)
    for (cam, fam, seed, act, pol), srows in S.items():
        if pol != policy or act != actuator or (family and fam != family):
            continue
        vals[cam].append(_metric(srows, metric))
    out = {}
    for cam, v in vals.items():
        v = [x for x in v if math.isfinite(x)]
        out[cam] = float(np.median(v)) if v else math.nan
    return out


def _paired(S, pa, pb, actuator, metric, family=None, n_boot=10_000):
    a = per_camera(S, pa, actuator, metric, family)
    b = per_camera(S, pb, actuator, metric, family)
    cams = sorted(set(a) & set(b))
    xa = np.array([a[c] for c in cams])
    xb = np.array([b[c] for c in cams])
    res = paired_endpoint(xa, xb, n_boot=n_boot, seed=0)
    res["cameras"] = len(cams)
    res["missing_pairs"] = int(np.sum(~(np.isfinite(xa) & np.isfinite(xb))))
    return res


def primary_endpoints(S) -> list[dict]:
    out = []
    for pid, metric, base, act in PE:
        r = _paired(S, "A", base, act, metric)
        out.append({"id": pid, "metric": metric, "comparison": f"A - {base}", "actuator": act, **r})
    adj = holm([e["wilcoxon_p"] for e in out])
    for e, p in zip(out, adj):
        e["p_holm"] = p
        e["significant_holm_0.05"] = bool(p < 0.05)
    return out


def _q(v):
    v = np.asarray([x for x in v if math.isfinite(x)], float)
    if len(v) == 0:
        return None
    return {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "n": int(len(v))}


def curves(S, policies) -> dict:
    """E1: median (IQR) true field p95 per policy, actuator and n (runs, not camera medians)."""
    out = {}
    for pol in policies:
        for act in ACTUATORS:
            ss = [s for k, s in S.items() if k[4] == pol and k[3] == act]
            out[f"{pol}/{act}"] = {str(n): _q([value_at(s, n) for s in ss]) for n in N_GRID}
    return out


def km(S, policies) -> dict:
    out = {}
    for pol in policies:
        for act in ACTUATORS:
            ss = [s for k, s in S.items() if k[4] == pol and k[3] == act]
            tt = [frames_to(s, 1.0, 30) for s in ss]
            t = [x[0] for x in tt]
            e = [x[1] for x in tt]
            out[f"{pol}/{act}"] = {"median_frames": km_median(t, e), "reached": int(sum(e)), "runs": len(ss),
                                   "curve": km_curve(t, e)}
    return out


def effort(S, sessions, policies) -> dict:
    """Attempts (captures, including rejected frames) to reach n = 10 and n = 30 accepted frames."""
    out = {}
    for pol in policies:
        for act in ACTUATORS:
            ss = [s for k, s in S.items() if k[4] == pol and k[3] == act]
            a10 = [next((r["attempts"] for r in s if r["n"] >= 10), math.nan) for s in ss]
            a30 = [next((r["attempts"] for r in s if r["n"] >= 30), math.nan) for s in ss]
            reached30 = sum(1 for s in ss if s and s[-1]["n"] >= 30)
            out[f"{pol}/{act}"] = {"attempts_to_10": _q(a10), "attempts_to_30": _q(a30), "runs": len(ss),
                                   "runs_reaching_30_frames": reached30}
    return out


def stop_quality(S, sessions, quantiles, tau, threshold=0.10) -> dict:
    """A's stop (in-loop record and replay), false stops (> 2 tau), wasted frames, risk-coverage."""
    out = {"tau_px": tau, "threshold": threshold}
    sess = {(s["camera"], s["family"], s["seed"], s["actuator"], s["policy"]): s for s in sessions}
    for pol in ("A", "A-gate", "A-outlier", "A-model", "A-nbv"):
        for act in ACTUATORS:
            ss = {k: s for k, s in S.items() if k[4] == pol and k[3] == act}
            n_stop, true_at, wasted, agree, fam_stop = [], [], [], 0, defaultdict(list)
            for k, s in ss.items():
                r = replay_stop(s, quantiles[act], tau, threshold)
                inloop = (sess.get(k, {}).get("stop") or {}).get("n")
                agree += int((r is None and inloop is None) or (r is not None and inloop == r["n"]))
                if r is None:
                    fam_stop[k[1]].append(None)
                    continue
                n_stop.append(r["n"])
                true_at.append(r["true_p95"])
                fam_stop[k[1]].append(r["n"])
                first = next((x["n"] for x in s if math.isfinite(x["true_p95"]) and x["true_p95"] < tau), None)
                if first is not None:
                    wasted.append(r["n"] - first)
            ta = np.asarray(true_at, float)
            out[f"{pol}/{act}"] = {
                "runs": len(ss), "stopped": len(n_stop), "stop_rate": len(n_stop) / len(ss) if ss else None,
                "replay_matches_inloop": agree,
                "frames_at_stop": _q(n_stop), "true_p95_at_stop": _q(true_at),
                "false_stop_rate": float(np.mean(ta > 2 * tau)) if len(ta) else None,
                "false_stops": int(np.sum(ta > 2 * tau)),
                "frac_stops_true_p95_gt_tau": float(np.mean(ta > tau)) if len(ta) else None,
                "wasted_frames": _q(wasted), "wasted_frames_defined_runs": len(wasted),
                "by_family": {f: {"runs": len(v), "stopped": sum(1 for x in v if x is not None),
                                  "median_frames_at_stop": float(np.median([x for x in v if x is not None]))
                                  if any(x is not None for x in v) else None} for f, v in sorted(fam_stop.items())},
            }
    rc = {}
    for act in ACTUATORS:
        ss = [s for k, s in S.items() if k[4] == "A" and k[3] == act]
        pts = []
        for thr in SWEEP:
            frames, fs, stopped = [], [], 0
            for s in ss:
                r = replay_stop(s, quantiles[act], tau, thr)
                if r is None:
                    frames.append(30)
                else:
                    stopped += 1
                    frames.append(r["n"])
                    fs.append(r["true_p95"] > 2 * tau)
            pts.append({"stop_p_exceed": thr, "stopped": stopped, "runs": len(ss),
                        "false_stop_rate": float(np.mean(fs)) if fs else None,
                        "false_stops_per_run": float(np.sum(fs) / len(ss)) if ss else None,
                        "mean_frames": float(np.mean(frames)) if frames else None})
        rc[act] = pts
    out["risk_coverage"] = rc
    return out


def honesty(rows, quantiles, tau) -> dict:
    """Reliability of the first-order map (coverage at chi2(2) quantiles), r = true p95 / U95 and
    the calibration of P_exceed (binned predicted vs observed frequency of true p95 > tau)."""
    out = {}
    cols = ("cov50", "cov80", "cov90", "cov95")
    for pol in ("A", "B0"):
        for act in ACTUATORS:
            rr = [r for r in rows if r["policy"] == pol and r["actuator"] == act and r["views"] is not None and r["views"] >= 6
                  and math.isfinite(r["U95"]) and r["U95"] > 0 and math.isfinite(r["true_p95"])]
            F = np.array([[r[c] for c in cols] for r in rr], float)
            rel = []
            for j, q in enumerate(COV_LEVELS):
                col = F[:, j] if len(F) else np.zeros(0)
                col = col[np.isfinite(col)]
                rel.append({"nominal": q, "empirical_mean": float(col.mean()) if len(col) else None, "rows": int(len(col))})
            ratio = np.array([r["true_p95"] / r["U95"] for r in rr])
            entry = {"reliability": rel,
                     "r": {"n": int(len(ratio)), "median": float(np.median(ratio)) if len(ratio) else None,
                           "p10": float(np.percentile(ratio, 10)) if len(ratio) else None,
                           "p90": float(np.percentile(ratio, 90)) if len(ratio) else None,
                           "frac_gt1": float(np.mean(ratio > 1)) if len(ratio) else None,
                           "hist_edges": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5, 10, 1e9],
                           "hist": np.histogram(ratio, bins=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5, 10, 1e9])[0].tolist()}}
            if pol == "A":
                pe = np.array([p_exceed(r["U95"], quantiles[act], tau) for r in rr])
                ex = np.array([r["true_p95"] > tau for r in rr])
                bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001]
                cal = []
                for lo, hi in zip(bins[:-1], bins[1:]):
                    sel = (pe >= lo) & (pe < hi)
                    if sel.any():
                        cal.append({"bin": [lo, min(hi, 1.0)], "rows": int(sel.sum()), "mean_pred": float(pe[sel].mean()),
                                    "observed": float(ex[sel].mean())})
                entry["p_exceed_calibration"] = cal
            for fam in FAMILIES:
                fr = [r for r in rr if r["family"] == fam]
                if fr:
                    entry.setdefault("r_by_family", {})[fam] = float(np.median([r["true_p95"] / r["U95"] for r in fr]))
                    entry.setdefault("cov95_by_family", {})[fam] = float(np.nanmean([r["cov95"] for r in fr]))
            out[f"{pol}/{act}"] = entry
    return out


def gains(sessions) -> dict:
    out = {}
    for act in ACTUATORS:
        pg, rg, tg = [], [], []
        for s in sessions:
            if s.get("policy") != "A" or s.get("actuator") != act:
                continue
            for e in s.get("realised") or []:
                if e.get("pred_gain") is None or e.get("realised_gain") is None:
                    continue
                pg.append(e["pred_gain"])
                rg.append(e["realised_gain"])
                tg.append(e.get("true_gain", math.nan))
        pg, rg, tg = np.array(pg, float), np.array(rg, float), np.array(tg, float)
        ok = np.isfinite(tg)
        from scipy.stats import spearmanr

        out[act] = {"views": int(len(pg)),
                    "spearman_pred_vs_realised_U95_drop": float(spearmanr(pg, rg).statistic) if len(pg) > 2 else None,
                    "spearman_pred_vs_true_p95_drop": float(spearmanr(pg[ok], tg[ok]).statistic) if ok.sum() > 2 else None,
                    "median_pred": float(np.median(pg)) if len(pg) else None,
                    "median_realised": float(np.median(rg)) if len(rg) else None,
                    "median_true_drop": float(np.nanmedian(tg)) if ok.any() else None,
                    "frac_realised_positive": float(np.mean(rg > 0)) if len(rg) else None}
    return out


def outliers(sessions) -> dict:
    out = {}
    for pol in ("A", "A-outlier", "A-gate", "A-model", "A-nbv"):
        for act in ACTUATORS:
            ss = [s for s in sessions if s.get("policy") == pol and s.get("actuator") == act]
            ca = sum(s.get("corrupted_accepted", 0) for s in ss)
            cd = sum(s.get("corrupted_dropped", 0) for s in ss)
            fd = sum(s.get("clean_dropped", 0) for s in ss)
            out[f"{pol}/{act}"] = {"corrupted_views_accepted": ca, "corrupted_views_dropped": cd,
                                   "recall": cd / ca if ca else None, "clean_views_dropped": fd}
    return out


def descriptive_pairs(S, policies, metrics=("p95@10", "p95@20", "p95@30", "frames<1px", "heldout@10")) -> dict:
    """A minus each other policy, per actuator, all families and per family (descriptive only)."""
    out = {}
    for base in policies:
        if base == "A":
            continue
        for act in ACTUATORS:
            for metric in metrics:
                key = f"A-{base}|{act}|{metric}" if not base.startswith("A-") else f"A vs {base}|{act}|{metric}"
                e = {"all": _paired(S, "A", base, act, metric, n_boot=2000)}
                for fam in FAMILIES:
                    e[fam] = _paired(S, "A", base, act, metric, fam, n_boot=2000)
                out[key] = e
    return out


def per_policy_table(S, policies) -> dict:
    out = {}
    for pol in policies:
        for act in ACTUATORS:
            for fam in (None,) + FAMILIES:
                ss = [s for k, s in S.items() if k[4] == pol and k[3] == act and (fam is None or k[1] == fam)]
                e = {}
                for n in (10, 20, 30):
                    e[f"p95@{n}"] = _q([value_at(s, n) for s in ss])
                    e[f"mean@{n}"] = _q([value_at(s, n, "true_mean") for s in ss])
                    e[f"fx_rel_pct@{n}"] = _q([value_at(s, n, "fx_rel_pct") for s in ss])
                    e[f"cx_err_px@{n}"] = _q([value_at(s, n, "cx_err_px") for s in ss])
                    e[f"cy_err_px@{n}"] = _q([value_at(s, n, "cy_err_px") for s in ss])
                    e[f"heldout_rms@{n}"] = _q([value_at(s, n, "heldout_rms") for s in ss])
                    e[f"rms@{n}"] = _q([value_at(s, n, "rms") for s in ss])
                out[f"{pol}/{act}/{fam or 'all'}"] = e
    return out


def a_stop_fixed(S, sessions, quantiles, tau) -> dict:
    """A-stop: A's trajectory read at a fixed budget = the median in-loop stop of A (per actuator)."""
    out = {}
    for act in ACTUATORS:
        ns = [((s.get("stop") or {}).get("n")) for s in sessions if s.get("policy") == "A" and s.get("actuator") == act]
        ns = [n for n in ns if n is not None]
        if not ns:
            continue
        nb = int(round(float(np.median(ns))))
        ss = [s for k, s in S.items() if k[4] == "A" and k[3] == act]
        at_fixed = [value_at(s, nb) for s in ss]
        at_stop = []
        for s in ss:
            r = replay_stop(s, quantiles[act], tau, 0.10)
            at_stop.append(value_at(s, 30) if r is None else r["true_p95"])
        out[act] = {"fixed_budget": nb, "true_p95_at_fixed_budget": _q(at_fixed),
                    "true_p95_at_own_stop_or_30": _q(at_stop),
                    "false_rate_fixed_budget_gt_2tau": float(np.mean(np.asarray(at_fixed) > 2 * tau))}
    return out


def b2_stop(S, sessions) -> dict:
    out = {}
    for pol in ("B2", "B2-k5"):
        for act in ACTUATORS:
            ss = [s for s in sessions if s.get("policy") == pol and s.get("actuator") == act]
            ns = [(s.get("stop") or {}).get("n") for s in ss]
            stopped = [n for n in ns if n is not None]
            tp = []
            for s in ss:
                n = (s.get("stop") or {}).get("n")
                if n is not None:
                    tp.append(value_at(S.get((s["camera"], s["family"], s["seed"], s["actuator"], pol), []), n))
            out[f"{pol}/{act}"] = {"runs": len(ss), "stopped": len(stopped), "frames_at_stop": _q(stopped),
                                   "true_p95_at_stop": _q(tp)}
    return out


def summarize(runs_dir: Path, stopcal_path: Path, tau: float | None = None) -> dict:
    rows = load_runs(runs_dir)
    sessions = load_sessions(runs_dir)
    sc = json.loads(stopcal_path.read_text(encoding="utf-8"))
    quantiles = sc["quantiles"]
    if tau is None:
        from ..config import load_thresholds

        tau = float(load_thresholds().tau_px)
    S = sessions_of(rows)
    policies = sorted({k[4] for k in S}, key=lambda p: (p != "A", p))
    meta = json.loads((runs_dir / "meta.json").read_text(encoding="utf-8")) if (runs_dir / "meta.json").exists() else {}
    errors = [s for s in sessions if "error" in s]
    summary = {
        "split": meta.get("split"), "runs": len(S), "rows": len(rows), "errors": len(errors),
        "policies": policies, "tau_px": tau, "stop_calibration_sha256": meta.get("stop_calibration_sha256"),
        "thresholds_sha256": meta.get("thresholds_sha256"), "source_tree_sha256": meta.get("code", {}).get("source_tree_sha256"),
        "primary_endpoints": primary_endpoints(S),
        "curves_E1": curves(S, policies),
        "frames_to_1px_E3": km(S, policies),
        "effort_attempts": effort(S, sessions, policies),
        "stop_quality": stop_quality(S, sessions, quantiles, tau),
        "a_stop_fixed_budget": a_stop_fixed(S, sessions, quantiles, tau),
        "b2_app_stop": b2_stop(S, sessions),
        "uncertainty_honesty_E5": honesty(rows, quantiles, tau),
        "predicted_vs_realised_E10": gains(sessions),
        "outlier_recall": outliers(sessions),
        "per_policy": per_policy_table(S, policies),
        "descriptive_pairs": descriptive_pairs(S, policies),
        "end_rules": _end_rules(sessions),
        "approvals": _approvals(sessions),
        "latency_local_ms": _latency(sessions),
    }
    return summary


def _end_rules(sessions):
    out = defaultdict(lambda: defaultdict(int))
    for s in sessions:
        out[f"{s.get('policy')}/{s.get('actuator')}"][(s.get("end") or {}).get("rule", "ERROR")] += 1
    return {k: dict(v) for k, v in out.items()}


def _approvals(sessions):
    out = defaultdict(lambda: defaultdict(int))
    for s in sessions:
        for a in s.get("approvals") or []:
            out[f"{s['policy']}/{s['family']}/{s['actuator']}"][f"{a['rule']}:{a['decision']}"] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


def _latency(sessions):
    ms = [s["call_ms_max"] for s in sessions if s.get("call_ms_max") is not None]
    med = [s["call_ms_median"] for s in sessions if s.get("call_ms_median") is not None]
    if not ms:
        return None
    return {"note": "local, 14 parallel workers, single-threaded OpenCV; not Lambda", "per_session_max_ms": _q(ms),
            "max_ms": float(np.max(ms)), "p99_of_session_max_ms": float(np.percentile(ms, 99)),
            "median_call_ms": _q(med)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="directory with meta.json and runs.csv[.gz] or runs_core.csv.gz, sessions.jsonl[.gz]")
    ap.add_argument("--stopcal", default="config/stop_calibration.json")
    ap.add_argument("--out", default=None, help="default <runs>/summary.json")
    a = ap.parse_args(argv)
    d = Path(a.runs)
    s = summarize(d, Path(a.stopcal))
    out = Path(a.out) if a.out else d / "summary.json"
    out.write_text(json.dumps(s, indent=1, default=float), encoding="utf-8")
    for e in s["primary_endpoints"]:
        ci = e["ci"]
        print(f"{e['id']} {e['metric']:10s} {e['comparison']:7s} {e['actuator']:5s} median A {e['median_a']:.3f} vs {e['median_b']:.3f}"
              f"  diff {e['median_diff']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  W p={e['wilcoxon_p']:.2e} Holm p={e['p_holm']:.2e}"
              f"  wins/ties/losses {e['wins']}/{e['ties']}/{e['losses']}  n={e['cameras']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
