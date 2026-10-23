"""Build docs/report.md and docs/devpost.md from their templates in docs/tools/ and the committed results, then docs/report.pdf.

Every number in the report comes from a results file (results/*, config/*, infra/deployed.json) through
the values computed here; the template holds only prose and ``{{key}}`` placeholders. An unknown key
fails the build.

    python docs/tools/build_report.py              # report.md + report.pdf
    python docs/tools/build_report.py --md-only

The PDF step needs ``markdown`` and ``playwright`` (with Chromium): see README "Documentation build".
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
RES = REPO / "results"
MINUS = "−"


# ---------------------------------------------------------------- formatting
def num(x: float, d: int = 2) -> str:
    s = f"{x:.{d}f}"
    return s.replace("-", MINUS) if s.startswith("-") and float(s) != 0 else s.lstrip("-")


def sgn(x: float, d: int = 2) -> str:
    s = num(x, d)
    return s if s.startswith(MINUS) or float(x) == 0 else "+" + s


def ci(c, d: int = 2) -> str:
    return f"[{sgn(c[0], d)}, {sgn(c[1], d)}]"


def pval(p: float) -> str:
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m} × 10{str(int(e)).translate(sup)}"


def sci(x: float, d: int = 1) -> str:
    """Large numbers as m × 10ⁿ (n >= 5), else with thousands separators."""
    if abs(x) < 1e5:
        return f"{x:,.0f}" if abs(x) >= 100 else num(x, 1)
    m, e = f"{x:.{d}e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m} × 10{str(int(e)).translate(sup)}"


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f} %"


def load(rel: str):
    p = REPO / rel
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return json.loads(p.read_text(encoding="utf-8"))


def md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    al = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else (":---:" if a == "c" else "---") for a in al]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def parse_md_table(text: str) -> list[dict]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(head, [c.strip() for c in ln.strip("|").split("|")])) for ln in lines[2:]]


# ---------------------------------------------------------------- values
def values() -> dict[str, str]:
    V: dict[str, str] = {}
    S = load("results/synth_test/summary.json")
    meta = load("results/synth_test/meta.json")
    post = load("results/synth_test/posthoc.json")
    cons = load("results/synth_test/consequence.json")["table"]
    gal = load("results/failure_gallery.json")
    pool = load("results/pool/summary.json")
    dev = load("results/synth_dev/stop_fit.json")
    dep = load("infra/deployed.json")
    th = load("config/thresholds.json")
    presets = load("config/presets.json")["presets"]
    cw = load("results/ops/cloudwatch_day2.json")

    # deployment
    V["url"] = dep["api_url"]
    V["image_tag"] = dep["image_tag"]
    V["image_digest"] = dep["image_digest"]
    V["base_digest"] = dep["base_image_digest"]
    V["memory_mb"] = str(dep["memory_mb"])
    V["timeout_s"] = str(dep["timeout_s"])
    V["region"] = dep["region"]
    V["opencv"] = meta["versions"]["opencv"]
    V["numpy"] = meta["versions"]["numpy"]
    V["python"] = meta["versions"]["python"]

    # TEST run meta
    V["runs"] = f"{S['runs']:,}"
    V["errors"] = str(S["errors"])
    V["rows"] = f"{S['rows']:,}"
    V["test_minutes"] = f"{meta['wall_s_last_invocation'] / 60:.0f}"
    V["test_workers"] = str(meta["workers"])
    V["src_sha"] = S["source_tree_sha256"][:12]
    V["th_sha"] = S["thresholds_sha256"][:12]
    V["stop_sha"] = S["stop_calibration_sha256"][:12]
    V["tau"] = num(S["tau_px"])
    V["two_tau"] = num(2 * S["tau_px"], 1)

    # DEV freeze
    V["dev_r_robot"] = num(dev["r_A"]["robot"]["median"])
    V["dev_r_human"] = num(dev["r_A"]["human"]["median"])
    V["dev_n_robot"] = str(dev["r_A"]["robot"]["n"])
    V["dev_n_human"] = str(dev["r_A"]["human"]["n"])
    V["dev_stop_at_half"] = f"{dev['tau_rule']['median_frames_to_stop_normal_robot_at_0p5']:.0f}"

    # primary endpoints
    pe = {p["id"]: p for p in S["primary_endpoints"]}
    rows = []
    for pid in [f"PE{i}" for i in range(1, 9)]:
        p = pe[pid]
        unit = "frames" if p["metric"] == "frames<1px" else "px"
        d = 1 if unit == "frames" else 2
        what = "frames to p95 < 1 px" if unit == "frames" else "p95 at n = 10 (px)"
        rows.append([pid, what, p["comparison"].replace("-", MINUS), p["actuator"],
                     f"{num(p['median_a'], d if unit == 'px' else 0)} vs {num(p['median_b'], d if unit == 'px' else 0)}",
                     f"{sgn(p['median_diff'], d)} {ci(p['ci'], d)}", f"{p['wins']}/{p['ties']}/{p['losses']}",
                     pval(p["p_holm"])])
    V["table_pe"] = md_table(["ID", "Endpoint", "Comparison", "Actuator", "Median A vs baseline",
                              "Median paired difference [95 % CI]", "A better / tie / worse", "Holm p"], rows,
                             "llllrrrr")
    V["pe_all_sig"] = "all eight" if all(p["significant_holm_0.05"] for p in pe.values()) else "not all"
    V["pe_max_holm"] = pval(max(p["p_holm"] for p in pe.values()))
    for pid, p in pe.items():
        d = 0 if p["metric"] == "frames<1px" else 2
        V[f"{pid}_a"] = num(p["median_a"], d)
        V[f"{pid}_b"] = num(p["median_b"], d)
        V[f"{pid}_d"] = sgn(p["median_diff"], 1 if d == 0 else 2)
        V[f"{pid}_ci"] = ci(p["ci"], 1 if d == 0 else 2)
        V[f"{pid}_holm"] = pval(p["p_holm"])
        V[f"{pid}_wins"] = str(p["wins"])
        V[f"{pid}_cams"] = str(p["cameras"])

    # in-sample RMS vs truth (motivation)
    pp = S["per_policy"]["A/robot/all"]
    V["rms10"] = num(pp["rms@10"]["median"], 3)
    V["p95_10_pooled"] = num(pp["p95@10"]["median"])
    V["rms_ratio"] = f"{pp['p95@10']['median'] / pp['rms@10']['median']:.0f}"

    # per-family A - B0
    dp = S["descriptive_pairs"]
    fams = ["normal", "wide", "tele", "webcam"]
    rows = []
    for act in ("robot", "human"):
        r = [act]
        for fam in fams:
            x = dp[f"A-B0|{act}|p95@10"][fam]
            r.append(f"{num(x['median_a'])} vs {num(x['median_b'])}<br>{sgn(x['median_diff'])} {ci(x['ci'])}")
        rows.append(r)
    V["table_family"] = md_table(["Actuator"] + fams, rows, "lrrrr")
    for act in ("robot", "human"):
        for fam in fams:
            V[f"fam_{act}_{fam}"] = sgn(dp[f"A-B0|{act}|p95@10"][fam]["median_diff"])

    # ablations
    abl = [("A-gate", "perception gate R0–R3"), ("A-outlier", "outlier dropping R7"),
           ("A-model", "lens-model switch R8/R9"), ("A-nbv", "next-view planning (random order instead)")]
    rows = []
    for pol, what in abl:
        r = [pol, what]
        for act in ("robot", "human"):
            for n in (10, 30):
                x = dp[f"A vs {pol}|{act}|p95@{n}"]["all"]
                r.append(f"{sgn(x['median_diff'])} {ci(x['ci'])}")
        rows.append(r)
    V["table_ablation"] = md_table(["Removed", "Component", "robot, n = 10", "robot, n = 30", "human, n = 10",
                                    "human, n = 30"], rows, "llrrrr")
    for pol, _ in abl:
        for act in ("robot", "human"):
            for n in (10, 30):
                x = dp[f"A vs {pol}|{act}|p95@{n}"]["all"]
                V[f"abl_{pol}_{act}_{n}"] = sgn(x["median_diff"])
                V[f"abl_{pol}_{act}_{n}_ci"] = ci(x["ci"])
    for act in ("robot", "human"):
        x = dp[f"A vs A-model|{act}|p95@30"]["wide"]
        V[f"model_wide30_{act}"] = sgn(x["median_diff"])
        V[f"model_wide30_{act}_ci"] = ci(x["ci"])
        x = dp[f"A vs A-nbv|{act}|p95@10"]["wide"]
        V[f"nbv_wide10_{act}"] = sgn(x["median_diff"])
    # wide n=10: the switch has not happened yet
    V["model_n10_all_zero"] = "yes" if all(
        dp[f"A vs A-model|{a}|p95@10"][f]["median_diff"] == 0 and dp[f"A vs A-model|{a}|p95@10"][f]["wins"] == 0
        for a in ("robot", "human") for f in fams) else "no"

    # human effort
    ef = S["effort_attempts"]
    V["att10_A_human"] = f"{ef['A/human']['attempts_to_10']['median']:.0f}"
    V["att10_Agate_human"] = f"{ef['A-gate/human']['attempts_to_10']['median']:.0f}"
    V["att10_B0_human"] = f"{ef['B0/human']['attempts_to_10']['median']:.0f}"
    V["att10_A_robot"] = f"{ef['A/robot']['attempts_to_10']['median']:.0f}"
    V["reach30_A_human"] = str(ef["A/human"]["runs_reaching_30_frames"])
    V["runs_A_human"] = str(ef["A/human"]["runs"])
    gr = gal["gate_rejections_A_test"]
    V["blur_rejects_human"] = f"{sum(v for k, v in gr.items() if k.endswith('human/R2_BLUR')):,}"
    V["blur_rejects_robot"] = f"{sum(v for k, v in gr.items() if k.endswith('robot/R2_BLUR')):,}"
    V["glare_rejects"] = f"{sum(v for k, v in gr.items() if k.endswith('R3_GLARE')):,}"

    # stop quality
    sq = S["stop_quality"]
    for act in ("robot", "human"):
        q = sq[f"A/{act}"]
        V[f"stop_{act}"] = f"{q['stopped']}/{q['runs']}"
        V[f"stop_frames_{act}"] = f"{q['frames_at_stop']['median']:.0f}"
        V[f"stop_frames_iqr_{act}"] = f"{q['frames_at_stop']['q25']:.0f}–{q['frames_at_stop']['q75']:.0f}"
        V[f"stop_truep95_{act}"] = num(q["true_p95_at_stop"]["median"])
        V[f"false_stops_{act}"] = str(q["false_stops"])
        V[f"stop_gt_tau_{act}"] = pct(q["frac_stops_true_p95_gt_tau"], 1)
        V[f"wasted_{act}"] = f"{q['wasted_frames']['median']:.0f}"
        for fam in fams:
            b = q["by_family"][fam]
            V[f"stop_{act}_{fam}"] = f"{b['stopped']}/{b['runs']}"
    V["stops_total"] = str(sq["A/robot"]["stopped"] + sq["A/human"]["stopped"])
    V["false_stops_total"] = str(sq["A/robot"]["false_stops"] + sq["A/human"]["false_stops"])
    rc = sq["risk_coverage"]
    rows = []
    for i, r in enumerate(rc["robot"]):
        h = rc["human"][i]
        rows.append([num(r["stop_p_exceed"]),
                     f"{r['stopped']}/{r['runs']}", "–" if r["false_stop_rate"] is None else pct(r["false_stop_rate"], 1),
                     num(r["mean_frames"], 1),
                     f"{h['stopped']}/{h['runs']}", "–" if h["false_stop_rate"] is None else pct(h["false_stop_rate"], 1),
                     num(h["mean_frames"], 1)])
    V["table_risk"] = md_table(["stop_p_exceed", "robot stops", "false-stop rate", "mean frames", "human stops",
                                "false-stop rate", "mean frames"], rows, "lrrrrrr")
    rc5 = [r for r in rc["robot"] if abs(r["stop_p_exceed"] - 0.5) < 1e-9][0]
    V["rc_robot_05_fsr"] = pct(rc5["false_stop_rate"], 1)
    V["rc_robot_05_frames"] = num(rc5["mean_frames"], 1)
    ast = S["a_stop_fixed_budget"]
    V["astop_budget"] = str(ast["robot"]["fixed_budget"])
    V["astop_false_robot"] = pct(ast["robot"]["false_rate_fixed_budget_gt_2tau"], 0)
    V["astop_false_human"] = pct(ast["human"]["false_rate_fixed_budget_gt_2tau"], 0)
    b2 = S["b2_app_stop"]
    V["b2_stops"] = str(sum(v["stopped"] for v in b2.values()))
    V["b2_runs"] = str(sum(v["runs"] for v in b2.values()))
    worst = gal["F9_worst_false_stops"]["worst3"]
    V["worst_stop"] = num(max(w["true_p95"] for w in worst))
    V["worst_stop_family"] = worst[0]["family"]

    # uncertainty honesty
    E5 = S["uncertainty_honesty_E5"]
    for act in ("robot", "human"):
        rel = {r["nominal"]: r["empirical_mean"] for r in E5[f"A/{act}"]["reliability"]}
        V[f"cov95_{act}"] = num(rel[0.95])
        V[f"cov50_{act}"] = num(rel[0.5])
        V[f"r_med_{act}"] = num(E5[f"A/{act}"]["r"]["median"])
        V[f"r_gt1_{act}"] = pct(E5[f"A/{act}"]["r"]["frac_gt1"])
        V[f"cov95_wide_{act}"] = num(E5[f"A/{act}"]["cov95_by_family"]["wide"])
        cal = E5[f"A/{act}"]["p_exceed_calibration"]
        V[f"pex_low_pred_{act}"] = num(cal[0]["mean_pred"], 3)
        V[f"pex_low_obs_{act}"] = num(cal[0]["observed"], 3)
    E10 = S["predicted_vs_realised_E10"]
    for act in ("robot", "human"):
        e = E10[act]
        V[f"e10_rho_u_{act}"] = num(e["spearman_pred_vs_realised_U95_drop"])
        V[f"e10_rho_t_{act}"] = num(e["spearman_pred_vs_true_p95_drop"])
        V[f"e10_pred_{act}"] = num(e["median_pred"], 3)
        V[f"e10_real_{act}"] = num(e["median_realised"], 3)
        V[f"e10_pos_{act}"] = pct(e["frac_realised_positive"])
        V[f"e10_ratio_{act}"] = f"{e['median_pred'] / e['median_realised']:.1f}"

    # outliers
    r7 = post["r7"]["counts"]
    for act in ("robot", "human"):
        c = r7[f"A/{act}"]
        V[f"r7_acc_{act}"] = str(c["corrupted_accepted"])
        V[f"r7_drop_{act}"] = str(c["corrupted_dropped_by_R7"])
        V[f"r7_recall_{act}"] = num(c["recall"], 3 if c["recall"] < 1 else 2)
        V[f"r7_clean_{act}"] = str(c["clean_dropped_by_R7"])
        V[f"r7_prec_{act}"] = num(c["precision"])
        eff = post["r7"]["effect_A_minus_A-outlier_p95@10"][act]
        V[f"r7_better_{act}"] = str(eff["A_better"])
        V[f"r7_worse_{act}"] = str(eff["A_worse"])
        V[f"r7_runs_{act}"] = str(eff["runs"])
        V[f"r7_mean_{act}"] = sgn(eff["mean_diff_px"])
    V["r6_false_A_human"] = str(post["r6_restarts"].get("A/human:approve", 0))
    V["r6_false_A_robot"] = str(post["r6_restarts"].get("A/robot:approve", 0))
    V["r6_false_total"] = str(sum(post["r6_restarts"].values()))

    # hypotheses: human vs robot
    hv = post["human_vs_robot"]
    for comp in ("A - B0", "A - B2", "A - B1"):
        k = comp.split()[-1]
        V[f"hv_{k}_robot"] = sgn(hv[comp]["robot"]["median_diff"])
        V[f"hv_{k}_human"] = sgn(hv[comp]["human"]["median_diff"])

    # mm consequence (Z = 1.0 m)
    rows = []
    for pol in ("A", "B0", "B1", "B2", "B3", "B4"):
        r = [pol]
        for act in ("robot", "human"):
            n10 = cons[f"{pol}/{act}/n10"]["Z1.0"]
            r += [num(n10["median_of_run_median_mm"], 3), num(n10["median_of_run_p95_mm"], 3)]
        for act in ("robot", "human"):
            st = cons.get(f"{pol}/{act}/stop")
            r.append(num(st["Z1.0"]["median_of_run_median_mm"], 3) if st else "–")
        for act in ("robot", "human"):
            r.append(num(cons[f"{pol}/{act}/n30"]["Z1.0"]["median_of_run_median_mm"], 3))
        rows.append(r)
    V["table_mm"] = md_table(["Policy", "R median @10", "R p95 @10", "H median @10", "H p95 @10",
                              "R median @stop", "H median @stop", "R median @30", "H median @30"], rows)
    V["mm_A_n10"] = num(cons["A/robot/n10"]["Z1.0"]["median_of_run_median_mm"], 3)
    V["mm_B0_n10"] = num(cons["B0/robot/n10"]["Z1.0"]["median_of_run_median_mm"], 3)
    wt = post["mm_wide_tail"]
    V["mm_wide_A_robot"] = num(wt["A/robot"]["median_of_run_p95_mm"], 1)
    V["mm_wide_A_human"] = num(wt["A/human"]["median_of_run_p95_mm"], 1)
    V["mm_wide_B4_robot"] = num(wt["B4/robot"]["median_of_run_p95_mm"], 1)
    V["mm_wide_B4_human"] = num(wt["B4/human"]["median_of_run_p95_mm"], 1)
    V["mm_wide_B0_robot"] = num(wt["B0/robot"]["median_of_run_p95_mm"], 1)
    full = load("results/synth_test/consequence.json")["table"]
    for pol in ("A", "B4"):
        for act in ("robot", "human"):
            V[f"mm_wide30_{pol}_{act}"] = num(full[f"{pol}/{act}/n30/wide"]["Z1.0"]["median_of_run_p95_mm"],
                                             2 if pol == "A" else 1)
    inv = load("results/synth_test/consequence_inversion.json")
    V["inv_tol"] = num(inv["tolerance_px"], 1)
    for pol in ("A", "B4", "B0"):
        for act in ("robot", "human"):
            x = inv["table"][f"{pol}/{act}/n10/wide"]
            V[f"inv_fail_{pol}_{act}"] = pct(x["median_endpoint_fail_share"], 2)
            V[f"inv_p95_{pol}_{act}"] = num(x["median_of_run_p95_mm_invertible_pairs"], 1)
    V["inv_other_fams_A"] = "none" if all(
        inv["table"][f"A/{act}/n10/{fam}"]["runs_with_failed_endpoints"] == 0
        for act in ("robot", "human") for fam in ("normal", "tele", "webcam")) else "some"

    # B4 / B2 descriptive
    V["B4_robot_10"] = num(dp["A-B4|robot|p95@10"]["all"]["median_b"])
    V["B2k5_robot_10"] = num(dp["A-B2-k5|robot|p95@10"]["all"]["median_b"])
    V["B0_robot_10"] = num(dp["A-B0|robot|p95@10"]["all"]["median_b"])
    V["B3_robot_10"] = num(dp["A-B3|robot|p95@10"]["all"]["median_b"])
    V["A_B3_robot"] = sgn(dp["A-B3|robot|p95@10"]["all"]["median_diff"])
    V["A_B3_human"] = sgn(dp["A-B3|human|p95@10"]["all"]["median_diff"])
    V["A_B4_robot"] = sgn(dp["A-B4|robot|p95@10"]["all"]["median_diff"])
    ne = post["no_estimate_at_n10"]["runs"]
    V["b2_stalled"] = str(sum(len(ne.get(k, [])) for k in ("B2/robot", "B2/human")))
    V["b2k5_stalled"] = str(sum(len(ne.get(k, [])) for k in ("B2-k5/robot", "B2-k5/human")))

    # failure set
    for case, key in (("fisheye130", "fish"), ("autofocus_drift", "af"), ("warped_board", "warp")):
        g = gal[case]
        V[f"{key}_runs"] = str(g["runs"])
        V[f"{key}_abst"] = str(g["correct_abstentions"])
        V[f"{key}_clean"] = str(g["clean_stops"])
        V[f"{key}_ends"] = ", ".join(f"{k.split('_')[0]} {v}" for k, v in sorted(g["end_rules"].items()))
        V[f"{key}_noclean"] = f"{g['runs'] - g['clean_stops']}/{g['runs']}"
    fe = gal["fisheye130"]["end_rules"]
    if set(fe) != {"R10_OUT_OF_SCOPE", "R6_DRIFT"}:
        raise SystemExit(f"fisheye end rules changed ({sorted(fe)}): update the fisheye sentence in the templates")
    V["fish_R10"] = str(fe["R10_OUT_OF_SCOPE"])
    V["fish_R6"] = str(fe["R6_DRIFT"])
    V["warp_r9_deny"] = str(gal["warped_board"]["approvals"].get("R9_WARP:deny", 0))
    V["fish_r8"] = str(gal["fisheye130"]["approvals"].get("R8_UNDERFIT:approve", 0))
    agent_dir = load("results/agent/g2_r9_sag3mm_approve_all.json")
    sess = agent_dir["sessions"][0]
    V["sag3_true"] = num(sess["final_true_p95_px"])
    V["sag3_u95"] = num(sess["final_U95_px"])
    deny = load("results/agent/g2_r9_sag3mm_deny_all.json")["sessions"][0]
    V["sag3_deny_true"] = num(deny["final_true_p95_px"], 1)
    af = load("results/agent/g2_autofocus_drift_factor2p5.json")["sessions"][0]
    V["af25_true"] = num(af["final_true_p95_px"])
    win = {x["preset"]: x for x in load("results/agent/g2_presets.json")["sessions"]}
    lam = {x["preset"]: x for x in load("results/agent/g2_presets_lambda_image_cpu029.json")["sessions"]}
    same = sum(1 for k in win if win[k]["final_true_p95_px"] == lam[k]["final_true_p95_px"])
    V["platform_same"] = f"{same} of {len(win)}"
    V["platform_wide_win"] = num(win["wide_robot_handheld"]["final_true_p95_px"])
    V["platform_wide_linux"] = num(lam["wide_robot_handheld"]["final_true_p95_px"])
    V["af_preset_true"] = num(win["autofocus_drift"]["final_true_p95_px"])
    V["af_preset_u95"] = num(win["autofocus_drift"]["final_U95_px"])
    V["af_preset_end"] = win["autofocus_drift"]["decision"]["rule"].split("_")[0]

    # METRIC pool
    def pool_pcts(ds):
        vals = []
        for b, blk in pool[ds]["budgets"].items():
            for cam, v in blk["P-agent_percentile_in_P-rand"]["heldout_rms"].items():
                vals.append(v)
        return vals
    for ds, key in (("d455", "d455"), ("kinect_synth", "kin")):
        p = pool_pcts(ds)
        V[f"{key}_pct_med"] = f"{statistics.median(p):.0f}"
        V[f"{key}_pct_min"] = f"{min(p):.0f}"
        V[f"{key}_pct_max"] = f"{max(p):.0f}"
        V[f"{key}_pct_below50"] = f"{sum(1 for x in p if x < 50)}/{len(p)}"
        rn = pool[ds]["ranges_non_random"]
        V[f"{key}_field_min"] = sci(rn["field_p95"]["min"])
        V[f"{key}_field_max"] = sci(rn["field_p95"]["max"])
        V[f"{key}_fx_min"] = sgn(rn["fx_rel_pct"]["min"], 0)
        V[f"{key}_fx_max"] = sgn(rn["fx_rel_pct"]["max"], 0)
        V[f"{key}_small"] = f"{rn['board_area_lt_4pct']['frames']}/{rn['board_area_lt_4pct']['of_full_board_pool_frames']}"
        r8 = pool[ds]["r8_check"]
        V[f"{key}_r8_switch"] = f"{sum(1 for v in r8.values() if v['would_switch_cv_criterion'])}/{len(r8)}"
    f11 = gal["F11_metric"]
    V["d455_frames"] = ", ".join(str(c["full_board"]) for c in f11["d455"])
    V["kin_frames"] = ", ".join(str(c["full_board"]) for c in f11["kinect_synth"])
    V["d455_cam4_frames"] = str(pool["d455"]["r8_check"]["4"]["frames"])

    # LLM row
    def llm_rows(path):
        return {r["Policy"]: r for r in parse_md_table((REPO / path).read_text(encoding="utf-8"))}
    L1 = llm_rows("results/llm_row/table.md")
    L2 = llm_rows("results/llm_row_extra/table.md")
    sel = [("A", L1, "A (planner, reference)"), ("B0", L1, "B0 (random poses)"),
           ("MCP-scripted", L1, "scripted MCP client, always rank 1"),
           ("LLM qwen2.5:7b", L1, "qwen2.5:7b"),
           ("LLM qwen2.5:7b@nohints", L2, "qwen2.5:7b, no “Next:” hints"),
           ("LLM qwen2.5:7b@nohints+pose", L2, "qwen2.5:7b, no hints, free pose argument"),
           ("LLM nemotron-3-nano:4b@nothink+h8+pose", L2, "nemotron-3-nano:4b, 8-message history"),
           ("LLM nemotron-3-nano:4b@nothink+pose", L1, "nemotron-3-nano:4b, full history")]
    rows = []
    for key, tab, label in sel:
        r = tab[key]
        rows.append([label, r["Valid / attempted"], r["Rank-1 agreement (all / after opening)"].replace(" / ", " / "),
                     r["Pose captures"], r["Exported"], r["Median true p95 at 10 frames (px)"]])
    V["table_llm"] = md_table(["Row (10 normal TEST cameras, robot, budget 10)", "Valid / attempted calls",
                               "Rank-1 agreement (all / after opening)", "Invented-pose captures", "Exported",
                               "Median true p95 at 10 frames (px)"], rows, "lrrrrr")
    V["llm_qwen_valid"] = L1["LLM qwen2.5:7b"]["Valid / attempted"]
    V["llm_a_p95"] = L1["A"]["Median true p95 at 10 frames (px)"]
    V["llm_b0_p95"] = L1["B0"]["Median true p95 at 10 frames (px)"]
    V["llm_qwen_p95"] = L1["LLM qwen2.5:7b"]["Median true p95 at 10 frames (px)"]
    V["llm_pose_agree"] = L2["LLM qwen2.5:7b@nohints+pose"]["Rank-1 agreement (all / after opening)"]
    V["llm_pose_caps"] = L2["LLM qwen2.5:7b@nohints+pose"]["Pose captures"]
    V["llm_a_pex"] = L1["A"]["Median final P_exceed"]
    llm_env = load("results/llm_row/summary.json")["env"]
    V["ollama"] = llm_env.get("ollama", "?")

    # operations: the Logs Insights export (results/ops_insights.json, written by infra/guard.py --insights)
    # when present, else the day-2 CloudWatch extract (results/ops/cloudwatch_day2.json)
    oi_path = REPO / "results" / "ops_insights.json"
    if oi_path.exists():
        oi = load("results/ops_insights.json")
        w = oi["windows"]["all"]
        routes = {(r["route"], r["mode"]): r for r in w["routes_200"]}
        nb = w["nbv_sim"][0]
        rep = w["lambda_report"][0]
        mem_key, src = "max_memory_mb", "results/ops_insights.json"
        V["cw_window"] = (f"{w['start'][:16].replace('T', ' ')} to {w['end'][:16].replace('T', ' ')} UTC, "
                          f"all traffic since the log group was created")
        cm, dv = w["cloudwatch_metrics"], w["derived"]
        V["ops_health"] = (f" In that window the handler raised {len(w['handler_exceptions'])} exceptions and "
                           f"returned {dv['handler_5xx']} server errors, CloudWatch recorded {cm['errors']} Lambda "
                           f"errors and no platform failures (timeouts or out-of-memory), and the only refusals "
                           f"were {dv['handler_4xx']} client errors (malformed, oversized or unknown requests and "
                           f"tampered tokens), "
                           f"about {dv['stage_throttle_429_est']} stage-throttle 429s and "
                           f"{dv['concurrency_cap_503']} concurrency-cap 503s from the load tests.")
    else:
        routes = {(r["route"], r["mode"]): r for r in cw["routes_handler_ms"]}
        nb = cw["nbv"][0]
        rep = cw["report"][0]
        mem_key, src = "max_mem_mb", "results/ops/cloudwatch_day2.json"
        V["cw_window"] = f"{cw['window_utc'][0][:16].replace('T', ' ')}–{cw['window_utc'][1][11:16]} UTC"
        V["ops_health"] = ""
    V["ops_src"] = src
    rows = []
    for (route, mode), label in ((("/api/step", "sim"), "/api/step (sim)"), (("/api/frame", "webcam"), "/api/frame (webcam)"),
                                 (("/api/approve", "sim"), "/api/approve (sim, incl. S3 + presign)"),
                                 (("/api/session", "sim"), "/api/session (sim, builds the camera maps)")):
        r = routes[(route, mode)]
        rows.append([label, str(r["n"]), f"{float(r['p50_ms']):,.0f}", f"{float(r['p95_ms']):,.0f}",
                     f"{float(r['max_ms']):,.0f}"])
    rows.append(["NBV planner inside a step", str(nb["n"]), f"{float(nb['p50_ms']):,.0f}",
                 f"{float(nb['p95_ms']):,.0f}", f"{float(nb['max_ms']):,.0f}"])
    V["table_latency"] = md_table(["Server-side handler time", "n", "p50 ms", "p95 ms", "max ms"], rows)
    V["step_p50_s"] = num(float(routes[("/api/step", "sim")]["p50_ms"]) / 1000, 1)
    V["step_max_s"] = num(float(routes[("/api/step", "sim")]["max_ms"]) / 1000, 1)
    V["cw_inv"] = f"{int(rep['invocations']):,}"
    V["cw_p50"] = f"{float(rep['p50_ms']):,.0f}"
    V["cw_p95"] = f"{float(rep['p95_ms']):,.0f}"
    V["cw_max"] = f"{float(rep['max_ms']):,.0f}"
    V["cw_cold"] = str(rep["cold_starts"])
    V["cw_init"] = f"{float(rep['init_avg_ms']):,.0f}"
    V["cw_mem"] = f"{float(rep[mem_key]):.0f}"
    V["cw_gbs"] = f"{float(rep['gb_s']):.1f}"
    gbs_per_inv = float(rep["gb_s"]) / float(rep["invocations"])
    price = 0.0000166667  # USD per GB-s, Lambda x86 first tier (AWS public price list)
    V["gbs_per_inv"] = num(gbs_per_inv, 2)
    V["cost_step"] = f"{gbs_per_inv * price:.7f}"
    V["cost_session30"] = f"{30 * gbs_per_inv * price:.4f}"
    ceiling = 5 * dep["memory_mb"] / 1024 * 86400
    V["ceiling_gbs"] = f"{ceiling / 1000:,.0f}k"
    V["ceiling_usd"] = f"{ceiling * price:.2f}"
    V["price"] = "0.0000166667"
    guard = REPO / "infra" / "guard.py"
    if guard.exists():
        g = guard.read_text(encoding="utf-8")
        d = {k: float(re.search(rf'"--{k}-gbs", type=float, default=([\d_.]+)', g).group(1).replace("_", ""))
             for k in ("warn", "throttle", "kill")}
        V["_guard_registered"] = guard_registered()
        when = ("runs hourly as a Windows scheduled task through the judging period and"
                if V["_guard_registered"] == "yes" else
                "can run hourly as a Windows scheduled task (the README has the one-line registration; at the time "
                "of writing it was run by hand) and")
        V["guard_sentence"] = (f" `infra/guard.py` {when} reads the last 24 h of CloudWatch usage: above "
                               f"{d['warn'] / 1000:.0f}k GB-s it "
                               f"warns, above {d['throttle'] / 1000:.0f}k it lowers the stage throttle, and above "
                               f"{d['kill'] / 1000:.0f}k (about US${d['kill'] * price:.2f} per day) it engages the kill "
                               f"switch and leaves a marker file; `--restore` undoes both.")
    else:
        V["guard_sentence"] = ""
        V["_guard_registered"] = "n/a"
    scan = load("results/ops/ecr_scan.json")
    sev = scan["finding_counts"]
    V["ecr_scan"] = (f"basic vulnerability scan on push: {sev.get('CRITICAL', 0)} critical, {sev.get('HIGH', 0)} high, "
                     f"{sev.get('MEDIUM', 0)} medium and {sev.get('LOW', 0)} low findings, all in the base image's "
                     f"Amazon Linux packages ({', '.join(sorted({q.split('-')[0] for v in scan['packages'].values() for q in v}))}), "
                     f"none in the packages the image adds (`results/ops/ecr_scan.json`)")
    V["_ecr_scan_tag"] = scan["image_tag"]
    lt_rows = []
    for fname, label in (("load_test_day2_c3.json", "3 × 15 steps, 3 concurrent"),
                         ("load_test_day2_c5.json", "5 × 10, 5 concurrent"),
                         ("load_test_day2.json", "10 × 10, 10 concurrent (spec load)"),
                         ("load_test_day2_jitter.json", "10 × 10, 10 concurrent, jittered retries"),
                         ("load_test_day2_busywaits.json", "10 × 10, 10 concurrent, with the UI's busy-wait")):
        d = load(f"results/ops/{fname}")
        lt_rows.append([label, pct(d["rate_429"]), pct(d["rate_503"]), f"{d['sessions_failed']}/{d['sessions']}",
                        f"{d['wall_s']:.0f}"])
    V["table_load"] = md_table(["Load (sessions × steps, clients)", "429 share", "503 share", "Sessions failed",
                                "Wall s"], lt_rows)
    seq = load("results/ops/latency_sequential_day2.json")["routes"]["/api/step"]
    V["client_step_p50"] = f"{seq['client_ms_p50']:,.0f}"
    V["client_step_p95"] = f"{seq['client_ms_p95']:,.0f}"
    V["local_session_max_med"] = f"{S['latency_local_ms']['per_session_max_ms']['median']:,.0f}"

    # thresholds appendix
    keys = [k for k in th if not k.startswith("_")]
    ncol = 3
    per = (len(keys) + ncol - 1) // ncol
    rows = []
    for i in range(per):
        r = []
        for c in range(ncol):
            j = c * per + i
            r += [f"`{keys[j]}`", json.dumps(th[keys[j]])] if j < len(keys) else ["", ""]
        rows.append(r)
    V["table_thresholds"] = md_table(["Key", "Value"] * ncol, rows, "lr" * ncol)
    V["table_presets"] = "; ".join(
        f"`{p['id']}` ({p['family']}, {p['actuator']}{', failure demo' if p['failure_demo'] else ''})"
        for p in presets) + "."

    # rule firings on TEST, agent (A) runs only: trace lines per rule and runs in which the rule fired
    lines, runs, n_a, r13_opening = {}, {}, 0, 0
    with gzip.open(RES / "synth_test" / "events.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["key"].endswith("|A"):
                n_a += 1
                seen = set()
                for t in d["trace"]:
                    m = re.match(r"#\d+ (R\d+)_[A-Z_]+:", t)
                    if m:
                        lines[m.group(1)] = lines.get(m.group(1), 0) + 1
                        seen.add(m.group(1))
                        if m.group(1) == "R13" and "opening pose" in t:
                            r13_opening += 1
                for r in seen:
                    runs[r] = runs.get(r, 0) + 1
    V["a_runs"] = str(n_a)
    for r in ("R0", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R11", "R12", "R13"):
        V[f"fired_{r}"] = f"{lines.get(r, 0):,}"
        V[f"fired_{r}_runs"] = f"{runs.get(r, 0):,}"
    V["fired_R13_opening"] = f"{r13_opening:,}"
    V["fired_R13_planned"] = f"{lines.get('R13', 0) - r13_opening:,}"
    V["r4_lines"] = V["fired_R4"]
    V["r4_runs"] = V["fired_R4_runs"]

    # example trace (verbatim) from the TEST events log
    V["trace_excerpt"], V["trace_key"] = trace_excerpt()

    # evidence baked?
    ev = REPO / "src" / "nextview" / "lambda_app" / "evidence"
    V["_evidence_baked"] = "yes" if any(p.suffix in (".png", ".json") for p in ev.glob("*")) else "no"
    return V


def guard_registered() -> str:
    """'yes' when the Windows scheduled task "NextView guard" exists on this machine, else 'no'."""
    if not shutil.which("schtasks"):
        return "no"
    r = subprocess.run(["schtasks", "/Query", "/TN", "NextView guard"], capture_output=True)
    return "yes" if r.returncode == 0 else "no"


def trace_excerpt() -> tuple[str, str]:
    key = "test|1025|wide|0|human|A"
    with gzip.open(RES / "synth_test" / "events.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["key"] == key:
                t = d["trace"]
                break
        else:
            raise SystemExit(f"trace {key} not found")
    want = [lambda s: " R2_BLUR:" in s and "blur along" in s,
            lambda s: " R13_NBV:" in s and "largest σ" in s,
            lambda s: " R5_OFF_TARGET:" in s,
            lambda s: " R7_OUTLIER:" in s,
            lambda s: " R0_NO_BOARD:" in s and "next candidate" in s,
            lambda s: " R8_UNDERFIT:" in s]
    picked: list[int] = []
    for pred in want:
        i = next(i for i, s in enumerate(t) if pred(s))
        picked.append(i)
    i8 = picked[-1]
    picked += [i8 + 1, i8 + 2]
    i11 = next(i for i, s in enumerate(t) if " R11_STOP:" in s)
    picked += [i11, i11 + 1]
    out, last = [], None
    for i in sorted(set(picked)):
        if last is not None and i > last + 1:
            out.append("   …")
        out.append(t[i])
        last = i
    return "\n".join(out), key


# ---------------------------------------------------------------- render
def render_md(V: dict[str, str], name: str = "report.template.md") -> str:
    tpl = (DOCS / "tools" / name).read_text(encoding="utf-8")
    missing = sorted({m for m in re.findall(r"\{\{([A-Za-z0-9_:.\-]+)\}\}", tpl) if m not in V})
    if missing:
        raise SystemExit(f"undefined template keys in {name}: {missing}")
    out = re.sub(r"\{\{([A-Za-z0-9_:.\-]+)\}\}", lambda m: V[m.group(1)], tpl)
    head = (f"<!-- Generated by docs/tools/build_report.py from docs/tools/{name} and the results "
            "files; edit the template, not this file. -->\n")
    return head + out


CSS_FILE = DOCS / "tools" / "report.css"


def render_pdf(md_text: str, pdf_path: Path) -> int:
    import markdown
    from playwright.sync_api import sync_playwright

    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    css = CSS_FILE.read_text(encoding="utf-8")
    html_doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><base href="{DOCS.as_uri()}/">'
                f"<title>NextView technical report</title><style>{css}</style></head><body>{body}</body></html>")
    tmp = DOCS / "tools" / "_report.html"
    tmp.write_text(html_doc, encoding="utf-8")
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page()
            page.goto(tmp.as_uri())
            page.wait_for_load_state("networkidle")
            page.pdf(path=str(pdf_path), format="A4", print_background=True, display_header_footer=True,
                     header_template="<span></span>",
                     footer_template='<div style="font-size:7.5pt;color:#64748b;width:100%;text-align:center;'
                                     'font-family:Segoe UI,Arial,sans-serif">NextView technical report · '
                                     '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                     margin={"top": "12mm", "bottom": "14mm", "left": "12.5mm", "right": "12.5mm"})
            b.close()
    finally:
        tmp.unlink(missing_ok=True)
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except ImportError:
        return -1


def checks(V: dict[str, str], md_text: str) -> list[str]:
    warn = []
    if V["_evidence_baked"] != "yes":
        warn.append("src/nextview/lambda_app/evidence has no baked results/figures: the live Evidence tab is empty")
    readme = REPO / "README.md"
    if readme.exists():
        rd = readme.read_text(encoding="utf-8")
        plain = {k: v.replace(" ", " ") for k, v in V.items()}
        need = {"deployed image digest": plain["image_digest"], "deployed image tag": plain["image_tag"],
                "base image digest": plain["base_digest"], "live URL": plain["url"],
                "PE1 medians": f"{plain['PE1_a']} vs {plain['PE1_b']}",
                "PE2 medians": f"{plain['PE2_a']} vs {plain['PE2_b']}",
                "PE1 difference": f"{plain['PE1_d']} {plain['PE1_ci']}",
                "PE2 difference": f"{plain['PE2_d']} {plain['PE2_ci']}",
                "PE1 Holm p": plain["PE1_holm"], "PE2 Holm p": plain["PE2_holm"],
                "PE3 medians": f"{plain['PE3_a']} vs {plain['PE3_b']}",
                "PE4 medians": f"{plain['PE4_a']} vs {plain['PE4_b']}",
                "TEST run count": plain["runs"], "TEST minutes": f"{plain['test_minutes']} minutes",
                "TEST workers": f"{plain['test_workers']} worker processes"}
        for what, s in need.items():
            if s not in rd:
                warn.append(f"README.md is out of date: {what} should read '{s}'")
    if V.get("_guard_registered") == "no":
        warn.append('the "NextView guard" scheduled task is not registered: the report says the guard was run by hand '
                    "(register it with the README line, then rebuild)")
    if V["_ecr_scan_tag"] != V["image_tag"]:
        warn.append(f"results/ops/ecr_scan.json describes image {V['_ecr_scan_tag']}, not the deployed "
                    f"{V['image_tag']}: refresh it with `python scripts/ecr_scan_summary.py`")
    if "VIDEO_URL_TO_ADD" in (DOCS / "devpost.md").read_text(encoding="utf-8"):
        warn.append("docs/devpost.md still has the video link placeholder VIDEO_URL_TO_ADD")
    return warn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md-only", action="store_true")
    a = ap.parse_args(argv)
    V = values()
    md_text = render_md(V)
    (DOCS / "report.md").write_text(md_text, encoding="utf-8", newline="\n")
    print("wrote docs/report.md")
    (DOCS / "devpost.md").write_text(render_md(V, "devpost.template.md"), encoding="utf-8", newline="\n")
    print("wrote docs/devpost.md")
    if not a.md_only:
        pages = render_pdf(md_text, DOCS / "report.pdf")
        print(f"wrote docs/report.pdf ({pages} pages)")
    for w in checks(V, md_text):
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
