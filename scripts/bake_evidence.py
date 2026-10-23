"""Bake the Evidence tab into the Lambda image: figures and headline numbers from the committed results.

Copies ``results/figures/E*.png`` into ``src/nextview/lambda_app/evidence/`` and writes
``evidence/results.json`` (served by ``GET /api/evidence``): the headline numbers, figure captions and
the failure gallery. Every value comes from ``docs/tools/build_report.py``'s ``values()``, which
reads only the results files, so the tab and the report quote the same numbers. The baked files
are build output (git-ignored); ``infra/deploy.ps1`` runs this script before ``docker build``.

    python scripts/bake_evidence.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "results" / "figures"
OUT = REPO / "src" / "nextview" / "lambda_app" / "evidence"

CAPTIONS = {
    "E1_field_p95_vs_frames.png": "Median true field p95 against accepted frames, per policy (IQR bands), robot | human. TEST, simulator ground truth.",
    "E2_forest_primary_endpoints.png": "Pre-registered endpoints PE1–PE8: paired median differences (agent minus baseline) with 95 % bootstrap CIs.",
    "E3_frames_to_1px.png": "Frames until the true field p95 first falls below 1 px (survival curves).",
    "E4_ablations.png": "Ablations: agent minus the agent without one component (descriptive).",
    "E5_reliability.png": "Coverage of the first-order uncertainty map (χ² ellipses at nominal q), and the histogram of r = true p95 / U95.",
    "E6_risk_coverage.png": "Risk–coverage of the stop threshold: false-stop rate against mean frames.",
    "E7_metric_d455_heldout.png": "METRIC D455 (real images): held-out reprojection RMS against budget; grey band = random frame draws.",
    "E7b_metric_d455_field_vs_factory.png": "METRIC D455: field difference from the factory intrinsics (a reference, not ground truth).",
    "E8_metric_kinect_field.png": "METRIC KinectV2 (synthetic, ground-truth intrinsics): true field p95 against budget.",
    "E8b_metric_kinect_heldout.png": "METRIC KinectV2: held-out reprojection RMS against budget.",
    "E9_mm_consequence.png": "Length error of 100 mm segments in the outer 30 % of the image, plane at Z = 1 m (median over runs).",
    "E10_predicted_vs_realised.png": "Predicted against realised gain of each captured view.",
    "E12_failure_gallery.png": "Failure cases: what the agent measured and did (images rendered by the simulator).",
}


def _report_values() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("build_report", REPO / "docs" / "tools" / "build_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.values()


def payload(V: dict[str, str]) -> dict:
    def pe(i: int, what: str) -> tuple[str, str]:
        return (f"PE{i} {what}",
                f"{V[f'PE{i}_a']} vs {V[f'PE{i}_b']}; difference {V[f'PE{i}_d']} {V[f'PE{i}_ci']}; Holm p {V[f'PE{i}_holm']}")

    headline = dict([
        ("Benchmark", f"pre-registered synthetic TEST split, {V['runs']} runs (100 simulated cameras × 3 seeds × "
                      f"2 actuators × 11 policies), {V['errors']} errors; graded against simulator ground truth"),
        pe(1, "true field p95 after 10 frames, px, robot: agent vs random poses (B0)"),
        pe(2, "the same, hand-held (human): agent vs B0"),
        pe(3, "frames until true p95 < 1 px, robot: agent vs B0 (31 = not reached)"),
        pe(4, "the same, human: agent vs B0"),
        pe(5, "true field p95 after 10 frames, px, robot: agent vs interactive-calibration emulation (B2)"),
        pe(6, "the same, human: agent vs B2"),
        pe(7, "true field p95 after 10 frames, px, robot: agent vs 12-pose template (B1)"),
        pe(8, "the same, human: agent vs B1"),
        ("Primary endpoints", f"{V['pe_all_sig']} favour the agent after Holm correction (largest corrected p "
                              f"{V['pe_max_holm']}); everything below is descriptive"),
        ("Stopping", f"{V['stops_total']} of 600 agent runs stopped (robot {V['stop_robot']}, human {V['stop_human']}), "
                     f"{V['false_stops_total']} false stops (true p95 > {V['two_tau']} px); telephoto never stopped "
                     f"(robot {V['stop_robot_tele']}, human {V['stop_human_tele']})"),
        ("Uncertainty map coverage", f"{V['cov95_robot']} (robot) and {V['cov95_human']} (human) of cells inside the "
                                     f"nominal 95 % ellipse: the first-order map is optimistic, so the stop "
                                     f"probability is calibrated on separate DEV cameras"),
        ("Predicted vs realised gain", f"median predicted {V['e10_pred_robot']} px against realised "
                                       f"{V['e10_real_robot']} px (robot): optimistic"),
        ("Perception gate", f"adds no accuracy (agent minus agent without the gate at 10 frames: "
                            f"{V['abl_A-gate_robot_10']} px robot, {V['abl_A-gate_human_10']} px human) and costs a "
                            f"hand-held operator {V['att10_A_human']} attempts instead of {V['att10_Agate_human']} "
                            f"for 10 frames"),
        ("Real data: METRIC D455", f"the agent's held-out RMS ranks at percentile {V['d455_pct_med']} (median; range "
                                   f"{V['d455_pct_min']}–{V['d455_pct_max']}) of 100 random frame draws: not "
                                   f"consistently better than random"),
        ("Real data: METRIC KinectV2 (synthetic)", f"percentile {V['kin_pct_med']} (median; range {V['kin_pct_min']}–"
                                                   f"{V['kin_pct_max']}); boards under 4 % of the frame leave the "
                                                   f"image edges unconstrained for every policy"),
        ("Failure set", f"correct abstention: fisheye {V['fish_abst']}/{V['fish_runs']}, autofocus drift "
                        f"{V['af_abst']}/{V['af_runs']}, warped board {V['warp_abst']}/{V['warp_runs']}; "
                        f"{int(V['fish_clean']) + int(V['af_clean']) + int(V['warp_clean'])} clean stops"),
        ("LLM over MCP", f"qwen2.5:7b (local, Ollama) drove the same tools: {V['llm_qwen_valid']} valid calls, median "
                         f"true p95 {V['llm_qwen_p95']} px at 10 frames (planner {V['llm_a_p95']}, random "
                         f"{V['llm_b0_p95']}); it adds no accuracy"),
        ("AWS operations", f"{V['cw_inv']} Lambda invocations measured; sim step p50 {V['step_p50_s']} s server-side; "
                           f"about US${V['cost_session30']} per 30-request session before the free tier"),
        ("Sources", "results/synth_test/summary.json, posthoc.json, consequence.json; results/failure_gallery.json; "
                    "results/pool/summary.json; results/llm_row/table.md; results/ops_insights.json. "
                    "Report: docs/report.pdf"),
    ])
    gallery = [
        {"id": "F1", "case": "Fisheye 130°",
         "behaviour": f"R8 lens-model switch approved in {V['fish_r8']} runs, then R10 out-of-scope abstention",
         "numbers": f"20 cameras; ended {V['fish_ends']}; correct abstention {V['fish_abst']}/{V['fish_runs']}, "
                    f"{V['fish_clean']} clean stops"},
        {"id": "F2", "case": "Glare", "behaviour": "R3 rejects the frame and names the tilt axis",
         "numbers": f"{V['glare_rejects']} glare rejections on TEST webcam runs; moiré is not modelled"},
        {"id": "F3", "case": "Imprecise human", "behaviour": "R5 accepts the off-target view and re-plans",
         "numbers": "robot and human results are reported separately"},
        {"id": "F4, F5", "case": "Small board, pose ambiguity",
         "behaviour": "R0/R1 ask to move closer; R4 forces a tilt on the next target",
         "numbers": f"R4 fired {V['r4_lines']} times in {V['r4_runs']} of 600 agent runs on TEST"},
        {"id": "F6", "case": "Autofocus drift", "behaviour": "R6 asks for approval to lock focus and restart",
         "numbers": f"20 cameras; ended {V['af_ends']}; correct abstention {V['af_abst']}/{V['af_runs']}; at the "
                    f"frozen factor R6 often misses drift and raised {V['r6_false_total']} false restarts on TEST"},
        {"id": "F7", "case": "Under-coverage", "behaviour": "visible in the coverage figure (E5)",
         "numbers": f"coverage {V['cov95_wide_robot']} at nominal 0.95 on wide lenses (robot)"},
        {"id": "F8", "case": "Sagged board, release-object fit denied",
         "behaviour": "R9 raised; the residual pattern persists; escalation at the budget",
         "numbers": f"20 cameras; R9 raised in {V['warp_r9_deny']} runs; ended {V['warp_ends']}; correct abstention "
                    f"{V['warp_abst']}/{V['warp_runs']}"},
        {"id": "F9", "case": "False stops", "behaviour": "shown honestly",
         "numbers": f"none on TEST; the worst stop was {V['worst_stop']} px ({V['worst_stop_family']})"},
        {"id": "F10", "case": "Rolling shutter", "behaviour": "not modelled", "numbers": "stated as a limitation"},
        {"id": "F11", "case": "METRIC boards", "behaviour": "boards under 4 % of the frame",
         "numbers": f"{V['d455_small']} D455 and {V['kin_small']} KinectV2 full-board pool frames; the real-data "
                    f"comparison is weak"},
    ]
    return {"available": True, "label": "Simulator-graded unless marked METRIC; only PE1–PE8 are pre-registered "
                                        "endpoints", "headline": headline, "figures": dict(CAPTIONS),
            "failure_gallery": gallery}


def bake(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_file() and p.name != ".gitkeep":
            p.unlink()
    figs = sorted(FIGS.glob("E*.png"))
    missing = sorted(set(CAPTIONS) - {p.name for p in figs})
    if missing:
        raise SystemExit(f"figures missing from results/figures: {missing}")
    for p in figs:
        shutil.copy2(p, out / p.name)
    data = payload(_report_values())
    data["figures"] = {k: v for k, v in data["figures"].items() if (out / k).exists()}
    (out / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"figures": len(figs), "headline": len(data["headline"]), "gallery": len(data["failure_gallery"]),
            "bytes": sum(p.stat().st_size for p in out.iterdir() if p.is_file())}


def main() -> int:
    r = bake()
    print(f"baked {r['figures']} figures, {r['headline']} headline rows, {r['gallery']} gallery entries "
          f"({r['bytes']:,} bytes) into {OUT.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
