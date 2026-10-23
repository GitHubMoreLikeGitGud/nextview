"""Draw docs/architecture.svg (hand-built SVG) and render docs/architecture.png (1600 px wide).

    python docs/tools/draw_architecture.py            # SVG + PNG (PNG needs Playwright + Chromium)
    python docs/tools/draw_architecture.py --svg-only

Deployment values (memory, timeout, throttle) are read from infra/deployed.json and
infra/deploy.ps1 so the picture follows the deployed stack.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
W, H = 1600, 1110
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"
MONO = "Consolas, Menlo, monospace"

# One colour per OpenCV 5 module (legend), plus roles.
MOD = {
    "objdetect": "#0f766e",
    "geometry": "#1d4ed8",
    "calib": "#7c3aed",
    "imgproc": "#c2410c",
    "core": "#475569",
}
AWS = "#d97706"
LOCAL = "#64748b"
HUMAN = "#b91c1c"
INK = "#0f172a"
MUTED = "#475569"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def wrap(text: str, width_px: float, size: float) -> list[str]:
    """Greedy wrap with an average glyph width of 0.53 em."""
    max_chars = max(8, int(width_px / (size * 0.53)))
    out: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            cand = word if not line else line + " " + word
            if len(cand) <= max_chars:
                line = cand
            else:
                if line:
                    out.append(line)
                line = word
        out.append(line)
    return out


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, *, fill="#ffffff", stroke=INK, sw=1.5, rx=10, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"{o} stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, *, size=15, weight="normal", fill=INK, anchor="start", family=FONT, italic=False):
        st = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{st}>{esc(s)}</text>')

    def para(self, x, y, s, width, *, size=14, lh=1.3, **kw) -> float:
        """Wrapped text; returns the y below the last line."""
        for i, line in enumerate(wrap(s, width, size)):
            self.text(x, y + i * size * lh, line, size=size, **kw)
        return y + len(wrap(s, width, size)) * size * lh

    def chip(self, x, y, module) -> float:
        """A coloured module label; returns its width."""
        w = 10 + len(module) * 7.4
        self.rect(x, y - 13, w, 18, fill=MOD[module], stroke=MOD[module], sw=1, rx=9)
        self.text(x + w / 2, y + 0.5, module, size=12, weight="bold", fill="#ffffff", anchor="middle")
        return w

    def tool(self, x, y, name, module, apis, width) -> float:
        """Tool name + module chips (on the next line if they do not fit), then the OpenCV calls."""
        self.text(x, y, name, size=15, weight="bold", family=MONO)
        if module:
            mods = module.split("+")
            need = sum(10 + len(m) * 7.4 + 5 for m in mods)
            cx = x + len(name) * 8.3 + 8
            if cx + need > x + width + 6:
                y += 21
                cx = x
            for m in mods:
                cx += self.chip(cx, y, m) + 5
        return self.para(x, y + 19, apis, width, size=13.5, fill=MUTED) + 8

    def arrow(self, d, *, color=INK, sw=2.2, dash=None, both=False, marker="a"):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        start = f' marker-start="url(#{marker}s)"' if both else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dd} marker-end="url(#{marker})"{start}/>')

    def person(self, cx, cy, color):
        self.add(f'<circle cx="{cx}" cy="{cy - 14}" r="9" fill="{color}"/>')
        self.add(f'<path d="M{cx - 15},{cy + 18} Q{cx - 15},{cy - 3} {cx},{cy - 3} Q{cx + 15},{cy - 3} {cx + 15},{cy + 18} Z" fill="{color}"/>')

    def robot(self, cx, cy, color):
        self.add(f'<rect x="{cx - 16}" y="{cy + 10}" width="32" height="8" rx="2" fill="{color}"/>')
        self.add(f'<path d="M{cx},{cy + 10} L{cx - 6},{cy - 8} L{cx + 12},{cy - 18}" fill="none" stroke="{color}" '
                 f'stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>')
        self.add(f'<circle cx="{cx - 6}" cy="{cy - 8}" r="4" fill="#ffffff" stroke="{color}" stroke-width="2"/>')
        self.add(f'<rect x="{cx + 10}" y="{cy - 26}" width="12" height="9" rx="1" fill="{color}"/>')

    def svg(self) -> str:
        defs = []
        for key, col in (("a", INK), ("h", HUMAN), ("w", AWS), ("l", LOCAL)):
            defs.append(f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
            defs.append(f'<marker id="{key}s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                f'role="img" aria-labelledby="t d">\n<title id="t">NextView architecture</title>\n'
                f'<desc id="d">Browser client, AWS API Gateway and Lambda container running the OpenCV 5 '
                f'perceive-decide-act loop with human approval gates, S3, CloudWatch and ECR, and the local MCP '
                f'server, LLM driver and benchmark harness.</desc>\n<defs>{"".join(defs)}</defs>\n'
                + "\n".join(self.parts) + "\n</svg>\n")


def deploy_values() -> dict:
    dep = json.loads((REPO / "infra" / "deployed.json").read_text(encoding="utf-8"))
    ps = (REPO / "infra" / "deploy.ps1").read_text(encoding="utf-8")
    rate = re.search(r"\$ThrottleRate\s*=\s*([\d.]+)", ps).group(1)
    burst = re.search(r"\$ThrottleBurst\s*=\s*(\d+)", ps).group(1)
    return {"memory": dep["memory_mb"], "timeout": dep["timeout_s"], "rate": rate.rstrip("0").rstrip("."),
            "burst": burst, "region": dep["region"]}


def draw() -> str:
    v = deploy_values()
    s = Svg()
    s.rect(0, 0, W, H, fill="#ffffff", stroke="#ffffff", sw=0, rx=0)

    # Header and legend
    s.text(24, 42, "NextView: architecture and agent loop", size=28, weight="bold")
    s.text(24, 68, "One request = one perceive → decide → act step. The same tool functions serve the AWS endpoint, "
                   "the benchmark and the local MCP server.", size=15, fill=MUTED)
    lx = 1010
    s.text(lx, 26, "OpenCV 5.0 module", size=12.5, weight="bold", fill=MUTED)
    cx = lx
    for m in MOD:
        cx += s.chip(cx, 46, m) + 6
    s.add(f'<line x1="{lx}" y1="64" x2="{lx + 34}" y2="64" stroke="{AWS}" stroke-width="2.5" stroke-dasharray="8 5"/>')
    s.text(lx + 40, 69, "AWS boundary", size=12.5, fill=MUTED)
    s.add(f'<line x1="{lx + 140}" y1="64" x2="{lx + 174}" y2="64" stroke="{LOCAL}" stroke-width="2.5" stroke-dasharray="3 4"/>')
    s.text(lx + 180, 69, "local only", size=12.5, fill=MUTED)
    s.add(f'<line x1="{lx + 262}" y1="64" x2="{lx + 296}" y2="64" stroke="{HUMAN}" stroke-width="2.5"/>')
    s.text(lx + 302, 69, "human control", size=12.5, fill=MUTED)

    # ---------------- Left: client and people ----------------
    s.rect(20, 92, 310, 548, fill="#f8fafc")
    s.text(36, 120, "Browser (single page)", size=15.5, weight="bold")
    tx = 36
    for tab in ("Simulator", "Webcam", "Evidence", "About"):
        w = 12 + len(tab) * 7.6
        s.rect(tx, 134, w, 24, fill="#e2e8f0", stroke="#94a3b8", sw=1, rx=12)
        s.text(tx + w / 2, 151, tab, size=12.5, anchor="middle")
        tx += w + 6
    y = 186
    y = s.para(36, y, "Holds the session as an HMAC-signed, compressed state token (expires after 6 h, at most "
                      "256 KB). The server rebuilds K and the distortion from the views on every step; "
                      "nothing is trusted from the client.", 280, size=13.5) + 8
    y = s.para(36, y, "Simulator: POST /api/step (JSON) returns the rendered frame, the trace line, candidate "
                      "table, uncertainty map and, in simulation only, the true error.", 280, size=13.5) + 8
    y = s.para(36, y, "Webcam: POST /api/frame with a JPEG of at most 300 KB. Decoded in memory for one step, "
                      "never stored or logged.", 280, size=13.5) + 8
    y = s.para(36, y, "Shows the ghost outline of the next board pose, the numeric instruction, the approval "
                      "card and the export links. Retries 429/503 with back-off.", 280, size=13.5)

    s.rect(20, 660, 310, 430, fill="#fff7f7", stroke=HUMAN)
    s.text(36, 688, "People and actuators", size=15.5, weight="bold", fill=HUMAN)
    s.person(58, 742, HUMAN)
    s.text(86, 732, "Approver", size=14.5, weight="bold")
    s.para(86, 751, "Approves or denies a lens-model change, a drift restart and every export, with a reason.",
           228, size=13)
    s.person(58, 852, INK)
    s.text(86, 842, "Operator (human mode)", size=14.5, weight="bold")
    s.para(86, 861, "Moves the board to the ghost outline: “board to bottom-left, ≈45 % of frame width, "
                    "tilt ~35°”.", 228, size=13)
    s.robot(58, 972, INK)
    s.text(86, 952, "Robot arm (robot mode)", size=14.5, weight="bold")
    s.para(86, 971, "Receives the target pose (rvec, tvec) directly; a person is needed only at the "
                    "approval gates.", 228, size=13)

    # ---------------- Centre: AWS ----------------
    s.rect(350, 92, 845, 998, fill="#fffbeb", stroke=AWS, sw=2.5, dash="10 6", opacity=0.55, rx=14)
    s.text(366, 118, f"AWS · {v['region']}", size=16, weight="bold", fill=AWS)

    s.rect(370, 130, 805, 56, fill="#ffffff", stroke=AWS, sw=2)
    s.text(386, 154, "API Gateway HTTP API", size=15.5, weight="bold")
    s.text(386, 175, f"$default route → Lambda proxy (payload 2.0, 29 s) · stage throttle {v['rate']} req/s, "
                     f"burst {v['burst']} · HTTPS", size=13.5, fill=MUTED)

    s.rect(370, 214, 805, 668, fill="#ffffff", stroke=AWS, sw=2)
    s.text(386, 240, f"Lambda container image (x86_64, {v['memory']} MB, {v['timeout']} s) · nextview.lambda_app",
           size=15.5, weight="bold")
    s.text(386, 261, "Router · verify state token (HMAC-SHA256) · size caps before parsing (413) · "
                     "one JSON trace line per request", size=13.5, fill=MUTED)

    col_w, top, col_h = 250, 278, 428
    cols = [(385, "PERCEIVE"), (652, "DECIDE"), (919, "ACT")]
    for x, title in cols:
        s.rect(x, top, col_w, col_h, fill="#f8fafc", stroke="#94a3b8", sw=1.2, rx=8)
        s.text(x + col_w / 2, top + 24, title, size=15, weight="bold", anchor="middle", fill=INK)
    x, y = 397, top + 50
    y = s.tool(x, y, "detect_board", "objdetect", "CharucoDetector.detectBoard, CharucoBoard.matchImagePoints",
               228)
    y = s.tool(x, y, "assess_frame", "geometry", "slanted-edge blur on the board's own square edges (px and "
               "direction) via findHomography; glare and board area via convexHull", 228)
    y = s.tool(x, y, "estimate_pose", "geometry", "solvePnPGeneric (SOLVEPNP_IPPE), solvePnPRefineLM, Rodrigues; "
               "IPPE ambiguity ratio", 228)
    s.para(x, y + 2, "Gate R0–R5: no board, too small, blur, glare → reject with a measured reason; "
                     "ambiguity, off-target → flag and re-plan", 228, size=13.5, fill=HUMAN)

    x, y = 664, top + 50
    y = s.tool(x, y, "calibrate", "calib", "calibrateCameraExtended (k5 or CALIB_RATIONAL_MODEL), "
               "calibrateCameraROExtended", 228)
    y = s.tool(x, y, "residual_analysis", None, "R6 drift, R7 outlier views, R8 radial under-fit, R9 board "
               "warp, R10 out of scope", 228)
    y = s.tool(x, y, "uncertainty_map", "geometry", "projectPoints Jacobians + Schur complement → σ per pixel "
               "(32×18), U95; undistortPoints with criteria", 228)
    s.tool(x, y, "stop_rule", None, "P(true field p95 > τ) from a DEV-fitted calibration → R11 stop, "
           "R12 budget", 228)

    x, y = 931, top + 50
    y = s.tool(x, y, "propose_next_views", "geometry", "300 candidate poses scored by predicted U95 drop minus "
               "effort (R13)", 228)
    y = s.tool(x, y, "instruction", None, "ghost outline + numeric instruction (human) or a pose (robot)", 228)
    y = s.tool(x, y, "sim_capture", "imgproc+core", "warpPerspective, remap, filter2D (blur PSF), "
               "imencode/imdecode (JPEG, imgcodecs)", 228)
    s.tool(x, y, "export_calibration", "core+imgproc", "FileStorage YAML in memory, applyColorMap report card",
           228)

    # loop arrows between stages
    mid = top + 200
    s.arrow(f"M{385 + col_w},{mid} L{652 - 4},{mid}", sw=3)
    s.arrow(f"M{652 + col_w},{mid} L{919 - 4},{mid}", sw=3)
    ly = top + col_h + 22
    s.arrow(f"M{919 + col_w / 2},{top + col_h} L{919 + col_w / 2},{ly} L{385 + col_w / 2},{ly} "
            f"L{385 + col_w / 2},{top + col_h + 4}", sw=3)
    s.text(777, ly + 22, "next frame (render or upload): the visual result changes the next target, tool call "
                         "or approval", size=13.5, anchor="middle", weight="bold", fill=INK)

    # approval gate
    gy = ly + 40
    s.rect(385, gy, 784, 106, fill="#fef2f2", stroke=HUMAN, sw=2)
    s.text(401, gy + 24, "Approval gate", size=15.5, weight="bold", fill=HUMAN)
    s.para(401, gy + 45, "switch_model (R8) · use_ro (R9) · drift_restart (R6) · accept_with_warnings (R12) · "
                         "export (R11) · out_of_scope (R10). The lens model never changes and nothing is exported "
                         "without an approval record {id, kind, decision, reason, ts} in the state and the trace.",
           750, size=13.5)
    s.arrow(f"M385,{gy + 53} L330,{gy + 53}", color=HUMAN, sw=2.5, both=True, marker="h")
    # browser <-> API <-> Lambda
    s.arrow("M330,158 L370,158", sw=2.5, both=True)
    s.arrow("M772,188 L772,212", sw=2.5, both=True)

    # services
    sy, sh = 912, 168
    s.rect(370, sy, 290, sh, fill="#ffffff", stroke=AWS, sw=2)
    s.text(384, sy + 24, "CloudWatch", size=15.5, weight="bold")
    s.para(384, sy + 46, "Logs: one JSON line per request (no corners, no images, no tokens). EMF metrics: "
                         "latency, NBV time, rejects, approvals, stops, cold starts. Logs Insights for p50/p95.",
           265, size=13)
    s.rect(675, sy, 290, sh, fill="#ffffff", stroke=AWS, sw=2)
    s.text(689, sy + 24, "S3 (private)", size=15.5, weight="bold")
    s.para(689, sy + 46, "On an approved export only: reports/ (OpenCV YAML, report-card PNG, JSON) and "
                         "traces/. Block Public Access, SSE-S3, TLS-only; presigned links expire after 1 h, "
                         "objects after 30 days.", 265, size=13)
    s.rect(980, sy, 195, sh, fill="#ffffff", stroke=AWS, sw=2)
    s.text(994, sy + 24, "ECR", size=15.5, weight="bold")
    s.para(994, sy + 46, "Container image pinned by digest; base image pinned too. Lifecycle keeps the last "
                         "3 images.", 170, size=13)
    s.arrow(f"M515,{gy + 106} L515,{sy - 2}", color=AWS, sw=2.2, marker="w")
    s.text(523, sy - 12, "logs + metrics", size=12.5, fill=AWS)
    s.arrow(f"M820,{gy + 106} L820,{sy - 2}", color=AWS, sw=2.2, marker="w")
    s.text(828, sy - 12, "approved export", size=12.5, fill=AWS)
    s.arrow(f"M1078,{sy} L1078,{gy + 108}", color=AWS, sw=2.2, marker="w")
    s.text(1086, sy - 12, "image", size=12.5, fill=AWS)

    # ---------------- Right: local ----------------
    s.rect(1215, 92, 365, 998, fill="#f8fafc", stroke=LOCAL, sw=2, dash="3 4", rx=14)
    s.text(1231, 118, "Local (not deployed)", size=16, weight="bold", fill=LOCAL)

    s.rect(1231, 132, 333, 250, fill="#ffffff", stroke=LOCAL, sw=1.6)
    s.text(1245, 156, "MCP server (stdio, FastMCP)", size=15.5, weight="bold")
    yy = s.para(1245, 177, "The same tools as the endpoint: sim_new_session, sim_capture, detect_board, "
                           "assess_frame, accept_view, calibrate*, uncertainty_map, residual_analysis, "
                           "propose_next_views, drop_view, export_calibration*.", 310, size=13)
    s.para(1245, yy + 6, "* gated: the server asks the human through MCP elicitation; a client that cannot "
                         "ask gets needs_approval and nothing changes.", 310, size=13, fill=HUMAN)

    s.rect(1231, 410, 333, 138, fill="#ffffff", stroke=LOCAL, sw=1.6)
    s.text(1245, 434, "LLM driver or any MCP client", size=15.5, weight="bold")
    s.para(1245, 455, "qwen2.5:7b through Ollama at temperature 0 picks among the planner's ranked candidates; "
                      "every call is logged (JSONL). Reported as an extra row, not the hosted agent.",
           310, size=13)
    s.arrow("M1397,408 L1397,384", color=LOCAL, sw=2.2, both=True, marker="l")

    s.rect(1231, 576, 333, 232, fill="#ffffff", stroke=LOCAL, sw=1.6)
    s.text(1245, 600, "Benchmark harness", size=15.5, weight="bold")
    s.para(1245, 621, "Simulator with exact ground truth; 11 policies (agent, ablations, random, template, "
                      "OpenCV interactive-calibration rules, coverage-greedy, dense grid) on pre-registered "
                      "splits, run with multiprocessing. Writes results/ (CSV, JSON, figures), from which "
                      "the report is generated.", 310, size=13)

    s.rect(1231, 836, 333, 150, fill="#ffffff", stroke=LOCAL, sw=1.6)
    s.text(1245, 860, "Real data: METRIC", size=15.5, weight="bold")
    s.para(1245, 881, "Zenodo 10.5281/zenodo.7976757, CC BY 4.0. Real D455 and synthetic KinectV2 ChArUco "
                      "frames, fetched with HTTP range reads (no full download); pool evaluation of the same "
                      "agent.", 310, size=13)
    s.arrow("M1397,836 L1397,812", color=LOCAL, sw=2.2, marker="l")
    s.para(1231, 1018, "Python 3.12 · opencv-python-headless 5.0.0.93 · numpy 2.5.1 in the image; scipy, "
                       "matplotlib, mcp, httpx and pytest locally.", 340, size=12.5, fill=MUTED)
    return s.svg()


def render_png(svg_path: Path, png_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        page.goto(svg_path.resolve().as_uri())
        page.screenshot(path=str(png_path), clip={"x": 0, "y": 0, "width": W, "height": H})
        b.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--svg-only", action="store_true")
    a = ap.parse_args(argv)
    svg_path, png_path = DOCS / "architecture.svg", DOCS / "architecture.png"
    svg_path.write_text(draw(), encoding="utf-8", newline="\n")
    print("wrote", svg_path)
    if not a.svg_only:
        render_png(svg_path, png_path)
        print("wrote", png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
