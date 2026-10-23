"""Evidence figures E1-E10 (spec 11.7) from the result files only.

    python -m nextview.bench.plots --synth results/synth_test --pool results/pool --out results/figures

Inputs: ``<synth>/summary.json`` (summarize), ``<synth>/consequence.json`` (consequence),
``<synth>/sessions.jsonl`` (predicted vs realised gains), ``<pool>/metric_*.json`` (run_pool).
Every number drawn comes from those files; nothing is typed in. Figures are PNG (and the data
behind each one is the JSON it was read from, which is the table view).

Colours: one fixed hue per policy, in the validated categorical order (A blue, B0 orange,
B1 aqua, B2 yellow, B3 magenta, B4 green, B2-k5 violet), never cycled; baselines also carry a
dash pattern and every line is labelled at its end, so identity never rests on colour alone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

__all__ = ["main"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
COLORS = {"A": "#2a78d6", "B0": "#eb6834", "B1": "#1baf7a", "B2": "#eda100", "B3": "#e87ba4", "B4": "#008300",
          "B2-k5": "#6250d6"}
DASH = {"A": "-", "B0": (0, (5, 2)), "B1": (0, (1, 1.5)), "B2": (0, (6, 2, 1, 2)), "B3": (0, (3, 1.5)),
        "B4": (0, (8, 2)), "B2-k5": (0, (2, 2))}
POOL_COLORS = {"P-agent": "#2a78d6", "P-first": "#eb6834", "P-tmpl": "#1baf7a", "P-cov": "#e87ba4", "P-rand": "#8a8984"}
ACTS = ("robot", "human")
MAIN = ("A", "B0", "B1", "B2", "B3", "B4")


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    })
    return plt


def _end_labels(ax, items, log=False, min_gap=0.055):
    """Direct labels at line ends, spread vertically so they do not collide. items: (x, y, text)."""
    items = [(x, y, t) for x, y, t in items if math.isfinite(y) and (y > 0 or not log)]
    if not items:
        return
    lo, hi = ax.get_ylim()
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    finv = (lambda v: 10 ** v) if log else (lambda v: v)
    span = f(hi) - f(lo)
    pos = sorted(((f(y), x, y, t) for x, y, t in items))
    placed = []
    for fy, x, y, t in pos:
        if placed and fy - placed[-1][0] < min_gap * span:
            fy = placed[-1][0] + min_gap * span
        placed.append((fy, x, y, t))
    for fy, x, y, t in placed:
        ax.annotate(t, (x, y), xytext=(x, finv(fy)), textcoords="data", va="center", fontsize=8, color=INK2,
                    xycoords="data", annotation_clip=False)
        ax.annotate("", (x, y))


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(p)


def e1(summary, out):
    plt = _mpl()
    cur = summary["curves_E1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, act in zip(axes, ACTS):
        ends = []
        for pol in MAIN:
            c = cur.get(f"{pol}/{act}")
            if not c:
                continue
            ns = sorted(int(n) for n in c if c[n])
            med = np.array([c[str(n)]["median"] for n in ns])
            lo = np.array([c[str(n)]["q25"] for n in ns])
            hi = np.array([c[str(n)]["q75"] for n in ns])
            ax.plot(ns, med, color=COLORS[pol], linestyle=DASH[pol], label=pol, zorder=3 if pol == "A" else 2)
            if pol in ("A", "B0"):
                ax.fill_between(ns, lo, hi, color=COLORS[pol], alpha=0.12, linewidth=0)
            ends.append((ns[-1] + 0.6, med[-1], pol))
        ax.set_yscale("log")
        _end_labels(ax, ends, log=True)
        ax.set_title(f"{act} actuator")
        ax.set_xlabel("accepted frames")
        ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=":")
    axes[0].set_ylabel("true field p95 (px, log)")
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    fig.suptitle("E1  Median true field p95 vs frames (IQR band for A and B0), synthetic TEST", fontsize=10, x=0.01, ha="left")
    return _save(fig, out, "E1_field_p95_vs_frames")


def e2(summary, out):
    plt = _mpl()
    pe = summary["primary_endpoints"]
    groups = [("true field p95 at n = 10, A minus baseline (px)", [e for e in pe if e["metric"] == "p95@10"]),
              ("frames until p95 < 1 px, A minus B0", [e for e in pe if e["metric"] == "frames<1px"])]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.2), gridspec_kw={"height_ratios": [6, 2.6]})
    for ax, (label, es) in zip(axes, groups):
        ys = np.arange(len(es))[::-1]
        for y, e in zip(ys, es):
            lo, hi = e["ci"]
            ax.plot([lo, hi], [y, y], color=COLORS["A"], linewidth=2)
            ax.plot([e["median_diff"]], [y], "o", color=COLORS["A"], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
            star = "  (Holm < 0.05)" if e["significant_holm_0.05"] else ""
            ax.text(1.01, y, f"{e['median_diff']:+.2f} [{lo:+.2f}, {hi:+.2f}]  Holm p = {e['p_holm']:.1e}{star}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{e['id']}  A - {e['comparison'].split(' - ')[1]}  {e['actuator']}" for e in es], fontsize=8)
        ax.set_ylim(-0.6, len(es) - 0.4)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_xlabel(label)
        lo_all = min(min(e["ci"][0] for e in es), 0)
        hi_all = max(max(e["ci"][1] for e in es), 0)
        pad = 0.08 * (hi_all - lo_all or 1)
        ax.set_xlim(lo_all - pad, hi_all + pad)
    fig.suptitle("E2  Pre-registered endpoints: paired median difference, 95 % bootstrap CI (negative = A better)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "E2_forest_primary_endpoints")


def e3(summary, out):
    plt = _mpl()
    km = summary["frames_to_1px_E3"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, act in zip(axes, ACTS):
        ends = []
        for pol in MAIN:
            k = km.get(f"{pol}/{act}")
            if not k:
                continue
            t = [p["t"] for p in k["curve"]]
            s = [p["S"] for p in k["curve"]]
            ax.step(t, s, where="post", color=COLORS[pol], linestyle=DASH[pol], label=pol)
            ends.append((t[-1] + 0.6, s[-1], pol))
        ax.set_title(f"{act} actuator")
        ax.set_xlabel("accepted frames")
        ax.set_ylim(0, 1.02)
        _end_labels(ax, ends)
    axes[0].set_ylabel("runs not yet below 1 px (KM)")
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    fig.suptitle("E3  Frames until the true field p95 is below 1 px (censored at 30)", fontsize=10, x=0.01, ha="left")
    return _save(fig, out, "E3_frames_to_1px")


def e4(summary, out):
    plt = _mpl()
    pp = summary["per_policy"]
    pols = ["A", "A-gate", "A-outlier", "A-model", "A-nbv", "B0"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, act in zip(axes, ACTS):
        x = np.arange(len(pols))
        for j, (n, shade) in enumerate(((10, 0.55), (30, 1.0))):
            vals = [((pp.get(f"{p}/{act}/all") or {}).get(f"p95@{n}") or {}).get("median", np.nan) for p in pols]
            cols = [COLORS["A"] if p.startswith("A") else COLORS["B0"] for p in pols]
            bars = ax.bar(x + (j - 0.5) * 0.38, vals, width=0.36, color=cols, alpha=shade, edgecolor=SURFACE, linewidth=2)
            for b, v in zip(bars, vals):
                if math.isfinite(v):
                    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                                ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(pols, fontsize=8)
        ax.set_title(f"{act}: light bar n = 10, dark bar n = 30")
    axes[0].set_ylabel("median true field p95 (px)")
    fig.suptitle("E4  Ablations: what each component contributes (synthetic TEST, all families)", fontsize=10, x=0.01, ha="left")
    return _save(fig, out, "E4_ablations")


def e5(summary, out):
    plt = _mpl()
    h = summary["uncertainty_honesty_E5"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle=":")
    for key, col, ls in (("A/robot", COLORS["A"], "-"), ("A/human", COLORS["A"], (0, (4, 2))),
                         ("B0/robot", COLORS["B0"], "-"), ("B0/human", COLORS["B0"], (0, (4, 2)))):
        rel = h.get(key, {}).get("reliability")
        if not rel:
            continue
        x = [r["nominal"] for r in rel]
        y = [r["empirical_mean"] for r in rel]
        ax.plot(x, y, "o", color=col, linestyle=ls, markersize=5, label=key)
    ax.set_xlim(0.4, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage q (chi-square(2) ellipse)")
    ax.set_ylabel("fraction of grid cells inside")
    ax.set_title("reliability of the first-order map (>= 6 views)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for key, col in (("A/robot", COLORS["A"]), ("A/human", "#256abf")):
        r = h.get(key, {}).get("r")
        if not r:
            continue
        edges = r["hist_edges"][:-1] + [15]
        cnt = np.asarray(r["hist"], float)
        frac = cnt / cnt.sum()
        centres = np.arange(len(frac))
        off = -0.2 if key.endswith("robot") else 0.2
        ax.bar(centres + off, frac, width=0.38, color=col, label=f"{key} (median r {r['median']:.2f})",
               edgecolor=SURFACE, linewidth=1)
    labels = ["<.25", ".25-.5", ".5-.75", ".75-1", "1-1.25", "1.25-1.5", "1.5-2", "2-3", "3-5", "5-10", ">10"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=30)
    ax.set_xlabel("r = true field p95 / U95")
    ax.set_ylabel("fraction of steps")
    ax.legend(fontsize=8)
    ax.set_title("r on TEST (stop calibration fitted on DEV)")
    fig.suptitle("E5  Uncertainty honesty", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "E5_reliability")


def e6(summary, out):
    plt = _mpl()
    rc = summary["stop_quality"]["risk_coverage"]
    tau = summary["stop_quality"]["tau_px"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for act, ls in (("robot", "-"), ("human", (0, (4, 2)))):
        pts = rc.get(act) or []
        x = [p["mean_frames"] for p in pts]
        y = [p["false_stops_per_run"] for p in pts]
        ax.plot(x, y, "o", color=COLORS["A"], linestyle=ls, markersize=5, label=f"A, {act}")
        for p, xi, yi in zip(pts, x, y):
            ax.annotate(f"{p['stop_p_exceed']:g}", (xi, yi), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xlabel("mean frames used (no stop = 30)")
    ax.set_ylabel(f"false stops per run (true p95 > {2 * tau:g} px)")
    ax.legend(fontsize=8)
    ax.set_title(f"E6  Risk-coverage of the stop threshold (labels = stop_p_exceed, tau {tau:g} px)", fontsize=9)
    return _save(fig, out, "E6_risk_coverage")


def _pool_fig(pool_files, metric, title, name, out, ylabel, log=True, clip=False):
    """One panel per camera. ``clip``: the y axis ends just above the P-rand 5-95 % band (or the
    policies' values when no band), and a point above that is drawn at the top edge as a
    triangle labelled with its value, so one diverged fit does not flatten the other lines."""
    plt = _mpl()
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    cams = []
    for f in pool_files:
        cams += json.loads(Path(f).read_text(encoding="utf-8"))["cameras"]
    if not cams:
        return None
    fig, axes = plt.subplots(1, len(cams), figsize=(3.0 * len(cams), 3.2), squeeze=False)
    for ax, c in zip(axes[0], cams):
        bs = sorted(int(b) for b in c["budgets"])
        series = {}
        for pol in ("P-first", "P-tmpl", "P-cov", "P-agent"):
            xs, ys = [], []
            for b in bs:
                e = c["budgets"][str(b)].get(pol)
                if e and e.get("ok") and math.isfinite(e.get(metric, math.nan)):
                    xs.append(b)
                    ys.append(e[metric])
            if xs:
                series[pol] = (xs, ys)
        xs = [b for b in bs if c["budgets"][str(b)].get("P-rand")]
        band = [c["budgets"][str(b)]["P-rand"][metric] for b in xs]
        top = None
        if clip:
            ref = [x["p95"] for x in band] or [y for _, ys in series.values() for y in ys]
            vals = sorted(y for _, ys in series.values() for y in ys)
            med = vals[len(vals) // 2] if vals else None
            top = max(ref + [y for y in vals if med is not None and y <= 20 * med]) * (1.6 if log else 1.1)
        for pol, (px, py) in series.items():
            yy = [min(y, top) if top else y for y in py]
            ax.plot(px, yy, "o-", color=POOL_COLORS[pol], markersize=4, label=pol, linewidth=1.6 if pol == "P-agent" else 1.2)
            for x, y in zip(px, py):
                if top and y > top:
                    ax.plot([x], [top], marker="^", color=POOL_COLORS[pol], markersize=7, clip_on=False)
                    ax.annotate(f"{pol} {y:.3g} px, off scale", (x, top), xytext=(6, -10), textcoords="offset points",
                                fontsize=7, color=INK2, annotation_clip=False)
        if xs:
            ax.fill_between(xs, [x["p05"] for x in band], [x["p95"] for x in band], color=POOL_COLORS["P-rand"], alpha=0.18,
                            linewidth=0, label="P-rand 5-95 %")
            ax.plot(xs, [x["median"] for x in band], color=POOL_COLORS["P-rand"], linewidth=1, linestyle=":")
        if log:
            ax.set_yscale("log")
        if top:
            ax.set_ylim(top=top)
        if log:
            lo, hi = ax.get_ylim()
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if hi / lo <= 100 else (1.0,)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{c['subset']} cam{c['camera']} (pool {c['pool']}, held-out {c['heldout']})", fontsize=8)
        ax.set_xlabel("frames (budget)")
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, name)


def e9(cons, out):
    plt = _mpl()
    t = cons["table"]
    pols = ["A", "B0", "B1", "B2", "B3", "B4", "A-gate", "A-outlier", "A-model", "A-nbv"]
    rows = []
    for pol in pols:
        for act in ACTS:
            r10 = t.get(f"{pol}/{act}/n10", {}).get("Z1.0", {})
            r30 = t.get(f"{pol}/{act}/n30", {}).get("Z1.0", {})
            rs = t.get(f"{pol}/{act}/stop", {}).get("Z1.0", {})
            if not r10:
                continue
            fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
            rows.append([pol, act, fmt(r10.get("median_of_run_median_mm")), fmt(r10.get("median_of_run_p95_mm")),
                         fmt(rs.get("median_of_run_median_mm")), fmt(r30.get("median_of_run_median_mm"))])
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(rows) + 1.0))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["policy", "actuator", "|dL| median\nn = 10", "|dL| p95\nn = 10",
                                            "|dL| median\nat stop", "|dL| median\nn = 30"], loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.25)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0:
            cell.set_text_props(color=INK, weight="bold")
            cell.set_height(cell.get_height() * 1.9)
    ax.set_title("E9  Length error of 100 mm segments in the outer 30 % of the image, plane at Z = 1.0 m (mm; median over runs)",
                 fontsize=9, loc="left")
    return _save(fig, out, "E9_mm_consequence")


def e10(sessions_path, out):
    plt = _mpl()
    pg, rg, tg = [], [], []
    from .files import load_sessions

    for s in load_sessions(sessions_path):
        if s.get("policy") != "A":
            continue
        for e in s.get("realised") or []:
            if e.get("pred_gain") is not None and e.get("realised_gain") is not None:
                pg.append(e["pred_gain"])
                rg.append(e["realised_gain"])
                tg.append(e.get("true_gain", math.nan))
    if not pg:
        return None
    pg, rg, tg = np.array(pg), np.array(rg), np.array(tg, float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pg), size=min(3000, len(pg)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, y, lab in ((axes[0], rg, "realised U95 drop after refit (px)"), (axes[1], tg, "realised true p95 drop (px)")):
        ok = np.isfinite(y[idx])
        ax.scatter(pg[idx][ok], y[idx][ok], s=8, color=COLORS["A"], alpha=0.35, linewidths=0)
        lim = np.nanpercentile(np.abs(np.concatenate([pg, y[np.isfinite(y)]])), 99)
        ax.plot([0, lim], [0, lim], color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xlim(-0.02 * lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color=INK2, linewidth=0.6)
        ax.set_xlabel("predicted U95 gain of the chosen view (px)")
        ax.set_ylabel(lab)
    fig.suptitle(f"E10  Predicted vs realised gain, policy A ({len(pg)} views; 3,000 shown)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out, "E10_predicted_vs_realised")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", default="results/synth_test")
    ap.add_argument("--pool", default="results/pool")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args(argv)
    out = Path(a.out)
    made = []
    sp = Path(a.synth) / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        made += [e1(s, out), e2(s, out), e3(s, out), e4(s, out), e5(s, out), e6(s, out)]
    cp = Path(a.synth) / "consequence.json"
    if cp.exists():
        made.append(e9(json.loads(cp.read_text(encoding="utf-8")), out))
    if any((Path(a.synth) / n).exists() for n in ("sessions.jsonl", "sessions.jsonl.gz")):
        made.append(e10(Path(a.synth), out))
    pdir = Path(a.pool)
    d455 = sorted(pdir.glob("metric_d455.json"))
    kin = sorted(pdir.glob("metric_kinect_synth.json"))
    if d455:
        made.append(_pool_fig(d455, "heldout_rms", "E7  METRIC D455 (real): held-out reprojection RMS vs budget", "E7_metric_d455_heldout",
                              out, "held-out RMS (px, log)", clip=True))
        made.append(_pool_fig(d455, "field_p95", "METRIC D455: field p95 difference from FACTORY intrinsics (a reference, not GT)",
                              "E7b_metric_d455_field_vs_factory", out, "field p95 vs factory (px, log)"))
    if kin:
        made.append(_pool_fig(kin, "field_p95", "E8  METRIC KinectV2 (synthetic, GT intrinsics): true field p95 vs budget",
                              "E8_metric_kinect_field", out, "true field p95 (px, log)"))
        made.append(_pool_fig(kin, "heldout_rms", "METRIC KinectV2: held-out reprojection RMS vs budget", "E8b_metric_kinect_heldout",
                              out, "held-out RMS (px, log)", clip=True))
    made = [m for m in made if m]
    print("\n".join(made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
