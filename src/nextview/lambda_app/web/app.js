"use strict";
/* NextView single-page UI: simulator, webcam, evidence and about tabs. No external assets. */
(() => {
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const isNum = (x) => typeof x === "number" && Number.isFinite(x);
  const fmt = (x, nd = 2) => (isNum(x) ? x.toFixed(nd) : "n/a");
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function h(tag, props, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : String(v));
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return e;
  }

  // ---------------------------------------------------------------- notices
  const notices = new Map();
  function notice(key, msg, kind) {
    if (msg) notices.set(key, { msg, kind });
    else notices.delete(key);
    const box = $("notice");
    const items = [...notices.values()];
    if (!items.length) { box.hidden = true; box.textContent = ""; return; }
    const top = items[items.length - 1];
    box.textContent = top.msg;
    box.className = "notice" + (top.kind === "error" ? " error" : "");
    box.hidden = false;
  }

  // ---------------------------------------------------------------- API with retry/backoff (spec 8.4)
  const BACKOFF_MS = [500, 1000, 2000, 4000]; // 5 tries in all
  let lastRequestAt = 0;
  async function api(path, body) {
    const cold = Date.now() - lastRequestAt > 5 * 60 * 1000; // first request, or the container may have been recycled
    const warm = cold ? setTimeout(() => notice("warm", "Warming up (first request starts the container)…"), 2500) : null;
    try {
      for (let attempt = 0; ; attempt++) {
        let resp = null, data = null, netErr = null;
        try {
          resp = await fetch(path, {
            method: body ? "POST" : "GET",
            headers: body ? { "content-type": "application/json" } : {},
            body: body ? JSON.stringify(body) : undefined,
            cache: "no-store",
          });
          const text = await resp.text();
          try { data = text ? JSON.parse(text) : null; } catch (_) { data = null; }
        } catch (e) { netErr = e; }
        lastRequestAt = Date.now();
        const status = resp ? resp.status : 0;
        const ownError = data && data.error && data.error.code;
        const retryable = netErr || status === 429 || (status >= 500 && ownError !== "INTERNAL");
        if (!retryable) {
          if (resp.ok) return data;
          const err = new Error((data && data.error && data.error.message) || `HTTP ${status}`);
          err.status = status; err.code = ownError || null;
          throw err;
        }
        if (attempt >= BACKOFF_MS.length) {
          const err = new Error(netErr ? `Network error (${netErr.message})` : `Service busy (HTTP ${status}); try again in a moment`);
          err.status = status; err.code = "BUSY";
          throw err;
        }
        notice("busy", `Busy — retrying (${attempt + 1}/${BACKOFF_MS.length})…`);
        await sleep(BACKOFF_MS[attempt] * (0.5 + Math.random())); // jitter keeps parallel clients from retrying in lockstep
      }
    } finally {
      if (warm) clearTimeout(warm);
      notice("warm", null);
      notice("busy", null);
    }
  }

  // A session start waits out a busy service the way auto-run does: after the retries above give up,
  // pause 5 s and try again, up to 6 times. The state lives in the returned token, so a repeat is harmless.
  const BUSY_WAITS = 6, BUSY_WAIT_MS = 5000;
  async function apiPatient(path, body) {
    for (let wait = 1; ; wait++) {
      try {
        return await api(path, body);
      } catch (e) {
        if (e.code !== "BUSY" || wait > BUSY_WAITS) throw e;
        notice("start", `Service busy: starting again in 5 s (wait ${wait}/${BUSY_WAITS})…`);
        await sleep(BUSY_WAIT_MS);
        notice("start", null);
      }
    }
  }

  // ---------------------------------------------------------------- colour maps
  const VIRIDIS = [[68, 1, 84], [72, 40, 120], [62, 73, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137],
    [53, 183, 121], [110, 206, 88], [181, 222, 43], [253, 231, 37]];
  function viridis(t) {
    t = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
    const i = Math.min(VIRIDIS.length - 2, Math.floor(t)), f = t - i;
    const a = VIRIDIS[i], b = VIRIDIS[i + 1];
    return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f));
  }

  // ---------------------------------------------------------------- overlays (image pixel coordinates)
  function drawOverlays(ctx, W, H, d, opt) {
    const s = W / d.size[0];
    if (opt.unc && d.unc && d.unc.sigma_px) {
      const [cols, rows] = d.unc.grid, v = d.unc.sigma_px;
      const fin = v.filter(isNum);
      if (fin.length) {
        const lo = Math.max(1e-3, Math.min(...fin)), hi = Math.max(lo * 1.0001, Math.max(...fin));
        const L = Math.log(lo), R = Math.log(hi) - L;
        const cw = W / cols, ch = H / rows;
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          const x = v[r * cols + c];
          if (!isNum(x)) continue;
          const [R8, G8, B8] = viridis((Math.log(Math.max(x, lo)) - L) / R);
          ctx.fillStyle = `rgba(${R8},${G8},${B8},0.55)`;
          ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
        }
        if (opt.legend) opt.legend(lo, hi);
      }
    }
    if (opt.coverage && d.coverage) {
      const [cols, rows] = d.coverage.grid, n = d.coverage.counts;
      const mx = Math.max(1, ...n), cw = W / cols, ch = H / rows;
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const k = n[r * cols + c];
        ctx.fillStyle = k === 0 ? "rgba(230,70,60,0.28)" : `rgba(40,170,230,${(0.12 + 0.5 * Math.log1p(k) / Math.log1p(mx)).toFixed(3)})`;
        ctx.fillRect(c * cw + 1, r * ch + 1, cw - 2, ch - 2);
      }
    }
    if (opt.corners && d.corners && d.corners.length) {
      ctx.lineWidth = 2 * Math.max(1, s);
      for (const [x, y] of d.corners) {
        ctx.beginPath();
        ctx.arc(x * s, y * s, 5 * Math.max(1, s), 0, 2 * Math.PI);
        ctx.strokeStyle = "rgba(0,0,0,0.75)"; ctx.lineWidth = 4; ctx.stroke();
        ctx.strokeStyle = "#39f59a"; ctx.lineWidth = 2; ctx.stroke();
      }
    }
    if (opt.ghost && d.ghost && d.ghost.length === 4) {
      ctx.save();
      ctx.beginPath();
      d.ghost.forEach(([x, y], i) => (i ? ctx.lineTo(x * s, y * s) : ctx.moveTo(x * s, y * s)));
      ctx.closePath();
      ctx.fillStyle = "rgba(255,190,40,0.10)"; ctx.fill();
      ctx.setLineDash([18, 12]); ctx.lineWidth = 6; ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.stroke();
      ctx.lineWidth = 3.5; ctx.strokeStyle = "#ffbe28"; ctx.stroke();
      ctx.setLineDash([]);
      const [x0, y0] = d.ghost[0];
      ctx.font = "600 26px system-ui, sans-serif";
      const label = "next target";
      const tx = Math.min(Math.max(8, x0 * s), W - 150), ty = Math.min(Math.max(30, y0 * s - 10), H - 8);
      ctx.fillStyle = "rgba(0,0,0,0.65)"; ctx.fillRect(tx - 4, ty - 24, ctx.measureText(label).width + 8, 30);
      ctx.fillStyle = "#ffbe28"; ctx.fillText(label, tx, ty);
      ctx.restore();
    }
  }

  // ---------------------------------------------------------------- race chart
  function lastPerX(points) {
    const m = new Map();
    for (const [x, y] of points) if (isNum(x) && isNum(y) && y > 0) m.set(x, y);
    return [...m.entries()].sort((a, b) => a[0] - b[0]);
  }
  function drawRace(canvas, history, tau, budget) {
    const dpr = 2, Wc = 640, Hc = 260;
    if (canvas.width !== Wc * dpr) { canvas.width = Wc * dpr; canvas.height = Hc * dpr; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, Wc, Hc);
    const ink = cssVar("--ink-2") || "#444", line = cssVar("--line") || "#ddd", accent = cssVar("--accent") || "#127a6e";
    const base = cssVar("--base") || "#8a6d3b", muted = cssVar("--muted") || "#777";
    const agent = lastPerX(history.map((h) => [h.views, h.true_p95]));
    const rnd = lastPerX(history.map((h) => [h.baseline_views, h.baseline_true_p95]));
    const L = 48, Rr = 14, T = 26, B = 34, pw = Wc - L - Rr, ph = Hc - T - B;
    const ys = [...agent, ...rnd].map((p) => p[1]);
    const ymin = Math.min(0.1, ...ys, tau * 0.8), ymax = Math.max(2, ...ys);
    const ly0 = Math.log10(ymin), ly1 = Math.log10(ymax);
    const xmax = Math.max(budget || 30, ...agent.map((p) => p[0]), ...rnd.map((p) => p[0]));
    const X = (v) => L + (v / xmax) * pw;
    const Y = (v) => T + ph - ((Math.log10(v) - ly0) / (ly1 - ly0)) * ph;
    ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = muted; ctx.strokeStyle = line; ctx.lineWidth = 1;
    for (const t of [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]) {
      if (t < ymin || t > ymax) continue;
      ctx.beginPath(); ctx.moveTo(L, Y(t)); ctx.lineTo(L + pw, Y(t)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(String(t), L - 6, Y(t) + 4);
    }
    const step = xmax > 20 ? 5 : 2;
    ctx.textAlign = "center";
    for (let v = 0; v <= xmax; v += step) ctx.fillText(String(v), X(v), T + ph + 16);
    ctx.fillText("accepted frames", L + pw / 2, Hc - 4);
    ctx.save(); ctx.translate(12, T + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("true field p95, px (log)", 0, 0); ctx.restore();
    for (const [v, lab, dy] of [[tau, "τ " + tau + " px", 13], [1, "1 px", -4]]) { // τ label below its line, 1 px above
      if (v < ymin || v > ymax) continue;
      ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = ink; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(L + pw, Y(v)); ctx.stroke(); ctx.restore();
      ctx.textAlign = "right"; ctx.fillStyle = ink; ctx.fillText(lab, L + pw - 2, Y(v) + dy);
    }
    const series = (pts, color, width) => {
      if (!pts.length) return;
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)))); ctx.stroke();
      for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(X(x), Y(y), 2.6, 0, 2 * Math.PI); ctx.fill(); }
    };
    series(rnd, base, 1.8);
    series(agent, accent, 2.6);
    ctx.textAlign = "left"; ctx.font = "600 12px system-ui, sans-serif";
    ctx.fillStyle = accent; ctx.fillText("● NextView agent", L, 14);
    ctx.fillStyle = base; ctx.fillText("● random baseline", L + 122, 14);
    if (!agent.length && !rnd.length) {
      ctx.fillStyle = muted; ctx.font = "13px system-ui, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("No ground-truth error yet (needs 3 accepted views)", L + pw / 2, T + ph * 0.3);
    }
  }

  // ---------------------------------------------------------------- residual field (approval evidence)
  function drawField(canvas, field) {
    const [cols, rows] = field.grid, vec = field.vectors;
    canvas.width = 640; canvas.height = 360;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 640, 360);
    const cw = 640 / cols, ch = 360 / rows;
    const mags = vec.filter((v) => v && isNum(v[0]) && isNum(v[1])).map((v) => Math.hypot(v[0], v[1])).sort((a, b) => a - b);
    if (!mags.length || mags[mags.length - 1] <= 0) return 0;
    const mx = mags[mags.length - 1], ref = Math.max(mags[Math.floor(0.9 * (mags.length - 1))], mx * 0.05);
    const k = (0.45 * Math.min(cw, ch)) / ref; // the 90th-percentile arrow spans ~half a cell; longer ones are clipped
    const cap = 0.95 * Math.min(cw, ch);
    ctx.strokeStyle = cssVar("--gate") || "#6a3fb5"; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 2;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const v = vec[r * cols + c];
      const x0 = (c + 0.5) * cw, y0 = (r + 0.5) * ch;
      if (!v || !isNum(v[0]) || !isNum(v[1])) { ctx.fillRect(x0 - 1, y0 - 1, 2, 2); continue; }
      let dx = v[0] * k, dy = v[1] * k;
      const len = Math.hypot(dx, dy);
      if (len > cap) { dx *= cap / len; dy *= cap / len; }
      const x1 = x0 + dx, y1 = y0 + dy;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      const a = Math.atan2(y1 - y0, x1 - x0);
      ctx.beginPath(); ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - 7 * Math.cos(a - 0.45), y1 - 7 * Math.sin(a - 0.45));
      ctx.lineTo(x1 - 7 * Math.cos(a + 0.45), y1 - 7 * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill();
    }
    return mx;
  }

  // ---------------------------------------------------------------- trace classes
  function traceClass(line) {
    const m = /^#\d+ ([A-Z0-9_]+)/.exec(line);
    const r = m ? m[1] : "";
    if (/^R[0-3]_/.test(r)) return "reject";
    if (/^R[45]_/.test(r)) return "flag";
    if (/^R(6|8|9|10|11|12)_/.test(r)) return "gate";
    if (r === "R13_NBV" || r === "R7_OUTLIER") return "plan";
    if (r === "APPROVAL") return "approval";
    if (r === "USER") return "user";
    return "sys";
  }
  const EVIDENCE_LABELS = {
    radial_trend_p: "radial trend p (corner-level screen)", cv_rms: "held-out RMS (px)", warp_score: "warp score", cv_gain: "CV gain",
    U95_px: "U95 (px)", p_exceed: "P(field p95 > τ)", tau_px: "τ (px)", views: "views", attempts: "attempts",
    model: "model", flags: "flags", label: "assessment", rms_px: "RMS (px)", hfov_deg: "estimated HFOV (°)",
    hfov_limit_deg: "HFOV limit (°)", median_px: "median prequential RMS (px)", factor: "drift factor",
    prequential_rms_series: "prequential RMS series (px)",
  };
  // Budget-card wording follows the stop rule: "within target" only below the stop threshold (R11's P < stop_p_exceed).
  function budgetAssessment(p, stopP) {
    if (!isNum(p) || !isNum(stopP)) return null;
    if (p < stopP) return "within target";
    if (p < 0.5) return `uncertain (P is above the stop threshold ${stopP}, so the target is not confirmed)`;
    return "likely above target — recommend recapture";
  }

  // ---------------------------------------------------------------- session controller
  class Controller {
    constructor(prefix, mode) {
      this.p = prefix; this.mode = mode;
      this.reset();
    }
    reset() {
      this.state = null; this.session = null; this.resp = null; this.traceSeen = new Set();
      this.choose = null; this.busy = false; this.auto = false; this.terminal = false; this.history = [];
      this.tau = 0.5; this.budget = 30; this.stopP = 0.1; this.minSwitch = null; this.lastPerception = null; this.lastUnc = null; this.lastCov = null;
      this.lastCal = null; this.ghost = null; this.pending = null; this.size = [1280, 720];
      $(this.p + "-trace").textContent = "";
      $(this.p + "-candidates").replaceChildren(h("p", { class: "muted", text: "Ranked once the first calibration exists." }));
      $(this.p + "-approval").hidden = true;
      $(this.p + "-instruction").textContent = "";
      $(this.p + "-metrics").textContent = "";
    }
    el(id) { return $(this.p + "-" + id); }
    addTrace(lines) {
      const ol = this.el("trace");
      for (const ln of lines || []) {
        if (!ln || this.traceSeen.has(ln)) continue;
        this.traceSeen.add(ln);
        ol.append(h("li", { class: traceClass(ln), text: ln }));
      }
      ol.scrollTop = ol.scrollHeight;
    }
    sys(msg) { this.addTrace([`— ${msg}`]); }

    startFrom(s) {
      this.reset();
      this.state = s.state; this.session = s.session; this.tau = s.thresholds.tau_px; this.budget = s.thresholds.budget_frames;
      if (isNum(s.thresholds.stop_p_exceed)) this.stopP = s.thresholds.stop_p_exceed;
      this.minSwitch = isNum(s.thresholds.model_switch_min_views) ? s.thresholds.model_switch_min_views : null;
      this.size = s.image_size;
      const what = s.preset ? `${s.preset.label} (${s.preset.family}), ${s.actuator}, seed ${s.seed}` : `webcam ${s.image_size.join("×")}`;
      this.sys(`session ${s.session_id} started: ${what}; budget ${s.thresholds.budget_frames} frames, τ ${s.thresholds.tau_px} px`);
      this.renderMetrics(null, null);
      this.setInstruction(this.mode === "sim" ? "Session ready. Press Step, or Auto-run to let the agent work through the session." :
        "Session ready. Hold the board in view and press Capture (or turn on Auto-capture).", "");
    }

    apply(r, rtt) {
      this.resp = r;
      if (r.state) this.state = r.state;
      this.session = r.session || this.session;
      this.history = r.history || this.history;
      if (r.perception) this.lastPerception = r.perception;
      if (r.uncertainty) this.lastUnc = r.uncertainty;
      if (r.coverage) this.lastCov = r.coverage;
      if (r.calibration) this.lastCal = r.calibration;
      this.ghost = r.plan && r.plan.target ? r.plan.target.ghost_quad : (r.status === "continue" ? this.ghost : null);
      this.pending = r.approval || null;
      this.terminal = ["exported", "discarded", "ended"].includes(r.status);
      this.addTrace(r.decision && r.decision.trace_all);
      this.renderInstruction(r);
      this.renderMetrics(r, rtt);
      this.renderCandidates(r);
      this.renderApproval(r.approval);
      if (r.export) this.renderExport(r.export);
      if (this.terminal) { this.auto = false; this.sys(`session ${r.status}`); }
    }

    setInstruction(text, cls, small) {
      const box = this.el("instruction");
      box.className = "instruction" + (cls ? " " + cls : "");
      box.replaceChildren(text || "");
      if (small) box.append(h("small", { text: small }));
    }
    renderInstruction(r) {
      const d = r.decision || {};
      const rule = d.rule || "";
      let cls = "";
      if (/^R[0-3]_/.test(rule)) cls = "reject";
      if (r.approval) cls = "gate";
      if (this.terminal) cls = "done";
      let text = d.instruction || "";
      if (this.terminal) text = { exported: "Calibration exported. Session complete.", discarded: "Session discarded.", ended: "Session ended." }[r.status] || text;
      else if (r.approval) text = "Approval needed: " + this.cardView(r.approval).question;
      const notes = (d.notes || []).filter(Boolean).join(" · ");
      this.setInstruction(text, cls, notes);
    }
    renderMetrics(r, rtt) {
      const s = this.session || {};
      const cal = this.lastCal, unc = this.lastUnc;
      const items = [
        ["step", r ? r.step : 0],
        ["views", `${s.views ?? 0} / ${s.budget ?? this.budget}`],
        ["attempts", `${s.attempts ?? 0} / ${s.max_attempts ?? 60}`],
        ["model", (s.model && s.model.name ? s.model.name : "k5") + (s.model && s.model.pinned ? " (pinned)" : "")],
        ["RMS px", cal ? fmt(cal.rms_px, 3) : "–"],
        ["U95 px", unc ? fmt(unc.U95_px, 3) : "–"],
        ["P(p95 > τ)", unc ? (isNum(unc.p_exceed) ? fmt(unc.p_exceed, 2) : "uncal.") : "–"],
        ["status", (r ? r.status : (s.status || "ready")).replace(/_/g, " ")],
      ];
      if (r && r.latency_ms) items.push(["server ms", Math.round(r.latency_ms.total)]);
      if (isNum(rtt)) items.push(["round trip ms", Math.round(rtt)]);
      if (s.flags && s.flags.length) items.push(["flags", s.flags.join(", ")]);
      this.el("metrics").replaceChildren(...items.map(([k, v]) => h("div", null, h("dt", { text: k }), h("dd", { text: String(v) }))));
    }
    renderCandidates(r) {
      const box = this.el("candidates");
      const plan = r.plan;
      if (!plan || !plan.candidates || !plan.candidates.length) {
        if (plan && plan.target) {
          box.replaceChildren(h("p", { class: "muted" }, `Opening pose: ${plan.target.desc || ""}. Candidates are ranked once 3 views give a first calibration.`));
        } else if (!r.approval) {
          box.replaceChildren(h("p", { class: "muted", text: this.terminal ? "Session finished." : "No plan this step." }));
        }
        return;
      }
      this.choose = null;
      const rows = plan.candidates.map((c) => h("tr", {
        class: c.rank === 1 ? "top" : null, "data-rank": c.rank, title: "Use this candidate for the next capture",
        onclick: (ev) => {
          const rank = Number(ev.currentTarget.getAttribute("data-rank"));
          this.choose = this.choose === rank ? null : rank;
          for (const tr of box.querySelectorAll("tbody tr")) tr.classList.toggle("chosen", Number(tr.getAttribute("data-rank")) === this.choose);
          box.querySelector(".choice").textContent = this.choose ? `Next capture uses candidate ${this.choose} (your override, logged).` : "Next capture uses the top candidate.";
        },
      }, h("td", { text: c.rank }), h("td", { text: c.desc || "" }),
      h("td", { class: "num", text: fmt(c.gain, 3) }), h("td", { class: "num", text: fmt(c.pred_U95_after, 3) }),
      h("td", { class: "num", text: fmt(c.effort, 2) }), h("td", { class: "num", text: fmt(c.score, 3) })));
      const table = h("table", { class: "cands" },
        h("thead", null, h("tr", null, h("th", { text: "#" }), h("th", { text: "pose" }), h("th", { text: "pred ΔU95" }),
          h("th", { text: "U95 after" }), h("th", { text: "effort" }), h("th", { text: "score" }))),
        h("tbody", null, rows));
      const kids = [table, h("p", { class: "realised choice", text: "Next capture uses the top candidate." })];
      const lr = plan.last_realised;
      if (lr && (isNum(lr.pred_gain) || isNum(lr.realised_gain))) {
        const p = h("p", { class: "realised" }, "Last view: predicted ΔU95 ", h("b", { text: fmt(lr.pred_gain, 3) }),
          " px, realised ", h("b", { text: fmt(lr.realised_gain, 3) }), " px");
        if (isNum(lr.true_gain)) p.append(", true Δp95 ", h("b", { text: fmt(lr.true_gain, 3) }), " px");
        kids.push(p);
      }
      box.replaceChildren(...kids);
    }
    cardView(card) {
      const ev = { ...(card.evidence || {}) };
      let question = card.question || "";
      const warnings = [];
      if (card.kind === "accept_with_warnings") {
        const a = budgetAssessment(ev.p_exceed, this.stopP);
        if (a) {
          if (ev.label && question.includes(ev.label)) question = question.replace(ev.label, a);
          ev.label = a;
        }
        const n = Number(ev.views), pinned = !!(this.session && this.session.model && this.session.model.pinned);
        if (ev.model === "k5" && !pinned && isNum(this.minSwitch) && n >= 3 && n < this.minSwitch) {
          warnings.push(`Lens-model check not run: the test for a lens model that does not fit (R8) needs ${this.minSwitch} views, ` +
            `and this session has ${n}. U95 covers only the uncertainty within the k5 model, so it can understate the true error ` +
            "when k5 does not fit the lens (wide lenses especially).");
        }
      }
      return { question, ev, warnings };
    }
    renderApproval(card) {
      const box = this.el("approval");
      if (!card) { box.hidden = true; box.textContent = ""; return; }
      const view = this.cardView(card);
      const ev = view.ev;
      const dl = h("dl");
      for (const [k, v] of Object.entries(ev)) {
        if (["residual_field", "K", "dist"].includes(k) || v == null) continue;
        let txt;
        if (k === "cv_rms" && typeof v === "object") txt = Object.entries(v).map(([m, x]) => `${m} ${fmt(x, 3)}`).join(" vs ");
        else if (Array.isArray(v)) txt = v.map((x) => (isNum(x) ? fmt(x, 3) : String(x))).join(", ") || "none";
        else if (isNum(v)) txt = Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(2) : fmt(v, 3).replace(/\.?0+$/, "");
        else txt = String(v);
        dl.append(h("dt", { text: EVIDENCE_LABELS[k] || k }), h("dd", { text: txt }));
      }
      if (ev.K) dl.append(h("dt", { text: "fx, fy, cx, cy" }), h("dd", { text: [ev.K[0][0], ev.K[1][1], ev.K[0][2], ev.K[1][2]].map((x) => fmt(x, 1)).join(", ") }));
      const kids = [h("h2", { text: `Approval needed · ${card.kind.replace(/_/g, " ")} (${card.rule || ""})` }),
        h("p", { class: "q", text: view.question })];
      for (const w of view.warnings) kids.push(h("p", { class: "warn", text: w }));
      if (ev.residual_field && ev.residual_field.vectors) {
        const cv = h("canvas", { "aria-label": "Mean residual vector per image cell" });
        kids.push(cv);
        const mx = drawField(cv, ev.residual_field);
        kids.push(h("p", { class: "cap", text: `Mean residual per image cell (${ev.residual_field.grid.join("×")}); arrows scaled to the 90th percentile, longest ${fmt(mx, 2)} px.` }));
      }
      kids.push(dl);
      const reason = h("textarea", { placeholder: "Reason (recorded with the decision in the trace)", maxlength: 500, "aria-label": "Reason" });
      kids.push(reason);
      const opts = h("div", { class: "opts" });
      for (const o of card.options || []) {
        opts.append(h("button", {
          type: "button", class: o.decision === "approve" || o.decision === "acknowledge" ? "approve" : null, text: o.label,
          onclick: () => this.decide(card.id, o.decision, reason.value),
        }));
      }
      kids.push(opts);
      box.replaceChildren(...kids);
      box.hidden = false;
    }
    renderExport(exp) {
      const box = this.el("export");
      const kids = [];
      const names = { yaml: "OpenCV YAML", png: "Report card PNG", json: "JSON record" };
      if (exp.store === "inline" && exp.files) {
        for (const [k, f] of Object.entries(exp.files)) {
          const bin = Uint8Array.from(atob(f.b64), (c) => c.charCodeAt(0));
          const url = URL.createObjectURL(new Blob([bin], { type: f.content_type }));
          kids.push(h("a", { class: "dl", href: url, download: f.name, text: names[k] || k }));
        }
        kids.push(h("p", { class: "muted", text: exp.note || "Files generated in this response." }));
      } else {
        for (const k of ["yaml", "png", "json"]) {
          const url = exp[k + "_url"];
          if (url) kids.push(h("a", { class: "dl", href: url, download: "", rel: "noopener", text: names[k] }));
        }
        kids.push(h("p", { class: "muted", text: exp.expires_s ? `Download links expire in ${Math.round(exp.expires_s / 60)} minutes.` : "Stored by the local server." }));
      }
      box.replaceChildren(...kids);
    }

    async call(path, body) {
      if (this.busy) return null;
      this.busy = true; updateButtons();
      const t0 = performance.now();
      try {
        const r = await api(path, body);
        this.lastError = null;
        notice("error", null);
        this.apply(r, performance.now() - t0);
        return r;
      } catch (e) {
        this.lastError = e.code || null;
        if (e.code !== "BUSY") this.auto = false; // auto-run waits out a busy service (see loop)
        const msg = e.code === "STATE_EXPIRED" ? "This session has expired; start a new one." : e.message;
        notice("error", msg, "error");
        this.sys(`request failed: ${msg}`);
        return null;
      } finally {
        this.busy = false; updateButtons();
        this.afterCall();
      }
    }
    afterCall() { }
    async decide(id, decision, reason) {
      const r = await this.call("/api/approve", { state: this.state, approval_id: id, decision, reason: reason || "" });
      if (r && this.auto && !this.pending && !this.terminal) this.loop();
    }
  }

  // ---------------------------------------------------------------- simulator tab
  class SimController extends Controller {
    constructor() {
      super("sim", "sim");
      this.canvas = $("sim-canvas"); this.frame = null; this.frameScale = 0.75;
      for (const id of ["sim-ov-corners", "sim-ov-ghost", "sim-ov-coverage", "sim-ov-unc"]) $(id).addEventListener("change", () => this.draw());
    }
    reset() {
      super.reset();
      this.frame = null;
      if (this.canvas) this.draw();
      $("sim-export").replaceChildren(h("p", { class: "muted", text: "Export is offered when the agent stops (and always needs your approval)." }));
      $("sim-consequence").replaceChildren(h("p", { class: "muted", text: "Appears after 3 accepted views." }));
      if ($("sim-race")) drawRace($("sim-race"), [], this.tau || 0.5, this.budget || 30);
    }
    apply(r, rtt) {
      super.apply(r, rtt);
      if (r.frame && r.frame.jpeg_b64) {
        const img = new Image();
        img.onload = () => { this.frame = img; this.draw(); };
        img.src = "data:image/jpeg;base64," + r.frame.jpeg_b64;
      } else this.draw();
      drawRace($("sim-race"), this.history, this.tau, this.budget);
      this.renderConsequence(r);
    }
    draw() {
      const c = this.canvas, ctx = c.getContext("2d"), W = c.width, H = c.height;
      ctx.fillStyle = "#0d0f0e"; ctx.fillRect(0, 0, W, H);
      $("sim-empty").hidden = !!this.frame || !!this.state;
      if (this.frame) ctx.drawImage(this.frame, 0, 0, W, H);
      const legend = $("sim-legend");
      legend.hidden = true;
      drawOverlays(ctx, W, H, {
        size: this.size, corners: this.lastPerception && this.lastPerception.corners, ghost: this.ghost,
        unc: this.lastUnc, coverage: this.lastCov,
      }, {
        corners: $("sim-ov-corners").checked, ghost: $("sim-ov-ghost").checked && !this.terminal,
        coverage: $("sim-ov-coverage").checked, unc: $("sim-ov-unc").checked,
        legend: (lo, hi) => { legend.hidden = false; $("sim-legend-lo").textContent = fmt(lo, 2); $("sim-legend-hi").textContent = fmt(hi, 2) + " (log)"; },
      });
    }
    renderConsequence(r) {
      const t = r.truth, cal = this.lastCal;
      if (!t || !cal || !isNum(t.field_p95_px)) return;
      const fx = cal.K[0][0];
      const mm = (px) => (px * 1000) / fx;
      const kids = [
        h("p", { class: "big" }, `Field p95 ${fmt(t.field_p95_px, 2)} px ≈ ${fmt(mm(t.field_p95_px), 2)} mm`),
        h("p", { text: "lateral error at the image edge at 1 m (true error from the simulator; Y = X·1000 / f̂x, f̂x = " + fmt(fx, 1) + " px)" }),
      ];
      const b = r.baseline && r.baseline.truth;
      if (b && isNum(b.field_p95_px)) {
        kids.push(h("p", { class: "vs" }, `Random baseline at ${r.baseline.views} frames: ${fmt(b.field_p95_px, 2)} px ≈ ${fmt(mm(b.field_p95_px), 2)} mm`));
      }
      if (isNum(t.fx_rel_err_pct)) kids.push(h("p", { class: "vs", text: `Focal length error ${fmt(t.fx_rel_err_pct, 3)} %` }));
      $("sim-consequence").replaceChildren(...kids);
    }
    async start() {
      const preset = $("sim-preset").value;
      const actuator = (document.querySelector('input[name="sim-actuator"]:checked') || {}).value;
      const seed = Math.max(0, Math.floor(Number($("sim-seed").value) || 0));
      const budget = Math.min(30, Math.max(3, Math.floor(Number($("sim-budget").value) || 30)));
      this.auto = false;
      if (this.busy) return;
      this.busy = true; updateButtons();
      const prep = setTimeout(() => this.setInstruction("Preparing the simulated camera (lens model and distortion maps)…", ""), 400);
      try {
        const s = await apiPatient("/api/session", { mode: "sim", preset, actuator, seed, budget });
        clearTimeout(prep);
        notice("error", null);
        this.startFrom(s);
        this.draw();
      } catch (e) {
        clearTimeout(prep);
        notice("error", e.message, "error");
      } finally { this.busy = false; updateButtons(); }
    }
    async step() {
      if (!this.state || this.terminal) return null;
      const body = { state: this.state, choose_candidate: this.choose };
      return this.call("/api/step", body);
    }
    async loop() {
      this.auto = true; updateButtons();
      let busyWaits = 0;
      while (this.auto && this.state && !this.terminal && !this.pending) {
        const r = await this.step();
        if (!r) {
          if (this.lastError !== "BUSY" || ++busyWaits > 6) { this.auto = false; break; }
          notice("auto", `Service busy: auto-run resumes in 5 s (wait ${busyWaits}/6)…`);
          await sleep(5000);
          notice("auto", null);
          continue;
        }
        busyWaits = 0;
        await sleep(150);
      }
      if (this.pending && this.auto) notice("auto", "Auto-run is waiting for your decision on the approval card.");
      updateButtons();
    }
    afterCall() { if (!this.pending) notice("auto", null); }
  }

  // ---------------------------------------------------------------- webcam tab
  const MAX_W = 1280, MAX_H = 720, JPEG_CAP = 300 * 1024;
  class CamController extends Controller {
    constructor() {
      super("cam", "webcam");
      this.video = $("cam-video"); this.overlay = $("cam-overlay"); this.stream = null; this.timer = null;
      this.cap = [1280, 720];
    }
    reset() {
      super.reset();
      if (this.overlay) this.drawOverlay();
      $("cam-export").replaceChildren(h("p", { class: "muted", text: "Offered when the agent stops; needs your approval. The stop probability is transferred from simulation." }));
    }
    async open() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        notice("error", "This browser does not expose a camera here (getUserMedia needs HTTPS or localhost).", "error");
        return;
      }
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" } });
      } catch (e) {
        notice("error", `Camera unavailable: ${e.message || e.name}`, "error");
        return;
      }
      notice("error", null);
      this.video.srcObject = this.stream;
      await this.video.play().catch(() => {});
      await new Promise((res) => (this.video.videoWidth ? res() : this.video.addEventListener("loadedmetadata", res, { once: true })));
      const vw = this.video.videoWidth, vh = this.video.videoHeight;
      const k = Math.min(1, MAX_W / vw, MAX_H / vh);
      this.cap = [Math.max(2, Math.round((vw * k) / 2) * 2), Math.max(2, Math.round((vh * k) / 2) * 2)];
      $("cam-stage").style.aspectRatio = `${vw} / ${vh}`;
      this.overlay.width = this.cap[0]; this.overlay.height = this.cap[1];
      $("cam-empty").hidden = true;
      this.sysCam(`camera ${vw}×${vh}; frames sent at ${this.cap.join("×")}`);
      updateButtons();
    }
    sysCam(msg) { this.addTrace([`— ${msg}`]); }
    close() {
      this.setAuto(false);
      if (this.stream) for (const t of this.stream.getTracks()) t.stop();
      this.stream = null; this.video.srcObject = null;
      $("cam-empty").hidden = false;
      updateButtons();
    }
    async start() {
      if (this.busy || !this.stream) return;
      this.busy = true; updateButtons();
      try {
        const s = await apiPatient("/api/session", { mode: "webcam", actuator: "human", seed: Math.floor(Math.random() * 1e6), image_size: this.cap });
        notice("error", null);
        this.startFrom(s);
        this.drawOverlay();
      } catch (e) { notice("error", e.message, "error"); } finally { this.busy = false; updateButtons(); }
    }
    async grab() {
      const [w, h_] = this.size;
      const c = document.createElement("canvas");
      c.width = w; c.height = h_;
      c.getContext("2d").drawImage(this.video, 0, 0, w, h_);
      for (const q of [0.85, 0.7, 0.55, 0.4]) {
        const blob = await new Promise((res) => c.toBlob(res, "image/jpeg", q));
        if (blob && blob.size <= JPEG_CAP) {
          const buf = new Uint8Array(await blob.arrayBuffer());
          let s = "";
          for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
          return btoa(s);
        }
      }
      throw new Error("could not encode the frame under 300 KB");
    }
    async capture() {
      if (!this.state || this.busy || this.terminal || this.pending || !this.stream) return null;
      let b64;
      try { b64 = await this.grab(); } catch (e) { notice("error", e.message, "error"); return null; }
      return this.call("/api/frame", { state: this.state, jpeg_b64: b64, choose_candidate: this.choose });
    }
    apply(r, rtt) { super.apply(r, rtt); this.drawOverlay(); }
    drawOverlay() {
      const c = this.overlay, ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      drawOverlays(ctx, c.width, c.height, { size: this.size, corners: this.lastPerception && this.lastPerception.corners, ghost: this.ghost },
        { corners: true, ghost: !this.terminal });
    }
    setAuto(on) {
      $("cam-auto").checked = !!on;
      if (this.timer) { clearInterval(this.timer); this.timer = null; }
      if (on) this.timer = setInterval(() => { if (!this.busy && !this.pending) this.capture(); }, 1500);
    }
    loop() { }
  }

  const sim = new SimController();
  const cam = new CamController();

  function updateButtons() {
    const running = sim.auto && !sim.pending;
    $("sim-start").disabled = sim.busy;
    $("sim-step").disabled = !sim.state || sim.busy || sim.terminal || running || !!sim.pending;
    $("sim-auto").disabled = !sim.state || sim.terminal || running || !!sim.pending;
    $("sim-pause").disabled = !sim.auto;
    $("sim-auto").classList.toggle("active", running);
    $("cam-open").disabled = !!cam.stream;
    $("cam-close").disabled = !cam.stream;
    $("cam-start").disabled = !cam.stream || cam.busy;
    $("cam-capture").disabled = !cam.stream || !cam.state || cam.busy || cam.terminal || !!cam.pending;
    $("cam-auto").disabled = !cam.stream || !cam.state || cam.terminal;
  }

  // ---------------------------------------------------------------- evidence tab
  const FIG_CAPTIONS = {
    E1: "Median true field p95 vs accepted frames per policy (IQR bands), robot | human",
    E2: "Paired median differences with 95% CIs (PE1–PE8)",
    E3: "Frames-to-1 px survival curves",
    E4: "Ablations",
    E5: "Coverage of the first-order uncertainty map, and the histogram of r = true p95 / U95",
    E6: "Risk–coverage: false-stop rate vs mean frames",
    E7: "METRIC D455: held-out RMS vs budget",
    E8: "METRIC KinectV2: true field error vs budget",
    E9: "Millimetre consequence table",
    E10: "Predicted vs realised gain",
    E11: "Latency and cost",
    E12: "Failure gallery",
  };
  // One figure at a time, with the API back-off: a burst of image requests would exceed the stage
  // throttle (2 req/s, burst 5) and the account concurrency cap, and come back as 429 or 503.
  function loadImagesInTurn(imgs) {
    let i = 0;
    const next = () => {
      if (i >= imgs.length) return;
      const img = imgs[i++], url = img.dataset.src;
      let tries = 0;
      img.onload = () => setTimeout(next, 250);
      img.onerror = () => {
        if (tries >= BACKOFF_MS.length) { next(); return; }
        setTimeout(() => { img.src = url + "?retry=" + (tries + 1); tries++; }, BACKOFF_MS[tries] * (0.5 + Math.random()));
      };
      img.src = url;
    };
    next();
  }
  let evidenceLoaded = false;
  async function loadEvidence() {
    if (evidenceLoaded) return;
    const box = $("evidence-body");
    try {
      const d = await api("/api/evidence");
      evidenceLoaded = true;
      const kids = [];
      if (!d.available) kids.push(h("p", { class: "muted", text: "The benchmark results are baked into the image at build time; this build has none yet." }));
      const scalars = Object.entries(d.headline && typeof d.headline === "object" ? d.headline : d)
        .filter(([k, v]) => !k.startsWith("_") && k !== "available" && (typeof v === "number" || typeof v === "string"));
      if (scalars.length) {
        kids.push(h("table", { class: "kv" }, h("tbody", null, scalars.slice(0, 24).map(([k, v]) => h("tr", null, h("td", { text: k }), h("td", { text: isNum(v) ? String(+v.toFixed(4)) : v }))))));
      }
      const caps = (d.figures && typeof d.figures === "object" && !Array.isArray(d.figures)) ? d.figures : {};
      const figs = d._figures || [];
      if (figs.length) {
        kids.push(h("div", { class: "figs" }, figs.map((name) => {
          const id = (/^(E\d+)/.exec(name) || [])[1];
          const cap = caps[name] || (id && FIG_CAPTIONS[id]) || name.replace(/\.png$/, "");
          return h("figure", null, h("img", { "data-src": "/evidence/" + encodeURIComponent(name), alt: cap }),
            h("figcaption", null, h("b", { text: id ? id + " " : "" }), cap));
        })));
      }
      const gal = Array.isArray(d.failure_gallery) ? d.failure_gallery : [];
      if (gal.length) {
        kids.push(h("h2", { text: "Failure gallery" }));
        kids.push(h("div", { class: "figs" }, gal.map((g) => h("figure", null,
          g.image ? h("img", { "data-src": "/evidence/" + encodeURIComponent(g.image), alt: g.case || g.id || "" }) : null,
          h("figcaption", null, h("b", { text: (g.id || "") + " " }), g.case || "",
            g.behaviour || g.agent ? h("div", { text: g.behaviour || g.agent }) : null,
            g.numbers ? h("div", { class: "muted", text: typeof g.numbers === "string" ? g.numbers : JSON.stringify(g.numbers) }) : null)))));
      }
      box.replaceChildren(...(kids.length ? kids : [h("p", { class: "muted", text: "No evidence in this build." })]));
      loadImagesInTurn([...box.querySelectorAll("img[data-src]")]);
    } catch (e) {
      box.replaceChildren(h("p", { class: "muted", text: "Could not load the evidence: " + e.message }));
    }
  }

  // ---------------------------------------------------------------- tabs
  const TABS = ["sim", "cam", "evidence", "about"];
  function showTab(name) {
    if (!TABS.includes(name)) name = "sim";
    for (const t of TABS) {
      $("tab-" + t).setAttribute("aria-selected", String(t === name));
      $("panel-" + t).hidden = t !== name;
    }
    if (name === "evidence") loadEvidence();
    if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  }
  for (const t of TABS) $("tab-" + t).addEventListener("click", () => showTab(t));

  // ---------------------------------------------------------------- wiring
  let presetList = [];
  function syncActuator() {
    const p = presetList.find((x) => x.id === $("sim-preset").value);
    if (!p) return;
    for (const r of document.querySelectorAll('input[name="sim-actuator"]')) r.checked = r.value === p.actuator;
  }
  $("sim-preset").addEventListener("change", syncActuator);
  $("sim-start").addEventListener("click", () => sim.start());
  $("sim-step").addEventListener("click", () => sim.step());
  $("sim-auto").addEventListener("click", () => sim.loop());
  $("sim-pause").addEventListener("click", () => { sim.auto = false; notice("auto", null); updateButtons(); });
  $("cam-open").addEventListener("click", () => cam.open());
  $("cam-close").addEventListener("click", () => cam.close());
  $("cam-start").addEventListener("click", () => cam.start());
  $("cam-capture").addEventListener("click", () => cam.capture());
  $("cam-auto").addEventListener("change", (e) => cam.setAuto(e.target.checked));

  async function boot() {
    showTab((location.hash || "#sim").slice(1));
    sim.draw();
    drawRace($("sim-race"), [], 0.5, 30);
    updateButtons();
    try {
      const cfg = await api("/api/presets");
      presetList = cfg.presets;
      $("sim-preset").replaceChildren(...cfg.presets.map((p) => h("option", { value: p.id, text: `${p.label}${p.failure_demo ? " (failure demo)" : ""}` })));
      $("sim-preset").value = cfg.default;
      $("sim-budget").max = cfg.thresholds.budget_frames;
      syncActuator();
      $("foot-status").textContent = `build ${cfg.version} · OpenCV ${cfg.opencv}`;
      $("about-version").textContent = `Build ${cfg.version}, OpenCV ${cfg.opencv}.`;
    } catch (e) {
      notice("error", "Could not load the presets: " + e.message, "error");
    }
  }
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));
  window.nextview = { sim, cam }; // for scripted walkthroughs
  boot();
})();
