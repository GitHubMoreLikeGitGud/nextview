# NextView benchmark pre-registration

Written 2026-10-23 (SGT), before the synthetic TEST split was run. Nothing in this file depends on
TEST data. Only numbers produced by the scripts named here are reported, including null and
negative results.

(Note added at publication: "spec x.y" refers to the author's unpublished design specification.)

## 1. What is frozen

| Item | Value |
|---|---|
| Repository base commit | `7d6e45e9466a55b9b45874f9a74dd6cdf41ab76b` (working tree not yet committed; the commit that adds this file records the frozen tree) |
| Source tree (sha256 over every `nextview/**/*.py`, path:hash lines) | `fec8bf19e559184c42d185452a9fce1d58d094e94c5a8fbcdf870a4a7692266b` |
| `config/thresholds.json` sha256 | `d59b2a94f80c81e6a164ef841717cf3ab20e92097ff53b2d910da55391dd5bc0` |
| `config/stop_calibration.json` sha256 | `2aadd414b7bf79a6562069044f4317cc4ed579e53a4c20ab17c5342ae55149be` |
| `config/presets.json` sha256 | `2bc75500beb6d610e6fd31ee86ffe9ef5b7c862a0ad94d74be6e0f66dcddac72` |
| Analysis code `nextview/bench/summarize.py` | `1568682f08655fff21ab0600a3ee05620f826c3bc2a86ea5de05242fb71b863a` |
| `nextview/bench/analysis.py` / `stats.py` | `fc43bdb1…0d428` / `138a29a0…2840` |
| Policies `nextview/bench/policies.py` / `common.py` | `36c72b10…60df` / `658c7a61…f032` |
| Agent `nextview/agent/loop.py` / `policy.py`, planner `nextview/tools/nbv.py` | `8ecd01e3…2da1` / `8aa42dab…a24d` / `71bf7bdc…f08015` |
| DEV outputs | `results/synth_dev/runs.csv` `386b4785…683a`, `results/synth_dev/stop_fit.json` `9d7dc86d…107f`, `results/dev/blur_curve.json` `5a164b0d…7736` |

The TEST run is launched from a byte-for-byte snapshot of this source tree and config; its
`results/synth_test/meta.json` records the same hashes (`code.source_tree_sha256`,
`thresholds_sha256`, `stop_calibration_sha256`). The agent, tool and simulator code is identical
to the code that produced the DEV split; between DEV and TEST only the benchmark package changed
(analysis files added, and `policies.py` now also records the loop's predicted vs realised gain).

## 2. Threshold freeze (DEV split only)

- **tau = 0.75 px** (spec default 0.5 px). Spec 2.6 rule: with the DEV-fitted stop calibration,
  the median frames-to-stop of A on the normal family under the robot actuator at tau = 0.5 px was
  31 (no stop within 30 frames in most runs) > 25, so tau = 0.75 px. Recorded in
  `config/thresholds.json` `_meta.tau_reason`.
- **max_edge_rise_px = 3.0 px** (spec default, unchanged). DEV blur curve
  (`python -m nextview.bench.blur_curve`, 40 DEV cameras x 6 poses x 10 blur levels, 2,207
  measured frames): the median corner error is 0.052 px unblurred and stays at or below 0.064 px
  in every 0.5 px rise bin up to 9 px. The spec 2.9 criterion (median error 2x the sharp value)
  is never met, so the rule gives no threshold and the default is kept. Blur in this simulator
  mainly costs detections (motion L = 10 px: 146/240 frames with >= 12 corners), not corner
  accuracy. Recorded in `_meta.blur_reason`.
- Every other threshold is the spec 2.9 value.

## 3. Stop calibration (spec 2.6)

`python -m nextview.bench.fit_stop --runs results/synth_dev/runs.csv --out config/stop_calibration.json`

r = true field p95 / U95 at every accepted-frame step of policy A with >= 6 views on DEV, per
actuator, pooled over the four families; 101 empirical quantiles. DEV sample sizes: robot 982,
human 929 steps. DEV median r: robot 1.35, human 1.30 (the first-order U95 is optimistic by about
a third; 71 % of steps have r > 1). Only A's steps are fitted because P_exceed is consulted only on
A-family trajectories; B0's r is reported in `stop_fit.json` for comparison.

## 4. Splits (spec 11.2)

- **DEV** (done): cameras 0-39 (10 per family: 0-9 normal, 10-19 wide, 20-29 tele, 30-39 webcam),
  policy seed 0, both actuators, A and B0. Used only for the freeze above.
- **TEST**: cameras 1000-1099 (1000-1024 normal, 1025-1049 wide, 1050-1074 tele, 1075-1099
  webcam) x policy seeds {0, 1, 2} x actuators {robot, human} x 11 policies = 6,600 runs. Every
  run goes to 30 accepted frames or 60 attempts, recording metrics after each accepted frame.
- **Failure set** (abstention, descriptive): fisheye130 (robot), webcam with autofocus drift
  (human), normal family with a sagged board 2-4 mm (corner level, robot, RO denied); 20 cameras
  each, A only.
- Nuisances in the synthetic splits: each family's noise, JPEG and near-focus defocus, occluders
  (p = 0.15) and post-detection corner corruption (p = 0.05) for every family; glare on for the
  glossy webcam family only (spec 6.3).

## 5. Policies (spec 11.3; details in the `nextview/bench/policies.py` docstring)

| ID | Implementation |
|---|---|
| A | the agent loop (`run_step` / `apply_approval`), approve_all for model changes |
| A-gate | A with rejection only on detection failure (< 6 corners or degenerate span) |
| A-outlier | A without R7 |
| A-model | A with model-change cards denied (deny_all) |
| A-nbv | A with the NBV ranking replaced by a random order over the same feasible candidates |
| A-stop | A's own trajectory read at a fixed budget = median in-loop stop of A (no extra run) |
| B0 | 3 shared opening poses, then the simulator's random feasible poses; accept >= 6 corners; k5 |
| B1 | 12-pose textbook template (own order), jittered repeats beyond 12; GT-framed |
| B2 | OpenCV `apps/interactive-calibration` (tag 5.0.0) rules over the B0 random pose stream, with the app's own `calibrateCamera` call (flags auto-tuning, 30 iterations); each rule cites its source line |
| B2-k5 | descriptive only: B2's acceptance and stop rules with the bench's k5 solver |
| B3 | coverage-greedy over the NBV lattice sample under the current estimate |
| B4 | dense 45-pose grid (5 x 3 centres x {frontal, yaw 30, pitch 30}, s = 0.5), farthest-point order; GT-framed |

All policies except B1 and B4 share the 3 seeded opening poses per (camera, seed). Captures are
indexed by attempt number, so all policies see the same actuator error and nuisances at the same
attempt (common random numbers). The bench gives the agent no wall-clock budget (deadline and
`nbv_time_budget_s` 1e5 s) so runs are deterministic; on Lambda the budgets may bind but change no
rule. After A's first R11 stop the stop point is recorded, the export is denied and the stop rule
is switched off, so A's trajectory continues to 30 frames exactly as without a stop (checked: the
offline replay of R11 reproduces the in-loop stop).

## 6. Pre-registered primary endpoints (Holm across the 8 tests)

All differences are A minus baseline (negative = A better).

| ID | Endpoint | Comparison | Actuator |
|---|---|---|---|
| PE1 / PE2 | true field p95 (px, 32 x 18 grid) after 10 accepted frames | A vs B0 | robot / human |
| PE3 / PE4 | accepted frames until the true field p95 is first < 1 px (not reached in 30 = 31) | A vs B0 | robot / human |
| PE5 / PE6 | true field p95 at n = 10 | A vs B2 | robot / human |
| PE7 / PE8 | true field p95 at n = 10 | A vs B1 | robot / human |

**Statistics.** Per camera, the median over the 3 policy seeds (100 paired values per endpoint).
`scipy.stats.wilcoxon`, two-sided, zero differences dropped. 95 % percentile bootstrap CI of the
median paired difference, 10,000 resamples of cameras, seed 0. Holm-Bonferroni across the 8
p-values, alpha 0.05. A run that ends before n frames (attempt cap, abstention) contributes its
last estimate; a camera with no estimate in any seed is dropped and counted. Robot and human are
always reported separately. `python -m nextview.bench.summarize --runs results/synth_test`.

## 7. Descriptive analyses (no hypothesis tests)

Per family and per ablation comparisons (paired medians with 2,000-resample CIs), E1 curves
(median and IQR of field p95 vs frames), Kaplan-Meier frames-to-1 px, attempts (human effort),
stop quality of A at tau = 0.75 px and P_exceed < 0.10 (frames at stop, false-stop rate = true
p95 at stop > 2 tau, wasted frames, risk-coverage for `stop_p_exceed` in {0.02, 0.05, 0.1, 0.2,
0.3, 0.5}), A-stop, B2's own stop, uncertainty honesty (coverage at chi-square(2) quantiles for
q in {0.5, 0.8, 0.9, 0.95}, r = true p95 / U95, binned P_exceed vs observed exceedance), outlier
recall of R7 on corrupted views, held-out reprojection RMS on 20 fixed noise-free GT views per
camera (pose refit), predicted vs realised gain, and the mm consequence table (200 segment pairs
of 100 mm in the outer 30 % of the image at Z = 0.5 and 1.0 m).

**Stated hypotheses, not results.** Gains concentrate in `wide` (model selection) and `webcam`
(gate); the NBV share is whatever A-nbv shows; gains shrink under the human actuator.

## 8. Real data (after TEST, descriptive)

METRIC (Zenodo 10.5281/zenodo.7976757, CC BY 4.0): D455 real (factory intrinsics as a
reference, not GT) and KinectV2 synthetic (GT intrinsics, principal point converted from the
files' 1-based convention by -1 px). Per camera, seed 0: 50 held-out frames, the rest the pool.
Budgets {6, 8, 10, 15, 20, 30}; P-rand (100 draws), P-first, P-tmpl, P-cov, P-agent; held-out
reprojection RMS and the field difference against the reference. Detections are in square units
of the 5 x 4 METRIC board.

---

## Addendum after the TEST run (no change to anything above)

- The TEST run used the frozen snapshot: `results/synth_test/meta.json` records source tree
  `fec8bf19…266b`, thresholds `d59b2a94…bc0`, stop calibration `2aadd414…49be` and
  `summarize.py` `1568682f…863a`, as listed in section 1. 6,600 runs, 0 errors.
- Line-ending-independent identifiers (sha256 over `path:sha256(file with CRLF -> LF)` lines,
  59 files): TEST source tree `6f56014c13b79b50ade6c30c63863ca93c302434dd96e2e915161fc2a8cedce9`.
  Git normalises line endings on commit, so the raw-byte hashes in section 1 hold for the working
  tree that ran, the normalised one for a checkout.
- `summary.json` was produced by the pre-registered `summarize.py` and re-produced byte for byte
  from the full output files after they were moved (below).
- Storage: the full TEST `runs.csv` (46 MB) and `sessions.jsonl` exceed the committed-results
  budget. The repository keeps `runs_core.csv.gz` (every row and metric, 3-4 significant
  digits, no per-row K and distortion), `estimates.csv.gz` (full estimates at n = 10, 20, 30) and
  `sessions.jsonl.gz`; the full files are regenerated exactly by
  `python -m nextview.bench.run_synthetic --split test` (deterministic). Scripts written after
  the run (`files.py`, `consequence.py`, `gallery.py`, `plots.py`, `run_pool.py`) only read outputs.
- The offline R11 replay reproduced the in-loop stop in 600 of 600 A runs.

## Addendum 2: corrections after review (post hoc; sections 1-8 unchanged)

Everything in this section was found or done after the TEST run. None of it changes a
pre-registered endpoint, threshold or analysis rule; post-hoc numbers are in
`results/synth_test/posthoc.json` (`python -m nextview.bench.posthoc`), and `summary.json`
remains the output of the unchanged `summarize.py`.

- **Undisclosed pre-flight check.** One minute before the TEST launch, a 132-job check ran on TEST
  data (cameras 1000-1001, all 11 policies, seeds 0-2, both actuators; `results/tmp/check_test`,
  finished 10:28 SGT) and wrote a summary with endpoint values. Its source tree (`fec8bf19…266b`),
  thresholds (`d59b2a94…bc0`) and stop calibration (`2aadd414…49be`) hashes are identical to the
  TEST run's, so nothing was tuned on it, but it should have been reported here.
- **Commit timing.** Spec 11.1 asks for this file to be committed before TEST. It was not: it is
  committed after the TEST run, together with the working tree. That sections 1-8 predate TEST
  rests on the hashes in section 1, which the TEST `meta.json` reproduces, not on a commit time.
- **Erratum to section 5.** After a denied stop the trajectory is not "exactly" as without a stop.
  Re-running TEST camera 1045 (wide, seed 2, robot, A) with the calibrated and an uncalibrated stop
  file gives the same poses, fx, fy, cx, cy and true field p95 on all 30 rows; after the stop at
  n = 14 the extra refit following the deny moves the distortion coefficients and U95 by under
  1e-4 relative and the true field mean by up to 1.3e-3 relative (n = 18). Three other stopped runs
  (cameras 1000, 1010, 1090) and one without a stop (1060) were identical. A's stops never occur
  before n = 14 on TEST, so the n = 10 endpoints are unaffected.
- **Outlier recall (spec 11.5 item 7) corrected; a post-hoc correction to a descriptive metric.**
  The bench counted a corrupted view only if it was still the last view after its step. R7 usually
  drops a corrupted view in the step that accepted it, so most corrupted views were never counted
  and their drops were booked as clean drops. The bookkeeping is fixed in `policies.py`, and a test
  forces a corrupted view. TEST is deterministic, so the 3,000 A-family jobs were re-run from the
  frozen snapshot with only `policies.py` changed (source tree `44c665ea…babb`, kept in
  `results/tmp/frozen_r7fix`). Every row of every job is identical to the TEST run, and every
  session field other than timing and the outlier counts is identical. The corrected counts were
  written into `sessions.jsonl.gz`, and `summary.json` was regenerated with the unchanged
  `summarize.py`; only its `outlier_recall` block changed. Corrected: A/robot 412 corrupted views
  accepted, 412 dropped by R7 (recall 1.00), 62 clean views dropped (precision 0.87); A/human 393
  accepted, 392 dropped and 1 cleared by an R6 restart (recall 0.997), 57 clean views dropped
  (precision 0.87). The uncorrected numbers were 46 of 46 and 41 of 42, with 428 and 408 "clean"
  drops. A-outlier, without R7, accepts 414 (robot) and 392 (human) corrupted views and drops none.
  R7's effect on accuracy, A minus A-outlier, true field p95 at n = 10: the paired camera median is
  0.00 px, 95 % CI [-0.25, 0.00] (robot) and 0.00 [-0.15, 0.00] (human). Run by run, A is better in
  103 of 300 robot runs and worse in 22 (175 within 0.01 px), mean -2.25 px, and better in 87 of 296
  human runs and worse in 21, mean -1.65 px. The pooled run medians are 1.23 vs 2.26 px (robot) and
  1.26 vs 2.29 px (human). R7 leaves most runs unchanged; where it acts it mostly helps, often by a
  lot.
- **Failure set re-run until the agent ends the session.** The harness stopped every run at 30
  accepted frames, which counts views R7 later dropped, while R12 fires on 30 views in the set. It
  therefore ended 26 of 60 sessions before the agent could escalate (autofocus drift 8/20, warped
  board 18/20), and the correct-abstention rate was not measured. The failure split now runs until
  the agent's own terminal card (R10, R12 on views or attempts, R6, or a clean R11 stop), from the
  same frozen snapshot with only `policies.py` changed (source tree `44c665ea…babb`). Correct
  abstention (the session ends in R10, R12 or R6, spec 11.5 item 6): fisheye130 20/20 (R10 15,
  R6 5), autofocus drift 20/20 (R12 13, of which 11 on views and 2 on attempts; R6 7), warped
  board 20/20 (R12 20: 18 on views, 2 on attempts); 0 clean stops and 0 runs cut by the harness.
  Every session that had ended on its own card before ends on the same card now.
- **R6 restarts on TEST.** TEST has no drift nuisance, yet R6 raised drift restarts, all approved:
  A/human 10, A-outlier/human 6, A-gate/human 4, A-model 25 (robot 5, human 20), A-nbv 17
  (robot 8, human 9); A/robot 0. These are false alarms.
- **Runs without an estimate at n = 10.** Section 6 mentions dropping whole cameras only (none was
  dropped). Individual runs also drop out of their camera's seed median when they have no
  estimate at n = 10: 3 A/human runs restarted by R6 at n = 9, 9 and 10 (cameras 1034, 1048, 1088);
  and 6 B2/robot, 2 B2/human and 8 B2-k5/robot runs that accepted only 1 or 2 frames in 60
  attempts, because the app's bad-angle check (|pitch| or |yaw| > 40 deg under its current camera
  matrix) rejected every later frame; on camera 1003 seed 2, for example, the one-view calibration
  gives fx = 2581 px. Dropping these B2 runs can only favour B2. Scoring the 3 A runs at their last
  estimate before n = 10 leaves PE2 unchanged (median difference -1.905 px, CI [-2.972, -1.090])
  and moves the PE6 median difference from -1.446 to -1.429 px (CI unchanged, [-2.063, -0.904]).
- **Third stated hypothesis (gains shrink under the human actuator): not supported as a general
  statement.** Paired median differences at n = 10, robot vs human: A - B0 -1.44 vs -1.91 px (larger
  under human), A - B2 -1.75 vs -1.45 px (smaller), A - B1 -2.15 vs -2.08 px (about equal).
- **Wide-lens tail of the mm table.** The overall median of per-run medians at n = 10 and Z = 1 m
  is 0.047 mm for A, but on the wide family A's median per-run p95 length error is 25.9 mm (robot)
  and 29.8 mm (human), against B4's 5.7 and 7.4 mm.
- **METRIC pool (section 8).** The D455 calibrations are degenerate: the board covers under 4 % of
  the frame in all 93 full-board pool frames, fx errors against the factory reference run from
  -58 % to +199 %, and the field difference from 188 px to 6.7e7 px across the non-random
  policies, so the field metric cannot separate the policies there. On KinectV2 (GT intrinsics) the
  true field p95 of the non-random policies runs from 7.3 to 5,756 px.
- **Local paths in `meta.json`.** The `stop_calibration` and `package_dir` fields of
  `results/synth_{dev,test,failure}/meta.json` held absolute paths of the machine that ran them.
  They now hold repository-relative paths (for example `results/tmp/frozen_test/nextview`), and
  `run_synthetic.py` and `fit_stop.py` write relative paths from now on. Only those path strings
  changed: no hash covers `meta.json`, and every hash it records is unchanged.
- **Frozen snapshots committed.** The snapshots that ran DEV (`frozen_dev`), TEST (`frozen_test`)
  and the R7 fix and failure split (`frozen_r7fix`) were kept in the git-ignored `results/tmp/`.
  They are now committed byte for byte in `results/prereg/frozen_snapshots.tar.xz`, and
  `python scripts/verify_prereg.py` recomputes the source-tree and config hashes from the archive,
  compares them with each run's `meta.json` and this file, prints the line-ending-independent tree
  hash, and lists the source files that changed since. The raw-byte hashes in section 1 were taken
  on Windows with CRLF line endings in some files, so a fresh checkout cannot reproduce them; the
  archive can. A new `.gitattributes` checks Python files out with LF line endings on every platform.
- **`summarize.py` input lookup.** After review, `summarize.py` was changed only in how it finds its
  inputs: it now reads the committed `runs_core.csv.gz` and `sessions.jsonl.gz` when the full
  files are absent (through `files.load_runs` / `files.load_sessions`). The analysis code is
  unchanged. Re-running it on the committed compact tables reproduces every endpoint decision;
  values can differ in their last digits because `runs_core.csv.gz` is rounded to 3-4
  significant digits.
- **Wording of section 3.** "The first-order U95 is optimistic by about a third" overstates what r
  shows. U95 is the 95th percentile of the per-cell 1-sigma (RMS) radius, so r = true p95 / U95
  would exceed 1 even for an exact first-order map. The under-coverage is measured by the chi-square
  (2 dof) coverage of the predicted error ellipses (`summary.json`, `uncertainty_honesty_E5`). The
  stop calibration is unaffected: it maps U95 to the exceedance probability whatever the reason
  for the gap.
- **Wide-lens mm tail, explained (post hoc).** `python -m nextview.bench.consequence --runs
  results/synth_test --inversion-only` (`results/synth_test/consequence_inversion.json`) projects
  every back-projected segment endpoint forward again with the same estimate. On the wide family
  at n = 10 (k5, before any switch) the agent's fit fails to invert a median 7.25 % (robot) and
  7.75 % (human) of the outer-band endpoints, against 0.75 % and 1.75 % for B4; on the pairs that do
  invert, the median per-run p95 is 6.1 vs 4.9 mm (robot) and 6.4 vs 4.3 mm (human). At n = 30,
  after the approved switch, A's wide-lens p95 is 0.18 mm (robot) against B4's 5.9 mm. The
  `E9_mm_consequence.png` figure was re-rendered with two-line column headers (layout only).

## Addendum 3: publication (no change to anything above)

- **The public history is a squash.** The repository was published as a single commit. The base
  commit named in section 1 (`7d6e45e…`) and the earlier development history are not published,
  and neither is the git-ignored `results/tmp/` (the pre-flight check and the snapshot folders
  named in Addendum 2). The published source tree is not the TEST snapshot: it carries the later
  changes that `scripts/verify_prereg.py` lists.
- **Where the frozen trees are.** The snapshots that ran DEV, TEST and the R7 fix and failure split
  are committed in `results/prereg/frozen_snapshots.tar.xz`. `python scripts/verify_prereg.py`
  reproduces every source-tree and config hash recorded in section 1 and in each run's
  `meta.json` from that archive, and lists the source files that changed since. Those hashes, not
  a commit, date sections 1-8.
- **`freeze_commit` stays null.** `config/thresholds.json` (`_meta.freeze_commit`) was to be filled
  by the commit that adds the file. It is left `null` because the file is frozen by its sha256
  (section 1), and writing a commit id into it would change that hash.
