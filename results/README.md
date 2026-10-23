# Benchmark results

Every number in the report and the Evidence tab is read from these files. Released under
CC BY 4.0 (METRIC-derived numbers carry the METRIC attribution in `ATTRIBUTION.md`).

| Path | What | Produced by |
|---|---|---|
| `dev/blur_curve.json` | DEV blur curve (corner error vs ESF rise) behind the `max_edge_rise_px` freeze | `python -m nextview.bench.blur_curve --out results/dev/blur_curve.json` |
| `synth_dev/` | DEV split (cameras 0-39, A and B0): `runs.csv`, `sessions.jsonl`, `events.jsonl.gz`, `meta.json`; `stop_fit.json` (stop calibration fit and tau rule) | `python -m nextview.bench.run_synthetic --split dev --out results/synth_dev`, then `python -m nextview.bench.fit_stop --runs results/synth_dev/runs.csv --out config/stop_calibration.json` |
| `synth_test/` | TEST split (cameras 1000-1099 x 3 seeds x 2 actuators x 11 policies); `summary.json` holds the pre-registered endpoints and every descriptive number; `consequence.json` the mm table. The full `runs.csv` (46 MB) is not committed: `runs_core.csv.gz` has every row and metric (no per-row K / distortion; metrics rounded to 3-4 significant digits), `estimates.csv.gz` the full estimates at n = 10, 20, 30; the runner regenerates the full table exactly, and `summarize` reads the compact files directly when the full ones are absent | `run_synthetic --split test`, `python -m nextview.bench.summarize --runs results/synth_test`, `python -m nextview.bench.consequence --runs results/synth_test` |
| `synth_failure/` | Failure set (fisheye130, autofocus drift, warped board; 20 cameras each, A). Each run goes until the agent's own terminal card (R10, R12 on views or attempts, R6, or a clean R11 stop); correct-abstention counts are in `failure_gallery.json` | `run_synthetic --split failure --out results/synth_failure` |
| `synth_test/consequence_inversion.json` | Post-hoc check of the mm table (descriptive): share of segment endpoints whose back-projection does not reproject within 0.1 px, and the length error over the pairs that do, per policy, actuator, n and family | `python -m nextview.bench.consequence --runs results/synth_test --inversion-only` (reads the committed `estimates.csv.gz`) |
| `prereg/frozen_snapshots.tar.xz` | Byte-for-byte copies of the source tree and config that ran DEV, TEST and the failure split (`frozen_dev`, `frozen_test`, `frozen_r7fix`) | `python scripts/verify_prereg.py` checks every recorded hash against it (`--pack` rebuilt it from `results/tmp/frozen_*`) |
| `ops/`, `ops_insights.json`, `ops/ecr_scan.json` | Load tests and latency (day 2), CloudWatch Logs Insights numbers for the report, and the ECR scan of the deployed image | `infra/load_test.py`, `python infra/guard.py --insights results/ops_insights.json`, `python scripts/ecr_scan_summary.py` (after a deploy) |
| `synth_test/posthoc.json` | Post-hoc descriptive numbers written after review (not pre-registered): R7 counts, recall and precision, R7's accuracy effect (A vs A-outlier), R6 restarts, runs without an estimate at n = 10 and a sensitivity check, robot vs human gains, the wide-lens mm tail | `python -m nextview.bench.posthoc --runs results/tmp/full/synth_test` (the committed `results/synth_test` tables give the same counts) |
| `pool/` | METRIC real (D455) and synthetic (KinectV2) pool protocol, per camera and `summary.json` | `python -m nextview.bench.run_pool --out results/pool` |
| `failure_gallery.json` | Measured numbers for failure cases F1-F11 | `python -m nextview.bench.gallery --out results` |
| `figures/` | E1-E10, E12 | `python -m nextview.bench.plots`, `python -m nextview.bench.gallery` |
| `llm_row/` | LLM row (spec 9) on TEST cameras 1000-1009 (normal, robot, budget 10): A, B0, MCP-scripted and qwen2.5:7b, which may only pick among the planner's candidates; the earlier runs that also offered `sim_capture`'s `pose` argument are kept under labels ending `pose` (`runs_pose_offered.json`). `table.md` is the report table; `summary.json`, `runs.csv`, `runs.json`; one JSONL call log per episode at `llm_row/<model-slug>/run_<camera_seed>.jsonl` (spec 9 named `results/llm/run_<seed>.jsonl`) | `python -m nextview.bench.llm_row --models qwen2.5:7b --out results/llm_row`, then `--from-runs results/llm_row/runs.json results/llm_row/runs_pose_offered.json --out results/llm_row` |
| `llm_row_extra/` | LLM row ablations: qwen2.5:7b without the tools' "Next:" hints, nemotron-3-nano:4b with 8 messages of history | `llm_row --models qwen2.5:7b@nohints --skip-scripted --out results/llm_row_extra`, merged the same way |
| `llm_row_think/` | nemotron-3-nano:4b with reasoning on, cameras 1000-1004 | `llm_row --models --cameras 1000-1004 --skip-scripted --out results/llm_row_think`, merged the same way |

`runs.csv` is long format: one row per (camera, seed, actuator, policy, accepted frame n) with the
estimate (fx, fy, cx, cy, d0..d7), its true field error over the 32 x 18 grid against the GT
camera, U95, coverage fractions, held-out RMS and, for the agent, the loop's P_exceed and the R11
stability inputs. The pre-registration is `docs/preregistration.md`.
