| Policy | Runs | Valid / tool calls | Valid / attempted | Repaired | Rank-1 agreement (all / after opening) | Pose captures | Exported | Median end views | Median true p95 at end (px) | Median true p95 at 10 frames (px) | Median final P_exceed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 10 | – | – | – | – | – | – | 10.0 | 0.741 | 0.741 | 0.533 |
| B0 | 10 | – | – | – | – | – | – | 10.0 | 1.568 | 1.568 | – |
| LLM qwen2.5:7b@nohints | 10 | 487/498 (98 %) | 487/498 (98 %) | 0 | 100 % / 100 % | 0 | 9/10 | 10.0 | 0.695 | 0.741 | 0.503 |
| LLM nemotron-3-nano:4b@nothink+h8+pose | 10 | 399/410 (97 %) | 399/518 (77 %) | 11 | 100 % / 100 % | 0 | 10/10 | 10.0 | 0.633 | 0.680 | 0.503 |
| LLM qwen2.5:7b@nohints+pose | 10 | 392/493 (80 %) | 392/493 (80 %) | 0 | 65 % / 55 % | 42 | 6/10 | 10.0 | 1.046 | 0.647 | 0.552 |
