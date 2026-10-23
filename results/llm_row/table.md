| Policy | Runs | Valid / tool calls | Valid / attempted | Repaired | Rank-1 agreement (all / after opening) | Pose captures | Exported | Median end views | Median true p95 at end (px) | Median true p95 at 10 frames (px) | Median final P_exceed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 10 | – | – | – | – | – | – | 10.0 | 0.741 | 0.741 | 0.533 |
| B0 | 10 | – | – | – | – | – | – | 10.0 | 1.568 | 1.568 | – |
| MCP-scripted | 10 | 506/506 (100 %) | – | – | – | – | 10/10 | 10.0 | 0.695 | 0.741 | 0.503 |
| LLM qwen2.5:7b | 10 | 410/420 (98 %) | 410/420 (98 %) | 0 | 100 % / 100 % | 0 | 10/10 | 10.0 | 0.695 | 0.741 | 0.503 |
| LLM qwen2.5:7b@pose | 10 | 410/423 (97 %) | 410/423 (97 %) | 0 | 100 % / 100 % | 0 | 10/10 | 10.0 | 0.695 | 0.741 | 0.503 |
| LLM nemotron-3-nano:4b@nothink+pose | 10 | 93/100 (93 %) | 93/193 (48 %) | 7 | 100 % / – | 0 | 0/10 | 3.0 | 7.936 | – | 1.000 |
