"""Summarise the ECR scan of the deployed image into results/ops/ecr_scan.json (cited by the report).

Reads the image tag from infra/deployed.json, waits for the scan-on-push result, and records the finding counts
per severity and the affected packages. AWS credentials come from the environment (AWS_PROFILE etc.).

    python scripts/ecr_scan_summary.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL", "UNDEFINED"]
ADDED = ("opencv", "numpy")  # what requirements-lambda.txt adds on top of the base image


def aws(*args: str) -> dict:
    p = subprocess.run(["aws", *args, "--output", "json"], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"aws {args[0]} {args[1]} failed: {p.stderr.strip()}")
    return json.loads(p.stdout or "{}")


def main() -> int:
    dep = json.loads((REPO / "infra" / "deployed.json").read_text(encoding="utf-8"))
    repo, tag, region = dep["function"], dep["image_tag"], dep["region"]
    for _ in range(40):
        j = aws("ecr", "describe-image-scan-findings", "--repository-name", repo, "--image-id", f"imageTag={tag}",
                "--region", region, "--max-items", "1000")
        if j.get("imageScanStatus", {}).get("status") == "COMPLETE":
            break
        time.sleep(5)
    else:
        raise SystemExit(f"scan of {tag} not complete: {j.get('imageScanStatus')}")
    f = j["imageScanFindings"]
    ids, pk = defaultdict(set), defaultdict(set)
    for x in f.get("findings", []):
        a = {k["key"]: k["value"] for k in x.get("attributes", [])}
        ids[x["severity"]].add(x["name"])
        pk[x["severity"]].add(a.get("package_name", "?"))
    counts = f.get("findingSeverityCounts", {})
    added = sorted(p for s in pk for p in pk[s] if p.lower().startswith(ADDED))
    out = {
        "what": "ECR basic image scan (scan on push) of the deployed image",
        "image_tag": tag, "image_digest": dep["image_digest"], "base_image_digest": dep["base_image_digest"],
        "scan_status": j["imageScanStatus"]["status"], "scan_completed": f.get("imageScanCompletedAt"),
        "vulnerability_source_updated": f.get("vulnerabilitySourceUpdatedAt"),
        "finding_counts": {s: counts[s] for s in ORDER if s in counts},
        "distinct_ids": {s: len(ids[s]) for s in ORDER if s in ids},
        "packages": {s: sorted(pk[s]) for s in ORDER if s in pk},
        "python_packages_with_findings": added,
        "note": "Findings count a CVE once per affected package. "
                + ("Every affected package is an Amazon Linux 2023 OS package of the AWS Lambda Python 3.12 base image "
                   "(pinned by digest in the Dockerfile); none is in the packages the image adds (opencv-python-headless, "
                   "numpy)." if not added else "Some findings are in packages the image adds; see python_packages_with_findings.")
                + " Written by scripts/ecr_scan_summary.py from `aws ecr describe-image-scan-findings` for this tag.",
    }
    path = REPO / "results" / "ops" / "ecr_scan.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{tag}: {out['finding_counts']}; added packages with findings: {added or 'none'}; wrote {path.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
