"""audit.csv.gz -> clean -> split -> leakage check -> splits.csv.gz + dataset_report.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.preprocessing.clean import clean
from ml.preprocessing.report import build_report
from ml.preprocessing.shortcuts import run_probes
from ml.preprocessing.split import find_leaks, make_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("data/manifests/audit.csv.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/splits.csv.gz"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/metrics/dataset_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = make_splits(clean(pd.read_csv(args.audit, dtype={"phash": str})), seed=args.seed)
    leaks = find_leaks(manifest)
    if any(leaks.values()):
        raise SystemExit(f"data leakage detected: {leaks}")
    report = build_report(manifest, leaks, run_probes(manifest[manifest.keep], seed=args.seed))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    args.report.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("original_images", "cleaned_images", "removed_by_reason", "splits", "leakage_check")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
