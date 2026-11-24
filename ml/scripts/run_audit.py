"""Audit every catalogued image and write data/manifests/audit.csv.gz."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ml.datasets.catalog import all_items
from ml.preprocessing.audit import audit_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/manifests/audit.csv.gz"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    items = list(all_items(args.raw))
    print(f"auditing {len(items)} images with {args.workers} workers")
    frame = audit_items(items, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.groupby(["source", "label", "status"]).size())


if __name__ == "__main__":
    main()
