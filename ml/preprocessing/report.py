"""Machine-readable dataset report built from the split manifest."""

from __future__ import annotations

import pandas as pd


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def build_report(manifest: pd.DataFrame, leaks: dict[str, int], probes: dict[str, dict]) -> dict:
    kept = manifest[manifest.keep]
    dropped = manifest[~manifest.keep]
    resolution = kept.width.astype(int).astype(str) + "x" + kept.height.astype(int).astype(str)
    return {
        "original_images": len(manifest),
        "cleaned_images": len(kept),
        "removed_images": len(dropped),
        "removed_by_reason": _counts(dropped.drop_reason),
        "exact_duplicates_removed": int((dropped.drop_reason == "exact_duplicate").sum()),
        "near_duplicate_clusters": int((kept.groupby("cluster").size() > 1).sum()),
        "images_in_near_duplicate_clusters": int((kept.groupby("cluster").cluster.transform("size") > 1).sum()),
        "class_distribution": {"real": int((kept.label == 0).sum()), "ai_generated": int((kept.label == 1).sum())},
        "by_source": {
            str(source): {
                "real": int((g.label == 0).sum()),
                "ai_generated": int((g.label == 1).sum()),
                "role": str(g.role.iloc[0]),
                "domain": str(g.domain.iloc[0]),
            }
            for source, g in kept.groupby("source")
        },
        "by_generator": _counts(kept[kept.label == 1].generator),
        "formats": {
            str(label): _counts(g.format) for label, g in kept.groupby(kept.label.map({0: "real", 1: "ai_generated"}))
        },
        "resolution_top": {k: int(v) for k, v in resolution.value_counts().head(10).items()},
        "splits": {
            split: {
                "real": int(((kept.split == split) & (kept.label == 0)).sum()),
                "ai_generated": int(((kept.split == split) & (kept.label == 1)).sum()),
            }
            for split in ("train", "val", "test", "external")
        },
        "leakage_check": leaks,
        "shortcut_probes": probes,
    }
