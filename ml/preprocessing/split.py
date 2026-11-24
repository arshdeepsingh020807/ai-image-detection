"""Reproducible train/val/test splits that keep near-duplicate clusters together."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.preprocessing.hashing import near_duplicate_groups

SPLITS = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)


def make_splits(
    cleaned: pd.DataFrame, seed: int = 42, fractions: tuple[float, ...] = DEFAULT_FRACTIONS
) -> pd.DataFrame:
    """Adds `split` to kept rows: train/val/test for the training pool, "external" for held-out sources.

    Whole clusters are assigned to one split, separately inside every (source, label) stratum so each
    split keeps the class and source mix. Rows that were dropped get split == "".
    """
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must sum to 1")
    out = cleaned.copy()
    out["split"] = ""
    out.loc[out.keep & (out.role == "external"), "split"] = "external"
    rng = np.random.default_rng(seed)
    pool = out[out.keep & (out.role == "train_pool")]
    for _, stratum in pool.groupby(["source", "label"], sort=True):
        sizes = stratum.groupby("cluster").size().sort_index()
        order = rng.permutation(len(sizes))
        clusters, counts = sizes.index.to_numpy()[order], sizes.to_numpy()[order]
        boundaries = np.cumsum(fractions) * counts.sum()
        position = np.cumsum(counts) - counts / 2  # midpoint of each cluster decides its split
        which = np.searchsorted(boundaries, position, side="right").clip(0, len(SPLITS) - 1)
        for cluster, idx in zip(clusters, which, strict=False):
            out.loc[stratum.index[stratum.cluster == cluster], "split"] = SPLITS[idx]
    return out


def find_leaks(manifest: pd.DataFrame, max_distance: int = 4) -> dict[str, int]:
    """Re-derive duplicate groups from the hashes and count groups that span more than one split."""
    kept = manifest[manifest.split != ""]
    clusters = pd.Series(near_duplicate_groups([int(h, 16) for h in kept.phash], max_distance), index=kept.index)
    return {
        "exact_duplicate_leaks": int((kept.groupby("md5").split.nunique() > 1).sum()),
        "near_duplicate_leaks": int((kept.split.groupby(clusters).nunique() > 1).sum()),
    }
