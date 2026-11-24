"""Cleaning policy: decides which audited images are kept and clusters near-duplicates."""

from __future__ import annotations

import pandas as pd

from detector_core.preprocess import ALLOWED_FORMATS
from ml.preprocessing.hashing import near_duplicate_groups

NEAR_DUPLICATE_BITS = 4


def clean(audit: pd.DataFrame, max_distance: int = NEAR_DUPLICATE_BITS) -> pd.DataFrame:
    """Adds `cluster` (near-duplicate group id), `keep` and `drop_reason` to the audit table.

    Rules, applied in this order (first match wins):
      corrupt            unreadable or truncated
      unsupported_format format the service would reject (e.g. AVIF, MPO) or animated
      exact_duplicate    same bytes as an earlier kept image
      conflicting_label  near-duplicate cluster contains both real and AI images
      external_leak      training-pool image whose near-duplicate cluster touches an external image
    """
    out = audit.sort_values(["role", "path"]).reset_index(drop=True).copy()  # external copies win ties
    out["drop_reason"] = ""

    def mark(mask: pd.Series, reason: str) -> None:
        out.loc[mask & (out.drop_reason == ""), "drop_reason"] = reason

    mark(out.status != "ok", "corrupt")
    decodable = out.status == "ok"
    mark(decodable & (~out.format.isin(ALLOWED_FORMATS) | (out.n_frames > 1)), "unsupported_format")
    survivors = out[out.drop_reason == ""]
    mark(out.index.isin(survivors[survivors.duplicated("md5", keep="first")].index), "exact_duplicate")

    out["cluster"] = -1
    usable = out.index[out.drop_reason == ""]
    out.loc[usable, "cluster"] = near_duplicate_groups([int(h, 16) for h in out.loc[usable, "phash"]], max_distance)

    live = out[out.drop_reason == ""]
    by_cluster = live.groupby("cluster")
    mixed = by_cluster.label.transform("nunique") > 1
    mark(out.index.isin(live[mixed].index), "conflicting_label")
    live = out[out.drop_reason == ""]
    touches_external = live.groupby("cluster").role.transform(lambda r: (r == "external").any())
    mark(out.index.isin(live[touches_external & (live.role == "train_pool")].index), "external_leak")
    out["keep"] = out.drop_reason == ""
    return out
