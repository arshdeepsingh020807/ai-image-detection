"""Shortcut probe: how well do file-level features alone predict the label?

An AUC near 0.5 means the metadata carries no label signal; a high AUC means a model could cheat
without looking at pixels, so the corresponding property must be neutralised in preprocessing.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

FEATURE_SETS = {
    "file_format": ["is_png", "width", "height", "file_size", "bits_per_pixel", "jpeg_quality", "has_exif"],
    "colour": ["mean_r", "mean_g", "mean_b"],
}


def add_metadata_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["is_png"] = (out.format == "PNG").astype(int)
    out["bits_per_pixel"] = out.file_size * 8 / (out.width * out.height)
    out["has_exif"] = out.has_exif.astype(int)
    return out


def probe(frame: pd.DataFrame, columns: list[str], *, max_per_class: int = 10_000, seed: int = 0) -> dict:
    """Cross-validated AUC/accuracy of a gradient-boosted model on class-balanced rows."""
    parts = [g.sample(min(len(g), max_per_class), random_state=seed) for _, g in frame.groupby("label")]
    sample = pd.concat(parts)
    labels = sample.label.to_numpy()
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    model = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    scores = cross_val_predict(model, sample[columns], labels, cv=folds, method="predict_proba")[:, 1]
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, scores > 0.5)),
        "n": len(sample),
        "features": columns,
    }


def run_probes(frame: pd.DataFrame, seed: int = 0) -> dict[str, dict]:
    """Probe each feature set on each (source) separately so one source's quirks don't hide in another."""
    prepared = add_metadata_features(frame)
    results: dict[str, dict] = {}
    for source, group in prepared.groupby("source"):
        if group.label.nunique() < 2:
            continue
        for name, columns in FEATURE_SETS.items():
            results[f"{source}/{name}"] = probe(group, columns, seed=seed)
    return results
