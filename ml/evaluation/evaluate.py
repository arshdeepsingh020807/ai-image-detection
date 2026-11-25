"""Turn a trained checkpoint into every metrics file of docs/contract/api.md.

Calibration (temperature) and the inconclusive band are fitted on the validation split only;
test, external and robustness numbers never influence any choice.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import Dataset

from detector_core.preprocess import prepare, to_rgb
from ml.evaluation.metrics import band_metrics, choose_band, classification_metrics, fit_temperature
from ml.evaluation.robustness import CONDITIONS, Transform, match_to_reference
from ml.training.data import make_loader

BAND_MAX_ERROR = 0.02


class ImageList(Dataset):
    def __init__(self, paths: list[str], transform: Transform | None = None):
        self.paths, self.transform = paths, transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as img:
            rgb = to_rgb(img)
        return torch.from_numpy(prepare(self.transform(rgb) if self.transform else rgb))


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    paths: list[str],
    device: torch.device,
    transform: Transform | None = None,
    batch_size: int = 64,
    workers: int = 4,
) -> np.ndarray:
    model.eval()
    loader = make_loader(ImageList(paths, transform), batch_size=batch_size, num_workers=workers)
    out = [model(torch.from_numpy(x).to(device)).squeeze(1).float().cpu() for x in loader]
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)


def downsample_curve(xs: np.ndarray, ys: np.ndarray, max_points: int = 200) -> tuple[list[float], list[float]]:
    """Keep first and last points and an even spread in between, so charts stay light but exact at the ends."""
    if len(xs) <= max_points:
        return xs.tolist(), ys.tolist()
    index = np.unique(np.linspace(0, len(xs) - 1, max_points).round().astype(int))
    return xs[index].tolist(), ys[index].tolist()


def curves(y: np.ndarray, prob: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, prob)
    precision, recall, _ = precision_recall_curve(y, prob)
    metrics = classification_metrics(y, prob)
    roc_x, roc_y = downsample_curve(fpr, tpr)
    pr_r, pr_p = downsample_curve(recall[::-1], precision[::-1])
    edges = np.linspace(0, 1, 11)
    return {
        "roc": {"fpr": roc_x, "tpr": roc_y, "auc": metrics["roc_auc"]},
        "pr": {"precision": pr_p, "recall": pr_r, "average_precision": metrics["pr_auc"]},
        "confusion": {"labels": ["real", "ai_generated"], "matrix": metrics["confusion_matrix"]},
        "confidence": {
            "bins": edges.tolist(),
            "real": np.histogram(prob[y == 0], edges)[0].tolist(),
            "ai_generated": np.histogram(prob[y == 1], edges)[0].tolist(),
        },
    }


def subgroup_metrics(frame: pd.DataFrame, prob: np.ndarray, column: str) -> dict[str, dict]:
    out = {}
    for value, idx in frame.reset_index(drop=True).groupby(column).groups.items():
        rows = frame.iloc[list(idx)]
        out[str(value)] = classification_metrics(rows.label.to_numpy(), prob[list(idx)])
    return out


def balanced_subset(frame: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    return pd.concat([g.sample(min(len(g), per_class), random_state=seed) for _, g in frame.groupby("label")])


def evaluate(
    model: torch.nn.Module,
    manifest: pd.DataFrame,
    device: torch.device,
    out_dir: Path,
    *,
    model_name: str,
    seed: int = 42,
    robustness_per_class: int = 1000,
    workers: int = 4,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = manifest[manifest.keep] if "keep" in manifest else manifest

    def probs(rows: pd.DataFrame, temperature: float, transform: Transform | None = None) -> np.ndarray:
        return expit(predict_logits(model, rows.path.tolist(), device, transform, workers=workers) / temperature)

    val = kept[kept.split == "val"].reset_index(drop=True)
    test = kept[kept.split == "test"].reset_index(drop=True)
    val_logits = predict_logits(model, val.path.tolist(), device, workers=workers)
    temperature = fit_temperature(val_logits, val.label.to_numpy())
    val_prob = expit(val_logits / temperature)
    low, high = choose_band(val_prob, val.label.to_numpy(), BAND_MAX_ERROR)

    test_prob = probs(test, temperature)
    y = test.label.to_numpy()
    test.assign(p_ai=test_prob).to_csv(out_dir / "test_predictions.csv", index=False)
    summary = {
        "model": model_name,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "temperature": temperature,
        "band": {"low": low, "high": high},
        "validation": classification_metrics(val.label.to_numpy(), val_prob),
        "test": classification_metrics(y, test_prob),
        "test_with_band": band_metrics(test_prob, y, low, high),
        "test_subgroups": {col: subgroup_metrics(test, test_prob, col) for col in ("domain", "source", "format")},
    }
    parts = curves(y, test_prob)

    external_sets, notes = [], []
    ext = kept[kept.split == "external"].reset_index(drop=True)
    for source, rows in ext.groupby("source"):
        in_scope = rows[rows.scope == "in_scope"] if "scope" in rows else rows
        p = probs(in_scope, temperature)
        external_sets.append(
            {
                "name": f"{source}",
                "protocol": "as_is",
                "n": len(in_scope),
                "metrics": classification_metrics(in_scope.label.to_numpy(), p),
            }
        )
        for generator, idx in in_scope.reset_index(drop=True).groupby("generator").groups.items():
            sub = in_scope.iloc[list(idx)]
            external_sets.append(
                {
                    "name": f"{source}/{generator}",
                    "protocol": "as_is",
                    "n": len(sub),
                    "metrics": classification_metrics(sub.label.to_numpy(), p[list(idx)]),
                }
            )
        out_of_scope = rows[rows.scope == "out_of_scope"] if "scope" in rows else rows.iloc[0:0]
        if len(out_of_scope):
            external_sets.append(
                {
                    "name": f"{source}/face_swap (out of scope)",
                    "protocol": "as_is",
                    "n": len(out_of_scope),
                    "metrics": classification_metrics(out_of_scope.label.to_numpy(), probs(out_of_scope, temperature)),
                }
            )
        real = in_scope[in_scope.label == 0]
        if len(real) and (in_scope.format != "JPEG").any():
            size = int(np.median(np.minimum(real.width, real.height)))
            quality = round(real.jpeg_quality.median()) if real.jpeg_quality.notna().any() else 90
            fake = in_scope[in_scope.label == 1]
            matched = np.concatenate(
                [
                    probs(real, temperature),
                    probs(fake, temperature, partial(match_to_reference, size=size, quality=quality)),
                ]
            )
            labels = np.r_[np.zeros(len(real)), np.ones(len(fake))]
            external_sets.append(
                {
                    "name": f"{source}",
                    "protocol": f"matched: AI images resized to {size}px and JPEG q{quality}",
                    "n": len(labels),
                    "metrics": classification_metrics(labels, matched),
                }
            )
            notes.append(
                f"{source}: real and AI images differ in file format or resolution; the matched protocol "
                "removes that difference and is the fairer number."
            )

    robust_rows = balanced_subset(test, robustness_per_class, seed).reset_index(drop=True)
    conditions = []
    for name, (description, transform) in CONDITIONS.items():
        p = probs(robust_rows, temperature, transform)
        m = classification_metrics(robust_rows.label.to_numpy(), p)
        conditions.append(
            {
                "name": name,
                "description": description,
                "accuracy": m["accuracy"],
                "roc_auc": m["roc_auc"],
                "false_positive_rate": m["false_positive_rate"],
                "n": m["n"],
            }
        )

    files = {
        "summary": summary,
        "roc": parts["roc"],
        "pr": parts["pr"],
        "confusion": parts["confusion"],
        "confidence": parts["confidence"],
        "external": {"sets": external_sets, "notes": notes},
        "robustness": {"conditions": conditions},
    }
    for name, payload in files.items():
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))
    return files
