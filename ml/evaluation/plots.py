"""Static figures for the README and viva, drawn only from artifacts/metrics/*.json."""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL, AI = "#2f6f9f", "#c0532b"


def _figure(title: str, size: tuple[float, float] = (5.5, 4.0)):
    fig, ax = plt.subplots(figsize=size, dpi=150)
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def training_curves(history: dict, out: Path) -> list[Path]:
    epochs = [e["epoch"] for e in history["epochs"]]
    written = []
    for key, label in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = _figure(f"Training vs validation {label.lower()}")
        ax.plot(epochs, [e[f"train_{key}"] for e in history["epochs"]], marker="o", label="Train", color=REAL)
        ax.plot(epochs, [e[f"val_{key}"] for e in history["epochs"]], marker="o", label="Validation", color=AI)
        finetune = [e["epoch"] for e in history["epochs"] if e["phase"] == "finetune"]
        if finetune:
            ax.axvline(finetune[0] - 0.5, color="grey", linestyle=":", linewidth=1)
            ax.text(finetune[0] - 0.45, ax.get_ylim()[1], " fine-tuning", va="top", fontsize=8, color="grey")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend(frameon=False)
        written.append(_save(fig, out / f"training_{key}.png"))
    return written


def confusion_matrix(confusion: dict, out: Path) -> list[Path]:
    matrix = confusion["matrix"]
    fig, ax = _figure("Confusion matrix (test split)", (4.5, 4.0))
    ax.imshow(matrix, cmap="Blues")
    total = sum(map(sum, matrix))
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i][j]}\n{matrix[i][j] / total:.1%}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if matrix[i][j] > total / 4 else "black",
            )
    names = ["Real", "AI-generated"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return [_save(fig, out / "confusion_matrix.png")]


def roc_curve(roc: dict, out: Path) -> list[Path]:
    fig, ax = _figure(f"ROC curve (AUC {roc['auc']:.3f})")
    ax.plot(roc["fpr"], roc["tpr"], color=AI)
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    return [_save(fig, out / "roc_curve.png")]


def pr_curve(pr: dict, out: Path) -> list[Path]:
    fig, ax = _figure(f"Precision-recall curve (AP {pr['average_precision']:.3f})")
    ax.plot(pr["recall"], pr["precision"], color=AI)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    return [_save(fig, out / "pr_curve.png")]


def confidence(hist: dict, out: Path) -> list[Path]:
    fig, ax = _figure("Predicted AI probability by true class (test split)")
    edges = hist["bins"]
    centres = [(a + b) / 2 for a, b in pairwise(edges)]
    width = (edges[1] - edges[0]) * 0.4
    ax.bar([c - width / 2 for c in centres], hist["real"], width, label="Real", color=REAL)
    ax.bar([c + width / 2 for c in centres], hist["ai_generated"], width, label="AI-generated", color=AI)
    ax.set_xlabel("Calibrated probability of AI-generated")
    ax.set_ylabel("Images")
    ax.legend(frameon=False)
    return [_save(fig, out / "confidence_distribution.png")]


def class_distribution(report: dict, out: Path) -> list[Path]:
    splits = ["train", "val", "test", "external"]
    fig, ax = _figure("Images per split after cleaning")
    real = [report["splits"][s]["real"] for s in splits]
    ai = [report["splits"][s]["ai_generated"] for s in splits]
    ax.bar(splits, real, label="Real", color=REAL)
    ax.bar(splits, ai, bottom=real, label="AI-generated", color=AI)
    ax.set_ylabel("Images")
    ax.legend(frameon=False)
    return [_save(fig, out / "class_distribution.png")]


def model_comparison(comparison: dict, out: Path) -> list[Path]:
    models = comparison["models"]
    fig, ax = _figure("Model comparison", (6.0, 4.0))
    names = [m["name"] for m in models]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [m["validation_auc"] for m in models], 0.4, label="Validation AUC", color=REAL)
    ax.bar([i + 0.2 for i in x], [m["external_auc"] for m in models], 0.4, label="External AUC", color=AI)
    ax.set_xticks(list(x), names, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False)
    return [_save(fig, out / "model_comparison.png")]


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


PLOTS: dict[str, Callable[[dict, Path], list[Path]]] = {
    "history": training_curves,
    "confusion": confusion_matrix,
    "roc": roc_curve,
    "pr": pr_curve,
    "confidence": confidence,
    "dataset_report": class_distribution,
    "model_comparison": model_comparison,
}


def plot_all(metrics_dir: Path, plots_dir: Path) -> list[Path]:
    """Draw every figure whose metrics file exists; missing files are skipped, never faked."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, draw in PLOTS.items():
        source = metrics_dir / f"{name}.json"
        if source.exists():
            written += draw(json.loads(source.read_text()), plots_dir)
    return written


if __name__ == "__main__":
    for path in plot_all(Path("artifacts/metrics"), Path("artifacts/plots")):
        print(path)
