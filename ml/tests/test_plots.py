import json

import pytest
from PIL import Image

from ml.evaluation import plots

FIXTURES = {
    "history": {
        "epochs": [
            {
                "epoch": 1,
                "phase": "head",
                "train_loss": 0.6,
                "val_loss": 0.5,
                "train_accuracy": 0.7,
                "val_accuracy": 0.75,
            },
            {
                "epoch": 2,
                "phase": "finetune",
                "train_loss": 0.3,
                "val_loss": 0.35,
                "train_accuracy": 0.9,
                "val_accuracy": 0.88,
            },
        ]
    },
    "confusion": {"labels": ["real", "ai_generated"], "matrix": [[40, 10], [5, 45]]},
    "roc": {"fpr": [0, 0.2, 1], "tpr": [0, 0.8, 1], "auc": 0.8},
    "pr": {"precision": [1, 0.8, 0.5], "recall": [0, 0.8, 1], "average_precision": 0.82},
    "confidence": {"bins": [i / 10 for i in range(11)], "real": [5] * 10, "ai_generated": [3] * 10},
    "dataset_report": {"splits": {s: {"real": 10, "ai_generated": 8} for s in ("train", "val", "test", "external")}},
    "model_comparison": {
        "models": [
            {"name": "a", "validation_auc": 0.9, "external_auc": 0.7},
            {"name": "b", "validation_auc": 0.95, "external_auc": 0.75},
        ]
    },
}
EXPECTED = {
    "training_accuracy.png",
    "training_loss.png",
    "confusion_matrix.png",
    "roc_curve.png",
    "pr_curve.png",
    "confidence_distribution.png",
    "class_distribution.png",
    "model_comparison.png",
}


def test_every_figure_is_drawn_as_a_valid_png(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    for name, payload in FIXTURES.items():
        (metrics / f"{name}.json").write_text(json.dumps(payload))
    written = plots.plot_all(metrics, tmp_path / "plots")
    assert {p.name for p in written} == EXPECTED
    for path in written:
        with Image.open(path) as img:
            assert img.format == "PNG" and img.width > 300 and img.height > 300


def test_missing_metrics_files_are_skipped_not_faked(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "roc.json").write_text(json.dumps(FIXTURES["roc"]))
    written = plots.plot_all(metrics, tmp_path / "plots")
    assert [p.name for p in written] == ["roc_curve.png"]
    assert plots.plot_all(tmp_path / "empty", tmp_path / "plots2") == []


def test_figures_change_with_the_data(tmp_path):
    a = plots.roc_curve(FIXTURES["roc"], tmp_path)[0].read_bytes()
    other = {"fpr": [0, 0.05, 1], "tpr": [0, 0.95, 1], "auc": 0.95}
    (tmp_path / "b").mkdir()
    b = plots.roc_curve(other, tmp_path / "b")[0].read_bytes()
    assert a != b


def test_malformed_payload_raises_instead_of_drawing_something_wrong(tmp_path):
    with pytest.raises(KeyError):
        plots.confusion_matrix({"labels": []}, tmp_path)
