import json

import pytest

from ml.scripts import update_readme as ur

TEMPLATE = "\n".join(f"## {n}\n<!-- generated:{n} -->\nold\n<!-- /generated:{n} -->" for n in ur.BLOCKS)

METRICS = {
    "summary": {
        "model": "m",
        "temperature": 1.234,
        "band": {"low": 0.2, "high": 0.8},
        "test": {
            "n": 1000,
            "accuracy": 0.9,
            "precision": 0.91,
            "recall": 0.89,
            "f1": 0.9,
            "specificity": 0.92,
            "false_positive_rate": 0.08,
            "false_negative_rate": 0.11,
            "roc_auc": 0.96,
            "pr_auc": 0.95,
            "log_loss": 0.25,
            "ece": 0.02,
            "confusion_matrix": [[460, 40], [55, 445]],
        },
        "test_with_band": {"coverage": 0.85, "accuracy_on_decided": 0.97, "inconclusive": 150},
    },
    "external": {
        "sets": [{"name": "X/DALL-E", "protocol": "as_is", "n": 2000, "metrics": {"accuracy": 0.7, "roc_auc": None}}],
        "notes": ["note one"],
    },
    "latency": {
        "cpu": {"p50_ms": 21.04, "p95_ms": 30.0, "threads": 1},
        "onnx_size_mb": 16.2,
        "runs": 50,
        "machine": "M",
    },
}


def test_render_fills_blocks_from_metrics_and_marks_missing_ones():
    out = ur.render(TEMPLATE, METRICS)
    assert (
        "| accuracy | 90.0% |" in out and "| ROC-AUC | 0.960 |" in out and "| ECE (calibration error) | 0.020 |" in out
    )
    assert "| Actually real | 460 | 40 |" in out and "| Actually AI | 55 | 445 |" in out
    assert "temperature 1.234" in out and "n = 1,000" in out
    assert "verdict on 85.0% of test images, 97.0% of which are correct; 150 are inconclusive" in out
    assert "| X/DALL-E | as_is | 2,000 | 70.0% | n/a |" in out and "- note one" in out
    assert "p50 21.0 ms" in out and "16.2 MB" in out
    assert out.count(ur.NOT_YET) == 3  # dataset, robustness and comparison have no metrics here
    assert "\nold\n" not in out


def test_render_is_idempotent_and_keeps_surrounding_text():
    once = ur.render("intro\n" + TEMPLATE + "\noutro", METRICS)
    assert ur.render(once, METRICS) == once and once.startswith("intro\n") and once.endswith("\noutro")


def test_missing_block_is_an_error():
    with pytest.raises(ValueError, match="generated:latency"):
        ur.render(TEMPLATE.replace("generated:latency", "x"), METRICS)


def test_dataset_block_uses_report_counts():
    report = {
        "original_images": 1200,
        "cleaned_images": 1000,
        "removed_images": 200,
        "removed_by_reason": {"exact_duplicate": 150, "corrupt": 50},
        "leakage_check": {"exact_duplicate_leaks": 0, "near_duplicate_leaks": 0},
        "by_source": {"S": {"real": 600, "ai_generated": 400, "domain": "faces", "role": "train_pool"}},
        "splits": {"train": {"real": 420, "ai_generated": 280}},
    }
    block = ur.dataset_block({"dataset_report": report})
    assert "1,200 images audited, 1,000 kept, 200 removed (exact duplicate 150, corrupt 50)" in block
    assert "| S | 600 | 400 | faces | train pool |" in block and "| train | 420 | 280 |" in block


def test_check_mode_detects_stale_readme(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "latency.json").write_text(json.dumps(METRICS["latency"]))
    readme = tmp_path / "README.md"
    readme.write_text(TEMPLATE)
    monkeypatch.setattr("sys.argv", ["x", "--readme", str(readme), "--metrics", str(metrics), "--check"])
    with pytest.raises(SystemExit) as stale:
        ur.main()
    assert "out of date" in str(stale.value.code)
    monkeypatch.setattr("sys.argv", ["x", "--readme", str(readme), "--metrics", str(metrics)])
    ur.main()
    monkeypatch.setattr("sys.argv", ["x", "--readme", str(readme), "--metrics", str(metrics), "--check"])
    with pytest.raises(SystemExit) as fresh:
        ur.main()
    assert fresh.value.code == 0


def test_repository_readme_matches_published_metrics():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text()
    assert ur.render(readme, ur.load_metrics(root / "artifacts/metrics")) == readme, (
        "run python -m ml.scripts.update_readme"
    )
