"""End-to-end smoke test of the experiment driver on a tiny synthetic dataset."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ml.scripts import run_experiments as rx
from ml.training.train import TrainConfig

ARCH = "mobilenetv3_small_050"


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    root = tmp_path_factory.mktemp("rx")
    rng = np.random.default_rng(0)
    rows = []
    layout = [("train", 24, "in_scope"), ("val", 10, "in_scope"), ("test", 10, "in_scope"), ("external", 8, "in_scope")]
    for split, n, scope in layout:
        for i in range(n):
            label = i % 2
            arr = np.asarray(Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).resize((256, 256)), int)
            if label:
                arr = arr + rng.integers(-25, 26, arr.shape)
            path = root / f"{split}{i}.jpg"
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path, quality=95)
            rows.append(
                {
                    "path": str(path),
                    "split": split,
                    "label": label,
                    "domain": "faces",
                    "source": "S",
                    "scope": scope,
                    "generator": "g" if label else "real",
                    "format": "JPEG",
                    "width": 256,
                    "height": 256,
                    "jpeg_quality": 95.0,
                    "keep": True,
                }
            )
    return pd.DataFrame(rows)


def cfg(tmp_path, **kw):
    return TrainConfig(
        **{
            "arch": ARCH,
            "pretrained": True,
            "epochs_head": 1,
            "epochs_finetune": 1,
            "batch_size": 8,
            "num_workers": 0,
            "device": "cpu",
            "out_dir": str(tmp_path / "ckpt"),
            **kw,
        }
    )


def test_compare_writes_a_row_per_candidate_with_real_measurements(tmp_path, manifest):
    payload = rx.compare(manifest, [ARCH], cfg(tmp_path, max_val=10), metrics_dir=tmp_path / "m")
    saved = json.loads((tmp_path / "m/model_comparison.json").read_text())
    assert saved == payload and len(saved["models"]) == 1
    row = saved["models"][0]
    assert row["name"] == ARCH and 0 <= row["validation_auc"] <= 1 and 0 <= row["external_auc"] <= 1
    assert row["onnx_size_mb"] > 0.5 and row["latency_ms"] > 0 and row["parameters_millions"] > 0.5
    assert row["train_images"] == 24 and saved["budget"]["max_val"] == 10


def test_unseen_generator_rows_exclude_png_and_out_of_scope(manifest):
    extra = manifest.iloc[[-1, -2]].assign(format="PNG")
    extra2 = manifest.iloc[[-3]].assign(scope="out_of_scope")
    rows = rx.unseen_generator_rows(pd.concat([manifest, extra, extra2]), 100, 0)
    assert (rows.split == "external").all() and (rows.format == "JPEG").all() and (rows.scope == "in_scope").all()
    assert len(rows) == 8


def test_final_produces_model_card_metrics_latency_and_plots(tmp_path, manifest):
    out = rx.final(
        manifest, cfg(tmp_path), metrics_dir=tmp_path / "m", model_dir=tmp_path / "model", plots_dir=tmp_path / "plots"
    )
    card = json.loads((tmp_path / "model/model_card.json").read_text())
    summary = json.loads((tmp_path / "m/summary.json").read_text())
    assert card["trained"] is True and card["architecture"] == ARCH and card["version"].startswith(ARCH)
    assert card["temperature"] == summary["temperature"] and card["band"] == summary["band"]
    assert card["test_metrics"]["accuracy"] == summary["test"]["accuracy"] == out["card"]["test_metrics"]["accuracy"]
    for name in ("history", "latency", "roc", "pr", "confusion", "confidence", "external", "robustness"):
        assert (tmp_path / "m" / f"{name}.json").exists(), name
    assert (tmp_path / "plots/training_loss.png").exists() and (tmp_path / "plots/roc_curve.png").exists()
    assert json.loads((tmp_path / "m/latency.json").read_text())["onnx_size_mb"] > 0.5


def test_compare_reuses_a_finished_row_only_for_the_same_budget(tmp_path, manifest, monkeypatch):
    base = cfg(tmp_path, max_val=10)
    first = rx.compare(manifest, [ARCH], base, metrics_dir=tmp_path / "m")["models"][0]
    calls = []
    monkeypatch.setattr(
        rx, "train", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError("retrained"))
    )
    again = rx.compare(manifest, [ARCH], base, metrics_dir=tmp_path / "m2")["models"][0]
    assert again == first and calls == []  # same budget: cached row, no training
    with pytest.raises(AssertionError, match="retrained"):
        rx.compare(manifest, [ARCH], cfg(tmp_path, max_val=10, epochs_finetune=2), metrics_dir=tmp_path / "m3")


def test_assemble_comparison_uses_finished_rows_and_rejects_mixed_budgets(tmp_path):
    budget = {"max_train": 10, "max_val": 5, "epochs_head": 1, "epochs_finetune": 2, "batch_size": 8, "seed": 1}
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "comparison_row.json").write_text(json.dumps({"budget": budget, "row": {"name": name}}))
    (tmp_path / "c").mkdir()  # a run that never finished is left out
    out = rx.assemble_comparison(tmp_path, metrics_dir=tmp_path / "m")
    assert [m["name"] for m in out["models"]] == ["a", "b"] and out["budget"] == budget
    assert json.loads((tmp_path / "m/model_comparison.json").read_text()) == out
    (tmp_path / "b/comparison_row.json").write_text(json.dumps({"budget": {**budget, "seed": 2}, "row": {"name": "b"}}))
    with pytest.raises(ValueError, match="different budgets"):
        rx.assemble_comparison(tmp_path, metrics_dir=tmp_path / "m")


def test_history_from_log_keeps_epoch_records_only(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        'Warning: x\n{"epoch": 1, "val_loss": 0.5}\n{"name": "row"}\nearly stopping\n{"epoch": 2, "val_loss": 0.3}\n'
    )
    assert rx.history_from_log(log) == {"epochs": [{"epoch": 1, "val_loss": 0.5}, {"epoch": 2, "val_loss": 0.3}]}
    log.write_text("nothing")
    with pytest.raises(ValueError, match="no epoch records"):
        rx.history_from_log(log)


def test_publish_evaluates_an_existing_checkpoint_without_training(tmp_path, manifest, monkeypatch):
    first = rx.final(
        manifest, cfg(tmp_path), metrics_dir=tmp_path / "m", model_dir=tmp_path / "model", plots_dir=tmp_path / "plots"
    )
    monkeypatch.setattr(rx, "train", lambda *a, **k: (_ for _ in ()).throw(AssertionError("trained")))
    again = rx.publish(
        manifest,
        cfg(tmp_path),
        tmp_path / "ckpt/best.pt",
        tmp_path / "ckpt/history.json",
        metrics_dir=tmp_path / "m2",
        model_dir=tmp_path / "model2",
        plots_dir=tmp_path / "plots2",
    )
    assert again["card"]["test_metrics"] == first["card"]["test_metrics"]  # same checkpoint, same numbers
    assert again["card"]["training"]["epochs_completed"] == 2 and again["card"]["training"]["train_images"] == 24
