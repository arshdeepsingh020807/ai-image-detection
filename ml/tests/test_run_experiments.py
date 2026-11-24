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
