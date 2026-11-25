import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from detector_core.preprocess import prepare
from ml.preprocessing.augment import jpeg_roundtrip
from ml.training import data as D
from ml.training import train as T

ARCH = "mobilenetv3_small_050"


def make_images(tmp_path, n=48, seed=0):
    """Class 1 carries fine high-frequency noise, class 0 is smooth: trivially learnable."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = i % 2
        base = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        img = Image.fromarray(base).resize((256, 256), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.int16)
        if label:
            arr = arr + rng.integers(-40, 41, arr.shape)
        fmt = "PNG" if i % 5 == 0 else "JPEG"
        path = tmp_path / f"{i}.{fmt.lower()}"
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
            path, fmt, **({"quality": 95} if fmt == "JPEG" else {})
        )
        split = "train" if i < n * 2 // 3 else "val"
        rows.append(
            {
                "path": str(path),
                "label": label,
                "format": fmt,
                "domain": "faces" if i % 3 else "general",
                "split": split,
            }
        )
    return pd.DataFrame(rows)


# ---------- data ----------


def test_train_samples_are_deterministic_per_epoch_and_index(tmp_path):
    rows = make_images(tmp_path, 8)
    ds = D.ManifestDataset(rows, train=True, seed=3)
    a, _ = ds[1]
    b, _ = ds[1]
    ds.set_epoch(1)
    c, _ = ds[1]
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert a.shape == (3, 224, 224) and a.dtype == torch.float32


def test_eval_samples_equal_the_serving_preprocessing(tmp_path):
    rows = make_images(tmp_path, 6)
    ds = D.ManifestDataset(rows, train=False)
    for i in range(6):
        x, y = ds[i]
        with Image.open(rows.path[i]) as img:
            expected = prepare(img.convert("RGB"))
        assert torch.equal(x, torch.from_numpy(expected)) and y.item() == rows.label[i]


def test_png_sources_are_jpeg_normalised_only_for_training(tmp_path):
    rows = make_images(tmp_path, 6)
    png_index = int(rows.index[rows.format == "PNG"][0])
    with Image.open(rows.path[png_index]) as img:
        original = img.convert("RGB")
    trained = D.ManifestDataset(rows, train=True).load(png_index)
    evaluated = D.ManifestDataset(rows, train=False).load(png_index)
    assert np.array_equal(np.asarray(trained), np.asarray(jpeg_roundtrip(original, 95)))
    assert np.array_equal(np.asarray(evaluated), np.asarray(original))
    jpg_index = int(rows.index[rows.format == "JPEG"][0])
    with Image.open(rows.path[jpg_index]) as img:
        assert np.array_equal(
            np.asarray(D.ManifestDataset(rows, train=True).load(jpg_index)), np.asarray(img.convert("RGB"))
        )


def test_balanced_weights_hand_case():
    rows = pd.DataFrame({"domain": ["f", "f", "f", "f", "g", "g"], "label": [0, 0, 0, 1, 0, 0]})
    w = D.balanced_weights(rows)
    assert np.allclose(w, np.array([1 / 3, 1 / 3, 1 / 3, 1, 1 / 2, 1 / 2]) / 3)
    for _, idx in rows.groupby(["domain", "label"]).groups.items():
        assert w[list(idx)].sum() == pytest.approx(1 / 3)


def test_balanced_sampler_draws_groups_equally_and_is_seeded():
    rows = pd.DataFrame({"domain": ["f"] * 90 + ["g"] * 10, "label": [0] * 80 + [1] * 10 + [0] * 10})
    draws = list(D.balanced_sampler(rows, seed=1))
    again = list(D.balanced_sampler(rows, seed=1))
    assert draws == again and len(draws) == 100
    many = np.array([i for s in range(30) for i in D.balanced_sampler(rows, seed=s)])
    groups = rows.domain.to_numpy()[many] + rows.label.astype(str).to_numpy()[many]
    shares = pd.Series(groups).value_counts(normalize=True)
    assert all(abs(v - 1 / 3) < 0.03 for v in shares)


def test_numpy_collate_stacks_tuples_and_plain_items():
    batch = [(torch.full((3, 2, 2), float(i)), torch.tensor(float(i % 2))) for i in range(4)]
    x, y = D.numpy_collate(batch)
    assert isinstance(x, np.ndarray) and x.shape == (4, 3, 2, 2) and x.dtype == np.float32
    assert y.tolist() == [0.0, 1.0, 0.0, 1.0] and x[2].max() == 2.0
    single = D.numpy_collate([torch.ones(2), torch.zeros(2)])
    assert single.shape == (2, 2) and single.sum() == 2


def test_multiworker_loader_returns_numpy_batches_equal_to_items(tmp_path):
    rows = make_images(tmp_path, 10)
    dataset = D.ManifestDataset(rows, train=False)
    batches = list(D.make_loader(dataset, batch_size=4, num_workers=2))
    xs = np.concatenate([x for x, _ in batches])
    assert all(isinstance(x, np.ndarray) for x, _ in batches) and xs.shape == (10, 3, 224, 224)
    assert np.array_equal(xs[7], dataset[7][0].numpy())
    assert np.concatenate([y for _, y in batches]).tolist() == rows.label.astype(float).tolist()


# ---------- training helpers ----------


def test_cosine_lr_closed_form():
    assert T.cosine_lr(0, 100, 1.0) == pytest.approx(1.0)
    assert T.cosine_lr(50, 100, 1.0) == pytest.approx(0.5)
    assert T.cosine_lr(100, 100, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert T.cosine_lr(0, 100, 1.0, warmup=10) == pytest.approx(0.1)
    assert T.cosine_lr(9, 100, 1.0, warmup=10) == pytest.approx(1.0)
    assert T.cosine_lr(55, 100, 2.0, warmup=10) == pytest.approx(2.0 * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert T.cosine_lr(500, 100, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_pick_device_respects_explicit_choice_and_auto_returns_available():
    assert T.pick_device("cpu").type == "cpu"
    assert T.pick_device("auto").type in {"cuda", "mps", "cpu"}


def test_sample_caps_proportionally_per_group():
    rows = pd.DataFrame({"domain": ["f"] * 80 + ["g"] * 20, "label": [0, 1] * 50, "x": range(100)})
    capped = T._sample(rows, 50, 0)
    assert len(capped) == 50
    assert (capped.domain == "f").sum() == 40
    assert T._sample(rows, None, 0) is rows and T._sample(rows, 500, 0) is rows


def test_config_from_yaml_with_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("arch: resnet50\nbatch_size: 16\n")
    cfg = T.TrainConfig.from_yaml(path, batch_size=8)
    assert cfg.arch == "resnet50" and cfg.batch_size == 8 and cfg.epochs_head == T.TrainConfig().epochs_head


# ---------- end-to-end training ----------


def cfg(tmp_path, **kw):
    base = {
        "arch": ARCH,
        "pretrained": True,
        "epochs_head": 1,
        "epochs_finetune": 4,
        "lr_head": 3e-3,
        "lr_finetune": 3e-3,
        "batch_size": 8,
        "num_workers": 0,
        "device": "cpu",
        "out_dir": str(tmp_path / "run"),
        "patience": 10,
    }
    return T.TrainConfig(**{**base, **kw})


def test_training_learns_saves_best_checkpoint_and_history(tmp_path):
    rows = make_images(tmp_path)
    result = T.train(cfg(tmp_path), rows, log=lambda *_: None)
    hist = result["history"]
    assert [h["phase"] for h in hist] == ["head", "finetune", "finetune", "finetune", "finetune"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert max(h["val_accuracy"] for h in hist) >= 0.8
    best_epoch = min(hist, key=lambda h: h["val_loss"])["epoch"]
    assert result["best"]["epoch"] == best_epoch
    saved = json.loads((tmp_path / "run/history.json").read_text())["epochs"]
    assert saved == hist and set(saved[0]) >= {
        "epoch",
        "phase",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "lr",
    }
    from ml.training.model import Detector

    model = Detector(ARCH, pretrained=False)
    model.load_state_dict(torch.load(result["checkpoint"]))
    loader = D.make_loader(D.ManifestDataset(rows[rows.split == "val"], train=False), batch_size=8)
    logits, labels = T.predict_logits(model, loader, torch.device("cpu"))
    loss, _ = T._loss_and_accuracy(logits, labels)
    assert loss == pytest.approx(result["best"]["val_loss"], abs=1e-4)  # checkpoint really is the best epoch


def test_progress_file_tracks_the_last_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "PROGRESS_EVERY", 1)
    rows = make_images(tmp_path, 24)
    T.train(cfg(tmp_path, epochs_finetune=1), rows, log=lambda *_: None)
    progress = json.loads((tmp_path / "run/progress.json").read_text())
    assert progress["state"] == "validating" and progress["phase"] == "finetune" and progress["epoch"] == 2
    assert progress["total_epochs"] == 2 and progress["batch"] == progress["batches"] == 16 // 8
    assert progress["images_per_second"] > 0 and 0 <= progress["running_accuracy"] <= 1
    assert not (tmp_path / "run/progress.tmp").exists()


def test_head_phase_leaves_backbone_untouched(tmp_path):
    rows = make_images(tmp_path, 16)
    from ml.training.model import Detector

    reference = Detector(ARCH, pretrained=True).backbone.state_dict()
    result = T.train(cfg(tmp_path, epochs_finetune=0, seed=42), rows, log=lambda *_: None)
    trained = torch.load(result["checkpoint"])
    for k, v in reference.items():
        if k.endswith(("weight", "bias")) and v.dtype.is_floating_point:
            assert torch.equal(trained["backbone." + k], v), k


def test_early_stopping_when_validation_stops_improving(tmp_path):
    rows = make_images(tmp_path, 24)
    result = T.train(cfg(tmp_path, lr_finetune=0.0, patience=2, epochs_finetune=6), rows, log=lambda *_: None)
    phases = [h["phase"] for h in result["history"]]
    assert phases.count("finetune") == 2  # zero learning rate never improves, so it stops after `patience` epochs


def test_empty_validation_split_is_a_clear_error(tmp_path):
    rows = make_images(tmp_path, 9).assign(split="train")
    with pytest.raises(ValueError, match="non-empty train and val"):
        T.train(cfg(tmp_path), rows, log=lambda *_: None)


def test_cosine_lr_stays_at_zero_after_schedule_end():
    assert T.cosine_lr(150, 100, 1.0) == pytest.approx(0.0, abs=1e-12)  # unclamped cosine would climb back to 0.5


def test_loss_and_accuracy_hand_case():
    logits = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    labels = np.array([1, 0, 0], dtype=np.float32)
    loss, acc = T._loss_and_accuracy(logits, labels)
    expected = -np.mean([np.log(1 / (1 + np.exp(-0.5))), np.log(1 - 1 / (1 + np.exp(0.5))), np.log(0.5)])
    assert loss == pytest.approx(expected, rel=1e-5) and acc == pytest.approx(2 / 3)  # logit 0 counts as AI


@pytest.mark.parametrize("cuda, mps, expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_auto_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert T.pick_device("auto").type == expected


def test_accuracy_threshold_is_logit_zero():
    _, acc = T._loss_and_accuracy(np.array([0.5, 0.0, -0.2], dtype=np.float32), np.array([1, 1, 0], dtype=np.float32))
    assert acc == 1.0  # 0.5 and 0.0 are AI, -0.2 is real
