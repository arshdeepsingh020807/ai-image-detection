import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from detector_core.preprocess import prepare
from ml.evaluation import evaluate as ev
from ml.evaluation.metrics import band_metrics, choose_band, classification_metrics, fit_temperature
from ml.evaluation.robustness import CONDITIONS


class EdgeEnergy(torch.nn.Module):
    """Deterministic stand-in detector: noisy (AI-labelled) images have more pixel-to-pixel energy.

    Its logits are informative but deliberately mis-scaled, so temperature and band fitting are non-trivial.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        energy = (x[..., 1:] - x[..., :-1]).abs().mean(dim=(1, 2, 3))
        return (12 * (energy - energy.median())).unsqueeze(1)


def write(path, label, fmt, size, seed):
    rng = np.random.default_rng(seed)
    arr = np.asarray(Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).resize(size), dtype=np.int16)
    if label:
        arr = arr + rng.integers(-3, 4, arr.shape)  # weak signal: classes overlap
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path, fmt)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    rows, seed = [], 0
    plan = [
        ("val", "S", "faces", "in_scope", None, 10),
        ("test", "S", "faces", "in_scope", None, 12),
        ("test", "G", "general", "in_scope", None, 6),
        ("external", "X", "faces", "in_scope", "gen_png", 6),
        ("external", "X", "faces", "out_of_scope", "swap", 4),
    ]
    for split, source, domain, scope, generator, n in plan:
        for i in range(n):
            label = i % 2 if generator is None else (1 if i % 2 else 0)
            fmt = "PNG" if (generator == "gen_png" and label) else "JPEG"
            path = root / f"{split}_{source}_{scope}_{i}.{fmt.lower()}"
            size = (512, 512) if fmt == "PNG" else (256, 256)
            write(path, label, fmt, size, seed)
            seed += 1
            rows.append(
                {
                    "path": str(path),
                    "split": split,
                    "source": source,
                    "domain": domain,
                    "scope": scope,
                    "label": label,
                    "generator": "real" if label == 0 else (generator or "g"),
                    "format": fmt,
                    "width": size[0],
                    "height": size[1],
                    "jpeg_quality": 95.0 if fmt == "JPEG" else np.nan,
                    "keep": True,
                }
            )
    manifest = pd.DataFrame(rows)
    model = EdgeEnergy()
    out = root / "metrics"
    files = ev.evaluate(model, manifest, torch.device("cpu"), out, model_name="tiny", robustness_per_class=4, workers=0)
    return model, manifest, out, files


def direct_logits(model, paths, transform=None):
    with torch.no_grad():
        arrays = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            arrays.append(prepare(transform(img) if transform else img))
        return model(torch.from_numpy(np.stack(arrays))).squeeze(1).numpy()


def test_all_contract_files_are_written(setup):
    _, _, out, _ = setup
    for name in ("summary", "roc", "pr", "confusion", "confidence", "external", "robustness"):
        assert json.loads((out / f"{name}.json").read_text())
    assert (out / "test_predictions.csv").exists()


def test_calibration_uses_validation_only_and_matches_independent_computation(setup):
    model, manifest, _, files = setup
    val = manifest[manifest.split == "val"]
    t = fit_temperature(direct_logits(model, val.path), val.label.to_numpy())
    assert files["summary"]["temperature"] == pytest.approx(t, rel=1e-4)
    test_t = fit_temperature(
        direct_logits(model, manifest[manifest.split == "test"].path),
        manifest[manifest.split == "test"].label.to_numpy(),
    )
    assert abs(t - test_t) > 1e-3  # fixture really distinguishes fitting on val from fitting on test
    low, high = choose_band(expit(direct_logits(model, val.path) / t), val.label.to_numpy(), ev.BAND_MAX_ERROR)
    assert files["summary"]["band"] == {"low": pytest.approx(low), "high": pytest.approx(high)}
    test = manifest[manifest.split == "test"]
    p = expit(direct_logits(model, test.path) / t)
    saved = pd.read_csv(setup[2] / "test_predictions.csv")
    assert np.allclose(saved.p_ai.to_numpy(), p, atol=1e-5)
    expected = classification_metrics(test.label.to_numpy(), p)
    assert files["summary"]["test"]["accuracy"] == expected["accuracy"]
    assert files["summary"]["test"]["roc_auc"] == pytest.approx(roc_auc_score(test.label, p))
    assert files["summary"]["test_with_band"] == pytest.approx(band_metrics(p, test.label.to_numpy(), low, high))


def test_curves_and_histogram_are_consistent(setup):
    _, manifest, _, files = setup
    test = manifest[manifest.split == "test"]
    roc, conf = files["roc"], files["confusion"]
    assert roc["fpr"][0] == 0 and roc["tpr"][0] == 0 and roc["fpr"][-1] == 1 and roc["tpr"][-1] == 1
    assert sum(map(sum, conf["matrix"])) == len(test)
    hist = files["confidence"]
    assert sum(hist["real"]) == (test.label == 0).sum() and sum(hist["ai_generated"]) == (test.label == 1).sum()
    assert len(hist["bins"]) == 11 and files["pr"]["recall"][0] <= files["pr"]["recall"][-1]


def test_subgroups_partition_the_test_split(setup):
    _, manifest, _, files = setup
    groups = files["summary"]["test_subgroups"]["domain"]
    assert (
        set(groups) == {"faces", "general"} and sum(g["n"] for g in groups.values()) == (manifest.split == "test").sum()
    )


def test_external_rows_cover_generators_out_of_scope_and_matched_protocol(setup):
    files = setup[3]
    sets = {(s["name"], s["protocol"].split(":")[0]): s for s in files["external"]["sets"]}
    assert ("X", "as_is") in sets and ("X/gen_png", "as_is") in sets and ("X/real", "as_is") in sets
    assert sets[("X", "as_is")]["n"] == 6 and sets[("X/face_swap (out of scope)", "as_is")]["n"] == 4
    matched = sets[("X", "matched")]
    assert "256px" in matched["protocol"] and "q95" in matched["protocol"] and matched["n"] == 6
    assert files["external"]["notes"]


def test_robustness_covers_every_condition_on_a_balanced_subset(setup):
    _, _, _, files = setup
    conditions = files["robustness"]["conditions"]
    assert [c["name"] for c in conditions] == list(CONDITIONS)
    assert all(c["n"] == 8 for c in conditions)  # 4 per class


def test_predict_logits_with_worker_processes_and_a_robustness_transform(setup):
    model, manifest, _, _ = setup
    paths = manifest[manifest.split == "test"].path.tolist()[:6]
    transform = CONDITIONS["jpeg_q50"][1]
    # one batch of 6: the stand-in centres on its batch median, so batches must match the direct computation
    with_workers = ev.predict_logits(model, paths, torch.device("cpu"), transform, batch_size=6, workers=2)
    assert np.allclose(with_workers, direct_logits(model, paths, transform), atol=1e-5)


def test_downsample_curve_keeps_ends_and_limits_size():
    xs, ys = np.linspace(0, 1, 1000), np.linspace(0, 1, 1000) ** 2
    dx, dy = ev.downsample_curve(xs, ys, max_points=50)
    assert len(dx) <= 50 and dx[0] == 0 and dx[-1] == 1 and dy[-1] == 1
    assert ev.downsample_curve(xs[:10], ys[:10]) == (xs[:10].tolist(), ys[:10].tolist())


def test_balanced_subset_caps_each_class():
    frame = pd.DataFrame({"label": [0] * 10 + [1] * 3})
    sub = ev.balanced_subset(frame, 5, 0)
    assert (sub.label == 0).sum() == 5 and (sub.label == 1).sum() == 3
