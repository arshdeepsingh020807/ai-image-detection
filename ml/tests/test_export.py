import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from detector_core.preprocess import MEAN, STD
from ml.scripts.export_onnx import export
from ml.training.model import Detector


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = Detector("mobilenetv3_small_050", pretrained=False).eval()
    with torch.no_grad():
        model.fc.weight.normal_()  # non-trivial head so parity is meaningful
    out = tmp_path_factory.mktemp("model")
    export(model, out, {"version": "test-1", "temperature": 1.7, "band": {"low": 0.3, "high": 0.8}, "trained": True})
    return model, out


def test_onnx_matches_pytorch_for_logit_and_features(exported):
    model, out = exported
    assert not any(m.training for m in model.modules())  # export must not flip the caller's model into train mode
    session = ort.InferenceSession(str(out / "detector.onnx"))
    x = np.random.default_rng(0).normal(size=(3, 3, 224, 224)).astype(np.float32)
    logit, features = session.run(None, {"image": x})
    with torch.no_grad():
        ref_logit, ref_features = model.forward_with_features(torch.from_numpy(x))
    assert logit.shape == (3, 1) and features.shape == tuple(ref_features.shape)
    assert np.abs(logit - ref_logit.numpy()).max() < 1e-4
    assert np.abs(features - ref_features.numpy()).max() < 1e-4


def test_dynamic_batch_gives_same_result_per_item(exported):
    _, out = exported
    session = ort.InferenceSession(str(out / "detector.onnx"))
    x = np.random.default_rng(1).normal(size=(4, 3, 224, 224)).astype(np.float32)
    batched = session.run(["logit"], {"image": x})[0]
    single = np.concatenate([session.run(["logit"], {"image": x[i : i + 1]})[0] for i in range(4)])
    assert np.allclose(batched, single, atol=1e-5)


def test_model_card_contents(exported):
    model, out = exported
    card = json.loads((out / "model_card.json").read_text())
    assert card["architecture"] == "mobilenetv3_small_050" and card["version"] == "test-1" and card["trained"] is True
    assert card["temperature"] == 1.7 and card["band"] == {"low": 0.3, "high": 0.8}
    assert np.allclose(card["cam_weights"], model.fc.weight[0].detach().numpy(), atol=1e-7)
    assert card["cam_bias"] == pytest.approx(model.fc.bias.item())
    assert card["input"]["mean"] == pytest.approx(MEAN.tolist()) and card["input"]["std"] == pytest.approx(STD.tolist())
    assert (card["input"]["short_side"], card["input"]["crop"]) == (256, 224)


def test_logit_reconstructs_from_card_weights_and_features(exported):
    """What the backend does: pool the ONNX feature map and apply card weights -> must equal the ONNX logit."""
    _, out = exported
    card = json.loads((out / "model_card.json").read_text())
    session = ort.InferenceSession(str(out / "detector.onnx"))
    x = np.random.default_rng(2).normal(size=(2, 3, 224, 224)).astype(np.float32)
    logit, features = session.run(None, {"image": x})
    rebuilt = features.mean(axis=(2, 3)) @ np.array(card["cam_weights"]) + card["cam_bias"]
    assert np.allclose(rebuilt, logit[:, 0], atol=1e-4)


def test_untrained_export_defaults_are_marked(tmp_path):
    export(Detector("mobilenetv3_small_050", pretrained=False), tmp_path, {})
    card = json.loads((tmp_path / "model_card.json").read_text())
    assert card["trained"] is False and card["version"] == "untrained-standin" and card["temperature"] == 1.0
