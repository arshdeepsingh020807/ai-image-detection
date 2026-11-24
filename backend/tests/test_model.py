import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.services.model import load_detector, try_load_detector
from backend.tests.helpers import CARD, MODEL_DIR, Reference, encode, noise_image
from detector_core.preprocess import decode_image, prepare


def copy_model(tmp_path: Path, **card_changes) -> Path:
    shutil.copy(MODEL_DIR / "detector.onnx", tmp_path / "detector.onnx")
    (tmp_path / "model_card.json").write_text(json.dumps({**CARD, **card_changes}))
    return tmp_path


@pytest.fixture(scope="module")
def detector():
    return load_detector(MODEL_DIR, threads=1)


def test_public_card_hides_cam_fields_and_adds_size(detector):
    public = detector.public_card
    assert "cam_weights" not in public and "cam_bias" not in public
    expected = {key: value for key, value in CARD.items() if key not in ("cam_weights", "cam_bias")}
    assert {key: value for key, value in public.items() if key != "onnx_size_mb"} == expected
    size_bytes = (MODEL_DIR / "detector.onnx").stat().st_size
    assert public["onnx_size_mb"] == pytest.approx(size_bytes / 1048576, abs=0.001)


def test_card_fields_parsed(detector):
    assert detector.card.temperature == CARD["temperature"] and detector.card.version == CARD["version"]
    assert detector.card.band.low == CARD["band"]["low"] and detector.card.trained is CARD["trained"]
    assert detector.cam_weights.dtype == np.float32 and detector.cam_weights.shape == (len(CARD["cam_weights"]),)


def test_run_matches_direct_onnxruntime(detector, reference: Reference):
    data = encode(noise_image(257, 300, seed=11), "PNG")
    logit, features = detector.run(prepare(decode_image(data)))
    ref_logit, ref_features = reference.outputs(data)
    assert logit == pytest.approx(ref_logit, abs=1e-5)
    assert features.shape == (len(CARD["cam_weights"]), 7, 7)
    assert np.allclose(features, ref_features, atol=1e-5)


def test_session_uses_the_configured_thread_count(tmp_path):
    detector = load_detector(copy_model(tmp_path), threads=2)
    options = detector.session.get_session_options()
    assert options.intra_op_num_threads == 2 and options.inter_op_num_threads == 1
    assert detector.session.get_providers() == ["CPUExecutionProvider"]


def test_mismatched_card_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        load_detector(copy_model(tmp_path, cam_weights=[0.1, 0.2]), threads=1)


@pytest.mark.parametrize("change", [{"temperature": 0}, {"band": {"low": 0.9, "high": 0.1}}, {"version": None}])
def test_invalid_card_values_are_rejected(tmp_path, change):
    with pytest.raises(ValidationError):
        load_detector(copy_model(tmp_path, **change), threads=1)


def test_try_load_returns_detector_when_valid(caplog):
    with caplog.at_level(logging.INFO, logger="backend"):
        assert try_load_detector(MODEL_DIR, threads=1) is not None
    assert any(record.getMessage() == "model loaded" for record in caplog.records)


@pytest.mark.parametrize("breakage", ["missing_dir", "missing_onnx", "corrupt_onnx", "corrupt_card", "wrong_card"])
def test_try_load_degrades_instead_of_raising(tmp_path, caplog, breakage):
    model_dir = copy_model(tmp_path)
    if breakage == "missing_dir":
        model_dir = tmp_path / "nope"
    elif breakage == "missing_onnx":
        (model_dir / "detector.onnx").unlink()
    elif breakage == "corrupt_onnx":
        (model_dir / "detector.onnx").write_bytes(b"\x00not a protobuf" * 10)
    elif breakage == "corrupt_card":
        (model_dir / "model_card.json").write_text("{broken")
    else:
        (model_dir / "model_card.json").write_text(json.dumps({**CARD, "cam_weights": [1.0]}))
    with caplog.at_level(logging.ERROR, logger="backend"):
        assert try_load_detector(model_dir, threads=1) is None
    assert [record.getMessage() for record in caplog.records] == ["model unavailable"]
    assert caplog.records[0].exc_info is not None
