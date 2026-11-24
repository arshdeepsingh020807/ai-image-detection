import base64
import io
import math
import tempfile

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image

from backend.app.schemas.responses import Band
from backend.tests.helpers import CARD, Reference, app_of, assert_error, declared_size_png, encode, noise_image, upload

SVG = b"<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'><rect width='100' height='100'/></svg>"


def animated(fmt: str) -> bytes:
    frames = [Image.new("RGB", (16, 16), colour) for colour in [(255, 0, 0), (0, 0, 255)]]
    return encode(frames[0], fmt, save_all=True, append_images=frames[1:], duration=40, loop=0)


# ---------- successful predictions ----------


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_supported_formats_match_independent_inference(shared_client, reference: Reference, fmt):
    data = encode(noise_image(300, 200, seed=5), fmt)
    response = upload(shared_client, data, filename=f"photo.{fmt.lower()}")
    assert response.status_code == 200, response.text
    body = response.json()
    expected_p_ai = reference.p_ai(data)
    assert body["probabilities"]["ai_generated"] == pytest.approx(expected_p_ai, abs=1e-6)
    assert body["probabilities"]["real"] == pytest.approx(1 - expected_p_ai, abs=1e-6)
    assert body["probabilities"]["real"] + body["probabilities"]["ai_generated"] == pytest.approx(1.0, abs=1e-12)
    assert body["confidence"] == pytest.approx(max(expected_p_ai, 1 - expected_p_ai), abs=1e-6)
    assert body["image"] == {"width": 300, "height": 200, "format": fmt, "size_bytes": len(data)}
    assert body["band"] == CARD["band"]
    assert body["model"] == {"architecture": CARD["architecture"], "version": CARD["version"], "trained": False}
    assert body["explanation"] is None


def test_response_shape_matches_contract(shared_client):
    body = upload(shared_client, encode(noise_image(50, 60), "PNG")).json()
    assert set(body) == {"label", "probabilities", "confidence", "band", "image", "explanation", "timing_ms", "model"}
    assert set(body["timing_ms"]) == {"decode", "preprocess", "inference", "explain", "total"}
    parts = sum(body["timing_ms"][k] for k in ("decode", "preprocess", "inference", "explain"))
    assert body["timing_ms"]["total"] >= parts > 0 and body["timing_ms"]["explain"] < 1.0
    assert body["label"] in ("real", "ai_generated", "inconclusive")


def test_temperature_is_applied(shared_client, reference: Reference, monkeypatch):
    card = app_of(shared_client).state.runtime.detector.card
    monkeypatch.setattr(card, "temperature", 0.25)
    data = encode(noise_image(128, 128, seed=8), "PNG")
    logit = reference.outputs(data)[0]
    body = upload(shared_client, data).json()
    assert body["probabilities"]["ai_generated"] == pytest.approx(1 / (1 + math.exp(-logit / 0.25)), abs=1e-6)


@pytest.mark.parametrize(
    "offset_low, offset_high, label",
    [(+1e-3, +2e-3, "real"), (-1e-3, +1e-3, "inconclusive"), (-2e-3, -1e-3, "ai_generated"), (0.0, 0.0, "real")],
)
def test_label_follows_the_band(shared_client, reference: Reference, monkeypatch, offset_low, offset_high, label):
    """Band edges are placed around this image's own (independently computed) p_ai to force each label."""
    data = encode(noise_image(96, 96, seed=4), "PNG")
    p_ai = reference.p_ai(data)
    band = Band(low=p_ai + offset_low, high=p_ai + offset_high)
    monkeypatch.setattr(app_of(shared_client).state.runtime.detector.card, "band", band)
    body = upload(shared_client, data).json()
    assert body["label"] == label and body["band"] == band.model_dump()


def test_wrong_extension_with_valid_content(shared_client):
    response = upload(shared_client, encode(noise_image(40, 40), "PNG"), filename="report.pdf")
    assert response.status_code == 200 and response.json()["image"]["format"] == "PNG"


def test_explain_returns_heatmap_and_hand_computed_region(shared_client):
    response = upload(shared_client, encode(noise_image(1000, 500), "JPEG"), explain="true")
    explanation = response.json()["explanation"]
    # 1000x500 -> 512x256 -> crop at (144, 16): x = 144/512, y = 16/256, w = 224/512, h = 224/256
    assert explanation["region"] == {"x": 0.28125, "y": 0.0625, "width": 0.4375, "height": 0.875}
    heatmap = Image.open(io.BytesIO(base64.b64decode(explanation["heatmap_png_base64"], validate=True)))
    assert heatmap.format == "PNG" and heatmap.mode == "RGBA" and heatmap.size == (224, 224)
    assert explanation["method"] == "grad-cam" and "not proof" in explanation["note"]
    assert response.json()["timing_ms"]["explain"] > 0


@pytest.mark.parametrize("value, explained", [("TRUE", True), (" false ", False), ("", False)])
def test_explain_flag_parsing(shared_client, value, explained):
    body = upload(shared_client, encode(noise_image(30, 30), "PNG"), explain=value).json()
    assert (body["explanation"] is not None) is explained


def test_explain_rejects_other_values(shared_client):
    assert_error(upload(shared_client, encode(noise_image(30, 30), "PNG"), explain="yes"), 400, "invalid_request")


def test_transparent_and_grayscale_inputs(shared_client):
    for img in (Image.new("RGBA", (64, 64), (255, 0, 0, 0)), Image.new("L", (64, 64), 100)):
        assert upload(shared_client, encode(img, "PNG")).status_code == 200


# ---------- rejected uploads ----------


@pytest.mark.parametrize(
    "data, filename, status, code",
    [
        (b"", "empty.png", 400, "empty_file"),
        (b"this is plain text, not a picture", "photo.jpg", 400, "invalid_image"),
        (SVG, "vector.svg", 400, "invalid_image"),
        (encode(Image.new("RGB", (8, 8)), "BMP"), "image.bmp", 415, "unsupported_format"),
        (encode(Image.new("RGB", (8, 8)), "TIFF"), "image.tiff", 415, "unsupported_format"),
        (animated("GIF"), "anim.gif", 415, "unsupported_format"),
        (animated("WEBP"), "anim.webp", 415, "animated_image"),
        (animated("PNG"), "anim.png", 415, "animated_image"),
    ],
)
def test_rejections(shared_client, data, filename, status, code):
    assert_error(upload(shared_client, data, filename=filename), status, code)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_truncated_images(shared_client, fmt):
    data = encode(noise_image(200, 200), fmt)
    assert_error(upload(shared_client, data[: len(data) // 2]), 400, "invalid_image")


def test_decompression_bomb_is_refused_cheaply(shared_client):
    data = declared_size_png(30000, 30000)  # 900 MP declared, beyond Pillow's own bomb limit
    assert len(data) < 200
    assert_error(upload(shared_client, data), 413, "image_too_large")


def test_huge_dimensions_over_the_pixel_limit(make_client):
    client = make_client(max_image_pixels=1_000_000)
    assert_error(upload(client, encode(Image.new("RGB", (1001, 1000)), "PNG")), 413, "image_too_large")
    assert upload(client, encode(Image.new("RGB", (1000, 1000)), "PNG")).status_code == 200


def test_default_pixel_limit_rejects_49_megapixels(shared_client):
    assert_error(upload(shared_client, declared_size_png(7000, 7000)), 413, "image_too_large")


def test_oversized_file(make_client):
    client = make_client(max_upload_bytes=1000)
    assert_error(upload(client, b"\x00" * 1001), 413, "file_too_large")
    exactly = encode(Image.new("RGB", (10, 10)), "PNG")
    assert len(exactly) < 1000 and upload(client, exactly).status_code == 200


def test_missing_file_field(shared_client):
    assert_error(shared_client.post("/api/predict", data={"explain": "true"}), 400, "missing_file")
    assert_error(shared_client.post("/api/predict"), 400, "missing_file")
    wrong_name = {"image": ("a.png", encode(noise_image(8, 8), "PNG"))}
    assert_error(shared_client.post("/api/predict", files=wrong_name), 400, "missing_file")


def test_json_body_is_not_an_upload(shared_client):
    assert_error(shared_client.post("/api/predict", json={"file": "aGVsbG8="}), 400, "missing_file")


# ---------- model unavailable ----------


def test_missing_model_degrades_every_model_route(make_client, tmp_path):
    client = make_client(model_dir=tmp_path / "no-model")
    health = client.get("/api/health")
    assert health.status_code == 503
    assert health.json()["status"] == "degraded" and health.json()["model_loaded"] is False
    assert health.json()["model_version"] is None
    assert_error(upload(client, encode(noise_image(20, 20), "PNG")), 503, "model_unavailable")
    assert_error(client.get("/api/model"), 503, "model_unavailable")
    assert client.get("/api/metrics").status_code == 200


# ---------- resources ----------


def test_no_temporary_files_even_for_large_uploads(make_client, tmp_path, monkeypatch):
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    client = make_client()
    data = encode(noise_image(1400, 1400, seed=2), "PNG")  # ~5.9 MB, far above Starlette's 1 MB spool threshold
    assert len(data) > 5_000_000
    assert upload(client, data, explain="true").status_code == 200
    assert list(spool.iterdir()) == []


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    fmt=st.sampled_from(["PNG", "JPEG", "WEBP"]),
    cut=st.integers(0, 400),
    flips=st.lists(st.tuples(st.integers(0, 399), st.integers(1, 255)), max_size=6),
)
def test_corrupted_uploads_never_cause_server_errors(shared_client, fmt, cut, flips):
    data = bytearray(encode(noise_image(24, 24, seed=1), fmt))
    for position, mask in flips:
        data[position % len(data)] ^= mask
    response = upload(shared_client, bytes(data[: max(cut, 1) if cut < len(data) else len(data)]))
    assert response.status_code in (200, 400, 413, 415), response.text
    assert ("error" in response.json()) is (response.status_code != 200)


def test_model_card_route(shared_client):
    body = shared_client.get("/api/model").json()
    assert "cam_weights" not in body and "cam_bias" not in body
    assert body["version"] == CARD["version"] and body["onnx_size_mb"] > 1
    assert np.isclose(body["temperature"], CARD["temperature"])
