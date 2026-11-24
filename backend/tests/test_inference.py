import base64
import io
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from backend.app.schemas.responses import Band
from backend.app.services import inference
from backend.app.services.inference import (
    analyse,
    calibrated_probability,
    crop_region,
    decode,
    label_for,
    probabilities,
)
from backend.app.services.model import load_detector
from backend.tests.helpers import MODEL_DIR, declared_size_png, encode, noise_image
from detector_core.preprocess import ImageRejected, resize_short_side


@pytest.fixture(scope="module")
def detector():
    return load_detector(MODEL_DIR, threads=1)


# ---------- calibrated_probability ----------


@pytest.mark.parametrize(
    "logit, temperature, expected",
    [
        (0.0, 1.0, 0.5),
        (2.0, 2.0, 0.7310585786300049),  # sigmoid(1)
        (-2.0, 2.0, 0.2689414213699951),  # sigmoid(-1)
        (3.0, 1.0, 0.9525741268224334),
        (-3.0, 0.5, 0.0024726231566347743),  # sigmoid(-6)
    ],
)
def test_calibrated_probability_matches_hand_computed_sigmoid(logit, temperature, expected):
    assert calibrated_probability(logit, temperature) == pytest.approx(expected, abs=1e-12)


def test_calibrated_probability_is_stable_at_extremes():
    assert calibrated_probability(10_000.0, 1.0) == 1.0
    assert calibrated_probability(-10_000.0, 1.0) == 0.0
    assert 0 < calibrated_probability(-700.0, 1.0) < 1e-300


def test_temperature_softens_but_never_flips_the_decision():
    sharp, soft = calibrated_probability(2.0, 1.0), calibrated_probability(2.0, 4.0)
    assert 0.5 < soft < sharp


@given(st.floats(-50, 50), st.floats(0.05, 20))
def test_calibrated_probability_symmetry_property(logit, temperature):
    p = calibrated_probability(logit, temperature)
    assert 0.0 <= p <= 1.0
    assert p + calibrated_probability(-logit, temperature) == pytest.approx(1.0, abs=1e-12)
    assert p == pytest.approx(0.5 * (1 + math.tanh(logit / temperature / 2)), abs=1e-12)  # sigmoid via tanh


def test_probabilities_sum_to_one():
    p = probabilities(0.3)
    assert p.ai_generated == 0.3 and p.real == pytest.approx(0.7) and p.real + p.ai_generated == pytest.approx(1.0)


# ---------- label_for ----------


@pytest.mark.parametrize(
    "p_ai, expected",
    [
        (0.0, "real"),
        (0.3, "real"),
        (0.3000001, "inconclusive"),
        (0.5, "inconclusive"),
        (0.6999999, "inconclusive"),
        (0.7, "ai_generated"),
        (1.0, "ai_generated"),
    ],
)
def test_label_band_boundaries_are_inclusive(p_ai, expected):
    assert label_for(p_ai, Band(low=0.3, high=0.7)) == expected


def test_label_degenerate_band_never_inconclusive():
    band = Band(low=0.5, high=0.5)
    assert [label_for(p, band) for p in (0.4999, 0.5, 0.5001)] == ["real", "real", "ai_generated"]


def test_band_rejects_inverted_or_out_of_range_bounds():
    for low, high in [(0.8, 0.2), (-0.1, 0.5), (0.5, 1.1)]:
        with pytest.raises(ValueError):
            Band(low=low, high=high)


# ---------- crop_region ----------


@pytest.mark.parametrize(
    "size, expected",
    [
        # 1000x500: short side 500 -> scale 0.512 -> 512x256; crop offset ((512-224)//2, (256-224)//2) = (144, 16)
        ((1000, 500), (144 / 512, 16 / 256, 224 / 512, 224 / 256)),
        ((500, 1000), (16 / 256, 144 / 512, 224 / 256, 224 / 512)),
        ((256, 256), (16 / 256, 16 / 256, 224 / 256, 224 / 256)),
        ((224, 224), (16 / 256, 16 / 256, 224 / 256, 224 / 256)),  # upscaled to 256 first
        # 4000x3000 -> 341x256 (round(4000*256/3000) = 341); offset ((341-224)//2, 16) = (58, 16)
        ((4000, 3000), (58 / 341, 16 / 256, 224 / 341, 224 / 256)),
    ],
)
def test_crop_region_hand_computed(size, expected):
    region = crop_region(*size)
    assert (region.x, region.y, region.width, region.height) == pytest.approx(expected, abs=1e-12)


@settings(max_examples=150, deadline=None)
@given(st.integers(1, 900), st.integers(1, 900))
def test_crop_region_matches_the_real_preprocessing_geometry(width, height):
    """Oracle: the size detector_core's resize actually produces, plus its centre-crop offsets."""
    resized_w, resized_h = resize_short_side(Image.new("L", (width, height))).size
    region = crop_region(width, height)
    assert region.width == pytest.approx(224 / resized_w) and region.height == pytest.approx(224 / resized_h)
    assert region.x == pytest.approx(((resized_w - 224) // 2) / resized_w)
    assert region.y == pytest.approx(((resized_h - 224) // 2) / resized_h)


@given(st.integers(1, 100_000), st.integers(1, 100_000))
def test_crop_region_is_a_centred_sub_rectangle(width, height):
    region = crop_region(width, height)
    for offset, extent in ((region.x, region.width), (region.y, region.height)):
        assert offset >= 0 and 0 < extent <= 224 / 256 and offset + extent <= 1
        assert abs(offset + extent / 2 - 0.5) <= 1 / 256  # centred to within one resized pixel
    short_extent = region.width if width <= height else region.height
    assert short_extent == pytest.approx(224 / 256)


# ---------- decode ----------


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_decode_reports_source_format(fmt):
    image, source_format = decode(encode(noise_image(30, 20), fmt), 1000)
    assert source_format == fmt and image.size == (30, 20) and image.mode == "RGB"


def test_decode_maps_decompression_bomb_to_image_too_large():
    data = declared_size_png(20000, 20000)  # 400 MP declared, beyond Pillow's hard limit
    with pytest.raises(ImageRejected) as exc:
        decode(data, 40_000_000)
    assert exc.value.code == "image_too_large"


def test_decode_keeps_other_rejection_codes():
    with pytest.raises(ImageRejected) as exc:
        decode(b"not an image", 1000)
    assert exc.value.code == "invalid_image"


# ---------- analyse ----------


def test_analyse_logit_matches_an_independent_onnx_session(detector, reference):
    ramp = np.tile(np.linspace(0, 255, 320, dtype=np.uint8), (240, 1))  # left-right asymmetric on purpose
    data = encode(Image.fromarray(ramp).convert("RGB"), "PNG")
    result = analyse(detector, data, explain=False, max_pixels=10**7)
    assert result.logit == pytest.approx(reference.outputs(data)[0], abs=1e-6)
    assert result.explanation is None
    assert result.image.model_dump() == {"width": 320, "height": 240, "format": "PNG", "size_bytes": len(data)}
    assert set(result.timing_ms) == {"decode", "preprocess", "inference", "explain"}
    assert all(value >= 0 for value in result.timing_ms.values())


def test_analyse_explanation_is_heatmap_and_region(detector):
    result = analyse(detector, encode(noise_image(1000, 500), "JPEG"), explain=True, max_pixels=10**7)
    assert result.explanation is not None
    assert result.explanation.region.model_dump() == pytest.approx(
        {"x": 0.28125, "y": 0.0625, "width": 0.4375, "height": 0.875}
    )
    heatmap = Image.open(io.BytesIO(base64.b64decode(result.explanation.heatmap_png_base64)))
    assert heatmap.format == "PNG" and heatmap.mode == "RGBA" and heatmap.size == (224, 224)
    assert result.explanation.method == "grad-cam" and result.explanation.note == inference.EXPLANATION_NOTE


def test_analyse_uses_the_displayed_orientation(detector):
    exif = Image.Exif()
    exif[0x0112] = 6  # stored 400x200, displayed 200x400
    result = analyse(detector, encode(noise_image(400, 200), "JPEG", exif=exif), explain=True, max_pixels=10**7)
    assert (result.image.width, result.image.height) == (200, 400)
    assert result.explanation is not None and result.explanation.region.height == pytest.approx(224 / 512)


def test_analyse_enforces_pixel_limit(detector):
    with pytest.raises(ImageRejected) as exc:
        analyse(detector, encode(Image.new("RGB", (100, 100)), "PNG"), explain=False, max_pixels=9_999)
    assert exc.value.code == "image_too_large"
    assert analyse(detector, encode(Image.new("RGB", (100, 100)), "PNG"), explain=False, max_pixels=10_000)


def test_analyse_timings_cover_the_stages(detector, monkeypatch):
    ticks = iter([1.0, 1.002, 1.005, 1.015, 1.0155])
    monkeypatch.setattr(inference.time, "perf_counter", lambda: next(ticks))
    timing = analyse(detector, encode(noise_image(50, 50), "PNG"), explain=False, max_pixels=10**7).timing_ms
    assert timing == pytest.approx({"decode": 2.0, "preprocess": 3.0, "inference": 10.0, "explain": 0.5})


def test_analyse_deterministic_and_input_sensitive(detector):
    a, b = encode(noise_image(64, 64, seed=1), "PNG"), encode(noise_image(64, 64, seed=2), "PNG")
    first, again = (
        analyse(detector, a, explain=False, max_pixels=10**7),
        analyse(detector, a, explain=False, max_pixels=10**7),
    )
    assert first.logit == again.logit
    assert first.logit != analyse(detector, b, explain=False, max_pixels=10**7).logit
    assert np.isfinite(first.logit)
