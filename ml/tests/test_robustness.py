import numpy as np
import pytest
from PIL import Image

from ml.evaluation import robustness as rb


def photo(w=400, h=300, seed=0):
    base = np.random.default_rng(seed).integers(0, 256, (h // 10, w // 10, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((w, h), Image.Resampling.BICUBIC)


@pytest.mark.parametrize("name", list(rb.CONDITIONS))
def test_every_condition_returns_rgb_is_deterministic_and_described(name):
    description, fn = rb.CONDITIONS[name]
    img = photo()
    a, b = fn(img), fn(img)
    assert description and a.mode == "RGB" and np.array_equal(np.asarray(a), np.asarray(b))


def test_original_is_identity_and_others_change_the_image():
    img = photo()
    assert np.array_equal(np.asarray(rb.CONDITIONS["original"][1](img)), np.asarray(img))
    for name, (_, fn) in rb.CONDITIONS.items():
        if name != "original":
            out = fn(img)
            assert out.size != img.size or not np.array_equal(np.asarray(out), np.asarray(img)), name


def test_geometry_of_resizing_and_cropping_conditions():
    img = photo(400, 300)
    assert rb.CONDITIONS["low_resolution"][1](img).size == (200, 150)
    assert rb.CONDITIONS["very_low_resolution"][1](img).size == (100, 75)
    assert rb.CONDITIONS["cropped"][1](img).size == (240, 180)
    assert rb.CONDITIONS["screenshot"][1](img).size == (300, 225)


def test_center_crop_fraction_takes_the_middle():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[40:60, 40:60] = 255
    out = np.asarray(rb.center_crop_fraction(Image.fromarray(arr), 0.2))
    assert out.shape == (20, 20, 3) and out.min() == 255


def test_jpeg_conditions_degrade_in_order():
    img = photo()
    ref = np.asarray(img, float)
    errors = [np.abs(np.asarray(rb.CONDITIONS[f"jpeg_q{q}"][1](img), float) - ref).mean() for q in (90, 70, 50, 30)]
    assert errors == sorted(errors)


def test_edited_is_brighter():
    img = Image.new("RGB", (64, 64), (100, 100, 100))
    assert np.asarray(rb.edited(img)).mean() > 105


def test_match_to_reference_downscales_then_compresses_and_never_upscales():
    big = photo(1024, 1024)
    out = rb.match_to_reference(big, size=200, quality=75)
    expected = rb.jpeg_roundtrip(big.resize((200, 200), Image.Resampling.LANCZOS), 75)
    assert out.size == (200, 200) and np.array_equal(np.asarray(out), np.asarray(expected))
    assert not np.array_equal(np.asarray(out), np.asarray(rb.match_to_reference(big, size=200, quality=95)))
    small = photo(150, 150)
    assert rb.match_to_reference(small, size=200, quality=75).size == (150, 150)
