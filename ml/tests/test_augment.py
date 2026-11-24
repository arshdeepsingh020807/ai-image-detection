import numpy as np
import pytest
from PIL import Image

from detector_core import preprocess as pp
from ml.preprocessing import augment as aug

OFF = aug.AugmentConfig(p_resample=0, p_jpeg=0, p_flip=0, p_color=0)


def photo(size=300, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (size // 10, size // 10, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((size, size), Image.Resampling.BICUBIC)


def psnr(a, b):
    mse = np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)
    return 99 if mse == 0 else 10 * np.log10(255**2 / mse)


def test_all_probabilities_zero_equals_plain_resize_and_random_crop():
    img = photo(300)
    out = aug.augment(img, np.random.default_rng(3), OFF)
    rng = np.random.default_rng(3)
    # Draw order in augment(): p_resample, crop offsets (2 ints), then the remaining probability draws.
    rng.random()
    expected = pp.to_array(pp.random_crop(pp.resize_short_side(img), rng))
    assert np.array_equal(out, expected)


def test_output_contract_and_seed_determinism():
    img = photo(512)
    a = aug.augment(img, np.random.default_rng(1))
    b = aug.augment(img, np.random.default_rng(1))
    c = aug.augment(img, np.random.default_rng(2))
    assert a.shape == (3, 224, 224) and a.dtype == np.float32 and np.isfinite(a).all()
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_flip_mirrors_the_crop_exactly():
    img = photo(256)
    cfg = aug.AugmentConfig(p_resample=0, p_jpeg=0, p_flip=1.0, p_color=0)
    flipped = aug.augment(img, np.random.default_rng(4), cfg)
    plain = aug.augment(img, np.random.default_rng(4), OFF)
    assert np.array_equal(flipped, plain[:, :, ::-1])


def test_jpeg_roundtrip_degrades_monotonically_and_returns_rgb():
    img = photo(128)
    scores = [psnr(img, aug.jpeg_roundtrip(img, q)) for q in (20, 50, 80, 95)]
    assert scores == sorted(scores) and scores[0] < scores[-1] - 3
    assert aug.jpeg_roundtrip(img, 90).mode == "RGB" and aug.jpeg_roundtrip(img, 90).size == img.size


def test_resample_roundtrip_keeps_size_and_removes_detail():
    img = photo(200, seed=5)
    noisy = Image.fromarray(np.random.default_rng(0).integers(0, 256, (200, 200, 3), dtype=np.uint8))
    out = aug.resample_roundtrip(noisy, 0.5)
    assert out.size == noisy.size
    assert np.asarray(out, float).std() < np.asarray(noisy, float).std() * 0.8
    assert aug.resample_roundtrip(img, 0.9).size == img.size


def test_quality_range_is_respected(monkeypatch):
    seen = []
    real = aug.jpeg_roundtrip
    monkeypatch.setattr(aug, "jpeg_roundtrip", lambda im, q: (seen.append(q), real(im, q))[1])
    cfg = aug.AugmentConfig(p_resample=0, p_jpeg=1.0, p_flip=0, p_color=0, jpeg_quality=(60, 95))
    rng = np.random.default_rng(0)
    for _ in range(300):
        aug.augment(photo(256), rng, cfg)
    assert min(seen) == 60 and max(seen) == 95  # both ends of the inclusive range occur


def test_colour_jitter_stays_within_configured_range():
    img = Image.new("RGB", (256, 256), (100, 100, 100))
    cfg = aug.AugmentConfig(p_resample=0, p_jpeg=0, p_flip=0, p_color=1.0, color_range=0.1)
    rng = np.random.default_rng(0)
    means = []
    for _ in range(200):
        arr = aug.augment(img, rng, cfg) * pp.STD[:, None, None] + pp.MEAN[:, None, None]
        means.append(arr.mean() * 255)
    assert min(means) >= 85 and max(means) <= 115 and np.std(means) > 1


@pytest.mark.parametrize("p_field", ["p_resample", "p_jpeg", "p_flip", "p_color"])
def test_probabilities_are_honoured(p_field):
    img = photo(256)
    changed = 0
    for seed in range(40):
        cfg = aug.AugmentConfig(**{**OFF.__dict__, p_field: 1.0})
        changed += not np.array_equal(
            aug.augment(img, np.random.default_rng(seed), cfg), aug.augment(img, np.random.default_rng(seed), OFF)
        )
    assert changed >= 38  # with p=1 the operation applies (almost) every time


@pytest.mark.parametrize("axis", [0, 1])
def test_resample_blurs_detail_along_both_axes(axis):
    stripes = (np.indices((200, 200))[axis] % 2 * 255).astype(np.uint8)  # 1-px stripes varying along one axis
    out = np.asarray(aug.resample_roundtrip(Image.fromarray(stripes).convert("RGB"), 0.5), float)
    assert out.std() < np.asarray(stripes, float).std() * 0.5
