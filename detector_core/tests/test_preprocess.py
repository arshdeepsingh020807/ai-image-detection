import io

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from detector_core import preprocess as pp


def encode(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def coord_image(width: int, height: int) -> Image.Image:
    """Pixel (x, y) holds (x % 256, y % 256, 7), so crops can be checked exactly."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = np.arange(width)[None, :] % 256
    arr[..., 1] = np.arange(height)[:, None] % 256
    arr[..., 2] = 7
    return Image.fromarray(arr)


# ---------- decode_image ----------


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_decode_accepts_supported_formats(fmt):
    img = pp.decode_image(encode(Image.new("RGB", (40, 30), (10, 20, 30)), fmt))
    assert img.mode == "RGB" and img.size == (40, 30)


def test_decode_png_preserves_exact_pixels():
    src = coord_image(32, 32)
    out = pp.decode_image(encode(src, "PNG"))
    assert np.array_equal(np.asarray(out), np.asarray(src))


@pytest.mark.parametrize(
    "data, code",
    [
        (b"", "empty_file"),
        (b"just some text pretending to be a.jpg", "invalid_image"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 10, "invalid_image"),
        (encode(Image.new("RGB", (5, 5)), "GIF"), "unsupported_format"),
        (encode(Image.new("RGB", (5, 5)), "BMP"), "unsupported_format"),
        (b"<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'/>", "invalid_image"),
    ],
)
def test_decode_rejects_bad_input_with_stable_code(data, code):
    with pytest.raises(pp.ImageRejected) as exc:
        pp.decode_image(data)
    assert exc.value.code == code


def test_decode_rejects_truncated_jpeg():
    good = encode(Image.effect_noise((200, 200), 80).convert("RGB"), "JPEG")
    with pytest.raises(pp.ImageRejected) as exc:
        pp.decode_image(good[: len(good) // 2])
    assert exc.value.code == "invalid_image"


def test_decode_rejects_animated_webp_and_apng():
    frames = [Image.new("RGB", (8, 8), c) for c in [(255, 0, 0), (0, 255, 0)]]
    for fmt in ("WEBP", "PNG"):
        data = encode(frames[0], fmt, save_all=True, append_images=frames[1:], duration=50, loop=0)
        with pytest.raises(pp.ImageRejected) as exc:
            pp.decode_image(data)
        assert exc.value.code == "animated_image", fmt


def test_decode_enforces_pixel_limit_before_decoding():
    data = encode(Image.new("1", (3000, 3000)), "PNG")  # tiny file, 9 MP
    with pytest.raises(pp.ImageRejected) as exc:
        pp.decode_image(data, max_pixels=1_000_000)
    assert exc.value.code == "image_too_large"
    assert pp.decode_image(data, max_pixels=9_000_000).size == (3000, 3000)  # boundary is inclusive


def test_decode_decompression_bomb_is_rejected_not_crashed():
    data = encode(Image.new("1", (20000, 20000)), "PNG")  # 400 MP declared, ~KBs on disk
    assert len(data) < 1_000_000
    with pytest.raises(pp.ImageRejected):
        pp.decode_image(data, max_pixels=10**9)


def test_decode_applies_exif_orientation():
    img = Image.new("RGB", (40, 20), (1, 2, 3))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees: stored 40x20 displays as 20x40
    out = pp.decode_image(encode(img, "JPEG", exif=exif))
    assert out.size == (20, 40)


# ---------- to_rgb ----------


def test_to_rgb_flattens_transparency_onto_white():
    img = Image.new("RGBA", (2, 1))
    img.putpixel((0, 0), (255, 0, 0, 0))  # fully transparent -> white
    img.putpixel((1, 0), (255, 0, 0, 255))  # opaque -> red
    out = np.asarray(pp.to_rgb(img))
    assert out[0, 0].tolist() == [255, 255, 255]
    assert out[0, 1].tolist() == [255, 0, 0]


def test_to_rgb_half_transparent_blend_matches_formula():
    img = Image.new("RGBA", (1, 1), (255, 0, 0, 128))
    out = np.asarray(pp.to_rgb(img))[0, 0].astype(int)
    alpha = 128 / 255
    expected = [round(255 * alpha + 255 * (1 - alpha)), round(255 * (1 - alpha)), round(255 * (1 - alpha))]
    assert np.abs(out - expected).max() <= 1


def test_to_rgb_palette_with_transparency_and_grayscale_and_cmyk():
    pal = Image.new("P", (2, 1))
    pal.putpalette([255, 0, 0] + [0, 0, 255] + [0] * 250)
    pal.putpixel((1, 0), 1)
    pal.info["transparency"] = 0
    out = np.asarray(pp.to_rgb(pal))
    assert out[0, 0].tolist() == [255, 255, 255] and out[0, 1].tolist() == [0, 0, 255]

    gray = np.asarray(pp.to_rgb(Image.new("L", (1, 1), 77)))[0, 0]
    assert gray.tolist() == [77, 77, 77]

    cmyk = pp.to_rgb(Image.new("CMYK", (3, 3), (0, 0, 0, 0)))  # no ink -> white
    assert cmyk.mode == "RGB" and np.asarray(cmyk)[0, 0].tolist() == [255, 255, 255]


def test_to_rgb_16bit_scales_to_full_range_and_returns_copy_for_rgb():
    arr = np.array([[0, 32768, 65535]], dtype=np.uint16)
    out = np.asarray(pp.to_rgb(Image.fromarray(arr)))
    assert out[0, 0, 0] == 0 and out[0, 2, 0] == 255 and 126 <= out[0, 1, 0] <= 128
    dark = np.asarray(pp.to_rgb(Image.fromarray(np.full((1, 1), 1000, dtype=np.uint16))))
    assert dark[0, 0, 0] == round(1000 / 65535 * 255)  # fixed scale: a dark image stays dark
    src = Image.new("RGB", (2, 2), (1, 2, 3))
    cp = pp.to_rgb(src)
    assert cp is not src and cp.tobytes() == src.tobytes()


# ---------- resizing and cropping ----------


def test_resize_short_side_noop_when_already_256():
    img = Image.new("RGB", (256, 400))
    assert pp.resize_short_side(img) is img


@pytest.mark.parametrize(
    "size, expected",
    [
        ((512, 1024), (256, 512)),
        ((100, 200), (256, 512)),
        ((300, 300), (256, 256)),
        ((4000, 3000), (341, 256)),
        ((10, 2000), (256, 51200)),
    ],
)
def test_resize_short_side_sizes(size, expected):
    assert pp.resize_short_side(Image.new("RGB", size)).size == expected


def test_center_crop_selects_the_exact_middle_region():
    out = np.asarray(pp.center_crop(coord_image(256, 256)))
    assert out.shape == (224, 224, 3)
    assert out[0, 0].tolist() == [16, 16, 7] and out[-1, -1].tolist() == [239, 239, 7]  # (256-224)//2 = 16


def test_center_crop_non_square_and_odd_margin():
    out = np.asarray(pp.center_crop(coord_image(300, 257)))
    assert out[0, 0].tolist() == [(300 - 224) // 2, (257 - 224) // 2, 7]


def test_random_crop_stays_in_bounds_is_seeded_and_varies():
    img = coord_image(256, 256)
    first = np.asarray(pp.random_crop(img, np.random.default_rng(5)))
    again = np.asarray(pp.random_crop(img, np.random.default_rng(5)))
    assert np.array_equal(first, again)
    rng = np.random.default_rng(0)
    corners = {tuple(np.asarray(pp.random_crop(img, rng))[0, 0, :2]) for _ in range(3000)}
    assert len(corners) > 20
    offsets_x = {x for x, _ in corners}
    offsets_y = {y for _, y in corners}
    assert min(offsets_x) == 0 and max(offsets_x) == 32  # 256 - 224 = 32 is the largest valid offset
    assert min(offsets_y) == 0 and max(offsets_y) == 32


# ---------- array conversion ----------


def test_to_array_matches_hand_computed_normalisation():
    arr = pp.to_array(Image.new("RGB", (2, 3), (255, 0, 128)))
    assert arr.shape == (3, 3, 2) and arr.dtype == np.float32
    expected = [(1 - 0.485) / 0.229, (0 - 0.456) / 0.224, (128 / 255 - 0.406) / 0.225]
    for channel in range(3):
        assert np.allclose(arr[channel], expected[channel], atol=1e-5)


def test_prepare_shape_dtype_and_determinism():
    img = coord_image(500, 333)
    a, b = pp.prepare(img), pp.prepare(img)
    assert a.shape == (3, 224, 224) and a.dtype == np.float32 and np.array_equal(a, b)


def test_prepare_on_256_image_equals_manual_centre_crop():
    img = coord_image(256, 256)
    manual = pp.to_array(img.crop((16, 16, 240, 240)))
    assert np.array_equal(pp.prepare(img), manual)


@settings(max_examples=60, deadline=None)
@given(w=st.integers(1, 700), h=st.integers(1, 700), mode=st.sampled_from(["RGB", "L", "RGBA", "P", "CMYK"]))
def test_prepare_invariants_for_any_size_and_mode(w, h, mode):
    arr = pp.prepare(pp.to_rgb(Image.new(mode, (w, h))))
    assert arr.shape == (3, 224, 224) and arr.dtype == np.float32 and np.isfinite(arr).all()
    assert arr.min() > -3 and arr.max() < 3.0


def test_downscale_is_antialiased_not_point_sampled():
    # 1-px checkerboard of 0/255: a proper filter averages to mid-grey, point sampling keeps 0 or 255.
    yy, xx = np.indices((512, 512))
    board = Image.fromarray((((xx + yy) % 2) * 255).astype(np.uint8)).convert("RGB")
    out = np.asarray(pp.resize_short_side(board))[8:-8, 8:-8]
    assert 100 < out.mean() < 155 and out.std() < 40


def test_upscale_is_smooth_not_blocky():
    ramp = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))
    out = np.asarray(pp.resize_short_side(Image.fromarray(ramp).convert("RGB")))[..., 0]
    steps = np.abs(np.diff(out[64].astype(int)))
    assert steps.max() <= 4  # nearest-neighbour would jump by ~2 then 0; bicubic keeps a steady ramp
    assert (steps > 0).mean() > 0.9
