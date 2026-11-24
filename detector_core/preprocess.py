"""Image decoding and preprocessing shared by training, evaluation, export and serving.

Pure Pillow + NumPy so the production service never imports PyTorch.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

SHORT_SIDE = 256
CROP = 224
ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageRejected(ValueError):
    """The bytes are not an image we accept. `code` is a stable machine-readable reason."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def decode_image(
    data: bytes,
    *,
    max_pixels: int = 40_000_000,
    allowed_formats: frozenset[str] = ALLOWED_FORMATS,
) -> Image.Image:
    """Decode bytes to a fully loaded RGB image, judging by content, never by file name."""
    if not data:
        raise ImageRejected("empty_file", "The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            if fmt not in allowed_formats:
                raise ImageRejected("unsupported_format", f"Unsupported image format: {fmt}.")
            width, height = img.size
            if width < 1 or height < 1 or width * height > max_pixels:
                raise ImageRejected("image_too_large", "Image dimensions exceed the allowed limit.")
            if getattr(img, "n_frames", 1) > 1:
                raise ImageRejected("animated_image", "Animated or multi-frame images are not supported.")
            img.load()
            return to_rgb(ImageOps.exif_transpose(img))
    except ImageRejected:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageRejected("invalid_image", "The file is not a valid, readable image.") from exc


def to_rgb(img: Image.Image) -> Image.Image:
    """Consistent RGB: transparency is flattened onto white, other modes are converted."""
    if img.mode == "RGB":
        return img.copy()
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.asarray(img, dtype=np.float32)
        return Image.fromarray(np.clip(arr / 65535 * 255, 0, 255).round().astype(np.uint8)).convert("RGB")
    return img.convert("RGB")


def resize_short_side(img: Image.Image, size: int = SHORT_SIDE) -> Image.Image:
    """Scale so the shorter side equals `size`; untouched when it already does.

    Training data is 256 px, so large uploads are brought to the same scale instead of
    cropping a tiny patch out of a multi-megapixel photo.
    """
    width, height = img.size
    short = min(width, height)
    if short == size:
        return img
    scale = size / short
    new_size = (max(size, round(width * scale)), max(size, round(height * scale)))
    return img.resize(new_size, Image.Resampling.BICUBIC)


def center_crop(img: Image.Image, size: int = CROP) -> Image.Image:
    width, height = img.size
    left, top = (width - size) // 2, (height - size) // 2
    return img.crop((left, top, left + size, top + size))


def random_crop(img: Image.Image, rng: np.random.Generator, size: int = CROP) -> Image.Image:
    width, height = img.size
    left = int(rng.integers(0, width - size + 1))
    top = int(rng.integers(0, height - size + 1))
    return img.crop((left, top, left + size, top + size))


def to_array(img: Image.Image) -> np.ndarray:
    """RGB image -> normalised float32 array of shape (3, H, W)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1).astype(np.float32, copy=False)


def prepare(img: Image.Image, crop: Callable[[Image.Image], Image.Image] = center_crop) -> np.ndarray:
    """Deterministic evaluation/serving path: short side 256, centre crop 224, normalise."""
    return to_array(crop(resize_short_side(img)))
