"""Training-time augmentation. Applied identically to both classes so no operation can identify a label.

Deliberately omitted: rotation and shear (interpolation smooths exactly the high-frequency traces the
detector relies on), vertical flips, strong colour jitter and blur.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance

from detector_core.preprocess import random_crop, resize_short_side, to_array


@dataclass(frozen=True)
class AugmentConfig:
    p_resample: float = 0.3
    resample_scale: tuple[float, float] = (0.6, 0.95)
    p_jpeg: float = 0.7
    jpeg_quality: tuple[int, int] = (60, 95)
    p_flip: float = 0.5
    p_color: float = 0.3
    color_range: float = 0.1


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def resample_roundtrip(img: Image.Image, scale: float) -> Image.Image:
    """Down- then up-scale, mimicking an image that was shrunk and re-enlarged somewhere upstream."""
    small = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS)
    return small.resize(img.size, Image.Resampling.BICUBIC)


def augment(img: Image.Image, rng: np.random.Generator, cfg: AugmentConfig = AugmentConfig()) -> np.ndarray:
    """RGB image -> normalised float32 (3, 224, 224) training sample."""
    img = resize_short_side(img)
    if rng.random() < cfg.p_resample:
        img = resample_roundtrip(img, float(rng.uniform(*cfg.resample_scale)))
    img = random_crop(img, rng)
    if rng.random() < cfg.p_flip:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < cfg.p_color:
        img = ImageEnhance.Brightness(img).enhance(1 + float(rng.uniform(-cfg.color_range, cfg.color_range)))
        img = ImageEnhance.Contrast(img).enhance(1 + float(rng.uniform(-cfg.color_range, cfg.color_range)))
    if rng.random() < cfg.p_jpeg:
        img = jpeg_roundtrip(img, int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1)))
    return to_array(img)
