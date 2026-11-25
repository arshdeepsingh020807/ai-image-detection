"""Image perturbations for the robustness suite and the matched external protocol. All are deterministic."""

from __future__ import annotations

import io
from collections.abc import Callable
from functools import partial

from PIL import Image, ImageEnhance, ImageFilter

from ml.preprocessing.augment import jpeg_roundtrip

Transform = Callable[[Image.Image], Image.Image]


def downscale(img: Image.Image, factor: float) -> Image.Image:
    return img.resize((max(1, round(img.width * factor)), max(1, round(img.height * factor))), Image.Resampling.LANCZOS)


def center_crop_fraction(img: Image.Image, fraction: float) -> Image.Image:
    w, h = round(img.width * fraction), round(img.height * fraction)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def screenshot_like(img: Image.Image) -> Image.Image:
    """Displayed at 75 % size, captured losslessly: resampling plus PNG, no JPEG traces."""
    buf = io.BytesIO()
    downscale(img, 0.75).filter(ImageFilter.UnsharpMask(radius=1, percent=60, threshold=2)).save(buf, "PNG")
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def edited(img: Image.Image) -> Image.Image:
    """Typical light photo edit: +15 % brightness, +10 % contrast, then saved as JPEG q90."""
    img = ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(1.15)).enhance(1.10)
    return jpeg_roundtrip(img, 90)


def identity(img: Image.Image) -> Image.Image:
    return img


def blurred(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(1))


# Named functions and partials only: DataLoader workers are spawned processes, so transforms must pickle.
CONDITIONS: dict[str, tuple[str, Transform]] = {
    "original": ("Unmodified test image", identity),
    "jpeg_q90": ("Re-saved as JPEG quality 90", partial(jpeg_roundtrip, quality=90)),
    "jpeg_q70": ("Re-saved as JPEG quality 70", partial(jpeg_roundtrip, quality=70)),
    "jpeg_q50": ("Re-saved as JPEG quality 50", partial(jpeg_roundtrip, quality=50)),
    "jpeg_q30": ("Re-saved as JPEG quality 30 (heavy compression)", partial(jpeg_roundtrip, quality=30)),
    "low_resolution": ("Downscaled to 50 % (the model then upsamples it)", partial(downscale, factor=0.5)),
    "very_low_resolution": ("Downscaled to 25 %", partial(downscale, factor=0.25)),
    "cropped": ("Central 60 % crop", partial(center_crop_fraction, fraction=0.6)),
    "screenshot": ("Screenshot-like: 75 % resample, sharpening, PNG", screenshot_like),
    "edited": ("Brightness/contrast edit then JPEG q90", edited),
    "blurred": ("Gaussian blur radius 1", blurred),
}


def match_to_reference(img: Image.Image, size: int, quality: int) -> Image.Image:
    """Matched external protocol: bring an image to the resolution and JPEG quality of the opposite class."""
    scale = size / min(img.size)
    return jpeg_roundtrip(downscale(img, scale) if scale < 1 else img, quality)
