import io
import math
import time
from dataclasses import dataclass
from typing import cast

from PIL import Image

from backend.app.schemas.responses import Band, Explanation, ImageFormat, ImageInfo, Label, Probabilities, Region
from backend.app.services.heatmap import render_heatmap
from backend.app.services.model import Detector
from detector_core.preprocess import CROP, SHORT_SIDE, ImageRejected, decode_image, prepare

EXPLANATION_NOTE = (
    "Brighter, more opaque areas influenced the AI-generated score more. This shows where the model looked, "
    "not proof of how the image was made."
)


@dataclass(frozen=True)
class Analysis:
    image: ImageInfo
    logit: float
    explanation: Explanation | None
    timing_ms: dict[str, float]


def calibrated_probability(logit: float, temperature: float) -> float:
    """sigmoid(logit / temperature), written so neither branch can overflow."""
    z = logit / temperature
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def probabilities(p_ai: float) -> Probabilities:
    return Probabilities(real=1.0 - p_ai, ai_generated=p_ai)


def label_for(p_ai: float, band: Band) -> Label:
    """`p_ai <= low` wins over `>= high`, so a degenerate band (low == high) never yields inconclusive."""
    if p_ai <= band.low:
        return "real"
    if p_ai >= band.high:
        return "ai_generated"
    return "inconclusive"


def crop_region(width: int, height: int) -> Region:
    """Fraction of the original image inside the model's centre crop.

    Mirrors the geometry of `preprocess.resize_short_side` + `center_crop` (tests compare against them).
    Resizing maps the whole image onto the whole resized image, so offsets divided by the resized size
    are already fractions of the original.
    """
    scale = SHORT_SIDE / min(width, height)
    # The long side scales to >= SHORT_SIDE on its own, so preprocess's max(SHORT_SIDE, ...) guard is not needed here.
    resized_w, resized_h = round(width * scale), round(height * scale)
    left, top = (resized_w - CROP) // 2, (resized_h - CROP) // 2
    return Region(x=left / resized_w, y=top / resized_h, width=CROP / resized_w, height=CROP / resized_h)


def _milliseconds(start: float, end: float) -> float:
    return round((end - start) * 1000, 3)


def decode(data: bytes, max_pixels: int) -> tuple[Image.Image, ImageFormat]:
    """`decode_image` plus the source format, with Pillow's decompression-bomb guard reported as a size problem."""
    try:
        image = decode_image(data, max_pixels=max_pixels)
    except ImageRejected as exc:
        if isinstance(exc.__cause__, Image.DecompressionBombError):
            raise ImageRejected("image_too_large", "Image dimensions exceed the allowed limit.") from exc
        raise
    with Image.open(io.BytesIO(data)) as header:  # decode_image returns an RGB copy without the source format
        return image, cast(ImageFormat, header.format)


def analyse(detector: Detector, data: bytes, *, explain: bool, max_pixels: int) -> Analysis:
    """Blocking decode -> preprocess -> inference (-> explanation); run it in a worker thread."""
    t0 = time.perf_counter()
    image, source_format = decode(data, max_pixels)
    t1 = time.perf_counter()
    tensor = prepare(image)
    t2 = time.perf_counter()
    logit, features = detector.run(tensor)
    t3 = time.perf_counter()
    explanation = (
        Explanation(
            heatmap_png_base64=render_heatmap(features, detector.cam_weights),
            region=crop_region(*image.size),
            note=EXPLANATION_NOTE,
        )
        if explain
        else None
    )
    t4 = time.perf_counter()
    return Analysis(
        image=ImageInfo(width=image.width, height=image.height, format=source_format, size_bytes=len(data)),
        logit=logit,
        explanation=explanation,
        timing_ms={
            "decode": _milliseconds(t0, t1),
            "preprocess": _milliseconds(t1, t2),
            "inference": _milliseconds(t2, t3),
            "explain": _milliseconds(t3, t4),
        },
    )
