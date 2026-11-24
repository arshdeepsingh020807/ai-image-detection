import base64
import io

import numpy as np
from PIL import Image

from detector_core.preprocess import CROP

MAX_ALPHA = 200  # the hottest cell stays slightly see-through so the photo remains visible beneath it
# Upper half of matplotlib's "inferno": lightness rises monotonically, so a stronger
# influence always reads as a brighter, warmer colour, also in greyscale and for colour-blind viewers.
_COLOUR_STOPS = np.array([0.0, 1 / 3, 2 / 3, 1.0])
_COLOURS = np.array([[120, 28, 109], [212, 72, 66], [250, 153, 8], [252, 255, 164]], dtype=np.float64)


def activation_map(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """ReLU(sum_c w_c * A_c) min-max scaled to [0, 1]; a flat map has nothing to highlight and becomes all zeros."""
    cam = np.maximum(np.tensordot(weights, features, axes=1), 0.0).astype(np.float32)
    low, high = float(cam.min()), float(cam.max())
    if high <= low:
        return np.zeros_like(cam)
    return (cam - low) / (high - low)


def colourise(intensity: np.ndarray) -> np.ndarray:
    """[0, 1] intensities -> RGBA uint8 with opacity proportional to intensity."""
    rgb = np.stack([np.interp(intensity, _COLOUR_STOPS, _COLOURS[:, c]) for c in range(3)], axis=-1)
    alpha = intensity[..., None] * MAX_ALPHA
    return np.concatenate([rgb, alpha], axis=-1).round().astype(np.uint8)


def render_heatmap(features: np.ndarray, weights: np.ndarray) -> str:
    """Grad-CAM for the pooled linear head, as a base64 CROP x CROP RGBA PNG aligned with the model's input crop."""
    cam = Image.fromarray(activation_map(features, weights))
    upsampled = np.asarray(cam.resize((CROP, CROP), Image.Resampling.BILINEAR))
    buffer = io.BytesIO()
    Image.fromarray(colourise(np.clip(upsampled, 0.0, 1.0))).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
