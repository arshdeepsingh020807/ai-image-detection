import base64
import io

import numpy as np
import pytest
from PIL import Image

from backend.app.services.heatmap import MAX_ALPHA, activation_map, colourise, render_heatmap
from backend.tests.helpers import CARD, Reference, encode, noise_image


def loop_cam(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reference CAM with explicit loops: ReLU(sum_c w_c A_c), then min-max."""
    channels, rows, cols = features.shape
    cam = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            cam[i, j] = max(0.0, sum(float(weights[c]) * float(features[c, i, j]) for c in range(channels)))
    return (cam - cam.min()) / (cam.max() - cam.min())


def bilinear_reference(grid: np.ndarray, size: int) -> np.ndarray:
    """Half-pixel-centre bilinear upsampling with edge clamping (the convention Pillow uses)."""
    n = grid.shape[0]
    src = np.clip((np.arange(size) + 0.5) * n / size - 0.5, 0, n - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = src - lo
    rows = grid[lo] * (1 - frac)[:, None] + grid[hi] * frac[:, None]
    return rows[:, lo] * (1 - frac)[None, :] + rows[:, hi] * frac[None, :]


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


# ---------- activation_map ----------


def test_activation_map_matches_loop_reference():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(16, 7, 7)).astype(np.float32)
    weights = rng.normal(size=16).astype(np.float32)
    assert np.allclose(activation_map(features, weights), loop_cam(features, weights), atol=1e-5)


def test_activation_map_hand_computed_with_relu():
    features = np.zeros((2, 7, 7), dtype=np.float32)
    features[0, 0, 0], features[1, 0, 0] = 4.0, 1.0  # 0.5*4 - 1*1 = 1
    features[0, 3, 3] = 6.0  # 0.5*6 = 3, the peak
    features[1, 6, 6] = 5.0  # -5, clipped to 0 by ReLU
    cam = activation_map(features, np.array([0.5, -1.0], dtype=np.float32))
    assert cam[3, 3] == 1.0 and cam[0, 0] == pytest.approx(1 / 3) and cam[6, 6] == 0.0
    assert cam.min() == 0.0 and cam.dtype == np.float32


def test_activation_map_min_max_uses_the_positive_floor():
    features = np.full((1, 7, 7), 2.0, dtype=np.float32)
    features[0, 0, 0] = 4.0
    cam = activation_map(features, np.array([1.0], dtype=np.float32))
    assert cam[0, 0] == 1.0 and cam[1, 1] == 0.0  # floor of 2 maps to 0, not 0.5


@pytest.mark.parametrize("fill, weight", [(1.0, -1.0), (0.0, 1.0), (3.0, 1.0)])
def test_activation_map_flat_maps_become_zero_without_nan(fill, weight):
    """All negative (ReLU -> 0), all zero, and constant positive maps carry no location."""
    cam = activation_map(np.full((3, 7, 7), fill, dtype=np.float32), np.full(3, weight, dtype=np.float32))
    assert np.array_equal(cam, np.zeros((7, 7))) and not np.isnan(cam).any()


# ---------- colourise ----------


def test_colourise_endpoints_and_alpha_proportional():
    rgba = colourise(np.array([[0.0, 0.25, 0.5, 1.0]]))
    assert rgba.dtype == np.uint8 and rgba.shape == (1, 4, 4)
    assert rgba[0, 0].tolist() == [120, 28, 109, 0]
    assert rgba[0, 3].tolist() == [252, 255, 164, 200]  # the hottest cell stays slightly see-through
    assert rgba[0, :, 3].tolist() == [0, round(0.25 * MAX_ALPHA), round(0.5 * MAX_ALPHA), MAX_ALPHA]


def test_colourise_hits_the_inner_anchors():
    rgba = colourise(np.array([[1 / 3, 2 / 3]]))
    assert rgba[0, 0, :3].tolist() == [212, 72, 66] and rgba[0, 1, :3].tolist() == [250, 153, 8]


def test_colourise_interpolates_between_anchors():
    mid = colourise(np.array([[0.5]]))[0, 0, :3].astype(float)  # halfway between the 1/3 and 2/3 anchors
    assert mid.tolist() == pytest.approx([(212 + 250) / 2, (72 + 153) / 2, (66 + 8) / 2], abs=0.51)


def test_colourmap_is_perceptually_ordered():
    """Brighter means stronger: luminance must rise with intensity, and hot is warm (red > blue)."""
    ramp = colourise(np.linspace(0, 1, 256)[None, :])[0].astype(float)
    luminance = relative_luminance(ramp[:, :3])
    one_blue_level = 0.0722 + 1e-9  # uint8 rounding may drop blue by one level while red and green hold
    assert (np.diff(luminance) >= -one_blue_level).all() and luminance[-1] - luminance[0] > 150
    assert (np.diff(luminance[::8]) > 0).all()
    assert (ramp[128:, 0] > ramp[128:, 2]).all()
    assert (np.diff(ramp[:, 3]) >= 0).all()


# ---------- render_heatmap ----------


def decode_png(encoded: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert image.format == "PNG"
    return image


def test_render_heatmap_is_224_rgba_and_matches_numpy_pipeline():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(8, 7, 7)).astype(np.float32)
    weights = rng.normal(size=8).astype(np.float32)
    image = decode_png(render_heatmap(features, weights))
    assert image.mode == "RGBA" and image.size == (224, 224)
    expected_alpha = np.clip(bilinear_reference(loop_cam(features, weights), 224), 0, 1) * MAX_ALPHA
    assert np.abs(np.asarray(image)[..., 3].astype(float) - expected_alpha).max() <= 1.0


def test_render_heatmap_puts_the_hotspot_where_the_features_are():
    features = np.zeros((1, 7, 7), dtype=np.float32)
    features[0, 1, 5] = 1.0  # row 1, column 5 -> upper right
    alpha = np.asarray(decode_png(render_heatmap(features, np.ones(1, dtype=np.float32))))[..., 3]
    row, col = np.unravel_index(alpha.argmax(), alpha.shape)
    assert 32 <= row < 64 and 160 <= col < 192  # cell (1, 5) covers rows 32-63 and columns 160-191
    assert alpha[200, 20] == 0


def test_render_heatmap_all_zero_cam_is_fully_transparent():
    image = decode_png(render_heatmap(np.zeros((4, 7, 7), dtype=np.float32), np.ones(4, dtype=np.float32)))
    assert np.asarray(image)[..., 3].max() == 0


def test_render_heatmap_on_real_model_features(reference: Reference):
    _, features = reference.outputs(encode(noise_image(300, 300), "PNG"))
    weights = np.asarray(CARD["cam_weights"], dtype=np.float32)
    alpha = np.asarray(decode_png(render_heatmap(features, weights)))[..., 3].astype(float)
    expected = np.clip(bilinear_reference(loop_cam(features, weights), 224), 0, 1) * MAX_ALPHA
    assert np.abs(alpha - expected).max() <= 1.0
