"""Per-image inspection used to build the dataset report and to probe for shortcuts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from detector_core.preprocess import to_rgb
from ml.datasets.catalog import Item
from ml.preprocessing.hashing import md5_file, phash_int

# IJG standard luminance quantisation table (quality 50); only its sum is used.
_BASE_LUMA_SUM = sum(
    [
        16,
        11,
        10,
        16,
        24,
        40,
        51,
        61,
        12,
        12,
        14,
        19,
        26,
        58,
        60,
        55,
        14,
        13,
        16,
        24,
        40,
        57,
        69,
        56,
        14,
        17,
        22,
        29,
        51,
        87,
        80,
        62,
        18,
        22,
        37,
        56,
        68,
        109,
        103,
        77,
        24,
        35,
        55,
        64,
        81,
        104,
        113,
        92,
        49,
        64,
        78,
        87,
        103,
        121,
        120,
        101,
        72,
        92,
        95,
        98,
        112,
        100,
        103,
        99,
    ]
)
BORDER = 4


def jpeg_quality(img: Image.Image) -> float:
    """Estimated IJG quality (1-100) from the luminance table; NaN if not a JPEG."""
    tables = getattr(img, "quantization", None)
    if img.format != "JPEG" or not tables:
        return float("nan")
    scale = 100 * sum(tables[0]) / _BASE_LUMA_SUM
    return (200 - scale) / 2 if scale <= 100 else 5000 / scale


def border_is_flat(arr: np.ndarray, width: int = BORDER, tolerance: float = 2.0) -> bool:
    """True when the outer frame is almost one constant colour (letterbox / padding / frame)."""
    frame = np.concatenate(
        [
            arr[:width].reshape(-1, 3),
            arr[-width:].reshape(-1, 3),
            arr[:, :width].reshape(-1, 3),
            arr[:, -width:].reshape(-1, 3),
        ]
    )
    return bool(frame.std(axis=0).max() < tolerance)


def inspect_file(path: Path) -> dict:
    record: dict = {"path": str(path), "file_size": path.stat().st_size}
    try:
        record["md5"] = md5_file(path)
        with Image.open(path) as img:
            record.update(
                format=img.format,
                mode=img.mode,
                width=img.width,
                height=img.height,
                jpeg_quality=jpeg_quality(img),
                has_exif=bool(img.getexif()),
                n_frames=getattr(img, "n_frames", 1),
            )
            rgb = to_rgb(img)  # decodes the full image, so truncated files fail here
        arr = np.asarray(rgb)
        record.update(
            phash=f"{phash_int(rgb):016x}",
            mean_r=float(arr[..., 0].mean()),
            mean_g=float(arr[..., 1].mean()),
            mean_b=float(arr[..., 2].mean()),
            flat_border=border_is_flat(arr),
            status="ok",
            reason="",
        )
    except Exception as exc:  # any decode failure quarantines the file with the reason recorded
        record.update(status="corrupt", reason=f"{type(exc).__name__}: {exc}"[:200])
    return record


def audit_items(items: list[Item], workers: int = 8) -> pd.DataFrame:
    meta = pd.DataFrame(
        [
            {
                "path": str(i.path),
                "source": i.source,
                "domain": i.domain,
                "label": i.label,
                "generator": i.generator,
                "role": i.role,
                "scope": i.scope,
            }
            for i in items
        ]
    )
    with ProcessPoolExecutor(workers) as pool:
        records = list(pool.map(inspect_file, [i.path for i in items], chunksize=256))
    return meta.merge(pd.DataFrame(records), on="path", how="left")
