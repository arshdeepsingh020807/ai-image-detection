import io
import math

import numpy as np
import pytest
from PIL import Image

from ml.datasets.catalog import Item
from ml.preprocessing import audit
from ml.preprocessing.hashing import md5_file


def noise_image(size=64, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8))


@pytest.mark.parametrize("quality", [30, 50, 60, 75, 85, 90, 95])
def test_jpeg_quality_estimate_matches_encoder_setting(quality):
    buf = io.BytesIO()
    noise_image().save(buf, "JPEG", quality=quality)
    est = audit.jpeg_quality(Image.open(io.BytesIO(buf.getvalue())))
    assert abs(est - quality) <= 2


def test_jpeg_quality_is_nan_for_non_jpeg():
    buf = io.BytesIO()
    noise_image().save(buf, "PNG")
    assert math.isnan(audit.jpeg_quality(Image.open(io.BytesIO(buf.getvalue()))))


def test_border_is_flat_detects_padding_and_ignores_content():
    padded = np.full((64, 64, 3), 10, dtype=np.uint8)
    padded[8:-8, 8:-8] = np.random.default_rng(1).integers(0, 256, (48, 48, 3))
    assert audit.border_is_flat(padded)
    assert not audit.border_is_flat(np.asarray(noise_image()))
    assert not audit.border_is_flat(np.random.default_rng(2).integers(0, 256, (64, 64, 3)).astype(np.uint8))


@pytest.mark.parametrize("side", ["top", "bottom", "left", "right"])
@pytest.mark.parametrize("channel", [0, 1, 2])
def test_border_not_flat_when_any_single_side_or_channel_varies(side, channel):
    arr = np.full((64, 64, 3), 10, dtype=np.uint8)
    noise = np.random.default_rng(3).integers(0, 256, (64, 64)).astype(np.uint8)
    region = {
        "top": (slice(0, 4), slice(None)),
        "bottom": (slice(-4, None), slice(None)),
        "left": (slice(8, -8), slice(0, 4)),
        "right": (slice(8, -8), slice(-4, None)),
    }[side]
    arr[region[0], region[1], channel] = noise[region]
    assert not audit.border_is_flat(arr)
    assert audit.border_is_flat(np.full((64, 64, 3), 10, dtype=np.uint8))


def test_inspect_file_records_each_channel_mean(tmp_path):
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 10, 100, 200
    path = tmp_path / "c.png"
    Image.fromarray(arr).save(path)
    rec = audit.inspect_file(path)
    assert (rec["mean_r"], rec["mean_g"], rec["mean_b"]) == (10.0, 100.0, 200.0)


def test_inspect_file_valid_jpeg(tmp_path):
    path = tmp_path / "ok.jpg"
    noise_image(80).save(path, "JPEG", quality=90)
    rec = audit.inspect_file(path)
    assert rec["status"] == "ok" and rec["reason"] == ""
    assert (rec["width"], rec["height"], rec["format"]) == (80, 80, "JPEG")
    assert rec["md5"] == md5_file(path) and rec["file_size"] == path.stat().st_size
    assert (
        abs(rec["jpeg_quality"] - 90) <= 2
        and len(rec["phash"]) == 16
        and int(rec["phash"], 16) >= 0
        and not rec["has_exif"]
    )
    arr = np.asarray(Image.open(path).convert("RGB"))
    assert rec["mean_r"] == pytest.approx(arr[..., 0].mean())


def test_inspect_file_flags_exif_and_png_has_nan_quality(tmp_path):
    exif = Image.Exif()
    exif[0x010F] = "Camera"
    jpg = tmp_path / "e.jpg"
    noise_image().save(jpg, "JPEG", exif=exif)
    png = tmp_path / "p.png"
    noise_image().save(png)
    assert audit.inspect_file(jpg)["has_exif"] is True
    rec = audit.inspect_file(png)
    assert rec["status"] == "ok" and math.isnan(rec["jpeg_quality"]) and rec["format"] == "PNG"


@pytest.mark.parametrize("name, content", [("zero.jpg", b""), ("text.jpg", b"not an image"), ("half.png", None)])
def test_inspect_file_quarantines_corrupt_files_with_reason(tmp_path, name, content):
    path = tmp_path / name
    if content is None:
        buf = io.BytesIO()
        noise_image(128).save(buf, "PNG")
        content = buf.getvalue()[: len(buf.getvalue()) // 2]
    path.write_bytes(content)
    rec = audit.inspect_file(path)
    assert rec["status"] == "corrupt" and rec["reason"]
    assert "phash" not in rec


def test_audit_items_merges_metadata_and_runs_in_parallel(tmp_path):
    items = []
    for i in range(6):
        p = tmp_path / f"{i}.png"
        noise_image(32, seed=i).save(p)
        items.append(Item(p, "src", "faces", i % 2, "gen", "train_pool"))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"xx")
    items.append(Item(bad, "src", "faces", 0, "gen", "train_pool"))
    df = audit.audit_items(items, workers=2)
    assert len(df) == 7 and (df.status == "ok").sum() == 6 and (df.status == "corrupt").sum() == 1
    assert df.set_index("path").loc[str(tmp_path / "3.png"), "label"] == 1
    assert df.md5.nunique() == 6 + 1
