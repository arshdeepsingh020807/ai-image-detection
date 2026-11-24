import csv
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from ml.datasets import extract_subset as ex


def encoded(size, fmt, seed=0):
    arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as z:
        for i in range(12):
            z.writestr(f"dataset/ai/{i:04d}.png", encoded((1024, 768), "PNG", i))
        for i in range(9):
            z.writestr(f"dataset/real/{i:04d}.jpg", encoded((300, 600), "JPEG", 100 + i))
        z.writestr("dataset/real/0000.png", encoded((512, 512), "PNG", 7))  # same stem as a jpg
        z.writestr("dataset/real/broken.jpg", b"\xff\xd8\xff garbage")
        z.writestr("dataset/real/readme.txt", b"hello")
        z.writestr("dataset/aid/0001.jpg", encoded((64, 64), "JPEG"))  # 'aid' is not a label folder
        z.writestr("dataset/other/0001.jpg", encoded((64, 64), "JPEG"))
    return path


@pytest.mark.parametrize(
    "member, label",
    [
        ("dataset/ai/1.jpg", 1),
        ("dataset/real/1.jpg", 0),
        ("x/AI/y/1.png", 1),
        ("dataset/aid/1.jpg", None),
        ("dataset/other/1.jpg", None),
        ("ai.jpg", None),
        ("real/ai/1.jpg", 0),
    ],
)
def test_label_of(member, label):
    assert ex.label_of(member) == label


def test_choose_members_is_balanced_seeded_and_filters_non_images():
    names = [f"d/ai/{i}.png" for i in range(50)] + [f"d/real/{i}.jpg" for i in range(30)] + ["d/real/x.txt"]
    a = ex.choose_members(names, 20, seed=1)
    assert len(a[0]) == 20 and len(a[1]) == 20 and all(n.endswith(".jpg") for n in a[0])
    assert a == ex.choose_members(list(reversed(names)), 20, seed=1)  # input order does not matter
    assert a != ex.choose_members(names, 20, seed=2)
    capped = ex.choose_members(names, 40, seed=1)
    assert len(capped[0]) == 30 and len(capped[1]) == 40  # cannot take more than exist


def test_output_name_is_collision_free():
    assert ex.output_name("dataset/real/0000.jpg") != ex.output_name("dataset/real/0000.png")
    assert ex.output_name("a/b/c.png") == "a__b__c.png.jpg"


def test_convert_resizes_to_training_scale_and_encodes_jpeg():
    data, original = ex.convert(encoded((1024, 768), "PNG"))
    assert original == {"orig_format": "PNG", "orig_width": 1024, "orig_height": 768}
    with Image.open(io.BytesIO(data)) as img:
        assert img.format == "JPEG" and img.size == (341, 256)  # short side 256, aspect kept


def test_extract_end_to_end(tmp_path, archive):
    out = tmp_path / "out"
    records = ex.extract(archive, out, per_class=8, seed=3, workers=2)
    ai, real = sorted((out / "ai").iterdir()), sorted((out / "real").iterdir())
    assert len(ai) == 8
    ok = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]
    assert len(real) == len([r for r in ok if "/real/" in r["member"]])
    assert all(min(Image.open(p).size) == 256 and Image.open(p).format == "JPEG" for p in ai + real)
    assert all(r["member"] == "dataset/real/broken.jpg" for r in failed)
    assert not any("aid" in r["member"] or "other" in r["member"] for r in records)
    with (out / "extraction_log.csv").open() as handle:
        log = list(csv.DictReader(handle))
    assert len(log) == len(records) == 16
    assert {row["orig_format"] for row in log if row["status"] == "ok" and "/ai/" in row["member"]} == {"PNG"}


def test_extract_is_reproducible(tmp_path, archive):
    a = {r["member"] for r in ex.extract(archive, tmp_path / "a", per_class=5, seed=9, workers=1)}
    b = {r["member"] for r in ex.extract(archive, tmp_path / "b", per_class=5, seed=9, workers=2)}
    assert a == b
