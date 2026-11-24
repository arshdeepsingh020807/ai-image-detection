"""Extract a seeded, class-balanced subset from a dataset zip without unpacking the whole archive.

Each image is decoded, brought to the training scale (short side 256, the same `resize_short_side`
used everywhere) and stored as JPEG q95, the dominant encoding of the other training sources. Both
classes get identical treatment, so original format and resolution (which differ by class in the
source) cannot leak into training. The original properties are kept in a CSV for the dataset report.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

from PIL import Image

from detector_core.preprocess import resize_short_side, to_rgb
from ml.datasets.catalog import IMAGE_SUFFIXES

STORED_QUALITY = 95
LABEL_FOLDERS = {"real": 0, "ai": 1}


def label_of(member: str) -> int | None:
    """Label from the first folder named exactly `real` or `ai` in the member path."""
    for part in PurePosixPath(member).parts[:-1]:
        if part.lower() in LABEL_FOLDERS:
            return LABEL_FOLDERS[part.lower()]
    return None


def choose_members(names: list[str], per_class: int, seed: int) -> dict[int, list[str]]:
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for name in sorted(names):
        label = label_of(name)
        if label is not None and PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES:
            by_label[label].append(name)
    rng = random.Random(seed)
    return {label: sorted(rng.sample(members, min(per_class, len(members)))) for label, members in by_label.items()}


def convert(data: bytes) -> tuple[bytes, dict]:
    with Image.open(io.BytesIO(data)) as img:
        original = {"orig_format": img.format, "orig_width": img.width, "orig_height": img.height}
        rgb = resize_short_side(to_rgb(img))
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=STORED_QUALITY)
    return buf.getvalue(), original


_archive: zipfile.ZipFile | None = None


def _open_archive(zip_path: str) -> None:
    global _archive  # one handle per worker process: reopening parses the whole central directory each time
    _archive = zipfile.ZipFile(zip_path)


def output_name(member: str) -> str:
    """Collision-free flat file name: `a/b/0001.png` -> `a__b__0001.png.jpg`."""
    return member.replace("/", "__") + ".jpg"


def _work(args: tuple[str, Path]) -> dict:
    member, out_path = args
    assert _archive is not None
    data = _archive.read(member)
    record = {"member": member, "path": str(out_path), "orig_bytes": len(data)}
    try:
        encoded, original = convert(data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(encoded)
        record.update(original, status="ok")
    except Exception as exc:  # unreadable members are recorded, not silently dropped
        record.update(status=f"failed: {type(exc).__name__}")
    return record


def extract(zip_path: Path, out_dir: Path, per_class: int, seed: int = 42, workers: int = 4) -> list[dict]:
    with zipfile.ZipFile(zip_path) as archive:
        chosen = choose_members(archive.namelist(), per_class, seed)
    jobs = [
        (member, out_dir / ("real" if label == 0 else "ai") / output_name(member))
        for label, members in chosen.items()
        for member in members
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers, initializer=_open_archive, initargs=(str(zip_path),)) as pool:
        records = list(pool.map(_work, jobs, chunksize=64))
    with (out_dir / "extraction_log.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in records for k in r}))
        writer.writeheader()
        writer.writerows(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-class", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    records = extract(args.zip, args.out, args.per_class, args.seed, args.workers)
    ok = sum(r["status"] == "ok" for r in records)
    print(f"extracted {ok}/{len(records)} images into {args.out}")


if __name__ == "__main__":
    main()
