"""Finish a large Kaggle download with parallel HTTP range requests.

Kaggle serves archives from a signed Cloud Storage URL that accepts byte ranges, but a single
connection is often throttled far below the line speed. This resumes an existing partial file:
the remaining bytes are fetched as N ranges in parallel (each resumable), appended in order and
verified against the expected size and, when the validator is an MD5, the full-file MD5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 1 << 20
API = "https://www.kaggle.com/api/v1/datasets/download/{slug}"


def kaggle_token() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN") or (Path.home() / ".kaggle" / "access_token").read_text().strip()
    if not token:
        raise SystemExit("no Kaggle token in KAGGLE_API_TOKEN or ~/.kaggle/access_token")
    return token


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # keep the Location instead of following it with our token
        return None


def signed_url(slug: str) -> str:
    request = urllib.request.Request(API.format(slug=slug), headers={"Authorization": f"Bearer {kaggle_token()}"})
    try:
        urllib.request.build_opener(_NoRedirect).open(request, timeout=60)
    except urllib.error.HTTPError as redirect:
        if redirect.code in (301, 302, 303, 307) and redirect.headers.get("Location"):
            return redirect.headers["Location"]
        raise
    raise SystemExit("Kaggle did not redirect to a download URL")


def split_ranges(start: int, end: int, parts: int) -> list[tuple[int, int]]:
    """Inclusive byte ranges covering [start, end) in `parts` nearly equal pieces (no empty pieces)."""
    total = end - start
    if total <= 0:
        return []
    parts = max(1, min(parts, total))
    edges = [start + total * i // parts for i in range(parts + 1)]
    return [(edges[i], edges[i + 1] - 1) for i in range(parts)]


def fetch_range(url: str, first: int, last: int, part: Path, retries: int = 30) -> Path:
    """Download bytes first..last into `part`, resuming whatever is already there."""
    want = last - first + 1
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            return part
        request = urllib.request.Request(url, headers={"Range": f"bytes={first + have}-{last}"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("ab") as out:
                if response.status != 206:
                    raise OSError(f"server ignored the range request (HTTP {response.status})")
                while block := response.read(CHUNK):
                    out.write(block)
        except OSError as exc:
            print(f"{part.name}: retry {attempt + 1} after {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(30, 2**attempt))
    if (part.stat().st_size if part.exists() else 0) < want:
        raise SystemExit(f"{part.name} incomplete after {retries} attempts")
    return part


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(8 * CHUNK):
            digest.update(block)
    return digest.hexdigest()


def remote_size(url: str) -> int:
    """Total object size from the Content-Range of a one-byte range request."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Range"].rsplit("/", 1)[1])


def finish(zip_path: Path, url: str, size: int, connections: int) -> None:
    if (actual := remote_size(url)) != size:
        raise SystemExit(f"size mismatch: server has {actual} bytes, expected {size}")
    have = zip_path.stat().st_size if zip_path.exists() else 0
    ranges = split_ranges(have, size, connections)
    parts = [zip_path.with_name(f"{zip_path.name}.part{i:02d}") for i in range(len(ranges))]
    with ThreadPoolExecutor(len(ranges) or 1) as pool:
        list(pool.map(lambda job: fetch_range(url, *job[0], job[1]), zip(ranges, parts, strict=True)))
    with zip_path.open("ab") as out:
        for part in parts:
            with part.open("rb") as handle:
                while block := handle.read(8 * CHUNK):
                    out.write(block)
    if zip_path.stat().st_size != size:
        raise SystemExit(f"size mismatch after download: {zip_path.stat().st_size} != {size}")
    for part in parts:
        part.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug")
    parser.add_argument("zip", type=Path)
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args()
    marker = json.loads(args.zip.with_name(args.zip.name + ".kaggle-partial").read_text())
    size, validator = int(marker["size"]), marker["validator"].strip('"')
    finish(args.zip, signed_url(args.slug), size, args.connections)
    if len(validator) == 32:
        digest = md5_of(args.zip)
        if digest != validator:
            raise SystemExit(f"MD5 mismatch: {digest} != {validator}")
        print("MD5 verified", flush=True)
    args.zip.with_name(args.zip.name + ".kaggle-partial").unlink()
    (args.zip.parent / ".complete").touch()
    print(f"complete: {args.zip} ({size} bytes)", flush=True)


if __name__ == "__main__":
    main()
