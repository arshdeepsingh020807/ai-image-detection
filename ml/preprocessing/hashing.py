"""Exact and perceptual hashing plus near-duplicate grouping."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

HASH_BITS = 64


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def phash_int(img: Image.Image) -> int:
    return int(str(imagehash.phash(img)), 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def near_duplicate_groups(hashes: list[int], max_distance: int = 4) -> list[int]:
    """Cluster id per input hash; images within `max_distance` bits (transitively) share a cluster.

    Pigeonhole: with max_distance+1 bands, two hashes differing in <= max_distance bits agree
    exactly on at least one band, so only same-band candidates are compared.
    """
    n_bands = max_distance + 1
    edges = [HASH_BITS * i // n_bands for i in range(n_bands + 1)]
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        mask = (1 << (hi - lo)) - 1
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, h in enumerate(hashes):
            buckets[(h >> lo) & mask].append(idx)
        for members in buckets.values():
            for i, a in enumerate(members):
                for c in members[i + 1 :]:
                    if find(a) != find(c) and hamming(hashes[a], hashes[c]) <= max_distance:
                        parent[find(c)] = find(a)
    return [find(i) for i in range(len(hashes))]
