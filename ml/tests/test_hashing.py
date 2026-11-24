import hashlib
import io
import random

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image, ImageDraw

from ml.preprocessing import hashing as h


def scene(seed: int, size: int = 128) -> Image.Image:
    """Synthetic 'photo': smooth random blobs, so perceptual hashes are meaningful."""
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (rng.randrange(256),) * 3)
    draw = ImageDraw.Draw(img)
    for _ in range(12):
        x, y = rng.randrange(size), rng.randrange(size)
        r = rng.randrange(10, 50)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=tuple(rng.randrange(256) for _ in range(3)))
    return img


def test_md5_known_vectors_and_chunk_boundaries(tmp_path):
    for name, data in {"hello": b"hello", "empty": b"", "big": bytes(range(256)) * 12000}.items():
        f = tmp_path / name
        f.write_bytes(data)
        assert h.md5_file(f) == hashlib.md5(data).hexdigest()
    assert h.md5_file(tmp_path / "hello") == "5d41402abc4b2a76b9719d911017c592"
    assert h.md5_file(tmp_path / "empty") == "d41d8cd98f00b204e9800998ecf8427e"
    assert h.md5_file(tmp_path / "big", chunk=7) == hashlib.md5(bytes(range(256)) * 12000).hexdigest()


def test_hamming_known_values():
    assert h.hamming(0b1011, 0b0001) == 2
    assert h.hamming(0, 2**64 - 1) == 64
    assert h.hamming(12345, 12345) == 0


def test_phash_is_stable_for_recompression_and_resize_but_differs_between_images():
    base = scene(1)
    buf = io.BytesIO()
    base.save(buf, "JPEG", quality=70)
    recompressed = Image.open(buf).convert("RGB")
    resized = base.resize((200, 200))
    assert h.hamming(h.phash_int(base), h.phash_int(recompressed)) <= 6
    assert h.hamming(h.phash_int(base), h.phash_int(resized)) <= 6
    unrelated = [h.hamming(h.phash_int(base), h.phash_int(scene(s))) for s in range(10, 20)]
    assert np.median(unrelated) > 12


def reference_phash(img: Image.Image) -> int:
    """pHash from its definition (Zauner 2010): 32x32 grayscale, 2-D DCT, top-left 8x8, median threshold."""
    from scipy.fft import dct

    gray = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = dct(dct(gray, axis=0, norm=None), axis=1, norm=None)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return sum(int(b) << (63 - i) for i, b in enumerate(bits))


def test_phash_matches_reference_implementation_on_most_bits():
    for seed in range(5):
        img = scene(seed)
        assert h.hamming(h.phash_int(img), reference_phash(img)) <= 4


def brute_force_groups(hashes, d):
    n = len(hashes)
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack += [y for y in range(n) if y not in comp and h.hamming(hashes[x], hashes[y]) <= d]
        seen |= comp
        groups.append(frozenset(comp))
    return set(groups)


def as_partition(labels):
    by = {}
    for idx, lab in enumerate(labels):
        by.setdefault(lab, set()).add(idx)
    return {frozenset(v) for v in by.values()}


def test_near_duplicate_groups_edge_cases():
    assert h.near_duplicate_groups([]) == []
    assert h.near_duplicate_groups([5]) == [0]
    assert len(set(h.near_duplicate_groups([7, 7, 7]))) == 1
    far = [0, 2**64 - 1]
    assert len(set(h.near_duplicate_groups(far))) == 2


def test_near_duplicate_distance_boundary_and_transitivity():
    a = 0
    b = 0b1111  # distance 4 from a
    c = 0b11111  # distance 1 from b, 5 from a
    d = 0b111111  # chain continues
    assert as_partition(h.near_duplicate_groups([a, b], 4)) == {frozenset({0, 1})}
    assert as_partition(h.near_duplicate_groups([a, 0b11111], 4)) == {frozenset({0}), frozenset({1})}  # distance 5
    assert as_partition(h.near_duplicate_groups([a, b, c, d], 4)) == {frozenset({0, 1, 2, 3})}  # linked through chain


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(2, 120), d=st.integers(0, 6))
def test_near_duplicate_groups_match_brute_force(seed, n, d):
    rng = random.Random(seed)
    hashes = []
    for _ in range(n):
        if hashes and rng.random() < 0.4:  # plant near-duplicates
            base = rng.choice(hashes)
            for bit in rng.sample(range(64), rng.randint(0, d + 1)):
                base ^= 1 << bit
            hashes.append(base)
        else:
            hashes.append(rng.getrandbits(64))
    assert as_partition(h.near_duplicate_groups(hashes, d)) == brute_force_groups(hashes, d)


def test_pairs_inside_a_large_shared_band_bucket_are_still_found():
    rng = random.Random(7)
    common_low = 0x1ABC  # every hash agrees on this band, so they all land in one big bucket
    hashes = [(rng.getrandbits(51) << 13) | common_low for _ in range(12)]
    hashes[9] = hashes[6] ^ (1 << 40)  # distance 1, but neither is among the first bucket members
    groups = as_partition(h.near_duplicate_groups(hashes, 4))
    assert groups == brute_force_groups(hashes, 4)
    assert frozenset({6, 9}) in groups
