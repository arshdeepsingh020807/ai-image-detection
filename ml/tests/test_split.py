import random

import pandas as pd
import pytest

from ml.preprocessing.clean import clean
from ml.preprocessing.hashing import hamming
from ml.preprocessing.split import find_leaks, make_splits


def synthetic(n_per=400, seed=0, dup_rate=0.15):
    rng = random.Random(seed)
    rows = []
    for source, role in [("A", "train_pool"), ("B", "train_pool"), ("X", "external")]:
        for label in (0, 1):
            made = []
            for i in range(n_per if source != "X" else 60):
                if made and rng.random() < dup_rate:
                    h = rng.choice(made)
                    for bit in rng.sample(range(64), rng.randint(0, 3)):
                        h ^= 1 << bit
                else:
                    h = rng.getrandbits(64)
                made.append(h)
                rows.append(
                    {
                        "path": f"{source}/{label}/{i}",
                        "source": source,
                        "label": label,
                        "role": role,
                        "status": "ok",
                        "format": "JPEG",
                        "n_frames": 1,
                        "md5": f"{source}{label}{i}",
                        "phash": f"{h:016x}",
                    }
                )
    return pd.DataFrame(rows)


def manifest(seed=42, **kw):
    return make_splits(clean(synthetic(**kw)), seed=seed)


def brute_force_cross_split_pairs(m, d=4):
    kept = m[m.split != ""].reset_index(drop=True)
    h = [int(x, 16) for x in kept.phash]
    sp = kept.split.tolist()
    return sum(1 for i in range(len(h)) for j in range(i + 1, len(h)) if sp[i] != sp[j] and hamming(h[i], h[j]) <= d)


def test_fractions_per_stratum_are_close_to_requested():
    m = manifest()
    pool = m[m.split.isin(["train", "val", "test"])]
    for _, g in pool.groupby(["source", "label"]):
        shares = g.split.value_counts(normalize=True)
        assert (
            abs(shares["train"] - 0.70) < 0.06
            and abs(shares["val"] - 0.15) < 0.06
            and abs(shares["test"] - 0.15) < 0.06
        )


def test_no_leaks_between_any_splits_verified_by_brute_force():
    m = manifest()
    assert brute_force_cross_split_pairs(m) == 0
    assert find_leaks(m) == {"exact_duplicate_leaks": 0, "near_duplicate_leaks": 0}


def test_find_leaks_detects_planted_leaks():
    m = manifest()
    kept = m[m.split.isin(["train", "test"])]
    a = kept[kept.split == "train"].iloc[0]
    b = kept[kept.split == "test"].iloc[0]
    m.loc[b.name, "md5"] = a.md5
    assert find_leaks(m)["exact_duplicate_leaks"] == 1
    m.loc[b.name, "phash"] = f"{int(a.phash, 16) ^ 0b11:016x}"
    assert find_leaks(m)["near_duplicate_leaks"] >= 1
    assert brute_force_cross_split_pairs(m) >= 1


def test_external_rows_are_external_and_dropped_rows_have_no_split():
    m = manifest()
    assert set(m[m.role == "external"].loc[m.keep, "split"]) == {"external"}
    assert (m.loc[~m.keep, "split"] == "").all()
    assert set(m[m.role == "train_pool"].loc[m.keep, "split"]) <= {"train", "val", "test"}


def test_same_seed_same_split_different_seed_differs():
    a, b, c = manifest(seed=1), manifest(seed=1), manifest(seed=2)
    assert a.split.tolist() == b.split.tolist()
    assert a.split.tolist() != c.split.tolist()


def test_every_cluster_lands_in_exactly_one_split():
    m = manifest()
    kept = m[m.keep]
    assert (kept.groupby("cluster").split.nunique() == 1).all()


def test_custom_fractions_and_validation():
    m = make_splits(clean(synthetic()), fractions=(0.5, 0.25, 0.25))
    pool = m[m.split.isin(["train", "val", "test"])]
    assert abs((pool.split == "train").mean() - 0.5) < 0.05
    with pytest.raises(ValueError):
        make_splits(clean(synthetic()), fractions=(0.5, 0.2, 0.2))


def test_all_splits_nonempty_in_every_stratum_and_input_unchanged():
    frame = clean(synthetic())
    before = frame.copy()
    m = make_splits(frame)
    pd.testing.assert_frame_equal(frame, before)
    for _, g in m[m.split.isin(["train", "val", "test"])].groupby(["source", "label"]):
        assert set(g.split) == {"train", "val", "test"}


def singleton_frame(n_real, n_fake):
    rng = random.Random(1)
    rows = [
        {
            "path": f"p{i}",
            "source": "A",
            "label": 0 if i < n_real else 1,
            "role": "train_pool",
            "status": "ok",
            "format": "JPEG",
            "n_frames": 1,
            "md5": f"m{i}",
            "phash": f"{rng.getrandbits(64):016x}",
        }
        for i in range(n_real + n_fake)
    ]
    return clean(pd.DataFrame(rows))


def test_exact_counts_on_singleton_clusters_follow_midpoint_rule():
    m = make_splits(singleton_frame(20, 0))
    assert m.split.value_counts().to_dict() == {"train": 14, "val": 3, "test": 3}  # 0.7/0.15/0.15 of 20


@pytest.mark.parametrize("seed", range(6))
def test_each_class_is_split_on_its_own(seed):
    m = make_splits(singleton_frame(40, 10), seed=seed)
    for label, n in [(0, 40), (1, 10)]:
        counts = m[m.label == label].split.value_counts()
        assert abs(counts["train"] - 0.7 * n) <= 1 and abs(counts["val"] - 0.15 * n) <= 1 and counts["test"] >= 1


def test_find_leaks_ignores_dropped_rows():
    m = manifest()
    kept_row = m[m.keep & (m.split == "train")].iloc[0]
    extra = kept_row.copy()
    extra["path"], extra["split"], extra["keep"] = "dropped_copy", "", False  # same bytes, but not part of any split
    m = pd.concat([m, extra.to_frame().T], ignore_index=True)
    assert find_leaks(m) == {"exact_duplicate_leaks": 0, "near_duplicate_leaks": 0}
