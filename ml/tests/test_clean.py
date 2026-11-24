import pandas as pd

from ml.preprocessing.clean import clean

FAR = [
    0x0123456789ABCDEF,
    0xFEDCBA9876543210,
    0xF0F0F0F00F0F0F0F,
    0x00FF00FF00FF00FF,
    0xAAAAAAAA55555555,
    0x1111111122222222,
    0x3C3C3C3CC3C3C3C3,
    0x8000000000000001,
    0x7FFFFFFF80000000,
    0x0F0F0F0FF0F0F0F0,
    0x5A5A5A5AA5A5A5A5,
    0x1248124812481248,
]


def row(path, phash, *, role="train_pool", label=0, status="ok", fmt="JPEG", frames=1, md5=None):
    return {
        "path": path,
        "role": role,
        "label": label,
        "status": status,
        "format": fmt,
        "n_frames": frames,
        "md5": md5 or f"md5-{path}",
        "phash": f"{phash:016x}",
    }


def build():
    rows = [
        row("a_keep", FAR[0]),
        row("b_dup_of_a", FAR[1], md5="md5-a_keep"),  # same bytes as a_keep (different hash value is ignored)
        row("c_corrupt", FAR[2], status="corrupt"),
        row("d_avif", FAR[3], fmt="AVIF"),
        row("e_animated", FAR[4], frames=3),
        row("f_near1", FAR[5]),
        row("f_near2", FAR[5] ^ 0b11),  # distance 2, same label
        row("h_conf1", FAR[6], label=0),
        row("h_conf2", FAR[6] ^ 0b1, label=1),  # near-dup, opposite labels
        row("j_leaky", FAR[7]),
        row("j_ext", FAR[7] ^ 0b111, role="external"),  # train image near an external one
        row("k_a_train_dup", FAR[8], md5="md5-shared"),
        row("k_z_ext_dup", FAR[9], role="external", md5="md5-shared"),
        row("z_single", FAR[10]),
    ]
    return pd.DataFrame(rows)


def test_each_policy_rule_fires_exactly_where_expected():
    out = clean(build()).set_index("path")
    expected = {
        "a_keep": "",
        "b_dup_of_a": "exact_duplicate",
        "c_corrupt": "corrupt",
        "d_avif": "unsupported_format",
        "e_animated": "unsupported_format",
        "f_near1": "",
        "f_near2": "",
        "h_conf1": "conflicting_label",
        "h_conf2": "conflicting_label",
        "j_leaky": "external_leak",
        "j_ext": "",
        "k_a_train_dup": "exact_duplicate",
        "k_z_ext_dup": "",
        "z_single": "",
    }
    assert out.drop_reason.to_dict() == expected
    assert out.keep.to_dict() == {k: v == "" for k, v in expected.items()}


def test_clusters_group_near_duplicates_and_mark_dropped_rows():
    out = clean(build()).set_index("path")
    assert out.cluster["f_near1"] == out.cluster["f_near2"]
    kept = out[out.keep]
    assert (
        kept.cluster.is_unique is False and kept.loc[["a_keep", "z_single", "j_ext", "k_z_ext_dup"]].cluster.is_unique
    )
    assert (
        out[~out.keep & out.drop_reason.isin(["corrupt", "unsupported_format", "exact_duplicate"])].cluster == -1
    ).all()


def test_distance_threshold_is_respected():
    frame = pd.DataFrame([row("p", FAR[0]), row("q", FAR[0] ^ 0b11111)])  # 5 bits apart
    assert clean(frame, max_distance=4).cluster.nunique() == 2
    assert clean(frame, max_distance=5).cluster.nunique() == 1


def test_input_is_not_mutated_and_empty_frame_is_handled():
    frame = build()
    before = frame.copy()
    clean(frame)
    pd.testing.assert_frame_equal(frame, before)
    empty = clean(frame.iloc[0:0])
    assert len(empty) == 0 and {"keep", "drop_reason", "cluster"} <= set(empty.columns)
