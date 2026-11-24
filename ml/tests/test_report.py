import pandas as pd

from ml.preprocessing.report import build_report


def manifest():
    rows = [
        # path, source, role, domain, label, generator, format, w, h, keep, reason, cluster, split
        ("a", "S1", "train_pool", "faces", 0, "real", "JPEG", 256, 256, True, "", 0, "train"),
        ("b", "S1", "train_pool", "faces", 0, "real", "JPEG", 256, 256, True, "", 1, "val"),
        ("c", "S1", "train_pool", "faces", 1, "gan", "PNG", 300, 300, True, "", 2, "train"),
        ("d", "S1", "train_pool", "faces", 1, "gan", "JPEG", 256, 256, True, "", 2, "train"),
        ("e", "S1", "train_pool", "faces", 1, "gan", "JPEG", 256, 256, False, "exact_duplicate", -1, ""),
        ("f", "S1", "train_pool", "faces", 0, "real", "AVIF", 256, 256, False, "unsupported_format", -1, ""),
        ("g", "S2", "external", "general", 1, "mj", "PNG", 1024, 1024, True, "", 3, "external"),
        ("h", "S2", "external", "general", 0, "real", "JPEG", 200, 200, True, "", 4, "external"),
        ("i", "S1", "train_pool", "faces", 0, "real", "JPEG", 256, 256, True, "", 5, "train"),
    ]
    cols = [
        "path",
        "source",
        "role",
        "domain",
        "label",
        "generator",
        "format",
        "width",
        "height",
        "keep",
        "drop_reason",
        "cluster",
        "split",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_report_matches_hand_counted_values():
    r = build_report(manifest(), {"exact_duplicate_leaks": 0, "near_duplicate_leaks": 0}, {"p": {"auc": 0.5}})
    assert (r["original_images"], r["cleaned_images"], r["removed_images"]) == (9, 7, 2)
    assert r["removed_by_reason"] == {"exact_duplicate": 1, "unsupported_format": 1}
    assert r["exact_duplicates_removed"] == 1
    assert r["near_duplicate_clusters"] == 1 and r["images_in_near_duplicate_clusters"] == 2
    assert r["class_distribution"] == {"real": 4, "ai_generated": 3}
    assert r["by_source"]["S1"] == {"real": 3, "ai_generated": 2, "role": "train_pool", "domain": "faces"}
    assert r["by_source"]["S2"]["role"] == "external"
    assert r["by_generator"] == {"gan": 2, "mj": 1}
    assert r["formats"] == {"real": {"JPEG": 4}, "ai_generated": {"JPEG": 1, "PNG": 2}}
    assert r["resolution_top"]["256x256"] == 4 and r["resolution_top"]["1024x1024"] == 1
    assert r["splits"]["train"] == {"real": 2, "ai_generated": 2}
    assert r["splits"]["val"] == {"real": 1, "ai_generated": 0}
    assert r["splits"]["test"] == {"real": 0, "ai_generated": 0}
    assert r["splits"]["external"] == {"real": 1, "ai_generated": 1}
    assert r["leakage_check"] == {"exact_duplicate_leaks": 0, "near_duplicate_leaks": 0}
    assert r["shortcut_probes"] == {"p": {"auc": 0.5}}


def test_report_is_json_serialisable():
    import json

    json.dumps(build_report(manifest(), {}, {}))
