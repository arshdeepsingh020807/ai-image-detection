import numpy as np
import pandas as pd

from ml.preprocessing import shortcuts as sc


def frame(n=600, seed=0, leak=None):
    rng = np.random.default_rng(seed)
    label = np.tile([0, 1], n // 2)
    df = pd.DataFrame(
        {
            "label": label,
            "source": "s",
            "format": "JPEG",
            "width": 256,
            "height": 256,
            "file_size": rng.integers(20_000, 40_000, n),
            "jpeg_quality": rng.uniform(80, 95, n),
            "has_exif": False,
            "mean_r": rng.uniform(0, 255, n),
            "mean_g": rng.uniform(0, 255, n),
            "mean_b": rng.uniform(0, 255, n),
        }
    )
    if leak == "format":
        df.loc[df.label == 1, "format"] = "PNG"
    if leak == "quality":
        df.loc[df.label == 1, "jpeg_quality"] = 99.0
    if leak == "size":
        df.loc[df.label == 1, "width"] = 300
    return df


def test_add_metadata_features_formulae():
    df = pd.DataFrame(
        {
            "format": ["PNG", "JPEG"],
            "file_size": [1000, 625],
            "width": [10, 10],
            "height": [10, 5],
            "has_exif": [True, False],
        }
    )
    out = sc.add_metadata_features(df)
    assert out.is_png.tolist() == [1, 0]
    assert out.bits_per_pixel.tolist() == [80.0, 100.0]  # 1000*8/100 and 625*8/50
    assert out.has_exif.tolist() == [1, 0]
    assert df.has_exif.tolist() == [True, False]  # input untouched


def test_probe_finds_no_signal_in_random_metadata():
    res = sc.run_probes(frame())
    assert abs(res["s/file_format"]["auc"] - 0.5) < 0.08
    assert abs(res["s/colour"]["auc"] - 0.5) < 0.08


def test_probe_detects_planted_format_and_resolution_leaks():
    assert sc.run_probes(frame(leak="format"))["s/file_format"]["auc"] > 0.99
    assert sc.run_probes(frame(leak="size"))["s/file_format"]["auc"] > 0.99
    assert sc.run_probes(frame(leak="quality"))["s/file_format"]["auc"] > 0.99
    leaked = sc.run_probes(frame(leak="format"))["s/file_format"]
    assert leaked["accuracy"] > 0.99 and leaked["n"] == 600
    assert abs(sc.run_probes(frame())["s/file_format"]["accuracy"] - 0.5) < 0.08


def test_probe_balances_classes_and_caps_sample():
    df = frame(1000)
    skewed = pd.concat([df[df.label == 0], df[df.label == 1].head(100)])
    res = sc.probe(sc.add_metadata_features(skewed), sc.FEATURE_SETS["colour"], max_per_class=80)
    assert res["n"] == 160  # 80 per class despite 500 vs 100 rows


def test_single_class_sources_are_skipped_and_nan_quality_is_handled():
    df = frame()
    df.loc[df.index % 3 == 0, "jpeg_quality"] = np.nan
    one_class = df.assign(source="only_real", label=0).head(50)
    res = sc.run_probes(pd.concat([df, one_class]))
    assert set(res) == {"s/file_format", "s/colour"}
