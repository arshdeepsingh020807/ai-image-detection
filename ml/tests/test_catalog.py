from pathlib import Path

from ml.datasets import catalog


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_deepdetect_labels_roles_and_suffix_filter(tmp_path):
    base = tmp_path / "deepdetect/ddata"
    touch(base / "train/real/a.jpg")
    touch(base / "train/fake/b.png")
    touch(base / "test/fake/c.JPG")
    touch(base / "test/real/notes.txt")
    items = {i.path.name: i for i in catalog.deepdetect(tmp_path)}
    assert set(items) == {"a.jpg", "b.png", "c.JPG"}
    assert items["a.jpg"].label == 0 and items["a.jpg"].generator == "real"
    assert items["b.png"].label == 1 and items["c.JPG"].label == 1
    assert items["b.png"].generator == "unknown_fake"
    assert all(i.role == "train_pool" and i.domain == "faces" and i.source == "deepdetect" for i in items.values())


def test_labeled_deepfake_generators_scope_and_external_role(tmp_path):
    base = tmp_path / "labeled_deepfake"
    for rel in ["Real/r.jpg", "DALL-E/d.jpg", "Midjourney/m.png", "FaceSwap/f.jpg", "Stable Diffusion/s.jpg"]:
        touch(base / rel)
    items = {i.path.name: i for i in catalog.labeled_deepfake(tmp_path)}
    assert items["r.jpg"].label == 0 and items["r.jpg"].generator == "real"
    assert items["d.jpg"].generator == "DALL-E" and items["s.jpg"].generator == "Stable Diffusion"
    assert items["f.jpg"].scope == "out_of_scope" and items["d.jpg"].scope == "in_scope"
    assert all(i.role == "external" for i in items.values())


def test_midjourney_imagenet_splits_are_all_external(tmp_path):
    base = tmp_path / "midjourney_external/Midjourney_Exp2"
    for rel in ["train/REAL/a.jpg", "valid/FAKE/b.png", "test/FAKE/c.png", "test/REAL/d.jpg", "extra/FAKE/e.png"]:
        touch(base / rel)
    items = {i.path.name: i for i in catalog.midjourney_imagenet(tmp_path)}
    assert set(items) == {"a.jpg", "b.png", "c.png", "d.jpg"}  # unknown split folders are ignored
    assert items["a.jpg"].label == 0 and items["b.png"].label == 1 and items["b.png"].generator == "midjourney"
    assert all(i.role == "external" and i.domain == "general" for i in items.values())


def test_ai_detection_subset_is_general_domain_training_pool(tmp_path):
    base = tmp_path / "ai_detection_subset"
    touch(base / "real/a.jpg")
    touch(base / "ai/b.jpg")
    items = {i.path.name: i for i in catalog.ai_detection(tmp_path)}
    assert items["a.jpg"].label == 0 and items["a.jpg"].generator == "real"
    assert items["b.jpg"].label == 1 and items["b.jpg"].generator == "mixed_mj_dalle_sd"
    assert all(i.role == "train_pool" and i.domain == "general" and i.source == "ai_detection" for i in items.values())


def test_missing_sources_yield_nothing_and_all_items_combines(tmp_path):
    assert list(catalog.all_items(tmp_path)) == []
    touch(tmp_path / "deepdetect/ddata/train/real/a.jpg")
    touch(tmp_path / "labeled_deepfake/Real/r.jpg")
    touch(tmp_path / "midjourney_external/Midjourney_Exp2/test/FAKE/m.png")
    touch(tmp_path / "ai_detection_subset/ai/g.jpg")
    assert {i.source for i in catalog.all_items(tmp_path)} == {
        "deepdetect",
        "ai_detection",
        "labeled_deepfake",
        "midjourney_imagenet",
    }
