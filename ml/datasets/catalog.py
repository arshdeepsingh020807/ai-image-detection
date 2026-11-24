"""Enumerates every image under data/raw with its label, domain and role.

label: 0 = real, 1 = AI-generated. role: "train_pool" images are split into train/val/test;
"external" images are never trained on and only used for evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FACE_SWAP_FOLDERS = {"DeepFaceLab", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"}


@dataclass(frozen=True)
class Item:
    path: Path
    source: str
    domain: str
    label: int
    generator: str
    role: str
    scope: str = "in_scope"


def _images(folder: Path) -> Iterator[Path]:
    if folder.is_dir():
        yield from sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def deepdetect(root: Path) -> Iterator[Item]:
    base = root / "deepdetect" / "ddata"
    for split in ("train", "test"):
        for kind, label in (("real", 0), ("fake", 1)):
            for path in _images(base / split / kind):
                yield Item(path, "deepdetect", "faces", label, "unknown_fake" if label else "real", "train_pool")


def labeled_deepfake(root: Path) -> Iterator[Item]:
    base = root / "labeled_deepfake"
    if not base.is_dir():
        return
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        label = 0 if folder.name == "Real" else 1
        scope = "out_of_scope" if folder.name in FACE_SWAP_FOLDERS else "in_scope"
        for path in _images(folder):
            yield Item(
                path, "labeled_deepfake", "faces", label, "real" if label == 0 else folder.name, "external", scope
            )


def ai_detection(root: Path) -> Iterator[Item]:
    """General-content subset extracted by `ml.datasets.extract_subset` (Midjourney, DALL-E, Stable Diffusion
    versus Pexels/Unsplash/WikiArt photos; the source does not label the generator per image)."""
    base = root / "ai_detection_subset"
    for kind, label in (("real", 0), ("ai", 1)):
        for path in _images(base / kind):
            yield Item(path, "ai_detection", "general", label, "mixed_mj_dalle_sd" if label else "real", "train_pool")


def midjourney_imagenet(root: Path) -> Iterator[Item]:
    base = root / "midjourney_external" / "Midjourney_Exp2"
    for split in ("train", "valid", "test"):
        for kind, label in (("REAL", 0), ("FAKE", 1)):
            for path in _images(base / split / kind):
                yield Item(path, "midjourney_imagenet", "general", label, "midjourney" if label else "real", "external")


def all_items(root: Path) -> Iterator[Item]:
    yield from deepdetect(root)
    yield from ai_detection(root)
    yield from labeled_deepfake(root)
    yield from midjourney_imagenet(root)
