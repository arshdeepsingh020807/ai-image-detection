"""Torch dataset over a split manifest. Augmentation randomness is keyed on (seed, epoch, index), so a
sample looks the same no matter which loader worker produces it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from detector_core.preprocess import prepare, to_rgb
from ml.preprocessing.augment import AugmentConfig, augment, jpeg_roundtrip

SOURCE_JPEG_QUALITY = 95  # the dominant encoding of the training sources
DEFAULT_AUGMENT = AugmentConfig()


class ManifestDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, train: bool, seed: int = 0, cfg: AugmentConfig = DEFAULT_AUGMENT):
        self.paths = rows.path.tolist()
        self.labels = rows.label.astype(np.float32).to_numpy()
        self.formats = rows.format.tolist()
        self.train, self.seed, self.cfg, self.epoch = train, seed, cfg, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, index: int) -> Image.Image:
        with Image.open(self.paths[index]) as img:
            rgb = to_rgb(img)
        # Uncompressed sources are rarer in one class; one q95 encode removes "has no JPEG traces" as a label cue.
        if self.train and self.formats[index] != "JPEG":
            rgb = jpeg_roundtrip(rgb, SOURCE_JPEG_QUALITY)
        return rgb

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.load(index)
        if self.train:
            arr = augment(img, np.random.default_rng((self.seed, self.epoch, index)), self.cfg)
        else:
            arr = prepare(img)
        return torch.from_numpy(arr), torch.tensor(self.labels[index])


def balanced_weights(rows: pd.DataFrame, keys: tuple[str, ...] = ("domain", "label")) -> np.ndarray:
    """Per-row sampling weight so every (domain, label) group is drawn equally often."""
    sizes = rows.groupby(list(keys))[keys[0]].transform("size").to_numpy(dtype=np.float64)
    weights = 1.0 / sizes
    return weights / weights.sum()


def balanced_sampler(rows: pd.DataFrame, seed: int) -> WeightedRandomSampler:
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(balanced_weights(rows)), num_samples=len(rows), replacement=True, generator=generator
    )
