"""Measure training-step and data-loading throughput so epoch budgets come from measurements."""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.training.data import ManifestDataset
from ml.training.model import CANDIDATES, Detector
from ml.training.train import pick_device


def step_throughput(arch: str, device: torch.device, batch: int, steps: int) -> float:
    model = Detector(arch, pretrained=False).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x, y = torch.randn(batch, 3, 224, 224, device=device), torch.ones(batch, device=device)
    for i in range(steps + 3):
        if i == 3:  # first steps include kernel compilation
            if device.type == "mps":
                torch.mps.synchronize()
            started = time.time()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    return batch * steps / (time.time() - started)


def loader_throughput(manifest: pd.DataFrame, workers: int, batch: int, batches: int) -> float:
    rows = manifest[manifest.split == "train"].head(batch * batches)
    loader = DataLoader(ManifestDataset(rows, train=True), batch_size=batch, num_workers=workers)
    started = time.time()
    count = sum(len(x) for x, _ in loader)
    return count / (time.time() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/splits.csv.gz")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()
    device = pick_device()
    result = {"device": str(device), "train_images_per_second": {}, "loader_images_per_second": {}}
    for arch in CANDIDATES:
        result["train_images_per_second"][arch] = round(step_throughput(arch, device, args.batch, args.steps), 1)
    manifest = pd.read_csv(args.manifest)
    for workers in (2, 4, 6):
        result["loader_images_per_second"][workers] = round(loader_throughput(manifest, workers, args.batch, 20), 1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
