"""Two-phase transfer learning: train the head on a frozen backbone, then fine-tune the backbone at a lower LR."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ml.training.data import ManifestDataset, balanced_sampler, make_loader
from ml.training.model import Detector


@dataclass
class TrainConfig:
    arch: str = "efficientnet_b0"
    pretrained: bool = True
    epochs_head: int = 2
    epochs_finetune: int = 8
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    unfreeze: int | bool = True  # True = whole backbone, int = last N parameter tensors
    patience: int = 3
    seed: int = 42
    max_train: int | None = None  # cap for quick model-comparison runs
    max_val: int | None = None
    num_workers: int = 4
    device: str = "auto"
    out_dir: str = "artifacts/checkpoints/run"

    @classmethod
    def from_yaml(cls, path: Path, **overrides) -> TrainConfig:
        return cls(**{**(yaml.safe_load(path.read_text()) or {}), **overrides})


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_lr(step: int, total: int, base: float, warmup: int = 0) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _sample(rows: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or len(rows) <= cap:
        return rows
    parts = [
        g.sample(max(1, round(cap * len(g) / len(rows))), random_state=seed)
        for _, g in rows.groupby(["domain", "label"], sort=True)
    ]
    return pd.concat(parts)


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(torch.from_numpy(x).to(device)).squeeze(1).float().cpu())
        labels.append(torch.from_numpy(y))
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def _loss_and_accuracy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    loss = nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    return float(loss), float(((logits >= 0) == (labels >= 0.5)).mean())


PROGRESS_EVERY = 25


def write_progress(path: Path, **fields) -> None:
    """Live status for dashboards; written atomically so a reader never sees a half-written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**fields, "updated": time.time()}))
    tmp.replace(path)


def progress_fields(
    state: str,
    phase: str,
    epoch: int,
    total_epochs: int,
    batches: int,
    losses: list[float],
    correct: int,
    seen: int,
    started: float,
) -> dict:
    return {
        "state": state,
        "phase": phase,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "batch": len(losses),
        "batches": batches,
        "images_per_second": seen / max(1e-9, time.time() - started),
        "running_loss": float(np.mean(losses[-50:])) if losses else None,
        "running_accuracy": correct / seen if seen else None,
    }


def train(cfg: TrainConfig, manifest: pd.DataFrame, log=print) -> dict:
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows = _sample(manifest[manifest.split == "train"], cfg.max_train, cfg.seed)
    val_rows = _sample(manifest[manifest.split == "val"], cfg.max_val, cfg.seed)
    if train_rows.empty or val_rows.empty:
        raise ValueError("manifest needs non-empty train and val splits")
    train_set = ManifestDataset(train_rows, train=True, seed=cfg.seed)
    loader_args = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = make_loader(train_set, sampler=balanced_sampler(train_rows, cfg.seed), drop_last=True, **loader_args)
    val_loader = make_loader(ManifestDataset(val_rows, train=False), shuffle=False, **loader_args)

    model = Detector(cfg.arch, pretrained=cfg.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()  # classes are balanced by the sampler, so no pos_weight
    history: list[dict] = []
    best = {"val_loss": float("inf"), "epoch": 0}
    epoch, total_epochs = 0, cfg.epochs_head + cfg.epochs_finetune
    for phase, epochs, lr, trainable in (
        ("head", cfg.epochs_head, cfg.lr_head, False),
        ("finetune", cfg.epochs_finetune, cfg.lr_finetune, cfg.unfreeze),
    ):
        if epochs == 0:
            continue
        model.set_backbone_trainable(trainable)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
        total, step, stale = epochs * len(train_loader), 0, 0
        for _ in range(epochs):
            epoch += 1
            train_set.set_epoch(epoch)
            model.train()
            started, losses, correct, seen = time.time(), [], 0, 0
            batches = len(train_loader)

            for x, y in train_loader:
                for group in optimizer.param_groups:
                    group["lr"] = cosine_lr(step, total, lr, warmup=min(100, total // 10))
                x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                losses.append(loss.item())
                correct += int(((logits.detach() >= 0) == (y >= 0.5)).sum())
                seen += len(y)
                if step % PROGRESS_EVERY == 0:
                    write_progress(
                        out / "progress.json",
                        **progress_fields(
                            "training", phase, epoch, total_epochs, batches, losses, correct, seen, started
                        ),
                    )
            write_progress(
                out / "progress.json",
                **progress_fields("validating", phase, epoch, total_epochs, batches, losses, correct, seen, started),
            )
            val_loss, val_acc = _loss_and_accuracy(*predict_logits(model, val_loader, device))
            record = {
                "epoch": epoch,
                "phase": phase,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "train_accuracy": correct / max(1, seen),
                "val_accuracy": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - started, 1),
            }
            history.append(record)
            log(json.dumps(record))
            if val_loss < best["val_loss"] - 1e-4:
                best = {"val_loss": val_loss, "epoch": epoch}
                torch.save(model.state_dict(), out / "best.pt")
                stale = 0
            else:
                stale += 1
                if phase == "finetune" and stale >= cfg.patience:
                    log(f"early stopping after epoch {epoch}")
                    break
    (out / "history.json").write_text(json.dumps({"epochs": history}, indent=2))
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return {
        "history": history,
        "best": best,
        "checkpoint": str(out / "best.pt"),
        "device": str(device),
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    }
