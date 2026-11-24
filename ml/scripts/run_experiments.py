"""Model comparison and final training.

python -m ml.scripts.run_experiments compare            # short equal-budget runs for every candidate
python -m ml.scripts.run_experiments final --arch NAME  # full run, evaluation, ONNX export, plots
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch

from ml.evaluation.evaluate import evaluate, predict_logits
from ml.evaluation.latency import measure
from ml.evaluation.metrics import classification_metrics
from ml.evaluation.plots import plot_all
from ml.scripts.export_onnx import export
from ml.training.model import CANDIDATES, Detector
from ml.training.train import TrainConfig, _sample, pick_device, train

METRICS = Path("artifacts/metrics")


def load_trained(arch: str, checkpoint: str, device: torch.device) -> Detector:
    model = Detector(arch, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def unseen_generator_rows(manifest: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """External in-scope JPEGs only: unseen generators without the PNG-vs-JPEG shortcut."""
    rows = manifest[(manifest.split == "external") & (manifest.scope == "in_scope") & (manifest.format == "JPEG")]
    return _sample(rows, cap, seed)


def compare(manifest: pd.DataFrame, archs: list[str], base: TrainConfig, metrics_dir: Path = METRICS) -> dict:
    device = pick_device(base.device)
    models = []
    for arch in archs:
        cfg = TrainConfig(**{**base.__dict__, "arch": arch, "out_dir": f"{base.out_dir}/{arch}"})
        result = train(cfg, manifest)
        model = load_trained(arch, result["checkpoint"], device)
        val = _sample(manifest[manifest.split == "val"], cfg.max_val, cfg.seed)
        external = unseen_generator_rows(manifest, cfg.max_val, cfg.seed)
        val_m = classification_metrics(
            val.label,
            torch.sigmoid(
                torch.from_numpy(predict_logits(model, val.path.tolist(), device, workers=cfg.num_workers))
            ).numpy(),
        )
        ext_m = classification_metrics(
            external.label,
            torch.sigmoid(
                torch.from_numpy(predict_logits(model, external.path.tolist(), device, workers=cfg.num_workers))
            ).numpy(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            timing = measure(export(model.cpu(), Path(tmp), {}), runs=30)
        models.append(
            {
                "name": arch,
                "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                "onnx_size_mb": timing["onnx_size_mb"],
                "validation_auc": val_m["roc_auc"],
                "external_auc": ext_m["roc_auc"],
                "false_positive_rate": val_m["false_positive_rate"],
                "validation_accuracy": val_m["accuracy"],
                "latency_ms": round(timing["cpu"]["p50_ms"], 1),
                "train_images": result["train_size"],
                "epochs": len(result["history"]),
            }
        )
        print(json.dumps(models[-1]))
    payload = {
        "models": models,
        "budget": {
            k: base.__dict__[k]
            for k in ("max_train", "max_val", "epochs_head", "epochs_finetune", "batch_size", "seed")
        },
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "model_comparison.json").write_text(json.dumps(payload, indent=2))
    return payload


def final(
    manifest: pd.DataFrame,
    cfg: TrainConfig,
    metrics_dir: Path = METRICS,
    model_dir: Path = Path("artifacts/models"),
    plots_dir: Path = Path("artifacts/plots"),
) -> dict:
    result = train(cfg, manifest)
    device = pick_device(cfg.device)
    model = load_trained(cfg.arch, result["checkpoint"], device)
    files = evaluate(model, manifest, device, metrics_dir, model_name=cfg.arch, seed=cfg.seed, workers=cfg.num_workers)
    shutil.copy(Path(cfg.out_dir) / "history.json", metrics_dir / "history.json")
    summary = files["summary"]
    card = {
        "version": f"{cfg.arch}-{datetime.now(UTC):%Y%m%d}",
        "trained": True,
        "temperature": summary["temperature"],
        "band": summary["band"],
        "training": {
            "train_images": result["train_size"],
            "val_images": result["val_size"],
            "device": result["device"],
            "best_epoch": result["best"]["epoch"],
            "config": cfg.__dict__,
        },
        "test_metrics": {k: summary["test"][k] for k in ("accuracy", "roc_auc", "f1", "false_positive_rate")},
    }
    onnx_path = export(model.cpu(), model_dir, card)
    (metrics_dir / "latency.json").write_text(json.dumps(measure(onnx_path), indent=2))
    plot_all(metrics_dir, plots_dir)
    return {"result": result, "card": card}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["compare", "final"])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/splits.csv.gz"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--arch", default="efficientnet_b0")
    parser.add_argument("--archs", nargs="+", default=list(CANDIDATES))
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"phash": str})
    manifest = manifest[manifest.keep]
    if args.mode == "compare":
        base = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
        compare(manifest, args.archs, base)
    else:
        cfg = TrainConfig.from_yaml(args.config, arch=args.arch) if args.config else TrainConfig(arch=args.arch)
        final(manifest, cfg)


if __name__ == "__main__":
    main()
