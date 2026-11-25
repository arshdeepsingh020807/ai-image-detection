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


BUDGET_KEYS = ("max_train", "max_val", "epochs_head", "epochs_finetune", "batch_size", "seed")


def compare(manifest: pd.DataFrame, archs: list[str], base: TrainConfig, metrics_dir: Path = METRICS) -> dict:
    """Equal-budget run per candidate. Each finished row is cached beside its checkpoint and reused on a
    re-run with the same budget, so a crash part-way never repeats finished models."""
    device = pick_device(base.device)
    budget = {k: base.__dict__[k] for k in BUDGET_KEYS}
    models = []
    for arch in archs:
        cfg = TrainConfig(**{**base.__dict__, "arch": arch, "out_dir": f"{base.out_dir}/{arch}"})
        cached = Path(cfg.out_dir) / "comparison_row.json"
        if cached.exists() and (saved := json.loads(cached.read_text()))["budget"] == budget:
            models.append(saved["row"])
            print(f"reusing finished comparison of {arch}")
            continue
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
        cached.write_text(json.dumps({"budget": budget, "row": models[-1]}, indent=2))
        print(json.dumps(models[-1]))
    payload = {"models": models, "budget": budget}
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "model_comparison.json").write_text(json.dumps(payload, indent=2))
    return payload


def assemble_comparison(compare_dir: Path, metrics_dir: Path = METRICS) -> dict:
    """model_comparison.json from the cached rows of finished equal-budget runs (`compare` writes them)."""
    saved = [json.loads(p.read_text()) for p in sorted(compare_dir.glob("*/comparison_row.json"))]
    budgets = {json.dumps(s["budget"], sort_keys=True) for s in saved}
    if len(budgets) > 1:
        raise ValueError("cached comparison rows were trained with different budgets")
    payload = {"models": [s["row"] for s in saved], "budget": saved[0]["budget"] if saved else None}
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "model_comparison.json").write_text(json.dumps(payload, indent=2))
    return payload


def publish(
    manifest: pd.DataFrame,
    cfg: TrainConfig,
    checkpoint: Path,
    history_path: Path,
    metrics_dir: Path = METRICS,
    model_dir: Path = Path("artifacts/models"),
    plots_dir: Path = Path("artifacts/plots"),
) -> dict:
    """Evaluate a trained checkpoint, export it with its model card, measure latency and draw the figures."""
    history = json.loads(history_path.read_text())["epochs"]
    device = pick_device(cfg.device)
    model = load_trained(cfg.arch, str(checkpoint), device)
    files = evaluate(model, manifest, device, metrics_dir, model_name=cfg.arch, seed=cfg.seed, workers=cfg.num_workers)
    shutil.copy(history_path, metrics_dir / "history.json")
    summary = files["summary"]
    card = {
        "version": f"{cfg.arch}-{datetime.now(UTC):%Y%m%d}",
        "trained": True,
        "temperature": summary["temperature"],
        "band": summary["band"],
        "training": {
            "train_images": int((manifest.split == "train").sum()),
            "val_images": int((manifest.split == "val").sum()),
            "device": str(device),
            "epochs_completed": len(history),
            "best_epoch": min(history, key=lambda e: e["val_loss"])["epoch"],
            "config": cfg.__dict__,
        },
        "test_metrics": {k: summary["test"][k] for k in ("accuracy", "roc_auc", "f1", "false_positive_rate")},
    }
    onnx_path = export(model.cpu(), model_dir, card)
    (metrics_dir / "latency.json").write_text(json.dumps(measure(onnx_path), indent=2))
    plot_all(metrics_dir, plots_dir)
    return {"card": card}


def history_from_log(log: Path) -> dict:
    """Epoch records exactly as the trainer logged them (used when a run stopped before writing history.json)."""
    epochs = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"epoch"')]
    if not epochs:
        raise ValueError(f"no epoch records in {log}")
    return {"epochs": epochs}


def final(manifest: pd.DataFrame, cfg: TrainConfig, **paths: Path) -> dict:
    result = train(cfg, manifest)
    return publish(manifest, cfg, Path(result["checkpoint"]), Path(cfg.out_dir) / "history.json", **paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["compare", "final", "evaluate", "table"])
    parser.add_argument("--checkpoint", type=Path, help="evaluate: trained state dict (history.json beside it)")
    parser.add_argument("--workers", type=int, help="override num_workers (does not change results)")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/splits.csv.gz"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--arch", default="efficientnet_b0")
    parser.add_argument("--archs", nargs="+", default=list(CANDIDATES))
    args = parser.parse_args()
    if args.mode == "table":
        assemble_comparison(Path((TrainConfig.from_yaml(args.config) if args.config else TrainConfig()).out_dir))
        return
    manifest = pd.read_csv(args.manifest, dtype={"phash": str})
    manifest = manifest[manifest.keep]
    if args.mode == "compare":
        base = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
        compare(manifest, args.archs, base)
    else:
        overrides = {"arch": args.arch} | ({"num_workers": args.workers} if args.workers is not None else {})
        cfg = TrainConfig.from_yaml(args.config, **overrides) if args.config else TrainConfig(**overrides)
        if args.mode == "final":
            final(manifest, cfg)
        else:
            publish(manifest, cfg, args.checkpoint, args.checkpoint.parent / "history.json")


if __name__ == "__main__":
    main()
