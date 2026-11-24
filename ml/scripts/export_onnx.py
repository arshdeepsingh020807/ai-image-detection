"""Export a Detector to ONNX (logit + final feature map) and write model_card.json next to it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from detector_core.preprocess import CROP, MEAN, SHORT_SIDE, STD
from ml.training.model import Detector


class _Export(torch.nn.Module):
    def __init__(self, model: Detector):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_with_features(x)


def export(model: Detector, out_dir: Path, card: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    path = out_dir / "detector.onnx"
    torch.onnx.export(
        _Export(model).eval(),
        (torch.zeros(1, 3, CROP, CROP),),
        path,
        input_names=["image"],
        output_names=["logit", "features"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=64)}},
        dynamo=True,
        external_data=False,  # single self-contained file for deployment
    )
    full_card = {
        "architecture": model.name,
        "input": {
            "name": "image",
            "shape": ["batch", 3, CROP, CROP],
            "short_side": SHORT_SIDE,
            "crop": CROP,
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
        },
        "outputs": {"logit": "AI-generated logit, shape [batch, 1]", "features": "final feature map [batch, C, 7, 7]"},
        "cam_weights": model.fc.weight[0].detach().numpy().astype(np.float64).tolist(),
        "cam_bias": float(model.fc.bias.item()),
        "temperature": 1.0,
        "band": {"low": 0.5, "high": 0.5},
        "trained": False,
        "version": "untrained-standin",
        **card,
    }
    (out_dir / "model_card.json").write_text(json.dumps(full_card, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="efficientnet_b0")
    parser.add_argument("--checkpoint", type=Path, help="state dict; omit to export an UNTRAINED stand-in")
    parser.add_argument("--card", type=Path, help="JSON with temperature, band, version, trained, ... to merge in")
    parser.add_argument("--out", type=Path, default=Path("artifacts/models"))
    args = parser.parse_args()
    model = Detector(args.arch, pretrained=False)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    export(model, args.out, json.loads(args.card.read_text()) if args.card else {})
    print(f"wrote {args.out}/detector.onnx")


if __name__ == "__main__":
    main()
