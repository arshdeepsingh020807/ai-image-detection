"""Detector = pretrained timm backbone -> global average pool -> dropout -> single linear logit.

Because the head is pool + linear, Grad-CAM on the last feature map reduces exactly to
CAM = sum_c w_c * A_c, which the serving layer computes from ONNX alone.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

CANDIDATES = {
    "mobilenetv3_large_100": "mobilenetv3_large_100.ra_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "resnet50": "resnet50.a1_in1k",
}


class Detector(nn.Module):
    def __init__(self, name: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.name = name
        self.backbone = timm.create_model(
            CANDIDATES.get(name, name), pretrained=pretrained, num_classes=0, global_pool=""
        )
        self.backbone.eval()  # eval so the probe below cannot alter pretrained BatchNorm statistics
        with torch.no_grad():  # channel count of the unpooled map (MobileNetV3 applies its wide conv after pooling)
            channels = self.backbone.forward_features(torch.zeros(1, 3, 64, 64)).shape[1]
        self.backbone.train()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, 1)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.backbone.forward_features(x)
        pooled = fmap.mean(dim=(2, 3))
        return self.fc(self.dropout(pooled)), fmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(x)[0]

    def set_backbone_trainable(self, trainable: bool | int) -> int:
        """Freeze the backbone (False), unfreeze everything (True), or unfreeze the last N parameter tensors.

        The head always stays trainable. BatchNorm layers of frozen blocks are kept in eval mode by `train()`.
        Returns the number of trainable backbone parameters.
        """
        params = list(self.backbone.parameters())
        n = len(params) if trainable is True else 0 if trainable is False else int(trainable)
        for i, p in enumerate(params):
            p.requires_grad = i >= len(params) - n
        return sum(p.numel() for p in params if p.requires_grad)

    def train(self, mode: bool = True) -> Detector:
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                    p.requires_grad for p in module.parameters()
                ):
                    module.eval()
        return self
