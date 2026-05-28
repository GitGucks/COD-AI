from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# Number of discrete buttons tracked per frame.
# Order: cross, circle, square, triangle, L1, R1, L2, R2, share, options, L3, R3, PS, touchpad
NUM_BUTTONS = 14


class GhostModel(nn.Module):
    """Behavioural cloning model: frame → controller state.

    Input:  RGB frame tensor (B, 3, 224, 224), normalised to ImageNet stats.
    Output: dict of named tensors, all in batch dimension B.
        left_stick   (B, 2)  tanh    [-1, 1]  left stick x, y
        right_stick  (B, 2)  tanh    [-1, 1]  right stick x, y
        triggers     (B, 2)  sigmoid [0, 1]   L2, R2 analogue
        buttons      (B, 14) sigmoid [0, 1]   per-button probability
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        backbone_out = 1280

        self.neck = nn.Sequential(
            nn.Linear(backbone_out, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.left_stick_head  = nn.Linear(256, 2)
        self.right_stick_head = nn.Linear(256, 2)
        self.triggers_head    = nn.Linear(256, 2)
        self.buttons_head     = nn.Linear(256, NUM_BUTTONS)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(frames)
        x = self.neck(features)
        return {
            "left_stick":  torch.tanh(self.left_stick_head(x)),
            "right_stick": torch.tanh(self.right_stick_head(x)),
            "triggers":    torch.sigmoid(self.triggers_head(x)),
            "buttons":     torch.sigmoid(self.buttons_head(x)),
        }


# ImageNet normalisation constants (used by both collect and ghost).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
