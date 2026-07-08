"""
Network architectures for coarse (SimpleHRNet) and refine (RefineNet) stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual block with two 3×3 convolutions."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class SimpleHRNet(nn.Module):
    """
    Coarse stage: produces a 128×128 heatmap from a 512×512 RGB input.

    Input  : (B, 3, 512, 512) in [0, 1]
    Output : (B, 1, 128, 128) — raw logits (no sigmoid).
    """

    HEATMAP_SIZE = 128

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.branch1 = BasicBlock(64)
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            BasicBlock(128),
        )
        self.fuse = nn.Conv2d(64 + 128, 128, 1)
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b2 = F.interpolate(b2, size=b1.shape[-2:],
                           mode="bilinear", align_corners=False)
        x = self.fuse(torch.cat([b1, b2], dim=1))
        x = self.head(x)
        return F.interpolate(x, size=(self.HEATMAP_SIZE, self.HEATMAP_SIZE),
                             mode="bilinear", align_corners=False)


class RefineNet(nn.Module):
    """
    Refine stage: predicts a normalized (dx, dy) offset from a 128×128 crop.
    The offset is constrained to [-1.0, 1.0] via Tanh to prevent directional bias.

    Input  : (B, 3, 128, 128) in [0, 1]
    Output : (B, 2) — normalized offset.
    """

    CROP_SIZE = 128

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        # 128 channels * 16 * 16 spatial dims = 32768
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256), 
            nn.ReLU(),
            nn.Dropout(0.1),  # Breaks fixed directional biases
            nn.Linear(256, 2),
            nn.Tanh()         # Constrains output between -1.0 and 1.0
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))