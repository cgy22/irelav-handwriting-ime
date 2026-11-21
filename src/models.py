# -*- coding: utf-8 -*-
"""模型定义。

两个模型，职责分开：
  * MnistNet —— 实验①的教材级 CNN，10 类，28x28，参数量约 0.3M
  * GlyphNet —— 实验②的 100 类字形网络，28x28，GAP 收尾，参数量约 0.2M

都是纯卷积 + BN + ReLU，不用任何 XPU 上可能缺失的算子（如某些 padding 模式、
grid_sample 之类），保证 bf16 下能稳定跑。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MnistNet(nn.Module):
    """实验①：经典 3 卷积 + 2 全连接。"""

    def __init__(self, n_classes: int = 10, width: int = 32, p_drop: float = 0.4):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1, bias=False), nn.BatchNorm2d(2 * w), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * w * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class GlyphNet(nn.Module):
    """实验②：100 类字形网络，GAP 收尾对位置更鲁棒。"""

    def __init__(self, n_classes: int = 100, width: int = 32, p_drop: float = 0.15):
        super().__init__()
        w = width

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(block(1, w), block(w, w), nn.MaxPool2d(2))          # 28 -> 14
        self.body = nn.Sequential(block(w, 2 * w), block(2 * w, 2 * w), nn.MaxPool2d(2))  # -> 7
        self.head = nn.Sequential(block(2 * w, 4 * w), block(4 * w, 4 * w))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(4 * w, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.head(self.body(self.stem(x)))
        h = self.pool(h).flatten(1)
        return self.fc(self.drop(h))


def build_model(kind: str, n_classes: int) -> nn.Module:
    kind = kind.lower()
    if kind == "mnist":
        return MnistNet(n_classes=n_classes)
    if kind == "glyph":
        return GlyphNet(n_classes=n_classes)
    raise ValueError(f"未知模型类型: {kind}")
