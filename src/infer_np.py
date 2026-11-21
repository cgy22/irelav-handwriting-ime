# -*- coding: utf-8 -*-
"""纯 numpy 推理：客户端不需要 PyTorch。

为什么可行
----------
GlyphNet 只有 4 个 3×3 卷积（无 bias，BN 已在导出时折叠掉）+ 2 次 2×2 最大池化
+ 全局平均池化 + 一个全连接，权重约 1.2 MB。这种规模用 numpy 的矩阵乘算，
单字 1~3 ms，比加载 PyTorch 更省事，也让分发包小了一个数量级。

用法:
    from src.infer_np import NumpyGlyphNet
    net = NumpyGlyphNet("data/model/glyph100.npz")
    probs = net.predict(gray28)      # (28,28) float32 -> (100,) softmax
"""
from __future__ import annotations

import json
import os

import numpy as np

_EPS = 1e-5


class NumpyGlyphNet:
    def __init__(self, npz_path: str, meta_path: str | None = None):
        self.path = npz_path
        self.meta_path = meta_path or os.path.splitext(npz_path)[0] + ".json"
        data = np.load(npz_path)
        self.w = [data[f"w{i}"] for i in range(6)]
        self.b = [data[f"b{i}"] for i in range(6)]
        self.fc_w = data["fc_w"]
        self.fc_b = data["fc_b"]

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.classes: list[str] = self.meta["classes"]

    # ------------------------------------------------------------ 基本算子
    @staticmethod
    def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """x: (H,W,Cin)  w: (Cout,Cin,3,3) -> (H,W,Cout)。"""
        h, wd, _ = x.shape
        p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="constant")
        out = np.zeros((h, wd, w.shape[0]), dtype=np.float32)
        for ky in range(3):
            for kx in range(3):
                patch = p[ky:ky + h, kx:kx + wd, :]          # (H,W,Cin)
                out += patch @ w[:, :, ky, kx].T             # (H,W,Cout)
        out += b
        return out

    @staticmethod
    def _pool2(x: np.ndarray) -> np.ndarray:
        h, w, c = x.shape
        h2, w2 = h // 2, w // 2
        return x[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, c).max(axis=(1, 3))

    # ------------------------------------------------------------ 前向
    def forward(self, gray28: np.ndarray) -> np.ndarray:
        """输入 (28,28) 0..1 白字黑底 -> 100 类 logits。"""
        x = np.asarray(gray28, dtype=np.float32)
        if x.shape != (28, 28):
            raise ValueError(f"输入必须是 28x28，收到 {x.shape}")
        x = x[:, :, None]                                     # (28,28,1)

        x = self._conv3x3(x, self.w[0], self.b[0]); x = np.maximum(x, 0)
        x = self._conv3x3(x, self.w[1], self.b[1]); x = np.maximum(x, 0)
        x = self._pool2(x)                                    # 14

        x = self._conv3x3(x, self.w[2], self.b[2]); x = np.maximum(x, 0)
        x = self._conv3x3(x, self.w[3], self.b[3]); x = np.maximum(x, 0)
        x = self._pool2(x)                                    # 7

        x = self._conv3x3(x, self.w[4], self.b[4]); x = np.maximum(x, 0)
        x = self._conv3x3(x, self.w[5], self.b[5]); x = np.maximum(x, 0)

        x = x.mean(axis=(0, 1))                               # GAP
        return self.fc_w @ x + self.fc_b

    def predict(self, gray28: np.ndarray) -> np.ndarray:
        logits = self.forward(gray28)
        z = logits - logits.max()
        e = np.exp(z)
        return e / e.sum()
