# -*- coding: utf-8 -*-
"""把训练好的权重导成纯 numpy 能读的 .npz（训练侧才需要 PyTorch）。

导出时把 BatchNorm 折叠进卷积：y = gamma*(x-mean)/sqrt(var+eps)+beta
=>  w' = w*scale,  b' = beta - mean*scale
这样推理只剩"卷积 + ReLU + 池化 + 全连接"，不需要任何深度学习框架。

用法: python -m src.export_npz
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.charset import CLASSES
from src.models import GlyphNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "runs", "exp2_final", "best.pt")
OUT_DIR = os.path.join(ROOT, "data", "model")
OUT_NPZ = os.path.join(OUT_DIR, "glyph100.npz")
OUT_META = os.path.join(OUT_DIR, "glyph100.json")

BLOCKS = ["stem.0", "stem.1", "body.0", "body.1", "head.0", "head.1"]


def fold(conv_w: np.ndarray, bn: dict, eps: float = 1e-5):
    """卷积权重 + BN 参数 -> 折叠后的 (w, b)。"""
    gamma = bn["weight"]
    beta = bn["bias"]
    mean = bn["running_mean"]
    var = bn["running_var"]
    scale = gamma / np.sqrt(var + eps)
    w = conv_w * scale.reshape(-1, 1, 1, 1)
    b = beta - mean * scale
    return w.astype(np.float32), b.astype(np.float32)


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu")
    sd = {k: v.detach().cpu().numpy() for k, v in ck["model"].items()}

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(BLOCKS):
        w, b = fold(sd[f"{name}.0.weight"], {
            "weight": sd[f"{name}.1.weight"],
            "bias": sd[f"{name}.1.bias"],
            "running_mean": sd[f"{name}.1.running_mean"],
            "running_var": sd[f"{name}.1.running_var"],
        })
        out[f"w{i}"] = w
        out[f"b{i}"] = b

    out["fc_w"] = sd["fc.weight"].astype(np.float32)
    out["fc_b"] = sd["fc.bias"].astype(np.float32)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **out)

    meta = {
        "arch": "glyph100",
        "blocks": BLOCKS,
        "classes": list(CLASSES),
        "n_classes": len(CLASSES),
        "width": int(out["w0"].shape[0]),
        "input": [1, 28, 28],
        "weights": os.path.basename(OUT_NPZ),
        "source": os.path.relpath(CKPT, ROOT).replace("\\", "/"),
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    size = os.path.getsize(OUT_NPZ) / 1024
    print(f"[ok] {OUT_NPZ}  {size:.0f} KB  ({meta['n_classes']} 类, width={meta['width']})")
    print(f"[ok] {OUT_META}")


if __name__ == "__main__":
    main()
