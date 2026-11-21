# -*- coding: utf-8 -*-
"""手写数据体检：检查标签与模型识别是否一致，并把样本渲染成一张核对图。

用途有两个：
  1. 采集过程中定期自查，及早发现"标签整体错位"这类系统性错误
  2. 报告里作为数据质量证据

用法:
    python -m src.inspect_user              # 检查 train 与 test
    python -m src.inspect_user train
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import userdata
from .charset import CLASSES, NUM_CLASSES
from .models import GlyphNet
from .train_common import amp_context, pick_device

EXP2 = r"F:\mnist\runs\exp2_ext\best.pt"
OUT = r"F:\mnist\runs\user_eval"
CELL = 56  # 每格像素


def _sheet(x: np.ndarray, labels, preds, probs, out_png: str, cols: int = 21) -> None:
    n = len(x)
    rows = max(1, (n + cols - 1) // cols)
    W = cols * (CELL + 4) + 4
    H = rows * (CELL + 26) + 4
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i in range(n):
        r, c = divmod(i, cols)
        ox, oy = 4 + c * (CELL + 4), 4 + r * (CELL + 26)
        g = Image.fromarray((x[i] * 255).astype(np.uint8)).resize((CELL, CELL), Image.NEAREST)
        canvas.paste(g.convert("RGB"), (ox, oy + 18))
        ok = labels[i] == preds[i]
        color = (30, 140, 60) if ok else (200, 40, 40)
        d.text((ox + 1, oy + 3), f"{labels[i]}→{preds[i]}", fill=color)
        d.rectangle([ox - 1, oy + 17, ox + CELL, oy + 18 + CELL], outline=color)
    canvas.save(out_png)
    print(f"核对图: {out_png}  ({n} 张)")


def inspect(split: str, device: torch.device, model) -> dict:
    x, y, raw = userdata.build_arrays(split)
    if len(x) == 0:
        print(f"[{split}] 空")
        return {"n": 0}
    with torch.no_grad():
        logits = []
        for i in range(0, len(x), 128):
            xb = torch.from_numpy(x[i : i + 128]).unsqueeze(1).to(device)
            with amp_context(device):
                logits.append(model(xb).float().cpu())
        logits = torch.cat(logits)
        probs = torch.softmax(logits, 1).numpy()
    preds_idx = probs.argmax(1)
    preds = [CLASSES[int(i)] for i in preds_idx]
    probs_max = probs.max(1)

    mismatch = [(raw[i], preds[i], float(probs_max[i])) for i in range(len(raw)) if raw[i] != preds[i]]
    match = len(raw) - len(mismatch)
    print(f"[{split}] {len(raw)} 条：标签与模型一致 {match} 条（{match/len(raw)*100:.1f}%），不一致 {len(mismatch)} 条")
    if mismatch:
        print("  不一致的前 10 条（标签 → 模型识别, 置信度）:")
        for lab, pr, p in mismatch[:10]:
            print(f"     {lab} → {pr}  ({p*100:.0f}%)")
    from collections import Counter

    print("  标签分布:", dict(sorted(Counter(raw).items())))
    os.makedirs(OUT, exist_ok=True)
    _sheet(x, raw, preds, probs_max, os.path.join(OUT, f"inspect_{split}.png"))
    return {"n": len(raw), "match": match, "mismatch": len(mismatch)}


def main() -> None:
    device = pick_device()
    if not os.path.exists(EXP2):
        print("找不到 exp2_ext/best.pt")
        return
    ck = torch.load(EXP2, map_location=device)
    model = GlyphNet(NUM_CLASSES).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    splits = sys.argv[1:] or ["train", "test"]
    summary = {}
    for sp in splits:
        summary[sp] = inspect(sp, device, model)
    print("\n体检结论:")
    for sp, d in summary.items():
        if d.get("n"):
            frac = d["mismatch"] / d["n"]
            verdict = "正常" if frac < 0.15 else "⚠ 异常，疑似标签整体错位"
            print(f"  {sp}: {d['n']} 条，不一致率 {frac*100:.1f}% → {verdict}")


if __name__ == "__main__":
    main()
