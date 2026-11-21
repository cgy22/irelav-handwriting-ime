# -*- coding: utf-8 -*-
"""在自己的手写测试集上评测（课程要求 3 的"自造测试集"部分）。

两次评测：
  1. 实验① 的 MnistNet 在用户手写**数字**上的准确率 —— 与 MNIST 官方测试集对比，量化泛化落差
  2. 实验② 的 GlyphNet（100 类）在用户全部手写样本上的准确率

用法: python -m src.eval_user
"""
from __future__ import annotations

import os

import numpy as np
import torch

from . import report, userdata
from .charset import CLASSES, CHAR2IDX, NUM_CLASSES
from .models import GlyphNet, MnistNet
from .train_common import amp_context, pick_device

EXP1 = r"F:\mnist\runs\exp1_mnist\best.pt"
EXP2 = r"F:\mnist\runs\exp2_ext\best.pt"
OUT = r"F:\mnist\runs\user_eval"
DIGITS = [str(d) for d in range(10)]


@torch.no_grad()
def predict(model, x, device, batch=64):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i : i + batch]).unsqueeze(1).to(device)
        with amp_context(device):
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def main() -> None:
    device = pick_device()
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    x_all, y_all, raw_all = userdata.build_arrays("test")
    log(f"用户测试集样本数: {len(x_all)}")
    if len(x_all) == 0:
        log("还没有测试集样本，请先在面板的「采集」里选择测试集写入。")
        with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    # ---- 1) MNIST 模型在用户数字上
    if os.path.exists(EXP1) and len(x_all):
        ck = torch.load(EXP1, map_location=device)
        m1 = MnistNet(10).to(device)
        m1.load_state_dict(ck["model"])
        mask = np.array([r in DIGITS for r in raw_all])
        xd = x_all[mask]
        if len(xd):
            preds1 = predict(m1, xd, device)
            true1 = np.array([int(r) for r in np.array(raw_all)[mask]], dtype=np.int64)
            acc1 = float((preds1 == true1).mean())
            log(f"\n[实验① MnistNet] 用户手写数字 {len(xd)} 张，准确率 {acc1*100:.2f}%")
            log(f"  对照：MNIST 官方测试集 99.58%  ->  泛化落差 {99.58 - acc1*100:+.2f} 个百分点")
            report.plot_confusion(true1, preds1, DIGITS, os.path.join(OUT, "user_digits_confusion.png"),
                                  "用户手写数字 · 混淆矩阵")
            report.plot_errors((xd * 255).astype(np.uint8), true1, preds1, DIGITS,
                               os.path.join(OUT, "user_digits_errors.png"), 24, "用户手写数字 · 错误样本")
            per = []
            for d in range(10):
                m = true1 == d
                if m.any():
                    per.append(f"{d}:{float((preds1[m] == d).mean())*100:.0f}%")
            log("  每类召回: " + " ".join(per))

    # ---- 2) 100 类模型在全部用户样本上
    if os.path.exists(EXP2):
        ck = torch.load(EXP2, map_location=device)
        m2 = GlyphNet(NUM_CLASSES).to(device)
        m2.load_state_dict(ck["model"])
        preds2 = predict(m2, x_all, device)
        acc2 = float((preds2 == np.asarray(y_all)).mean())
        log(f"\n[实验② GlyphNet] 用户全部手写 {len(x_all)} 张，严格准确率 {acc2*100:.2f}%")
        report.plot_confusion(np.asarray(y_all), preds2, CLASSES, os.path.join(OUT, "user_all_confusion.png"),
                              "用户手写集 · 100 类混淆矩阵")
        report.plot_errors((x_all * 255).astype(np.uint8), np.asarray(y_all), preds2, CLASSES,
                           os.path.join(OUT, "user_all_errors.png"), 40, "用户手写集 · 错误样本")
        log(f"  错误样本图: {os.path.join(OUT, 'user_all_errors.png')}")

    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
