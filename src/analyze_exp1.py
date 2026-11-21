# -*- coding: utf-8 -*-
"""实验①结果分析：曲线、混淆矩阵、错误画廊、每类指标。

用法: python -m src.analyze_exp1
产出 runs/exp1_mnist/ 下的 loss_curves.png / confusion.png / errors.png / per_class.png
"""
from __future__ import annotations

import csv
import os

import numpy as np

from . import report

OUT = r"F:\mnist\runs\exp1_mnist"
LABELS = list("0123456789")


def read_metrics(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main() -> None:
    rows = read_metrics(os.path.join(OUT, "metrics.csv"))
    report.plot_curves(rows, os.path.join(OUT, "loss_curves.png"), "实验① MNIST 基线 CNN")

    z = np.load(os.path.join(OUT, "test_pred.npz"))
    images, labels, preds = z["images"], z["labels"], z["preds"]
    acc = float((labels == preds).mean())
    print(f"测试集准确率: {acc*100:.2f}%  (n={len(labels)})")

    cm = report.plot_confusion(labels, preds, LABELS, os.path.join(OUT, "confusion.png"), "实验① 混淆矩阵")
    n_err = report.plot_errors(images, labels, preds, LABELS, os.path.join(OUT, "errors.png"), 32, "实验① 识别错误样本")
    rows_pr = report.plot_class_report(labels, preds, LABELS, os.path.join(OUT, "per_class.png"))

    print(f"错误样本数: {n_err}")
    print("\n最容易被混淆的类别对（按错误数）:")
    pairs = []
    for t in range(10):
        for p in range(10):
            if t != p and cm[t, p] > 0:
                pairs.append((cm[t, p], LABELS[t], LABELS[p]))
    for cnt, t, p in sorted(pairs, reverse=True)[:8]:
        print(f"  {t} -> {p}: {cnt} 次")

    print("\n每类召回率最低的 3 个:")
    for r in sorted(rows_pr, key=lambda d: d["recall"])[:3]:
        print(f"  '{r['class']}': recall={r['recall']*100:.1f}% support={r['support']}")

    # 损失曲线收尾状态
    last = rows[-1]
    print(
        f"\n末轮: train_loss={last['train_loss']} test_loss={last['test_loss']} "
        f"gap={float(last['test_loss'])-float(last['train_loss']):+.4f}"
    )


if __name__ == "__main__":
    main()
