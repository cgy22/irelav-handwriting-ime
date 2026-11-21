# -*- coding: utf-8 -*-
"""实验②结果分析：三套评估集曲线、100 类混淆矩阵、易混字符对、错误画廊。

用法: python -m src.analyze_exp2
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import report  # noqa: E402
from .charset import CLASSES, CLASS2IDENT, IDX2CHAR  # noqa: E402

OUT = r"F:\mnist\runs\exp2_ext"
SETS = {"mnist": "MNIST 测试集", "emnist": "EMNIST letters", "synth": "字体合成留出集"}
_C2I = np.asarray(CLASS2IDENT, dtype=np.int64)


def read_metrics(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_all(rows: List[Dict[str, str]], out_png: str) -> None:
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))

    ax[0].plot(ep, [float(r["train_loss"]) for r in rows], "k-", lw=2, label="train loss")
    for k, name in SETS.items():
        ax[0].plot(ep, [float(r[f"{k}_loss"]) for r in rows], "o-", ms=3, lw=1.4, label=f"{name} loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("损失曲线（训练 vs 三套测试）" if report._CJK else "Loss")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    for k, name in SETS.items():
        ax[1].plot(ep, [float(r[f"{k}_acc_id"]) * 100 for r in rows], "o-", ms=3, lw=1.5, label=name)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("identity acc (%)")
    ax[1].set_title("字形身份准确率（忽略大小写）" if report._CJK else "Identity accuracy")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def source_report(name: str) -> Dict[str, float]:
    z = np.load(os.path.join(OUT, f"pred_{name}.npz"))
    labels, preds = z["labels"], z["preds"]
    id_true = _C2I[labels]
    id_pred = _C2I[preds]
    exact = float((labels == preds).mean())
    ident = float((id_true == id_pred).mean())
    print(f"  {SETS[name]:14s} n={len(labels):6d}  严格准确率={exact*100:5.2f}%  身份准确率={ident*100:5.2f}%")
    return {"exact": exact, "identity": ident}


def confusion_and_errors(name: str) -> None:
    z = np.load(os.path.join(OUT, f"pred_{name}.npz"))
    images, labels, preds = z["images"], z["labels"], z["preds"]
    cm = report.plot_confusion(labels, preds, CLASSES, os.path.join(OUT, f"confusion_{name}.png"),
                               f"实验② 混淆矩阵 · {SETS[name]}")
    report.plot_errors(images, labels, preds, CLASSES, os.path.join(OUT, f"errors_{name}.png"), 40,
                       f"实验② 错误样本 · {SETS[name]}")

    pairs = []
    n = cm.shape[0]
    for t in range(n):
        for p in range(n):
            if t != p and cm[t, p] > 0:
                pairs.append((int(cm[t, p]), CLASSES[t], CLASSES[p]))
    print(f"\n  {SETS[name]} 最容易混淆的 10 对:")
    for cnt, t, p in sorted(pairs, reverse=True)[:10]:
        print(f"    '{t}' -> '{p}': {cnt}")


def main() -> None:
    rows = read_metrics(os.path.join(OUT, "metrics.csv"))
    plot_all(rows, os.path.join(OUT, "curves.png"))

    print("三套评估集最终结果:")
    for name in SETS:
        source_report(name)

    last = rows[-1]
    print(f"\n末轮 mean identity acc = {float(last['mean_acc_id'])*100:.2f}%")
    print("\n逐集详析:")
    for name in ("emnist", "synth"):
        confusion_and_errors(name)


if __name__ == "__main__":
    main()
