# -*- coding: utf-8 -*-
"""出图：损失/准确率曲线、混淆矩阵、错误样本画廊。

所有图都存成 PNG 供报告直接引用；中文标签用微软雅黑，缺字时自动退回英文。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_CJK = None
for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"):
    try:
        path = font_manager.findfont(name, fallback_to_default=False)
        _CJK = name
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def plot_curves(csv_rows: Sequence[Dict[str, float]], out_png: str, title: str = "") -> None:
    """训练/测试损失 + 测试准确率双图。"""
    ep = [int(r["epoch"]) for r in csv_rows]
    tr = [float(r["train_loss"]) for r in csv_rows]
    te = [float(r["test_loss"]) for r in csv_rows]
    acc = [float(r["test_acc"]) * 100 for r in csv_rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ep, tr, "o-", label="train loss", lw=1.8, ms=4)
    ax[0].plot(ep, te, "s-", label="test loss", lw=1.8, ms=4)
    ax[0].set_xlabel("epoch" if not _CJK else "训练轮次 epoch")
    ax[0].set_ylabel("loss" if not _CJK else "损失")
    ax[0].set_title("损失曲线" if _CJK else "Loss")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].plot(ep, acc, "^-", color="tab:green", lw=1.8, ms=4)
    ax[1].set_xlabel("epoch" if not _CJK else "训练轮次 epoch")
    ax[1].set_ylabel("accuracy (%)" if not _CJK else "准确率 (%)")
    ax[1].set_title("测试集准确率" if _CJK else "Test accuracy")
    ax[1].grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], out_png: str, title: str = ""
) -> np.ndarray:
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    row = cm.sum(1, keepdims=True)
    norm = np.divide(cm, np.maximum(row, 1), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 8) if n <= 12 else (12, 11))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8 if n > 12 else 10)
    ax.set_yticklabels(labels, fontsize=8 if n > 12 else 10)
    ax.set_xlabel("预测 pred" if _CJK else "pred")
    ax.set_ylabel("真实 true" if _CJK else "true")
    ax.set_title(title or ("混淆矩阵（按行归一化）" if _CJK else "Confusion matrix"))
    fig.colorbar(im, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return cm


def plot_errors(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    out_png: str,
    max_cells: int = 40,
    title: str = "",
) -> int:
    """错误样本画廊：每格一张图，标注 真值→预测。返回错误总数。"""
    wrong = np.nonzero(y_true != y_pred)[0]
    if len(wrong) == 0:
        return 0
    idx = wrong[:max_cells]
    ncol = 8
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.35, nrow * 1.65))
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k >= len(idx):
            continue
        i = idx[k]
        ax.imshow(images[i], cmap="gray")
        ax.set_title(
            f"{labels[int(y_true[i])]}→{labels[int(y_pred[i])]}", fontsize=9, color="crimson"
        )
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return int(len(wrong))


def plot_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str], out_png: str
) -> List[Dict[str, float]]:
    """每类 precision / recall / F1 横向条形图。"""
    n = len(labels)
    rows = []
    for c in range(n):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": labels[c], "precision": prec, "recall": rec, "f1": f1, "support": tp + fn})

    fig, ax = plt.subplots(figsize=(max(6, n * 0.28), 4.2))
    xs = np.arange(n)
    ax.bar(xs - 0.25, [r["precision"] for r in rows], 0.25, label="precision")
    ax.bar(xs, [r["recall"] for r in rows], 0.25, label="recall")
    ax.bar(xs + 0.25, [r["f1"] for r in rows], 0.25, label="F1")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("每类 P / R / F1" if _CJK else "Per-class P/R/F1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return rows
