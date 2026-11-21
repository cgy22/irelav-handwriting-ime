# -*- coding: utf-8 -*-
"""数据层：四路数据（MNIST / EMNIST / 字体合成 / 用户笔迹）统一成数组 + 批次迭代器。

设计要点
--------
* **不用 torchvision，也不用 DataLoader 多进程**：数据总量只有几百 MB，全量常驻
  内存后用 numpy 花式索引切 batch 最快，而且避免了 Windows 上 spawn 4 个进程
  各拷一份数据的开销。
* 标签用 (y1, y2) 两个 int 表示：y2 = -1 是硬标签；y2 >= 0 表示在 y1/y2 之间
  各 0.5 的软标签（EMNIST 大小写合并标注就靠这个）。
* 增强在 batch 级用 torch 张量做（grid_sample + 形态学），比逐样本 PIL 快得多。
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import canon, idx_reader, synth
from .charset import CLASSES, CHAR2IDX, NUM_CLASSES, case_pair

CACHE_DIR = r"F:\mnist\data\cache"


# ------------------------------------------------------------------ 基础工具
def canon_many(imgs: np.ndarray) -> np.ndarray:
    """把 N 张不同尺寸的灰度图批量规范化到 28x28（float32, 0..1）。"""
    out = np.empty((len(imgs), 28, 28), dtype=np.float32)
    for i in range(len(imgs)):
        out[i] = canon.canonicalize(imgs[i])
    return out


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.npz")


def _load_cache(name: str) -> Optional[Dict[str, np.ndarray]]:
    p = _cache_path(name)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _save_cache(name: str, **arrays: np.ndarray) -> None:
    p = _cache_path(name)
    np.savez_compressed(p, **arrays)
    print(f"[cache] 已写入 {p} ({os.path.getsize(p) / 2**20:.1f} MB)")


# ------------------------------------------------------------------ 各数据源
def build_mnist(split: str = "train", use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MNIST -> (imgs float32 [N,28,28], y1, y2=-1)。走规范化管线。"""
    name = f"mnist_{split}_canon"
    if use_cache:
        z = _load_cache(name)
        if z is not None:
            return z["x"], z["y1"], z["y2"]
    x, y = idx_reader.load_mnist(split)
    xc = canon_many(x)
    y1 = np.array([CHAR2IDX[str(int(v))] for v in y], dtype=np.int64)
    y2 = np.full_like(y1, -1)
    if use_cache:
        _save_cache(name, x=xc.astype(np.float32), y1=y1, y2=y2)
    return xc, y1, y2


def build_emnist_letters(split: str = "train", use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EMNIST letters -> 26 类大小写合并。y1=小写形, y2=大写形（各 0.5）。"""
    name = f"emnist_letters_{split}_canon"
    if use_cache:
        z = _load_cache(name)
        if z is not None:
            return z["x"], z["y1"], z["y2"]
    x, y = idx_reader.load_emnist("letters", split)  # y 已转成 0..25
    xc = canon_many(x)
    y1 = np.empty(len(y), dtype=np.int64)
    y2 = np.empty(len(y), dtype=np.int64)
    for i, v in enumerate(y):
        lo, up = case_pair(chr(ord("a") + int(v)))
        y1[i] = CHAR2IDX[lo]
        y2[i] = CHAR2IDX[up]
    if use_cache:
        _save_cache(name, x=xc.astype(np.float32), y1=y1, y2=y2)
    return xc, y1, y2


def build_synth_cached(per_class: int = 400, seed: int = 20261121, use_cache: bool = True):
    """字体合成 -> 100 类硬标签。首次生成较慢，结果缓存到 npz。"""
    name = f"synth_{per_class}"
    if use_cache:
        z = _load_cache(name)
        if z is not None:
            return z["x"], z["y1"], z["y2"]
    t0 = time.time()
    x, y = synth.build_synth(CLASSES, per_class, seed=seed)
    print(f"[synth] 生成 {len(x)} 张，用时 {time.time() - t0:.1f}s")
    y2 = np.full_like(y, -1)
    if use_cache:
        _save_cache(name, x=x, y1=y.astype(np.int64), y2=y2)
    return x, y.astype(np.int64), y2


# --------------------------------------------------------------- 批次与增强
def _affine_matrices(n: int, gen: torch.Generator, dev: torch.device) -> torch.Tensor:
    ang = (torch.rand(n, generator=gen) * 2 - 1) * (9.0 * np.pi / 180.0)
    scale = 0.90 + torch.rand(n, generator=gen) * 0.20
    tx = (torch.rand(n, generator=gen) * 2 - 1) * 0.10
    ty = (torch.rand(n, generator=gen) * 2 - 1) * 0.10
    shear = (torch.rand(n, generator=gen) * 2 - 1) * 0.12
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos / scale
    theta[:, 0, 1] = (-sin + shear) / scale
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin / scale
    theta[:, 1, 1] = cos / scale
    theta[:, 1, 2] = ty
    return theta.to(dev)


def augment_batch(x: torch.Tensor, gen: torch.Generator, strength: float = 1.0) -> torch.Tensor:
    """batch 级在线增强：仿射 + 粗细变化 + 噪声。x 在 CPU 上 float32 0..1。"""
    n = x.shape[0]
    theta = _affine_matrices(n, gen, x.device)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")

    # 笔画粗细：对随机一半样本做膨胀/腐蚀
    mask = torch.rand(n, 1, 1, 1, generator=gen) < 0.5
    if mask.any():
        thick = F.max_pool2d(x, 3, stride=1, padding=1)
        thin = -F.max_pool2d(-x, 3, stride=1, padding=1)
        pick = torch.rand(n, 1, 1, 1, generator=gen)
        alt = torch.where(pick < 0.5, thick, thin)
        alpha = 0.45 * strength
        x = torch.where(mask, x * (1 - alpha) + alt * alpha, x)

    if strength > 0:
        x = x + torch.randn(x.shape, generator=gen) * (0.02 * strength)
    return x.clamp_(0.0, 1.0)


def make_batches(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    order: np.ndarray,
    batch_size: int,
    device: torch.device,
    train: bool,
    gen: Optional[torch.Generator] = None,
    strength: float = 1.0,
):
    """按 order 切 batch，返回 (xb[bf16/fp32 on device, 1,28,28], y1, y2)。"""
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        xb = torch.from_numpy(x[idx]).unsqueeze(1)  # float32 on CPU
        if train and gen is not None:
            xb = augment_batch(xb, gen, strength)
        # 注意：XPU 上 channels_last 会触发 level_zero 39 号显存错误（已实测复现），
        # 所以这里统一用默认 NCHW。bf16 带来的收益已足够（比 CPU 快约 8 倍）。
        xb = xb.to(device, non_blocking=True)
        yield xb, torch.from_numpy(y1[idx]).to(device), torch.from_numpy(y2[idx]).to(device)


def class_balanced_order(y1: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """按类别反频率采样，缓解"数字/字母样本远多于标点"的失衡。"""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y1, minlength=NUM_CLASSES).astype(np.float64)
    w = np.zeros_like(counts)
    nz = counts > 0
    w[nz] = 1.0 / np.sqrt(counts[nz])
    p = w[y1]
    p = p / p.sum()
    return rng.choice(len(y1), size=n_samples, replace=True, p=p)


def build_soft_targets(y1: torch.Tensor, y2: torch.Tensor, n_classes: int = NUM_CLASSES) -> torch.Tensor:
    """(B, n_classes) 软标签：y2<0 时 one-hot，否则两项各 0.5。"""
    t = torch.zeros(y1.shape[0], n_classes, device=y1.device, dtype=torch.float32)
    hard = y2 < 0
    t.scatter_(1, y1.unsqueeze(1), torch.where(hard, 1.0, 0.5).unsqueeze(1).float())
    if (~hard).any():
        t.scatter_(1, y2[~hard].unsqueeze(1), 0.5)
    return t


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=1)
    return -(target * logp).sum(dim=1).mean()


def group_ce(
    logits: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, label_smoothing: float = 0.15
) -> torch.Tensor:
    """身份组交叉熵（identity-group CE）。

    对"大小写合并"的样本（y2>=0），只要求概率质量落在 {y1, y2} 这个组里：
        loss = -log(p(y1) + p(y2))
    而不是强迫模型在两种形态之间 50/50 对半分。

    这一点很关键：孤立字形的形状本来就不携带大小写信息，用 50/50 软标签等于
    让模型"学会犹豫"，损失降不下去、字形判别力也被稀释。按组算质量后，模型
    可以专心把 mass 放在正确身份上，大小写交给 UI 的几何启发式去裁决。
    """
    logp = torch.log_softmax(logits.float(), dim=1)
    lp1 = logp.gather(1, y1.view(-1, 1)).squeeze(1)
    hard = y2 < 0

    # 组内对数概率（硬标签就是该类本身）
    tgt = lp1.clone()
    if (~hard).any():
        lp2 = logp[~hard].gather(1, y2[~hard].view(-1, 1)).squeeze(1)
        tgt[~hard] = torch.logsumexp(torch.stack([lp1[~hard], lp2], dim=1), dim=1)

    # 标签平滑：log 概率上界被托住，避免 group CE 无约束地把 logits 推到爆
    # （实测：不加平滑时 logits 膨胀，BN 统计量承载了"校准"作用，
    #  评估曲线出现 ±20% 的震荡）
    if label_smoothing > 0:
        loss = -(1.0 - label_smoothing) * tgt - label_smoothing * logp.mean(dim=1)
    else:
        loss = -tgt
    return loss.mean()


# ------------------------------------------------------------------ 数据集汇总
def load_ext_data(per_class: int = 400, cache: bool = True):
    """实验②的全部训练数据。"""
    mx, my1, my2 = build_mnist("train", use_cache=cache)
    ex, ey1, ey2 = build_emnist_letters("train", use_cache=cache)
    sx, sy1, sy2 = build_synth_cached(per_class, use_cache=cache)
    x = np.concatenate([mx, ex, sx], axis=0)
    y1 = np.concatenate([my1, ey1, sy1], axis=0)
    y2 = np.concatenate([my2, ey2, sy2], axis=0)
    print(f"[data] 训练集合计 {len(x)} 张 (MNIST {len(mx)} / EMNIST {len(ex)} / synth {len(sx)})")
    return x, y1, y2


if __name__ == "__main__":
    t0 = time.time()
    x, y1, y2 = build_mnist("train")
    print("MNIST canon:", x.shape, x.dtype, f"{time.time()-t0:.1f}s")
    t1 = time.time()
    ex, ey1, ey2 = build_emnist_letters("train")
    print("EMNIST letters canon:", ex.shape, f"{time.time()-t1:.1f}s")
    print("软标签样例:", ex[0].shape, ey1[0], ey2[0])
