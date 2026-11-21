# -*- coding: utf-8 -*-
"""IDX 格式读取器（MNIST / EMNIST），不依赖 torchvision。

两个必须知道的坑
----------------
1. EMNIST 的 IDX 图像是**转置存储**的，直接 reshape(28,28) 会得到躺倒的字，
   必须做一次 transpose（本模块 transpose=True 时自动处理）。已用样本图目视验证：
   只需转置，不需要 flip。
2. EMNIST-letters 的标签是 1..26，不是 0..25，使用前要减 1。
"""
from __future__ import annotations

import gzip
import os
from typing import Tuple

import numpy as np

DATA_DIR = os.environ.get("MNIST_DATA_DIR", r"F:\mnist\mnist_data")


def _read_idx(path: str) -> np.ndarray:
    """读取一个 .gz 压缩的 IDX 文件，返回 uint8 数组。"""
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF
        dims = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8)
    if ndim == 1:
        return arr.reshape(dims[0]).astype(np.uint8)
    return arr.reshape(dims).astype(np.uint8)


def _paths(prefix: str, split: str) -> Tuple[str, str]:
    img = os.path.join(DATA_DIR, f"{prefix}-{split}-images-idx3-ubyte.gz")
    lab = os.path.join(DATA_DIR, f"{prefix}-{split}-labels-idx1-ubyte.gz")
    return img, lab


def load_mnist(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """MNIST。返回 (images[N,28,28] uint8, labels[N] int64)。"""
    tag = "train" if split == "train" else "t10k"
    img_p = os.path.join(DATA_DIR, f"mnist-{tag}-images-idx3-ubyte.gz")
    lab_p = os.path.join(DATA_DIR, f"mnist-{tag}-labels-idx1-ubyte.gz")
    return _read_idx(img_p), _read_idx(lab_p).astype(np.int64)


def load_emnist(
    kind: str = "balanced", split: str = "train", transpose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """EMNIST。kind ∈ {balanced, letters}。

    * transpose=True 时做转置矫正，得到正立字形
    * kind='letters' 时标签 1..26 会被转成 0..25
    """
    prefix = f"emnist-{kind}"
    img_p, lab_p = _paths(prefix, split)
    images = _read_idx(img_p)
    labels = _read_idx(lab_p).astype(np.int64)
    if transpose:
        images = np.transpose(images, (0, 2, 1))
    if kind == "letters":
        labels = labels - 1
    return images, labels


def load_emnist_mapping_chars(kind: str = "balanced") -> list:
    """读取并使用 mapping 表；letters 无 mapping 文件时退化为 a..z。"""
    from .charset import load_emnist_mapping

    for name in (f"emnist-{kind}-mapping.txt", f"emnist-balanced-mapping.txt"):
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return load_emnist_mapping(p)
    if kind == "letters":
        return [chr(ord("a") + i) for i in range(26)]
    raise FileNotFoundError(f"找不到 {kind} 的 mapping 文件: {DATA_DIR}")


if __name__ == "__main__":
    x, y = load_mnist("train")
    print("MNIST train", x.shape, y.shape, x.dtype, "labels", np.unique(y))
    b, yb = load_emnist("balanced", "train")
    print("EMNIST balanced", b.shape, "labels", yb.min(), yb.max(), np.unique(yb).size)
    l, yl = load_emnist("letters", "train")
    print("EMNIST letters", l.shape, "labels", yl.min(), yl.max(), np.unique(yl).size)
    print("mapping:", load_emnist_mapping_chars("balanced"))
