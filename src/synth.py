# -*- coding: utf-8 -*-
"""字体合成数据工厂：给 100 类里"没有公开手写数据"的字符提供覆盖。

现实约束
--------
MNIST / EMNIST 只覆盖数字与字母，标点符号、Æ æ Œ œ Ê ê 没有任何公开手写
数据集。所以这些类只能靠系统字体渲染 + 强形变来"假装笔迹"。

做法
----
1. 多字体渲染（Arial / Times / Calibri / Segoe UI / Georgia / Verdana / Trebuchet / Courier，
   含粗体、斜体），保证笔画风格多样
2. 强形变：弹性扭曲（模拟书写抖动）+ 旋转 + 缩放 + 剪切 + 笔画粗细变化 + 模糊
3. 走**和真笔迹完全相同**的 canon.canonicalize 管线，保证域对齐

局限（必须写进报告）
------------------
字体合成拿到的是"印刷体 + 噪声"，不是真实笔迹的运动学特征（起笔、连笔、
压力梯度）。所以它负责"覆盖到"，真实精度主要靠用户手写微调来补。
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

from .canon import canonicalize

FONT_DIR = r"C:\Windows\Fonts"

# 覆盖常规/粗体/斜体三类风格
FONT_NAMES: List[str] = [
    "arial.ttf", "arialbd.ttf", "ariali.ttf",
    "times.ttf", "timesbd.ttf", "timesi.ttf",
    "calibri.ttf", "calibrib.ttf", "calibrii.ttf",
    "segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf",
    "georgia.ttf", "georgiab.ttf",
    "verdana.ttf", "verdanab.ttf",
    "trebuc.ttf", "trebucbd.ttf",
    "cour.ttf", "courbd.ttf",
]


def available_fonts() -> List[str]:
    out = []
    for n in FONT_NAMES:
        p = os.path.join(FONT_DIR, n)
        if os.path.exists(p):
            out.append(p)
    if not out:
        raise FileNotFoundError("系统字体目录里没找到可用字体")
    return out


_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(path, size)
    return _cache[key]


def render_char(
    ch: str,
    font_path: str,
    canvas: int = 96,
    font_size: int = 62,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """把一个字符渲染成 canvas x canvas 的灰度图（白字黑底）。"""
    rng = rng or np.random.default_rng()
    img = Image.new("L", (canvas, canvas), 0)
    d = ImageDraw.Draw(img)
    f = _font(font_path, font_size)
    try:
        d.text((canvas / 2, canvas / 2), ch, fill=255, font=f, anchor="mm")
    except Exception:
        return None
    a = np.asarray(img, dtype=np.float32)
    if a.max() < 20:  # 该字体缺这个字形
        return None
    return a


def _elastic(a: np.ndarray, rng: np.random.Generator, alpha: float, sigma: float) -> np.ndarray:
    """弹性形变：用平滑随机位移场重采样，模拟书写抖动。"""
    h, w = a.shape
    dx = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), sigma) * alpha
    dy = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([yy + dy, xx + dx])
    out = ndimage.map_coordinates(a, coords, order=1, mode="constant", cval=0.0)
    return out


def _affine(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """随机旋转 + 剪切 + 缩放（对包围盒整体作用）。"""
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="L")
    w, h = img.size
    ang = float(rng.uniform(-9, 9))
    shear = float(rng.uniform(-0.22, 0.22))
    scale = float(rng.uniform(0.86, 1.12))
    cx, cy = w / 2, h / 2
    # PIL 的 transform 用输出->输入的逆映射
    th = np.deg2rad(ang)
    ca, sa = np.cos(th), np.sin(th)
    sh = np.tan(shear) * 0.0 + shear
    m = np.array(
        [[ca / scale, (sa - sh) / scale, 0.0], [-sa / scale, ca / scale, 0.0]],
        dtype=np.float32,
    )
    m[0, 2] = cx - (m[0, 0] * cx + m[0, 1] * cy)
    m[1, 2] = cy - (m[1, 0] * cx + m[1, 1] * cy)
    out = img.transform((w, h), Image.AFFINE, tuple(m.flatten()), resample=Image.BILINEAR)
    return np.asarray(out, dtype=np.float32)


def augment_glyph(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """一套随机形变，模拟真实笔迹的多样性。"""
    out = a.copy()
    if rng.random() < 0.85:
        out = _elastic(out, rng, alpha=float(rng.uniform(0.8, 2.4)), sigma=float(rng.uniform(6, 12)))
    out = _affine(out, rng)
    # 笔画粗细：膨胀/腐蚀
    if rng.random() < 0.55:
        # 用半透明叠加代替硬膨胀/腐蚀，避免 @ 这类密集字形糊成一团
        if rng.random() < 0.5:
            alt = ndimage.grey_dilation(out, size=(3, 3))
        else:
            alt = ndimage.grey_erosion(out, size=(3, 3))
        k = float(rng.uniform(0.25, 0.55))
        out = np.clip(out * (1 - k) + alt * k, 0, 255)
    # 轻微模糊 + 噪声
    if rng.random() < 0.35:
        img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")
        img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 0.9))))
        out = np.asarray(img, dtype=np.float32)
    out = out + rng.normal(0, 4.0, out.shape)
    return np.clip(out, 0, 255)


def synth_one(ch: str, fonts: Sequence[str], rng: np.random.Generator) -> np.ndarray | None:
    fp = fonts[int(rng.integers(len(fonts)))]
    size = int(rng.integers(50, 78))
    a = render_char(ch, fp, canvas=96, font_size=size, rng=rng)
    if a is None:
        # 换字体重试
        for fp2 in fonts:
            a = render_char(ch, fp2, canvas=96, font_size=size, rng=rng)
            if a is not None:
                break
        if a is None:
            return None
    a = augment_glyph(a, rng)
    return canonicalize(a)


def build_synth(classes: Sequence[str], per_class: int, seed: int = 1234) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成数据集。返回 (images[N,28,28] float32 0..1, labels[N] int32)。"""
    from .charset import CHAR2IDX

    fonts = available_fonts()
    rng = np.random.default_rng(seed)
    imgs: List[np.ndarray] = []
    labs: List[int] = []
    missing: Dict[str, int] = {}
    for ch in classes:
        made = 0
        tries = 0
        while made < per_class and tries < per_class * 6:
            tries += 1
            img = synth_one(ch, fonts, rng)
            if img is None:
                continue
            imgs.append(img)
            labs.append(CHAR2IDX[ch])
            made += 1
        if made < per_class:
            missing[ch] = per_class - made
    if missing:
        print(f"[synth] 警告：{len(missing)} 个字符样本不足: {missing}")
    x = np.stack(imgs).astype(np.float32) if imgs else np.zeros((0, 28, 28), np.float32)
    y = np.asarray(labs, dtype=np.int32)
    return x, y


if __name__ == "__main__":
    from .charset import CLASSES

    fonts = available_fonts()
    print(f"可用字体 {len(fonts)} 个")
    # 每个字符试渲染一次，报告缺失
    rng = np.random.default_rng(0)
    bad = []
    for ch in CLASSES:
        ok = any(render_char(ch, f, font_size=60) is not None for f in fonts)
        if not ok:
            bad.append(ch)
    print("无可用字形的字符:", bad if bad else "无")
    x, y = build_synth(CLASSES, 4, seed=7)
    print("合成样本:", x.shape, y.shape, "值域", float(x.min()), float(x.max()))
