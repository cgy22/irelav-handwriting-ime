# -*- coding: utf-8 -*-
"""统一的字形规范化管线（全项目唯一入口）。

MNIST / EMNIST / 字体合成 / 用户笔迹 四个数据源必须走同一套变换，否则模型
学到的差异来自"图像管线不一致"而不是"人的书写习惯"，泛化实验就失去意义。

变换步骤：
  1) 二值化取墨迹，求包围盒
  2) 等比缩放，使**长边 = 20px**（保持长宽比，绝不拉伸）
  3) 按**质心**居中放入 28x28
  4) 反色为白字黑底、归一化到 0..1

保持长宽比是关键：标点 "-"、"|"、"'" 的形状信息全在长宽比里，一拉伸就全糊了。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

OUT = 28
BOX = 20
INK_THRESHOLD = 16


def _as_float(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.size and a.max() <= 1.0:
        a = a * 255.0
    return a


def ink_mask(a: np.ndarray, thresh: float = INK_THRESHOLD) -> np.ndarray:
    return a > thresh


def canonicalize(img: np.ndarray, out: int = OUT, box: int = BOX) -> np.ndarray:
    """任意灰度图 -> out x out 规范图（float32, 0..1, 白字黑底）。"""
    a = _as_float(img)
    m = ink_mask(a)
    if not m.any():
        return np.zeros((out, out), dtype=np.float32)

    ys, xs = np.nonzero(m)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop = a[y0 : y1 + 1, x0 : x1 + 1]
    h, w = crop.shape

    s = box / float(max(h, w))
    nh = max(1, int(round(h * s)))
    nw = max(1, int(round(w * s)))
    glyph = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8), mode="L").resize(
        (nw, nh), Image.LANCZOS
    )
    g = np.asarray(glyph, dtype=np.float32)

    # 质心居中的落点
    gm = g > INK_THRESHOLD
    if gm.any():
        gy, gx = np.nonzero(gm)
        cy = float((gy * g[gm]).sum() / g[gm].sum())
        cx = float((gx * g[gm]).sum() / g[gm].sum())
    else:
        cy, cx = (nh - 1) / 2.0, (nw - 1) / 2.0

    canvas = np.zeros((out, out), dtype=np.float32)
    top = int(round(out / 2.0 - cy))
    left = int(round(out / 2.0 - cx))

    # 逐区域拷贝，避免越界
    sy0, sx0 = max(0, -top), max(0, -left)
    dy0, dx0 = max(0, top), max(0, left)
    hh = min(nh - sy0, out - dy0)
    ww = min(nw - sx0, out - dx0)
    if hh > 0 and ww > 0:
        canvas[dy0 : dy0 + hh, dx0 : dx0 + ww] = g[sy0 : sy0 + hh, sx0 : sx0 + ww]
    return np.clip(canvas / 255.0, 0.0, 1.0)


def normalize_strokes(strokes):
    """容忍两种笔迹输入：[[(x,y),...], ...] 与 [(x,y), ...]（单笔被 JSON 拍平）。

    前端正常传的是"笔画列表"，但 PowerShell / 手写测试脚本很容易把外层数组拍平，
    与其在后面报 TypeError，不如在这里统一收口。
    """
    if not strokes:
        return []
    try:
        head = strokes[0][0]
    except (TypeError, IndexError):
        return []
    if isinstance(head, (int, float)):
        # 外层被拍平：整段点序列当作一笔
        return [list(strokes)]
    return [list(s) for s in strokes if s]


def render_strokes(
    strokes: Sequence[Sequence[Sequence[float]]],
    size: int = 224,
    width_ratio: float = 0.09,
    width: float | None = None,
    pad: int = 12,
    supersample: int = 2,
) -> np.ndarray:
    """把笔迹点序列渲染成灰度图（白字黑底 0..255）。

    strokes: [[(x, y), ...], ...] 或 [[(x, y, pressure), ...], ...]
    坐标单位任意，内部按所有点总包围盒等比缩放到画布。
    """
    strokes = normalize_strokes(strokes)
    pts = [p for s in strokes for p in s]
    if not pts:
        return np.zeros((size, size), dtype=np.float32)

    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w = max(x1 - x0, 1e-6)
    h = max(y1 - y0, 1e-6)
    scale = (size - 2 * pad) / max(w, h)
    if width is None:
        # 线宽与字形尺寸成比例：真笔迹的笔宽约为字高的 6%~10%，
        # 固定像素宽度会在缩放到 28x28 后细成 1px 的幽灵线。
        width = max(2.0, width_ratio * max(w, h))

    ss = max(1, int(supersample))
    S = size * ss
    img = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(img)

    def to_px(x: float, y: float):
        return (pad + (x - x0) * scale) * ss, (pad + (y - y0) * scale) * ss

    for s in strokes:
        if not s:
            continue
        if len(s) == 1:
            cx, cy = to_px(float(s[0][0]), float(s[0][1]))
            r = max(1.0, width * ss / 2.0)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
            continue
        for a, b in zip(s, s[1:]):
            pa = to_px(float(a[0]), float(a[1]))
            pb = to_px(float(b[0]), float(b[1]))
            press = 1.0
            if len(b) > 2 and float(b[2]) > 0:
                press = float(b[2])
            lw = max(1, int(round(width * ss * (0.65 + 0.7 * press))))
            d.line([pa, pb], fill=255, width=lw, joint="curve")
            # 圆头端点
            r = lw / 2.0
            for (px, py) in (pa, pb):
                d.ellipse([px - r, py - r, px + r, py + r], fill=255)

    if ss > 1:
        img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def strokes_to_canonical(
    strokes: Sequence[Sequence[Sequence[float]]],
    size: int = 224,
    width_ratio: float = 0.09,
) -> np.ndarray:
    return canonicalize(render_strokes(strokes, size=size, width_ratio=width_ratio))


def stroke_metrics(strokes: Sequence[Sequence[Sequence[float]]]) -> Dict[str, float]:
    """与大小写判定相关的几何量（原图像素坐标，未归一化）。"""
    strokes = normalize_strokes(strokes)
    pts = [p for s in strokes for p in s]
    if not pts:
        return {"height": 0.0, "width": 0.0, "aspect": 0.0, "n_strokes": 0, "ink": 0.0}
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return {
        "height": h,
        "width": w,
        "aspect": (w / h) if h > 1e-6 else 0.0,
        "n_strokes": float(len(strokes)),
        "ink": float(sum(len(s) for s in strokes)),
        "x0": min(xs),
        "x1": max(xs),
        "y0": min(ys),
        "y1": max(ys),
    }
