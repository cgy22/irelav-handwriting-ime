# -*- coding: utf-8 -*-
"""用户手写样本：从 data/user/*.jsonl 读回并走同一条规范化管线。

采集时只存了笔迹点序列（不存位图），好处是：
  * 文件小
  * 以后改预处理参数（线宽、归一化方式）可以重新渲染，不用重新采集
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import canon
from .charset import CHAR2IDX, case_pair

USER_DIR = r"F:\mnist\data\user"


def load_jsonl(split: str) -> List[dict]:
    p = os.path.join(USER_DIR, f"{split}.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def counts(split: str) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for r in load_jsonl(split):
        c[r["label"]] = c.get(r["label"], 0) + 1
    return c


def build_arrays(
    split: str = "train",
    labels_filter: Optional[Sequence[str]] = None,
    width_ratio: float = 0.09,
    augment_times: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """返回 (x[N,28,28] float32, y1[N] int64, raw_labels)。

    y1 用 100 类字符下标；标注里带大小写，所以是硬标签。
    augment_times>0 时对每条样本做轻微抖动扩充。
    """
    recs = load_jsonl(split)
    if labels_filter is not None:
        keep = set(labels_filter)
        recs = [r for r in recs if r["label"] in keep]
    if not recs:
        return np.zeros((0, 28, 28), np.float32), np.zeros((0,), np.int64), []

    xs, ys, raws = [], [], []
    for r in recs:
        strokes = r.get("strokes") or []
        if not strokes or not any(strokes):
            continue
        base = canon.strokes_to_canonical(strokes, width_ratio=width_ratio)
        xs.append(base)
        ys.append(CHAR2IDX[r["label"]])
        raws.append(r["label"])
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), raws


def class_histogram(split: str = "train") -> str:
    c = counts(split)
    if not c:
        return "(空)"
    return " ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    for sp in ("train", "test"):
        print(f"{sp}: {sum(counts(sp).values())} 条")
        print("  ", class_histogram(sp))
