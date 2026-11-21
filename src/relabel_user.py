# -*- coding: utf-8 -*-
"""按给定序列重贴标签（救回"整体错位"的采集数据）。

典型场景：面板按"目标字符"打标签，用户却按 0–9 顺序写了两轮，
于是 21 条样本的标签全是 0/1，但字形是 0..9 0..9 0。
这时按顺序重贴标签即可，不用重写。

用法:
    python -m src.relabel_user train "0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0"
    python -m src.relabel_user train --auto      # 按 0..9 循环自动推断
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter

from . import userdata
from .charset import CHAR2IDX


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    split = sys.argv[1]
    recs = userdata.load_jsonl(split)
    if not recs:
        print(f"{split} 是空的")
        return

    if len(sys.argv) >= 3 and sys.argv[2] == "--auto":
        labels = [str(i % 10) for i in range(len(recs))]
        print("自动推断序列: 0..9 循环")
    else:
        raw = sys.argv[2] if len(sys.argv) >= 3 else ""
        labels = [c for c in raw.replace(",", "").replace(" ", "")]
        if len(labels) != len(recs):
            print(f"标签数 {len(labels)} 与样本数 {len(recs)} 不符，拒绝执行")
            return

    bad = [c for c in labels if c not in CHAR2IDX]
    if bad:
        print(f"未知字符: {sorted(set(bad))}，拒绝执行")
        return

    path = userdata.os.path.join(userdata.USER_DIR, f"{split}.jsonl")
    shutil.copyfile(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        for r, lab in zip(recs, labels):
            r["label"] = lab
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"已重贴 {len(recs)} 条（备份: {path}.bak）")
    print("新的标签分布:", dict(sorted(Counter(labels).items())))


if __name__ == "__main__":
    main()
