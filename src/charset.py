# -*- coding: utf-8 -*-
"""字符集定义（100 类）。

100 = 10(数字) + 26(大写) + 26(小写) + 32(半角符号) + 6(Æ æ Œ œ Ê ê)

为什么大小写不单独交给模型
--------------------------
孤立字形做完包围盒归一化后，"c" 与 "C" 会变成同一张图，模型从形状上
不可能区分大小写。因此本工程的分工是：

  * 模型：负责字形身份（100 类软输出，允许 top-k 候选）
  * UI  ：负责大小写（相对 x-height 的几何启发式 + 候选列表 + Shift 强制）

评估时同时给出两个口径：
  * exact   ：严格 100 类准确率（大小写必须都对）
  * identity：忽略大小写后的字形身份准确率（71 个身份类）
"""
from __future__ import annotations

from typing import Dict, List, Tuple

DIGITS: List[str] = [str(d) for d in range(10)]
UPPER: List[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
LOWER: List[str] = [chr(c) for c in range(ord("a"), ord("z") + 1)]
# 32 个 ASCII 半角标点（用户要求覆盖 @ # $ % & 等常用符号，不含 © 之类）
SYMBOLS: List[str] = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
SPECIALS: List[str] = ["Æ", "æ", "Œ", "œ", "Ê", "ê"]

CLASSES: List[str] = DIGITS + UPPER + LOWER + SYMBOLS + SPECIALS
NUM_CLASSES: int = len(CLASSES)

CHAR2IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX2CHAR: Dict[int, str] = {i: c for i, c in enumerate(CLASSES)}

assert len(CHAR2IDX) == NUM_CLASSES, "字符集存在重复项"

# ---------------------------------------------------------------- 大小写配对
_CASE_UP: Dict[str, str] = {lo: up for lo, up in zip(LOWER, UPPER)}
_CASE_UP.update({"æ": "Æ", "œ": "Œ", "ê": "Ê"})
_CASE_DOWN: Dict[str, str] = {v: k for k, v in _CASE_UP.items()}


def case_pair(ch: str) -> Tuple[str, str]:
    """返回 (小写形, 大写形)；非大小写字母则返回 (ch, ch)。"""
    if ch in _CASE_UP:
        return ch, _CASE_UP[ch]
    if ch in _CASE_DOWN:
        return _CASE_DOWN[ch], ch
    return ch, ch


def is_cased(ch: str) -> bool:
    return ch in _CASE_UP or ch in _CASE_DOWN


def to_identity(ch: str) -> str:
    """忽略大小写后的身份标签，用于 identity 口径与候选去重。"""
    return case_pair(ch)[0]


IDENTITY_CLASSES: List[str] = sorted({to_identity(c) for c in CLASSES})
IDENT2IDX: Dict[str, int] = {c: i for i, c in enumerate(IDENTITY_CLASSES)}
CLASS2IDENT: List[int] = [IDENT2IDX[to_identity(c)] for c in CLASSES]
NUM_IDENTITIES: int = len(IDENTITY_CLASSES)


def soft_target_for_char(ch: str, mass: float = 1.0) -> List[Tuple[int, float]]:
    """构造软标签：大小写字母在两种形态间均分。

    用于 EMNIST（大小写合并标注）这类"只知道字形、不知道大小写"的数据。
    """
    lo, up = case_pair(ch)
    if lo == up:
        return [(CHAR2IDX[ch], mass)]
    return [(CHAR2IDX[lo], mass * 0.5), (CHAR2IDX[up], mass * 0.5)]


# ------------------------------------------------------------- EMNIST 映射表
def load_emnist_mapping(path: str) -> List[str]:
    """解析 emnist-*-mapping.txt。

    文件格式为每行 ``类索引 ASCII码``（空格或逗号分隔），例如::

        0 48
        1 49
        ...
        46 116

    第一列是类别下标，第二列才是字符的 ASCII 码。返回按类别下标排列的字符列表。
    """
    chars: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                code = int(parts[-1])
            except ValueError:
                continue
            chars.append(chr(code))
    return chars


def describe() -> str:
    return (
        f"字符集: {NUM_CLASSES} 类 = {len(DIGITS)}数字 + {len(UPPER)}大写 + "
        f"{len(LOWER)}小写 + {len(SYMBOLS)}符号 + {len(SPECIALS)}特殊; "
        f"身份类: {NUM_IDENTITIES}"
    )


if __name__ == "__main__":
    print(describe())
    print("符号:", "".join(SYMBOLS))
    print("前 12 类:", CLASSES[:12])
    print("末 8 类:", CLASSES[-8:])
