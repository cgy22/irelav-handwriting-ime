# -*- coding: utf-8 -*-
"""把 TonhTze AI 的 logo SVG 转成"纯图形"版本（去掉 <text>，全部变成路径）。

背景
----
原 SVG 的 "TonhTze" 是 <text> 元素，依赖 Cascadia Code 字体；换台机器就可能
变成方框。这里用 fontTools 取字形轮廓，把文字整体替换成 <path>。

字体选择
--------
原设计用 Cascadia Code Bold，本机没装；用 Consolas Bold（同为等宽粗体，
字面宽度接近）替代，保证仍然是"代码字体"的味道。

用法: python tools/make_logos.py
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from fontTools.pens.transformPen import TransformPen
from fontTools.misc.transform import Transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = r"F:\dornhub.eu.org\htdocs\resources\images\TonhTzeAI"
OUT_DIR = os.path.join(ROOT, "assets")
FONT_PATH = r"C:\Windows\Fonts\consolab.ttf"

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def text_to_path(font: TTFont, text: str, size: float, x: float, y: float, fill: str) -> str:
    """把一行文字转成 SVG path 的 d 属性。"""
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    upem = font["head"].unitsPerEm
    scale = size / upem
    hmtx = font["hmtx"]

    x_cursor = 0.0
    parts: list[str] = []
    for ch in text:
        gname = cmap.get(ord(ch))
        if gname is None:
            continue
        pen = SVGPathPen(glyph_set)
        # 字体坐标 y 向上，SVG y 向下：先翻转，再平移
        t = Transform().translate(x + x_cursor, y).scale(scale, -scale)
        glyph_set[gname].draw(TransformPen(pen, t))
        d = pen.getCommands()
        if d:
            parts.append(d)
        x_cursor += hmtx[gname][0] * scale
    return " ".join(parts)


def convert(src: str, dst: str) -> None:
    tree = ET.parse(src)
    root = tree.getroot()
    replaced = 0

    for parent in root.iter():
        for node in list(parent):
            if not node.tag.endswith("text"):
                continue
            content = "".join(node.itertext())
            if not content.strip():
                parent.remove(node)          # 纯空格的占位文本直接删掉
                continue
            size = float(node.get("font-size", "16"))
            fill = node.get("fill", "#000000")
            opacity = node.get("fill-opacity", "1")
            transform = node.get("transform", "")
            m = re.match(r"matrix\(([-\d.eE]+)\s+0\s+0\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\)",
                         transform.strip())
            if m:
                x, y = float(m.group(3)), float(m.group(4))
            else:
                x, y = float(node.get("x", "0")), float(node.get("y", "0"))

            d = text_to_path(TTFont(FONT_PATH) if False else _font(), content, size, x, y, fill)
            if not d:
                parent.remove(node)
                continue

            path = ET.Element(f"{{{SVG_NS}}}path", {
                "d": d,
                "fill": fill,
                "fill-opacity": opacity,
                "fill-rule": "nonzero",
            })
            parent.insert(list(parent).index(node), path)
            parent.remove(node)
            replaced += 1

    tree.write(dst, encoding="utf-8", xml_declaration=True)
    print(f"[ok] {os.path.basename(dst)}  替换文本元素 {replaced} 个")


_FONT: TTFont | None = None


def _font() -> TTFont:
    global _FONT
    if _FONT is None:
        _FONT = TTFont(FONT_PATH)
    return _FONT


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(FONT_PATH):
        print(f"[x] 找不到字体 {FONT_PATH}")
        return
    for name in ("TonhTzeAI-logo-v1-light.svg", "TonhTzeAI-logo-v1-dark.svg"):
        src = os.path.join(SRC_DIR, name)
        if not os.path.exists(src):
            print(f"[x] 缺少 {src}")
            continue
        out = os.path.join(OUT_DIR, name.replace("TonhTzeAI-logo-v1-", "tonhtze-ai-"))
        convert(src, out)


if __name__ == "__main__":
    main()
