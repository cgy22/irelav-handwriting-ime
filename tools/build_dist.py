# -*- coding: utf-8 -*-
"""打包生产环境：生成 dist/TonhTzeIME/（可整目录复制到别的机器）。

包含：
  native/    ime_shell.exe + WebView2 依赖 + 图标（已内嵌）
  app/       server.py / desktop.py / win32_helper.py / static/
  src/       推理与模型定义（训练脚本也一并带上，方便复现）
  data/      words.json（词表）
  runs/exp2_final/best.pt（发货权重）
  start_ime.cmd / 托盘说明

用法:
    python tools/build_dist.py            # 打包
    python tools/build_dist.py --clean    # 先清空 dist 再打包
"""
from __future__ import annotations

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist", "TonhTzeIME")

INCLUDE_DIRS = ["app", "src"]
INCLUDE_FILES = ["start_ime.cmd"]
DATA_FILES = [os.path.join("data", "words.json")]
MODEL_FILES = [os.path.join("runs", "exp2_final", "best.pt")]

SKIP_PATTERNS = ("__pycache__", ".pyc", ".runtime", ".vendor")


def ignore(_dir, names):
    return [n for n in names if any(p in n for p in SKIP_PATTERNS)]


def copy_tree(rel: str) -> None:
    src = os.path.join(ROOT, rel)
    dst = os.path.join(DIST, rel)
    if not os.path.exists(src):
        print(f"[!] 跳过（不存在）: {rel}")
        return
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    if "--clean" in sys.argv and os.path.isdir(DIST):
        # 只清源码与资源，保留 native（重新编译代价高）
        for name in ("app", "src", "data", "runs"):
            p = os.path.join(DIST, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        print("[clean] 已清理 dist 中的程序与数据目录")

    os.makedirs(DIST, exist_ok=True)
    for d in INCLUDE_DIRS:
        copy_tree(d)
    for f in INCLUDE_FILES:
        copy_tree(f)
    for f in DATA_FILES + MODEL_FILES:
        copy_tree(f)

    exe = os.path.join(DIST, "native", "ime_shell.exe")
    if not os.path.exists(exe):
        print("[!] 还没有原生外壳：先运行 python tools/build_shell_cs.py")

    total = 0
    for base, _dirs, files in os.walk(DIST):
        for f in files:
            total += os.path.getsize(os.path.join(base, f))
    print(f"[ok] 打包完成: {DIST}")
    print(f"     体积 {total/2**20:.1f} MB，入口 {os.path.join(DIST, 'start_ime.cmd')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
