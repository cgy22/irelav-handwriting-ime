# -*- coding: utf-8 -*-
"""编译单文件启动器 TonhTzeIME.exe。

产物：dist/TonhTzeIME/TonhTzeIME.exe（约 10 KB，双击即启动全部组件）
     仓库根目录也会放一份，方便直接从源码目录启动。

用法: python tools/build_launcher.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "app", "shell", "launcher", "Launcher.cs")
ICON = os.path.join(ROOT, "app", "static", "icon.ico")
OUT_DIR = os.path.join(ROOT, "dist", "TonhTzeIME")

CSC_CANDIDATES = [
    r"C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\Roslyn\csc.exe",
    r"C:\Program Files\Microsoft Visual Studio\17\Community\MSBuild\Current\Bin\Roslyn\csc.exe",
    r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
]


def main() -> int:
    if not os.path.exists(SRC):
        print(f"[x] 缺少 {SRC}")
        return 1
    csc = next((c for c in CSC_CANDIDATES if os.path.exists(c)), None)
    if not csc:
        print("[x] 找不到 csc.exe")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "TonhTzeIME.exe")
    args = [
        csc, "/nologo", "/target:winexe", "/platform:anycpu", "/optimize+",
        "/langversion:7.3", f"/out:{out}", SRC,
        "/r:System.dll", "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll",
    ]
    if os.path.exists(ICON):
        args.insert(5, f"/win32icon:{ICON}")

    print("[build] 编译启动器 ...")
    proc = subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not os.path.exists(out):
        print((proc.stdout or "") + (proc.stderr or ""))
        print("[x] 编译失败")
        return 1

    # 仓库根目录也放一份，双击即可从源码目录启动
    shutil.copy2(out, os.path.join(ROOT, "TonhTzeIME.exe"))
    print(f"[ok] {out} ({os.path.getsize(out) // 1024} KB)")
    print(f"[ok] 已在仓库根目录生成 TonhTzeIME.exe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
