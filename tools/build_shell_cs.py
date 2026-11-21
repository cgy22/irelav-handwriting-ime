# -*- coding: utf-8 -*-
"""用系统自带的 csc.exe 构建 C# 原生外壳 ime_shell.exe。

不依赖 .NET SDK：Windows 自带 .NET Framework 4.8 的 csc，
WebView2 的 net462 程序集来自 .vendor/webview2/pkg。

用法: python tools/build_shell_cs.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE = os.path.join(ROOT, "native")
PKG = os.path.join(ROOT, ".vendor", "webview2", "pkg")
OUT_DIR = os.path.join(ROOT, "dist", "TonhTzeIME", "native")
OUT_EXE = os.path.join(OUT_DIR, "ime_shell.exe")

CSC = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
FW = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319"


def main() -> int:
    if not os.path.exists(CSC):
        print("[x] 找不到 csc.exe")
        return 1
    core = os.path.join(PKG, "lib", "net462", "Microsoft.Web.WebView2.Core.dll")
    winforms = os.path.join(PKG, "lib", "net462", "Microsoft.Web.WebView2.WinForms.dll")
    loader = os.path.join(PKG, "build", "native", "x64", "WebView2Loader.dll")
    for p in (core, winforms, loader):
        if not os.path.exists(p):
            print(f"[x] 缺少 {p}")
            return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    args = [
        CSC, "/nologo", "/target:winexe", "/platform:x64", "/optimize+",
        f"/out:{OUT_EXE}",
        f"/r:{os.path.join(FW, 'System.dll')}",
        f"/r:{os.path.join(FW, 'System.Drawing.dll')}",
        f"/r:{os.path.join(FW, 'System.Windows.Forms.dll')}",
        f"/r:{core}",
        f"/r:{winforms}",
        f"/win32manifest:{os.path.join(NATIVE, 'app.manifest')}",
        os.path.join(NATIVE, "Program.cs"),
    ]
    print("[build] 编译 Program.cs ...")
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if out.strip():
        print(out.strip()[-3000:])
    if proc.returncode != 0 or not os.path.exists(OUT_EXE):
        print("[x] 编译失败")
        return 1

    # 运行时需要的托管依赖与加载器
    for src in (core, winforms, loader):
        shutil.copy2(src, os.path.join(OUT_DIR, os.path.basename(src)))
    print(f"[ok] {OUT_EXE} ({os.path.getsize(OUT_EXE)//1024} KB)")
    print(f"[ok] 依赖已复制到 {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
