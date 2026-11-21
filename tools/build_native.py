# -*- coding: utf-8 -*-
"""构建原生外壳 ime_shell.exe（MSVC + WebView2 SDK）。

为什么用 Python 而不是 .cmd：批处理文件会被 cmd 按 ANSI 解析，
文件里的中文注释会把命令行拆坏；Python 脚本 UTF-8 安全，也方便反复调用。

用法:
    python tools/build_native.py
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE = os.path.join(ROOT, "native")
SDK = os.path.join(ROOT, ".vendor", "webview2", "pkg", "build", "native")
OUT_DIR = os.path.join(ROOT, "dist", "TonhTzeIME", "native")
OUT_EXE = os.path.join(OUT_DIR, "ime_shell.exe")


def find_vcvars() -> str | None:
    patterns = [
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def main() -> int:
    vcvars = find_vcvars()
    if not vcvars:
        print("[x] 找不到 vcvars64.bat（需要 VS 的 C++ 桌面工作负载）")
        return 1
    header = os.path.join(SDK, "include", "WebView2.h")
    if not os.path.exists(header):
        print(f"[x] 找不到 WebView2 SDK: {SDK}")
        print("    先运行: python tools/fetch_webview2.py")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    obj_dir = os.path.join(OUT_DIR, "obj")
    os.makedirs(obj_dir, exist_ok=True)
    lib = os.path.join(SDK, "x64", "WebView2LoaderStatic.lib")

    # 全部用 ASCII 拼命令行，避免任何编码问题
    cl = (
        f'cl /nologo /std:c++17 /O2 /EHsc /MD /utf-8 /DUNICODE /D_UNICODE '
        f'/I"{SDK}\\include" '
        f'"{NATIVE}\\ime_shell.cpp" '
        f'/Fe:"{OUT_EXE}" /Fo:"{obj_dir}\\" '
        f'/link /SUBSYSTEM:WINDOWS "{lib}" '
        f'user32.lib gdi32.lib ole32.lib shell32.lib dwmapi.lib'
    )
    cmd = f'call "{vcvars}" >nul && {cl}'
    print("[build] 编译 ime_shell.cpp ...")
    # 用 shell=True 让 cmd 自己解析引号，否则引号会被二次转义成 \"
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out.strip()[-4000:])
    if proc.returncode != 0 or not os.path.exists(OUT_EXE):
        print("[x] 编译失败")
        return 1
    print(f"[ok] 已生成 {OUT_EXE} ({os.path.getsize(OUT_EXE)//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
