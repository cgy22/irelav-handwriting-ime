# -*- coding: utf-8 -*-
"""把便携版压成"单文件 exe"。

做法：先把 dist/TonhTzeIME-portable 打成一个 zip，再把这个 zip 作为资源
嵌进一个很小的 C# 启动器，编译成 dist/TonhTzeIME-single.exe。

首次运行会解压到 %LOCALAPPDATA%\\TonhTzeIME\\<版本>\\，之后直接复用（秒开）。
卸载 = 删掉那个目录。

用法: 先 python tools/build_portable.py，再 python tools/pack_single.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "dist", "TonhTzeIME-portable")
SRC_CS = os.path.join(ROOT, "app", "shell", "launcher", "Bootstrap.cs")
ICON = os.path.join(ROOT, "app", "static", "icon.ico")
PAYLOAD = os.path.join(ROOT, ".vendor", "payload.zip")
OUT = os.path.join(ROOT, "dist", "TonhTzeIME-single.exe")

CSC_CANDIDATES = [
    r"C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\Roslyn\csc.exe",
    r"C:\Program Files\Microsoft Visual Studio\17\Community\MSBuild\Current\Bin\Roslyn\csc.exe",
    r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
]

SKIP_DIRS = {".runtime", "__pycache__"}


def make_payload() -> None:
    if not os.path.isdir(SRC_DIR):
        raise SystemExit("[x] 先运行 python tools/build_portable.py")
    os.makedirs(os.path.dirname(PAYLOAD), exist_ok=True)
    if os.path.exists(PAYLOAD):
        os.remove(PAYLOAD)

    print("[1/2] 打包 payload ...")
    n = 0
    with zipfile.ZipFile(PAYLOAD, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for base, dirs, files in os.walk(SRC_DIR):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if f.lower() == "tonhtzeime.exe":
                    continue        # 单文件里不需要便携版自己的启动器
                full = os.path.join(base, f)
                rel = os.path.relpath(full, SRC_DIR)
                z.write(full, rel)
                n += 1
    size = os.path.getsize(PAYLOAD) / 2**20
    print(f"      {n} 个文件 -> {size:.1f} MB")


def compile_stub() -> None:
    csc = next((c for c in CSC_CANDIDATES if os.path.exists(c)), None)
    if not csc:
        raise SystemExit("[x] 找不到 csc.exe")

    print("[2/2] 编译单文件启动器 ...")
    args = [
        csc, "/nologo", "/target:winexe", "/platform:anycpu", "/optimize+",
        "/langversion:7.3", f"/out:{OUT}", SRC_CS,
        f"/resource:{PAYLOAD},payload.zip",
        "/r:System.dll", "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll",
        "/r:System.IO.Compression.dll", "/r:System.IO.Compression.FileSystem.dll",
    ]
    if os.path.exists(ICON):
        args.insert(5, f"/win32icon:{ICON}")
    proc = subprocess.run(args, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not os.path.exists(OUT):
        print((proc.stdout or "") + (proc.stderr or ""))
        raise SystemExit("[x] 编译失败")

    print(f"[ok] {OUT} ({os.path.getsize(OUT) / 2**20:.1f} MB)")


def main() -> int:
    make_payload()
    compile_stub()
    return 0


if __name__ == "__main__":
    sys.exit(main())
