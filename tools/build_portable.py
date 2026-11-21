# -*- coding: utf-8 -*-
"""便携版（离线组装）：直接复制本机 Python 运行时 + numpy + Pillow。

为什么不用 python.org 的 embeddable 包：
  1) 下载慢且不稳定（实测卡在轮子下载上）；
  2) embeddable 是 3.13，而本机验证过的 numpy/Pillow 是 3.15 的 ABI，混用会崩。
直接拷贝本机运行时，与已经验证通过的环境完全一致，且零下载。

产物：dist/TonhTzeIME-portable/
        runtime\\        约 80 MB 的 Python 运行时（stdlib + numpy + Pillow）
        app\\ src\\ data\\ native\\
        TonhTzeIME.exe  启动器（优先使用 runtime\\pythonw.exe）

用法: python tools/build_portable.py
"""
from __future__ import annotations

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "dist", "TonhTzeIME-portable")
RUNTIME = os.path.join(OUT, "runtime")
SRC = sys.base_prefix

TOP_FILES = [
    "python.exe", "pythonw.exe", "python3.dll",
    f"python{sys.version_info.major}{sys.version_info.minor}.dll",
    "vcruntime140.dll", "vcruntime140_1.dll",
]

# stdlib 里不需要的大件（tkinter、测试套件、pip 引导等）
LIB_SKIP = {
    "site-packages", "test", "tkinter", "turtle.py", "turtledemo", "idlelib",
    "ensurepip", "pydoc_data", "lib2to3", "curses", "dbm", "distutils", "__pycache__",
}

# DLLs 目录里可以不要的（tk/tcl、调试符号）
DLL_SKIP_PREFIX = ("_tkinter", "tcl", "tk8", "tk9", "_test")

# 需要带的第三方包
PACKAGES = ["numpy", "numpy.libs", "PIL"]


def copytree(src: str, dst: str, skip_names: set[str] | None = None,
             skip_prefix: tuple[str, ...] = ()) -> None:
    skip_names = skip_names or set()
    for entry in os.listdir(src):
        if entry in skip_names:
            continue
        if skip_prefix and entry.lower().startswith(skip_prefix):
            continue
        s = os.path.join(src, entry)
        d = os.path.join(dst, entry)
        if os.path.isdir(s):
            os.makedirs(d, exist_ok=True)
            copytree(s, d, skip_names, skip_prefix)
        else:
            if entry.endswith((".pyc", ".pdb", ".lib")):
                continue
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)


def dist_info_of(name: str) -> list[str]:
    sp = os.path.join(SRC, "Lib", "site-packages")
    key = name.lower().split(".")[0]
    return [e for e in os.listdir(sp)
            if e.lower().startswith(key) and e.endswith(".dist-info")]


def main() -> int:
    print(f"[1/4] 清理 {OUT}")
    shutil.rmtree(OUT, ignore_errors=True)   # 整目录重建，避免上一次的残留（比如 94 MB 的 data）
    shutil.rmtree(RUNTIME, ignore_errors=True)
    os.makedirs(RUNTIME, exist_ok=True)

    print("[2/4] 复制 Python 运行时 ...")
    for f in TOP_FILES:
        src = os.path.join(SRC, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(RUNTIME, f))
        else:
            print(f"      [!] 缺少 {f}")

    dlls_src = os.path.join(SRC, "DLLs")
    if os.path.isdir(dlls_src):
        copytree(dlls_src, os.path.join(RUNTIME, "DLLs"), skip_prefix=DLL_SKIP_PREFIX)

    copytree(os.path.join(SRC, "Lib"), os.path.join(RUNTIME, "Lib"), LIB_SKIP)

    print("[3/4] 复制 numpy / Pillow ...")
    sp_src = os.path.join(SRC, "Lib", "site-packages")
    sp_dst = os.path.join(RUNTIME, "Lib", "site-packages")
    os.makedirs(sp_dst, exist_ok=True)
    for pkg in PACKAGES:
        s = os.path.join(sp_src, pkg)
        if os.path.isdir(s):
            copytree(s, os.path.join(sp_dst, pkg))
            print(f"      + {pkg}")
    for pkg in ("numpy", "pillow"):
        for di in dist_info_of(pkg):
            copytree(os.path.join(sp_src, di), os.path.join(sp_dst, di))
            print(f"      + {di}")

    print("[4/4] 组装程序 ...")
    for name in ("app", "src", "native"):
        s = os.path.join(ROOT, name)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(OUT, name), dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # data/ 只带运行必需的词表与 numpy 权重（采集样本和缓存属于开发数据）
    data_dst = os.path.join(OUT, "data")
    os.makedirs(data_dst, exist_ok=True)
    for rel in ("words.json", os.path.join("model", "glyph100.npz"),
                os.path.join("model", "glyph100.json")):
        src = os.path.join(ROOT, "data", rel)
        if os.path.exists(src):
            dst = os.path.join(data_dst, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    for f in ("TonhTzeIME.exe", "LICENSE", "README.md", "THIRD-PARTY-NOTICES.md"):
        s = os.path.join(ROOT, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(OUT, f))
    shutil.rmtree(os.path.join(OUT, "runs"), ignore_errors=True)
    shutil.rmtree(os.path.join(OUT, ".runtime"), ignore_errors=True)

    total, files = 0, 0
    for base, _d, fs in os.walk(OUT):
        for f in fs:
            files += 1
            total += os.path.getsize(os.path.join(base, f))
    print(f"[ok] {OUT}")
    print(f"     {files} 个文件，{total / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
