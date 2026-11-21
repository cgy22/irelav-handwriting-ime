# -*- coding: utf-8 -*-
"""下载 WebView2 SDK（NuGet 包）到 .vendor/webview2/。

只下载到工作区的 .vendor 目录，不安装任何东西到系统。

用法: python tools/fetch_webview2.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, ".vendor", "webview2")
HEADER = os.path.join(VENDOR, "pkg", "build", "native", "include", "WebView2.h")


def main() -> int:
    if os.path.exists(HEADER):
        print(f"[ok] WebView2 SDK 已存在: {HEADER}")
        return 0

    os.makedirs(VENDOR, exist_ok=True)
    print("[1/2] 查询 NuGet 最新稳定版 ...")
    index_url = "https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/index.json"
    with urllib.request.urlopen(index_url, timeout=30) as r:
        import json
        versions = [v for v in json.load(r)["versions"] if "-" not in v]
    ver = versions[-1]
    print(f"      {ver}")

    pkg_zip = os.path.join(VENDOR, "pkg.zip")
    url = f"https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/{ver}/microsoft.web.webview2.{ver}.nupkg"
    print("[2/2] 下载并解包 ...")
    urllib.request.urlretrieve(url, pkg_zip)
    size_mb = os.path.getsize(pkg_zip) / 2**20
    with zipfile.ZipFile(pkg_zip) as z:
        z.extractall(os.path.join(VENDOR, "pkg"))
    os.remove(pkg_zip)
    print(f"[ok] 已解压 {size_mb:.1f} MB 到 {os.path.join(VENDOR, 'pkg')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
