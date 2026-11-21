# -*- coding: utf-8 -*-
"""输入法后端：HTTP 服务 + 推理 + 采集落盘。

只依赖标准库 http.server（零额外依赖）：单字推理 CPU 上 5~15ms，一次请求一个线程足够。

接口
----
GET  /                 -> 静态页面
GET  /api/state        -> 模型信息 / 各类已采集数量
POST /api/recognize    -> {strokes, canvas, line_heights} -> 候选
POST /api/collect      -> {strokes, label, split} -> 保存一条手写样本
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import canon  # noqa: E402
from src.charset import CLASSES, DIGITS, case_pair, to_identity  # noqa: E402
from src.models import GlyphNet, MnistNet  # noqa: E402

STATIC = os.path.join(_HERE, "static")
USER_DIR = os.path.join(_ROOT, "data", "user")
PORT = int(os.environ.get("IME_PORT", "8765"))

_lock = threading.Lock()


# --------------------------------------------------------------------- 模型
class Recognizer:
    def __init__(self) -> None:
        self.model = None
        self.classes: List[str] = []
        self.name = "未加载"
        self._load()

    def _load(self) -> None:
        exp2 = os.path.join(_ROOT, "runs", "exp2_ext", "best.pt")
        exp1 = os.path.join(_ROOT, "runs", "exp1_mnist", "best.pt")
        if os.path.exists(exp2):
            ck = torch.load(exp2, map_location="cpu")
            classes = list(ck.get("classes", CLASSES))
            m = GlyphNet(len(classes))
            m.load_state_dict(ck["model"])
            self.model, self.classes, self.name = m, classes, "exp2_ext 100类"
        elif os.path.exists(exp1):
            ck = torch.load(exp1, map_location="cpu")
            m = MnistNet(10)
            m.load_state_dict(ck["model"])
            self.model, self.classes, self.name = m, list(DIGITS), "exp1_mnist 仅数字"
        else:
            return
        self.model.eval()
        print(f"[rec] 已加载 {self.name}，{len(self.classes)} 类", flush=True)

    def reload(self) -> None:
        with _lock:
            self._load()

    @torch.no_grad()
    def infer(self, gray28: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("模型未加载")
        x = torch.from_numpy(gray28)[None, None].float()
        with _lock:
            logits = self.model(x)
        return torch.softmax(logits.float(), 1)[0].numpy()


def case_prior(height: float, line_heights: List[float]) -> float:
    """用"当前字高 / 本行已写字高中位数"估计大小写（对数几率偏移）。

    孤立字形做完包围盒归一化后丢失了相对高度，模型无法从形状判断大小写，
    所以大小写只能在这里补：明显高于本行中位数 -> 偏大写；明显低 -> 偏小写。
    """
    hs = [float(h) for h in line_heights if h and float(h) > 1]
    if not hs or height <= 1:
        return 0.0
    med = float(np.median(hs))
    if med <= 1:
        return 0.0
    r = height / med
    if r >= 1.30:
        return 1.6
    if r >= 1.15:
        return 0.7
    if r <= 0.82:
        return -0.8
    return 0.0


def candidates(probs: np.ndarray, classes: List[str], prior: float, k: int = 6):
    """大小写先验重排 + 按身份去重，返回候选列表。"""
    logits = np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float64)
    for i, ch in enumerate(classes):
        if ch.isupper() or ch in "ÆŒÊ":
            logits[i] += prior
        elif ch.islower() or ch in "æœê":
            logits[i] -= prior
    order = np.argsort(-logits)
    seen: Dict[str, bool] = {}
    out = []
    for idx in order:
        ch = classes[int(idx)]
        ident = to_identity(ch)
        if ident in seen:
            continue
        seen[ident] = True
        lo, up = case_pair(ch)
        out.append(
            {
                "ch": ch,
                "p": round(float(probs[int(idx)]), 4),
                "alt": up if ch == lo else lo,
            }
        )
        if len(out) >= k:
            break
    return out


# --------------------------------------------------------------------- 采集
def _ensure_user_dir() -> None:
    os.makedirs(USER_DIR, exist_ok=True)


def collect_path(split: str) -> str:
    _ensure_user_dir()
    return os.path.join(USER_DIR, f"{split}.jsonl")


def append_sample(split: str, label: str, strokes, canvas: Dict[str, float]) -> None:
    with _lock:
        with open(collect_path(split), "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"label": label, "strokes": strokes, "canvas": canvas, "ts": time.time()},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _read_all(split: str) -> List[dict]:
    p = collect_path(split)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def counts_by_class(split: str) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for r in _read_all(split):
        c[r["label"]] = c.get(r["label"], 0) + 1
    return c


# --------------------------------------------------------------------- HTTP
REC = Recognizer()


class Handler(BaseHTTPRequestHandler):
    server_version = "HandwritingIME/0.1"

    def log_message(self, fmt, *args):
        return

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _static(self, rel: str):
        path = os.path.normpath(os.path.join(STATIC, rel))
        if not path.startswith(STATIC) or not os.path.isfile(path):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(os.path.splitext(path)[1], "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path  # 去掉 ?query，避免 /?mode=collect 被当成未知路径
        if path in ("/", "/index.html"):
            self._static("index.html")
        elif path.startswith("/static/"):
            self._static(path[len("/static/") :])
        elif path == "/api/state":
            tr = counts_by_class("train")
            te = counts_by_class("test")
            self._json(
                {
                    "pid": os.getpid(),
                    "model": REC.name,
                    "classes": REC.classes,
                    "count_train": sum(tr.values()),
                    "count_test": sum(te.values()),
                    "counts_train": tr,
                    "counts_test": te,
                }
            )
        elif path == "/api/reload":
            REC.reload()
            self._json({"ok": True, "model": REC.name})
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._json({"error": "bad json"}, 400)
            return
        try:
            if self.path == "/api/recognize":
                self._handle_recognize(payload)
            elif self.path == "/api/collect":
                self._handle_collect(payload)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # 任何内部异常都回 JSON，不让连接被直接掐断
            import traceback
            traceback.print_exc()
            self._json({"error": f"服务端异常: {type(e).__name__}: {e}"}, 500)

    def _handle_recognize(self, payload):
        strokes = payload.get("strokes") or []
        if not strokes or not any(strokes):
            self._json({"top": [], "ms": 0})
            return
        if REC.model is None:
            self._json({"error": "模型未就绪"}, 503)
            return
        t0 = time.perf_counter()
        gray = canon.strokes_to_canonical(strokes)
        probs = REC.infer(gray)
        met = canon.stroke_metrics(strokes)
        heights = payload.get("line_heights") or []
        prior = case_prior(met["height"], heights)
        if not heights:
            # 行首没有参照高度：默认偏小写（符合输入法习惯），Shift 可强制大写
            prior = -0.3
        top = candidates(probs, REC.classes, prior)
        ms = (time.perf_counter() - t0) * 1000
        hint = "upper" if prior > 0.5 else ("lower" if prior < -0.5 else "unknown")
        self._json({"top": top, "ms": round(ms, 1), "case_hint": hint, "height": round(met["height"], 1)})

    def _handle_collect(self, payload):
        label = str(payload.get("label") or "")
        split = str(payload.get("split") or "train")
        strokes = payload.get("strokes") or []
        if not label or not strokes:
            self._json({"error": "缺少 label 或 strokes"}, 400)
            return
        if split not in ("train", "test"):
            split = "train"
        append_sample(split, label, strokes, payload.get("canvas") or {})
        self._json({"ok": True, "label": label, "split": split, "count": counts_by_class(split).get(label, 0)})


def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[srv] 手写输入法: http://127.0.0.1:{PORT}  pid={os.getpid()}", flush=True)
    print(f"[srv] 模型: {REC.name}", flush=True)
    print(f"[srv] 采集目录: {USER_DIR}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[srv] 已停止")


if __name__ == "__main__":
    main()
