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
from urllib.parse import parse_qs, urlparse

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
USER_DIR = os.environ.get("MNIST_USER_DIR", os.path.join(_ROOT, "data", "user"))
PORT = int(os.environ.get("IME_PORT", "8765"))

_lock = threading.Lock()

# ------------------------------------------------------------------ 词表
WORDS_PATH = os.path.join(_ROOT, "data", "words.json")
_WORDS = []
_PREFIXES = set()
_BY_PREFIX = {}


def load_words() -> None:
    """载入人造语词表（dh=罗马字拼写, cn=中文义, type=词性）。

    同时建立"所有合法前缀"集合：它既用于候选词，也用于反向约束单字识别——
    如果当前前缀是 "ton"，那么下一个字母只有能构成合法词的才应该被优先。
    """
    global _WORDS, _PREFIXES, _BY_PREFIX
    if not os.path.exists(WORDS_PATH):
        print("[words] 找不到 data/words.json，候选词功能关闭", flush=True)
        return
    with open(WORDS_PATH, "r", encoding="utf-8") as f:
        _WORDS = json.load(f)
    pref, by = set(), {}
    for i, w in enumerate(_WORDS):
        dh = w["dh"]
        for k in range(1, len(dh) + 1):
            p = dh[:k]
            pref.add(p)
            by.setdefault(p, []).append(i)
    _PREFIXES, _BY_PREFIX = pref, by
    print(f"[words] 载入 {len(_WORDS)} 条，合法前缀 {len(pref)} 个", flush=True)


def suggest(prefix: str, limit: int = 12):
    """按前缀返回候选词：完全相等优先，其次短词优先。"""
    ids = _BY_PREFIX.get(prefix, [])
    ids = sorted(ids, key=lambda i: (0 if _WORDS[i]["dh"] == prefix else 1, len(_WORDS[i]["dh"]), _WORDS[i]["dh"]))
    return [_WORDS[i] for i in ids[:limit]]


def lm_bonus(classes, prefix: str) -> np.ndarray:
    """用词表反向约束单字：只有能让前缀继续成为合法词的字母才加分。

    前缀本身不合法、或没有任何候选能延续时，不加分（避免把用户带进死路）。
    """
    bonus = np.zeros(len(classes))
    if not prefix or prefix not in _PREFIXES:
        return bonus
    hit = 0
    for i, ch in enumerate(classes):
        if prefix + ch in _PREFIXES:
            bonus[i] += 2.0
            hit += 1
    return bonus if hit else np.zeros(len(classes))


# --------------------------------------------------------------------- 模型
class Recognizer:
    def __init__(self) -> None:
        self.model = None
        self.classes: List[str] = []
        self.name = "未加载"
        self._load()

    def _load(self) -> None:
        final = os.path.join(_ROOT, "runs", "exp2_final", "best.pt")
        exp2 = os.path.join(_ROOT, "runs", "exp2_ext", "best.pt")
        exp1 = os.path.join(_ROOT, "runs", "exp1_mnist", "best.pt")
        if os.path.exists(final):
            ck = torch.load(final, map_location="cpu")
            classes = list(ck.get("classes", CLASSES))
            m = GlyphNet(len(classes))
            m.load_state_dict(ck["model"])
            self.model, self.classes, self.name = m, classes, "exp2_final 100类(含你的笔迹)"
        elif os.path.exists(exp2):
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


def char_group(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    if ch.islower() or ch in "æœê":
        return "lower"
    if ch.isupper() or ch in "ÆŒÊ":
        return "upper"
    return "symbol"


# 手写里真正"同形异类"的组：不靠上下文不可能稳定区分
AMBIG_SETS = [set(x) for x in (
    "0Oo", "1Il|", "2Zz", "5Ss", "8B", "6Gb", "9gq", "7T", "4A", "3E",
)]


def ambiguity_bonus(classes: List[str], ctx: str) -> np.ndarray:
    """只在同形组内部按上下文定向加权。

    不能对整类做加权（那会把 "3" 也推向字母），只对"同一张图对应多个类"的组生效：
    上下文是数字就抬同形组里的数字，是字母就抬字母。+2.4 对数几率足以翻转
    0.7 vs 0.16 这种差距，同时不影响无关字符。
    """
    ctx = (ctx or "")[-8:]
    digits = sum(c.isdigit() for c in ctx)
    letters = sum(c.isalpha() for c in ctx)
    bonus = np.zeros(len(classes))
    if not ctx or (digits and letters):
        return bonus
    prefer_digit = digits > 0
    for group in AMBIG_SETS:
        members = [i for i, ch in enumerate(classes) if ch in group]
        if len(members) < 2:
            continue
        for i in members:
            ch = classes[i]
            # 只抬"真正想要的类别"：字母上下文只抬字母，数字上下文只抬数字。
            # 不能把符号（| / l 这类）一起抬，否则 I 永远压不过 |。
            hit = ch.isdigit() if prefer_digit else ch.isalpha()
            if hit:
                bonus[i] += 3.2
    return bonus


def candidates(probs: np.ndarray, classes: List[str], prior: float, k: int = 6, ctx: str = "",
               payload_prefix: str = ""):
    """大小写先验 + 上下文先验重排，并按身份去重。"""
    logits = np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float64)
    for i, ch in enumerate(classes):
        if ch.isupper() or ch in "ÆŒÊ":
            logits[i] += prior
        elif ch.islower() or ch in "æœê":
            logits[i] -= prior
    logits += ambiguity_bonus(classes, ctx)
    logits += lm_bonus(classes, payload_prefix)
    # 展示用的置信度必须跟着重排后的分数走，否则会出现"第一候选 16%、第二候选 69%"的怪象
    adj = np.exp(logits - logits.max())
    adj /= adj.sum()

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
                "p": round(float(adj[int(idx)]), 4),
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
    """读取样本；与训练侧保持一致，跳过 meta.json 里被隔离的历史错标样本。

    否则会出现"面板显示 0 已写满 11 个、训练侧却只有 0 个"的错位，
    用户永远补不到这个类。
    """
    p = collect_path(split)
    if not os.path.exists(p):
        return []
    cutoff = 0.0
    meta = os.path.join(USER_DIR, "meta.json")
    if os.path.exists(meta):
        try:
            with open(meta, "r", encoding="utf-8") as f:
                cutoff = float(json.load(f).get("exclude_before", 0.0) or 0.0)
        except Exception:
            cutoff = 0.0
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if float(r.get("ts", 0) or 0) >= cutoff:
                out.append(r)
    return out


def counts_by_class(split: str) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for r in _read_all(split):
        c[r["label"]] = c.get(r["label"], 0) + 1
    return c


# --------------------------------------------------------------------- HTTP
REC = Recognizer()
load_words()


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
        elif path == "/api/suggest":
            q = parse_qs(urlparse(self.path).query)
            prefix = (q.get("prefix") or [""])[0]
            limit = int((q.get("limit") or ["12"])[0])
            self._json({"prefix": prefix, "items": suggest(prefix, limit) if prefix else []})
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
            elif self.path == "/api/undo":
                self._handle_undo(payload)
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
        top = candidates(probs, REC.classes, prior, ctx=payload.get("context") or "",
                         payload_prefix=payload.get("prefix") or "")
        ms = (time.perf_counter() - t0) * 1000
        hint = "upper" if prior > 0.5 else ("lower" if prior < -0.5 else "unknown")
        self._json({"top": top, "ms": round(ms, 1), "case_hint": hint, "height": round(met["height"], 1)})

    def _handle_undo(self, payload):
        """删掉最后一条刚存错的样本（采集时写错/标签不对时用）。"""
        split = str(payload.get("split") or "train")
        if split not in ("train", "test"):
            split = "train"
        p = collect_path(split)
        if not os.path.exists(p):
            self._json({"ok": False, "error": "没有可撤销的样本"})
            return
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if not lines:
            self._json({"ok": False, "error": "没有可撤销的样本"})
            return
        try:
            removed = json.loads(lines[-1]).get("label")
        except json.JSONDecodeError:
            removed = "?"
        with open(p, "w", encoding="utf-8") as f:
            if lines[:-1]:
                f.write("\n".join(lines[:-1]) + "\n")
        self._json({"ok": True, "removed": removed, "count": counts_by_class(split).get(removed, 0)})

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
