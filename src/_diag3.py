# -*- coding: utf-8 -*-
"""临时诊断：试三种 XPU 首轮分配规避手法。用法: python src/_diag3.py <variant>"""
import sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, r"F:\mnist")
from src import data, idx_reader
from src.models import MnistNet

variant = sys.argv[1] if len(sys.argv) > 1 else "warmup"
bs = 256
x, y = idx_reader.load_mnist("train")
x = x[:1024].astype(np.float32) / 255.0
y = y[:1024]
dev = torch.device("xpu")
try:
    if variant == "warmup":
        # 先把 GPU 从低功耗状态唤醒
        a = torch.randn(64, 64, device=dev)
        for _ in range(300):
            a = a @ a.T * 0.001
        torch.xpu.synchronize()
    elif variant == "frac":
        torch.xpu.set_per_process_memory_fraction(0.5)
    elif variant == "claim":
        big = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
        big.zero_()
        torch.xpu.synchronize()
        del big
        torch.xpu.empty_cache()

    m = MnistNet(10).to(dev)
    opt = AdamW(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    order = np.arange(len(x))
    n = 0
    for xb, y1, _ in data.make_batches(x, y, np.full_like(y, -1), order, bs, dev, True, gen):
        with torch.autocast("xpu", dtype=torch.bfloat16):
            logits = m(xb)
        loss = nn.functional.cross_entropy(logits.float(), y1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n += 1
        if n >= 4:
            break
    torch.xpu.synchronize()
    print(f"OK {variant}", flush=True)
except Exception as e:
    print(f"FAIL {variant} {type(e).__name__} {str(e)[:60]}", flush=True)
