# -*- coding: utf-8 -*-
"""临时诊断：XPU 首轮分配稳定性扫描。用法: python src/_diag2.py <bs> <dtype>"""
import sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, r"F:\mnist")
from src import data, idx_reader
from src.models import MnistNet

bs = int(sys.argv[1])
dtype = sys.argv[2]
x, y = idx_reader.load_mnist("train")
x = x[:2048].astype(np.float32) / 255.0
y = y[:2048]
dev = torch.device("xpu")
try:
    m = MnistNet(10).to(dev)
    opt = AdamW(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    order = np.arange(len(x))
    n = 0
    t0 = time.time()
    for xb, y1, _ in data.make_batches(x, y, np.full_like(y, -1), order, bs, dev, True, gen):
        if dtype == "bf16":
            with torch.autocast("xpu", dtype=torch.bfloat16):
                logits = m(xb)
        else:
            logits = m(xb)
        loss = nn.functional.cross_entropy(logits.float(), y1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n += 1
        if n >= 8:
            break
    torch.xpu.synchronize()
    print(f"OK {bs} {dtype} {n}步 {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"FAIL {bs} {dtype} {type(e).__name__} {str(e)[:70]}", flush=True)
