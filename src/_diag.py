# -*- coding: utf-8 -*-
"""临时诊断脚本：定位 XPU 训练时的 DEVICE_LOST。用完可删。"""
import sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, r"F:\mnist")
from src import data, idx_reader
from src.models import MnistNet

cfg = sys.argv[1] if len(sys.argv) > 1 else "bf16+cl"
x, y = idx_reader.load_mnist("train")
x = x[:4096].astype(np.float32) / 255.0
y = y[:4096]
dev = torch.device("xpu")
m = MnistNet(10).to(dev)
if "cl" in cfg:
    m = m.to(memory_format=torch.channels_last)
opt = AdamW(m.parameters(), lr=1e-3)
gen = torch.Generator().manual_seed(0)
order = np.arange(len(x))
print("cfg:", cfg, flush=True)
step = 0
t0 = time.time()
try:
    for xb, y1, _ in data.make_batches(x, y, np.full_like(y, -1), order, 256, dev, True, gen):
        if "bf16" in cfg:
            with torch.autocast("xpu", dtype=torch.bfloat16):
                logits = m(xb)
        else:
            logits = m(xb)
        loss = nn.functional.cross_entropy(logits.float(), y1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 4 == 0:
            torch.xpu.synchronize()
            print(f"  step {step} loss={float(loss.detach()):.3f}", flush=True)
        if step >= 12:
            break
    print(f"OK steps={step} {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"FAIL at step {step}: {type(e).__name__} {str(e)[:120]}", flush=True)
