# -*- coding: utf-8 -*-
"""诊断：真实数据管线下的 XPU 长跑稳定性。

对比两种喂数据方式：
  A) 每步 torch.from_numpy(...).to(device)      —— 每步新建 device 张量
  B) 预分配 device 缓冲 + copy_                 —— 不产生新的 device 分配
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r"F:\mnist")
from src import data
from src.models import GlyphNet

MODE = sys.argv[1] if len(sys.argv) > 1 else "A"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
BS = 192

x = np.random.rand(BS * 8, 28, 28).astype(np.float32) * 0.6
y1 = np.random.randint(0, 100, len(x)).astype(np.int64)
y2 = np.full_like(y1, -1)
gen = torch.Generator().manual_seed(0)

dev = torch.device("xpu")
m = GlyphNet(100).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=5e-5, foreach=False)
order = np.arange(len(x))
buf = torch.empty(BS, 1, 28, 28, device=dev) if MODE == "B" else None

t0 = time.time()
steps = 0
try:
    while steps < STEPS:
        for xb_cpu, b1, b2 in data.make_batches(x, y1, y2, order, BS, dev, True, gen):
            if MODE == "B":
                buf.copy_(xb_cpu, non_blocking=False)
                xb = buf
            else:
                xb = xb_cpu
            with torch.autocast("xpu", dtype=torch.bfloat16):
                logits = m(xb)
            loss = data.group_ce(logits, b1, b2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            if steps >= STEPS:
                break
    torch.xpu.synchronize()
    dt = time.time() - t0
    print(f"OK mode={MODE} steps={steps} {dt/steps*1000:.1f} ms/step {steps*BS/dt:.0f} img/s", flush=True)
except Exception as e:  # noqa: BLE001
    dt = time.time() - t0
    print(f"FAIL mode={MODE} at step={steps} ({dt:.1f}s): {type(e).__name__} {str(e)[:80]}", flush=True)
