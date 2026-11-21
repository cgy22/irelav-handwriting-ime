# -*- coding: utf-8 -*-
"""诊断：XPU 启动失败的进程内重试策略是否可靠。

失败都发生在首个 batch，所以重试成本很低（几秒）。这里验证：
连续 6 次尝试内能否恢复，以及恢复后的吞吐。
"""
import gc
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, r"F:\mnist")
from src import data
from src.models import GlyphNet

BATCH = 192
x = np.random.rand(192 * 4, 28, 28).astype(np.float32) * 0.6
y1 = np.random.randint(0, 100, len(x)).astype(np.int64)
y2 = np.full_like(y1, -1)
gen = torch.Generator().manual_seed(0)

for attempt in range(6):
    m = opt = None
    try:
        m = GlyphNet(100).to("xpu")
        opt = AdamW(m.parameters(), lr=1e-3)
        order = np.arange(len(x))
        steps, t0 = 0, time.time()
        for xb, b1, b2 in data.make_batches(x, y1, y2, order, BATCH, "xpu", True, gen):
            with torch.autocast("xpu", dtype=torch.bfloat16):
                logits = m(xb)
            loss = data.group_ce(logits, b1, b2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            if steps >= 4:
                break
        torch.xpu.synchronize()
        dt = time.time() - t0
        print(f"OK attempt={attempt} steps={steps} {dt:.2f}s -> {steps*BATCH/dt:.0f} img/s steps", flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print(f"FAIL attempt={attempt}: {type(e).__name__} {str(e)[:60]}", flush=True)
        m = opt = None
        gc.collect()
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass
        time.sleep(0.4)
else:
    print("ALL_FAIL", flush=True)
