# -*- coding: utf-8 -*-
"""诊断：eval 模式 vs train 模式（batch 统计）的准确率差异，判断是否 BN 统计量失效。"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r"F:\mnist")
from src import data
from src.charset import NUM_CLASSES
from src.models import GlyphNet
from src.train_common import amp_context, evaluate

EPOCHS = 4
BS = 192


def acc_with_batch_stats(model, x, y1, x_small=4000):
    """用 batch 自己的统计量（train 模式）算准确率，但不改动 running stats。"""
    bufs = {k: v.detach().clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches" in k}
    model.train()
    n = min(len(x), x_small)
    correct = 0
    with torch.no_grad():
        for i in range(0, n, 256):
            xb = torch.from_numpy(x[i : i + 256]).unsqueeze(1).to("xpu")
            with amp_context(torch.device("xpu")):
                logits = model(xb)
            correct += int((logits.argmax(1).cpu().numpy() == y1[i : i + 256]).sum())
    model.load_state_dict(bufs, strict=False)
    model.eval()
    return correct / n


x, y1, y2 = data.load_ext_data(per_class=400)
ex, ey1, ey2 = data.build_emnist_letters("test")

model = GlyphNet(NUM_CLASSES).to("xpu")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-5, foreach=False)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
gen = torch.Generator().manual_seed(0)

for ep in range(1, EPOCHS + 1):
    model.train()
    order = data.class_balanced_order(y1, 150_000, seed=1000 + ep)
    t0 = time.time()
    for xb, b1, b2 in data.make_batches(x, y1, y2, order, BS, "xpu", True, gen):
        with amp_context(torch.device("xpu")):
            logits = model(xb)
        loss = data.group_ce(logits, b1, b2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False)
        opt.step()
    sched.step()
    m_eval = evaluate(model, ex, ey1, ey2, torch.device("xpu"), n_classes=NUM_CLASSES)
    a_batch = acc_with_batch_stats(model, ex, ey1)
    print(
        f"epoch {ep}: eval模式 id={m_eval['acc_id']*100:.1f}%  "
        f"batch统计 id={a_batch*100:.1f}%  ({time.time()-t0:.0f}s)",
        flush=True,
    )
