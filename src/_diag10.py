# -*- coding: utf-8 -*-
"""诊断：扫描学习率，看评估曲线是否平滑（判断是否 LR 过高导致震荡）。"""
import sys
import time

import torch

sys.path.insert(0, r"F:\mnist")
from src import data
from src.charset import NUM_CLASSES
from src.models import GlyphNet
from src.train_common import amp_context, evaluate, pick_device

LR = float(sys.argv[1]) if len(sys.argv) > 1 else 3e-4
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
STRENGTH = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
BS = 256

dev = pick_device()
x, y1, y2 = data.load_ext_data(per_class=400)
ex, ey1, ey2 = data.build_emnist_letters("test")
sx, sy1, sy2 = data.build_synth_cached(per_class=60, seed=999)

model = GlyphNet(NUM_CLASSES).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-5, foreach=False)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
gen = torch.Generator().manual_seed(0)
rng = __import__('numpy').random.default_rng(7)



def recalibrate_bn(model, x, dev, bs=256, n_batches=60):
    """用干净（未增强）数据重算 BN 的 running stats。

    训练时输入永远是被仿射/加粗/加噪过的，running stats 天然偏向"脏数据"分布；
    测试集是干净的，统计量不匹配会直接吃掉准确率。用干净数据做一次累积平均
    （momentum=None）可以把这层偏差抹掉，且几乎不花时间。
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    saved = [(b.momentum, b.num_batches_tracked.clone()) for b in bns]
    for b in bns:
        b.momentum = None
        b.num_batches_tracked.zero_()
    model.train()
    idx = rng.choice(len(x), n_batches * bs, replace=False)
    with torch.no_grad():
        for i in range(n_batches):
            part = idx[i * bs : (i + 1) * bs]
            xb = torch.from_numpy(x[part]).unsqueeze(1).to(dev)
            model(xb)
    for b, (m, n) in zip(bns, saved):
        b.momentum = m
        b.num_batches_tracked.copy_(n)
    model.eval()

for ep in range(1, EPOCHS + 1):
    model.train()
    order = data.class_balanced_order(y1, 150_000, seed=1000 + ep)
    t0 = time.time()
    for xb, b1, b2 in data.make_batches(x, y1, y2, order, BS, dev, True, gen, strength=STRENGTH):
        with amp_context(dev):
            logits = model(xb)
        loss = data.group_ce(logits, b1, b2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False)
        opt.step()
    sched.step()
    recalibrate_bn(model, x, dev)
    me = evaluate(model, ex, ey1, ey2, dev, n_classes=NUM_CLASSES)
    ms = evaluate(model, sx, sy1, sy2, dev, n_classes=NUM_CLASSES)
    mi = evaluate(model, *data.build_mnist("test"), dev, n_classes=NUM_CLASSES)
    print(
        f"LR={LR:g} str={STRENGTH:g} epoch {ep}: mnist={mi['acc_id']*100:.1f}% "
        f"emnist={me['acc_id']*100:.1f}% synth={ms['acc_id']*100:.1f}% ({time.time()-t0:.0f}s)",
        flush=True,
    )
