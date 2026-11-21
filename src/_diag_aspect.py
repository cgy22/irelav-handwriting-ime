# -*- coding: utf-8 -*-
"""实验：各向异性增强幅度（±18% vs ±30%）对长宽比鲁棒性的影响。

用法: python src/_diag_aspect.py 0.30 6
"""
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r"F:\mnist")
from src import canon, data
from src.charset import NUM_CLASSES
from src.models import GlyphNet
from src.train_common import amp_context, evaluate, pick_device, recalibrate_bn

AR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
BS = 256
LR = 5e-4

dev = pick_device()
x, y1, y2 = data.load_ext_data(per_class=400)
ex, ey1, ey2 = data.build_emnist_letters("test")
mx, my1, my2 = data.build_mnist("test")

model = GlyphNet(NUM_CLASSES).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-5, foreach=False)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
gen = torch.Generator().manual_seed(0)

for ep in range(1, EPOCHS + 1):
    model.train()
    order = data.class_balanced_order(y1, 150_000, seed=1000 + ep)
    t0 = time.time()
    for xb, b1, b2 in data.make_batches(
        x, y1, y2, order, BS, dev, True, gen, strength=1.0, aspect_range=AR
    ):
        with amp_context(dev):
            logits = model(xb)
        loss = data.group_ce(logits, b1, b2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False)
        opt.step()
    sched.step()
    recalibrate_bn(model, x, dev, seed=100 + ep)
    m1 = evaluate(model, mx, my1, my2, dev, n_classes=NUM_CLASSES)
    m2 = evaluate(model, ex, ey1, ey2, dev, n_classes=NUM_CLASSES)
    print(f"AR=±{AR:.2f} ep{ep}: mnist={m1['acc_id']*100:.2f}% emnist={m2['acc_id']*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)


# ---- 长宽比鲁棒性测试
def distort(img, fx, fy):
    a = (img * 255).astype(np.float32)
    ys, xs = np.nonzero(a > 16)
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    im = Image.fromarray(crop.astype(np.uint8)).resize(
        (max(4, int(round(w * fx))), max(4, int(round(h * fy)))), Image.LANCZOS)
    big = np.zeros((120, 120), np.uint8)
    gw, gh = im.size
    big[60 - gh // 2:60 - gh // 2 + gh, 60 - gw // 2:60 - gw // 2 + gw] = np.asarray(im)
    return canon.canonicalize(big)


model.eval()  # evaluate() 结束时会把模型切回 train，这里必须显式切回 eval
torch.save({"model": model.state_dict(), "classes": None}, rf"F:\mnist\runs\_aspect_ar{AR:.2f}.pt")
idx = np.random.default_rng(0).choice(len(mx), 2000, replace=False)
xs, ys_lab = mx[idx], my1[idx]


@torch.no_grad()
def acc(xx):
    c = 0
    for i in range(0, len(xx), 512):
        xb = torch.from_numpy(np.stack(xx[i:i + 512])).unsqueeze(1).to(dev)
        with amp_context(dev):
            p = model(xb).argmax(1).cpu().numpy()
        c += int((p == ys_lab[i:i + 512]).sum())
    return c / len(xx)


base = acc(list(xs))
print(f"\nAR=±{AR:.2f} 基准 {base*100:.2f}%")
for f in (0.65, 0.80, 1.25, 1.50):
    v = acc([distort(im, 1.0, f) for im in xs])
    h = acc([distort(im, f, 1.0) for im in xs])
    print(f"  纵向×{f}: {v*100:5.2f}%   横向×{f}: {h*100:5.2f}%")
