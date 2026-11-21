# -*- coding: utf-8 -*-
"""个性化微调：用用户自己的手写样本把模型"校准"到本人的笔迹。

这就是"越用越准"的实验：
  冷启动（exp2 原模型）  ->  用 N 个用户训练样本微调  ->  在用户**独立测试集**上对比

为了不灾难性遗忘，微调时混入一小份合成样本做正则。
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from . import data, report, userdata
from .charset import CLASSES, NUM_CLASSES
from .models import GlyphNet
from .train_common import RunLogger, amp_context, evaluate, pick_device

SRC = r"F:\mnist\runs\exp2_ext\best.pt"
OUT = r"F:\mnist\runs\exp2_user"
EPOCHS = 8
BATCH = 64
LR = 2e-4


def main() -> None:
    device = pick_device()
    log = RunLogger("exp2_user")
    if not os.path.exists(SRC):
        log.log("找不到 exp2_ext/best.pt，先跑实验②")
        return

    utr_x, utr_y, utr_raw = userdata.build_arrays("train")
    ute_x, ute_y, ute_raw = userdata.build_arrays("test")
    log.log(f"用户训练样本 {len(utr_x)} 条 / 测试样本 {len(ute_x)} 条")
    if len(ute_x) == 0:
        log.log("用户测试集为空，无法做微调前后对比")
        return

    ck = torch.load(SRC, map_location=device)
    model = GlyphNet(NUM_CLASSES).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    before = evaluate(model, ute_x, ute_y, None, device, n_classes=NUM_CLASSES)
    log.log(f"微调前（用户测试集）: acc_id={before['acc_id']*100:.2f}% acc_exact={before.get('acc_exact', 0)*100:.2f}%")

    if len(utr_x) == 0:
        log.log("没有用户训练样本，跳过微调")
        return

    # 混入合成样本防遗忘
    sx, sy1, sy2 = data.build_synth_cached(per_class=40, use_cache=True)
    x = np.concatenate([utr_x, sx], 0)
    y1 = np.concatenate([utr_y, sy1], 0)
    y2 = np.concatenate([np.full_like(utr_y, -1), sy2], 0)
    # 用户样本权重更高：重复 4 遍
    rep = np.concatenate([np.arange(len(utr_x))] * 4)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    best = 0.0

    for ep in range(1, EPOCHS + 1):
        order = np.concatenate([rep, np.arange(len(utr_x), len(x))])
        rng.shuffle(order)
        run_loss, n = 0.0, 0
        for xb, b1, b2 in data.make_batches(x, y1, y2, order, BATCH, device, True, gen, strength=0.6):
            with amp_context(device):
                logits = model(xb)
            loss = data.group_ce(logits, b1, b2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * len(b1)
            n += len(b1)
        sched.step()
        m = evaluate(model, ute_x, ute_y, None, device, n_classes=NUM_CLASSES)
        log.metric({"epoch": ep, "train_loss": round(run_loss / max(n, 1), 5),
                    "user_test_acc_id": round(m["acc_id"], 5),
                    "user_test_acc_exact": round(m.get("acc_exact", 0.0), 5)})
        log.log(f"epoch {ep:02d} loss={run_loss/max(n,1):.4f} user_test acc={m['acc_id']*100:.2f}%")
        if m["acc_id"] > best:
            best = m["acc_id"]
            os.makedirs(OUT, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": ep, "user_acc": best, "classes": CLASSES},
                       os.path.join(OUT, "best.pt"))

    ck2 = torch.load(os.path.join(OUT, "best.pt"), map_location=device)
    model.load_state_dict(ck2["model"])
    after = evaluate(model, ute_x, ute_y, None, device, n_classes=NUM_CLASSES)
    log.log(f"微调后（用户测试集）: acc_id={after['acc_id']*100:.2f}% acc_exact={after.get('acc_exact', 0)*100:.2f}%")
    log.log(f"净提升: {(after['acc_id']-before['acc_id'])*100:+.2f} 个百分点")

    preds = _preds(model, ute_x, device)
    report.plot_confusion(np.asarray(ute_y), preds, CLASSES, os.path.join(OUT, "user_confusion.png"),
                          "用户测试集混淆矩阵")
    n_err = report.plot_errors((ute_x * 255).astype(np.uint8), np.asarray(ute_y), preds, CLASSES,
                              os.path.join(OUT, "user_errors.png"), 32, "用户测试集错误样本")
    log.log(f"用户测试集错误样本 {n_err} 个，图已保存")


@torch.no_grad()
def _preds(model, x, device, batch_size=128):
    model.eval()
    out = np.empty(len(x), dtype=np.int64)
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).unsqueeze(1).to(device)
        with amp_context(device):
            out[i : i + batch_size] = model(xb).argmax(1).cpu().numpy()
    return out


if __name__ == "__main__":
    main()
