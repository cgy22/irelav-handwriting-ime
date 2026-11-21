# -*- coding: utf-8 -*-
"""最终训练：把用户手写样本并入训练集，从实验②权重继续微调。

与实验②的区别：
  1. 训练池 = MNIST + EMNIST + 字体合成 + **用户手写（重复 8 倍，保证不被淹没）**
  2. 从 runs/exp2_ext/best.pt 初始化，学习率降到 2e-4，少跑几轮
  3. 每轮除三套基准评估集外，额外评估**用户测试集**（若已采集）

用法:
    python -m src.train_final            # 默认 6 epoch
    python -m src.train_final 10         # 指定 epoch
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from . import data, userdata
from .charset import CLASSES, NUM_CLASSES
from .models import GlyphNet
from .train_common import RunLogger, amp_context, evaluate, pick_device, recalibrate_bn

EXP2 = r"F:\mnist\runs\exp2_ext\best.pt"
OUT = r"F:\mnist\runs\exp2_final"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BS = 256
LR = 2e-4
BASE_PER_EPOCH = 120_000
USER_REPEAT = 8


def main() -> None:
    device = pick_device()
    os.makedirs(OUT, exist_ok=True)
    log = RunLogger("exp2_final")

    x, y1, y2 = data.load_ext_data(per_class=400)
    ux, uy, uraw = userdata.build_arrays("train")
    log.log(f"用户训练样本 {len(ux)} 条（重复 {USER_REPEAT} 倍参与训练）")
    if len(ux):
        x = np.concatenate([x, ux], 0)
        y1 = np.concatenate([y1, uy], 0)
        y2 = np.concatenate([y2, np.full_like(uy, -1)], 0)

    ev = {
        "mnist": data.build_mnist("test"),
        "emnist": data.build_emnist_letters("test"),
        "synth": data.build_synth_cached(per_class=60, seed=999),
    }
    ute = userdata.build_arrays("test")
    if len(ute[0]):
        ev["user"] = ute
        log.log(f"用户测试集 {len(ute[0])} 条")

    model = GlyphNet(NUM_CLASSES).to(device)
    if os.path.exists(EXP2):
        ck = torch.load(EXP2, map_location=device)
        model.load_state_dict(ck["model"])
        log.log(f"从实验②权重初始化（epoch {ck.get('epoch')}, mean {ck.get('mean_acc_id', 0)*100:.2f}%）")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-5, foreach=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    best = 0.0
    t_start = time.time()

    n_base = len(x) - len(ux)

    for ep in range(1, EPOCHS + 1):
        model.train()
        base_order = data.class_balanced_order(y1[:n_base], BASE_PER_EPOCH, seed=2000 + ep)
        parts = [base_order]
        if len(ux):
            user_idx = np.tile(np.arange(n_base, n_base + len(ux)), USER_REPEAT)
            parts.append(user_idx)
        order = np.concatenate(parts)
        rng.shuffle(order)

        t0 = time.time()
        run_loss, n = 0.0, 0
        for xb, b1, b2 in data.make_batches(x, y1, y2, order, BS, device, True, gen):
            with amp_context(device):
                logits = model(xb)
            loss = data.group_ce(logits, b1, b2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False)
            opt.step()
            run_loss += float(loss.detach()) * len(b1)
            n += len(b1)
        sched.step()
        recalibrate_bn(model, x, device, seed=200 + ep)

        row = {"epoch": ep, "train_loss": round(run_loss / max(n, 1), 5),
               "lr": round(float(sched.get_last_lr()[0]), 6)}
        msg, score = [], 0.0
        for name, (ex, ey1, ey2) in ev.items():
            m = evaluate(model, ex, ey1, ey2, device, n_classes=NUM_CLASSES)
            row[f"{name}_acc_id"] = round(m["acc_id"], 5)
            row[f"{name}_acc_top5"] = round(m["acc_top5_id"], 5)
            if "acc_exact" in m:
                row[f"{name}_acc_exact"] = round(m["acc_exact"], 5)
            msg.append(f"{name}={m['acc_id']*100:.2f}%")
            score += m["acc_id"]
        score /= len(ev)
        row["mean_acc_id"] = round(score, 5)
        row["sec"] = round(time.time() - t0, 1)
        log.metric(row)
        log.log(f"epoch {ep:02d}/{EPOCHS} loss={row['train_loss']:.4f} | " + " | ".join(msg) + f" | {row['sec']}s")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": ep, "mean_acc_id": score, "classes": CLASSES},
                       os.path.join(OUT, "best.pt"))
            log.log(f"   ^ 新最佳 {score*100:.2f}%，已保存")

    log.log(f"完成，最佳 {best*100:.2f}%，耗时 {(time.time()-t_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
