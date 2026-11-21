# -*- coding: utf-8 -*-
"""实验②：100 类扩展字符集模型。

数据配比（每 epoch 按类别反频率采样 15 万张）：
  * MNIST 6 万（数字，硬标签）
  * EMNIST letters 12.5 万（字母，大小写 50/50 软标签）
  * 字体合成 4 万（覆盖标点与 Æ æ Œ œ Ê ê，硬标签）

评估三套：MNIST 测试集 / EMNIST letters 测试集 / 合成留出集（不同随机种子）。
同时报告 exact（100 类）与 identity（71 个身份类，忽略大小写）两个口径。
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from . import data
from .charset import CLASSES, NUM_CLASSES
from .models import GlyphNet
from .train_common import RunLogger, amp_context, evaluate, pick_device

EPOCHS = 12
BATCH = 256
LR = 3e-3
WD = 5e-5
SAMPLES_PER_EPOCH = 150_000
OUT = r"F:\mnist\runs\exp2_ext"


def main() -> None:
    device = pick_device()
    os.makedirs(OUT, exist_ok=True)
    log = RunLogger("exp2_ext")

    x, y1, y2 = data.load_ext_data(per_class=400)
    ev = {
        "mnist": data.build_mnist("test"),
        "emnist": data.build_emnist_letters("test"),
        "synth": data.build_synth_cached(per_class=60, seed=999, use_cache=True),
    }
    for k, (ex, ey1, ey2) in ev.items():
        log.log(f"评估集 {k}: {len(ex)} 张")

    model = GlyphNet(NUM_CLASSES).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log.log(f"模型 GlyphNet 参数量 {n_par/1e3:.1f}K, 设备 {device}, 训练集 {len(x)}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = torch.Generator().manual_seed(0)
    best = 0.0
    t_start = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        order = data.class_balanced_order(y1, SAMPLES_PER_EPOCH, seed=1000 + ep)
        t0 = time.time()
        run_loss, run_n = 0.0, 0
        for xb, b1, b2 in data.make_batches(
            x, y1, y2, order, BATCH, device, train=True, gen=gen, strength=1.0
        ):
            with amp_context(device):
                logits = model(xb)
            target = data.build_soft_targets(b1, b2, NUM_CLASSES)
            loss = data.soft_ce(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run_loss += float(loss.detach()) * len(b1)
            run_n += len(b1)
        sched.step()
        tr_loss = run_loss / max(run_n, 1)

        row = {"epoch": ep, "train_loss": round(tr_loss, 5), "lr": round(float(sched.get_last_lr()[0]), 6)}
        parts = []
        score = 0.0
        for name, (ex, ey1, ey2) in ev.items():
            m = evaluate(model, ex, ey1, ey2, device, n_classes=NUM_CLASSES)
            row[f"{name}_loss"] = round(m["loss"], 5)
            row[f"{name}_acc_id"] = round(m["acc_id"], 5)
            row[f"{name}_acc_top5"] = round(m["acc_top5_id"], 5)
            if "acc_exact" in m:
                row[f"{name}_acc_exact"] = round(m["acc_exact"], 5)
            parts.append(
                f"{name}: loss={m['loss']:.3f} id={m['acc_id']*100:.1f}% top5={m['acc_top5_id']*100:.1f}%"
            )
            score += m["acc_id"]
        score /= len(ev)
        row["mean_acc_id"] = round(score, 5)
        row["sec"] = round(time.time() - t0, 1)
        log.metric(row)
        log.log(f"epoch {ep:02d}/{EPOCHS} train_loss={tr_loss:.4f} | " + " | ".join(parts) + f" | {row['sec']}s")

        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": ep, "mean_acc_id": score, "classes": CLASSES},
                       os.path.join(OUT, "best.pt"))
            log.log(f"   ^ 新最佳（mean identity acc {score*100:.2f}%），已保存")

    log.log(f"训练完成，最佳 mean identity acc {best*100:.2f}%，总耗时 {(time.time()-t_start)/60:.1f} 分钟")

    ck = torch.load(os.path.join(OUT, "best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    for name, (ex, ey1, ey2) in ev.items():
        m = evaluate(model, ex, ey1, ey2, device, n_classes=NUM_CLASSES)
        log.log(f"复载最佳权重 {name}: {m}")
        np.savez_compressed(
            os.path.join(OUT, f"pred_{name}.npz"),
            images=(ex * 255).astype(np.uint8),
            labels=ey1.astype(np.int64),
            labels2=ey2.astype(np.int64),
            preds=_collect_preds(model, ex, device),
        )


@torch.no_grad()
def _collect_preds(model, x, device, batch_size=512):
    model.eval()
    out = np.empty(len(x), dtype=np.int64)
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).unsqueeze(1).to(device)
        with amp_context(device):
            logits = model(xb)
        out[i : i + batch_size] = logits.argmax(1).cpu().numpy()
    return out


if __name__ == "__main__":
    main()
