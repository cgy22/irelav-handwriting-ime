# -*- coding: utf-8 -*-
"""实验①：MNIST 基线 CNN（课程要求 1、2 的主体）。

刻意保持"教材形态"：
  * 用原始 28x28 MNIST（不套用扩展模型的规范化管线），保证和经典实验可比
  * 只用 MNIST 自身的 6 万训练样本
  * 每个 epoch 同时记录训练集损失与测试集损失，用于画两条曲线

产出：runs/exp1_mnist/{log.txt, metrics.csv, best.pt, test_pred.npz}
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import data, idx_reader
from .models import MnistNet
from .train_common import RunLogger, amp_context, pick_device

EPOCHS = 12
BATCH = 256
LR = 1.5e-3
WD = 1e-4
OUT = r"F:\mnist\runs\exp1_mnist"


@torch.no_grad()
def eval_epoch(model, x, y, device, batch_size=512):
    model.eval()
    n = len(x)
    loss_sum, correct = 0.0, 0
    preds = np.empty(n, dtype=np.int64)
    lf = nn.CrossEntropyLoss(reduction="sum")
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).unsqueeze(1).to(device)
        yb = torch.from_numpy(y[i : i + batch_size]).to(device)
        with amp_context(device):
            logits = model(xb)
        loss_sum += float(lf(logits.float(), yb))
        p = logits.argmax(1)
        preds[i : i + batch_size] = p.cpu().numpy()
        correct += int((p == yb).sum())
    return loss_sum / n, correct / n, preds


def main() -> None:
    device = pick_device()
    os.makedirs(OUT, exist_ok=True)
    log = RunLogger("exp1_mnist")

    xtr, ytr = idx_reader.load_mnist("train")
    xte, yte = idx_reader.load_mnist("test")
    xtr = xtr.astype(np.float32) / 255.0
    xte = xte.astype(np.float32) / 255.0
    log.log(f"MNIST 训练 {xtr.shape} / 测试 {xte.shape}")

    model = MnistNet(10).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log.log(f"模型 MnistNet 参数量 {n_par/1e3:.1f}K, 设备 {device}")

    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    best_acc = 0.0
    t_start = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(xtr))
        run_loss, run_n = 0.0, 0
        t0 = time.time()
        for xb_np, y1, _y2 in data.make_batches(
            xtr, np.zeros_like(ytr), np.full_like(ytr, -1),
            order, BATCH, device, train=True, gen=gen,
        ):
            yb = torch.from_numpy(ytr[order[run_n : run_n + len(xb_np)]]).to(device)
            with amp_context(device):
                logits = model(xb_np)
            loss = nn.functional.cross_entropy(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach()) * len(yb)
            run_n += len(yb)
        sched.step()

        tr_loss = run_loss / run_n
        te_loss, te_acc, _ = eval_epoch(model, xte, yte, device)
        dt = time.time() - t0
        row = {
            "epoch": ep,
            "train_loss": round(tr_loss, 5),
            "test_loss": round(te_loss, 5),
            "test_acc": round(te_acc, 5),
            "lr": round(float(sched.get_last_lr()[0]), 6),
            "sec": round(dt, 1),
        }
        log.metric(row)
        log.log(
            f"epoch {ep:02d}/{EPOCHS}  train_loss={tr_loss:.4f}  test_loss={te_loss:.4f}  "
            f"test_acc={te_acc*100:.2f}%  ({dt:.1f}s)"
        )
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "acc": te_acc, "classes": list("0123456789")},
                os.path.join(OUT, "best.pt"),
            )

    log.log(f"训练完成，最佳测试准确率 {best_acc*100:.2f}%，总耗时 {(time.time()-t_start)/60:.1f} 分钟")

    ck = torch.load(os.path.join(OUT, "best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    te_loss, te_acc, preds = eval_epoch(model, xte, yte, device)
    log.log(f"复载最佳权重复核：test_loss={te_loss:.4f} test_acc={te_acc*100:.2f}%")
    np.savez_compressed(
        os.path.join(OUT, "test_pred.npz"),
        images=(xte * 255).astype(np.uint8),
        labels=yte.astype(np.int64),
        preds=preds,
    )


if __name__ == "__main__":
    main()
