# -*- coding: utf-8 -*-
"""训练公共件：设备选择（含 XPU 重试回落）、运行日志、评估。"""
from __future__ import annotations

import csv
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from .charset import CLASS2IDENT, NUM_IDENTITIES

RUNS_DIR = r"F:\mnist\runs"
_CLASS2IDENT = np.asarray(CLASS2IDENT, dtype=np.int64)


def pick_device(prefer: str | None = None, verbose: bool = True) -> torch.device:
    """选设备。默认 CPU。

    实测结论（本机 Arc 140V + 驱动 32.0.101.8860 + torch 2.15.0+xpu）：
    XPU 报 level_zero 39/40/20 号错误的真正原因不是驱动缺陷，而是**CPU 被占满**。
    Level Zero 的命令队列/事件登记需要 CPU 时间片，一旦 8 个核都被别的进程
    （尤其是同时跑的 CPU 训练）吃满，分配和同步就会超时并报
    OUT_OF_DEVICE_MEMORY / OUT_OF_RESOURCES / DEVICE_LOST。
    把 CPU 训练停掉之后，同一个模型在同一进程里立刻恢复正常（5873 img/s）。

    因此：**训练时不要同时跑 CPU 重负载**（这是必须遵守的使用约束）。
    """
    prefer = prefer or os.environ.get("MNIST_DEVICE", "xpu")
    if prefer == "xpu" and torch.xpu.is_available():
        for attempt in range(2):
            try:
                t = torch.randn(16, 16, device="xpu")
                (t @ t).sum().item()
                torch.xpu.synchronize()
                if verbose:
                    print(f"[device] XPU: {torch.xpu.get_device_name(0)}")
                return torch.device("xpu")
            except Exception as e:  # noqa: BLE001
                print(f"[device] XPU 第 {attempt + 1} 次初始化失败: {str(e)[:80]}")
                try:
                    torch.xpu.empty_cache()
                except Exception:
                    pass
    if verbose:
        print("[device] 使用 CPU")
    return torch.device("cpu")


def amp_context(device: torch.device):
    """XPU 上用 bf16 自动混合精度（实测比 fp32 快 3 倍且不需要 GradScaler）。"""
    if device.type == "xpu":
        return torch.autocast("xpu", dtype=torch.bfloat16)
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=False)


class RunLogger:
    """同时写人类可读 log.txt 和机器可读 metrics.csv。"""

    def __init__(self, name: str, root: str = RUNS_DIR):
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.log_path = os.path.join(self.dir, "log.txt")
        self.csv_path = os.path.join(self.dir, "metrics.csv")
        self._csv_fields = None

    def log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def metric(self, row: Dict[str, object]) -> None:
        fields = list(row.keys())
        new_file = not os.path.exists(self.csv_path)
        if new_file:
            self._csv_fields = fields
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._csv_fields or fields)
            if new_file:
                w.writeheader()
            w.writerow(row)


@torch.no_grad()
def recalibrate_bn(model: torch.nn.Module, x: np.ndarray, device: torch.device,
                   bs: int = 256, n_batches: int = 60, seed: int = 7) -> None:
    """用**干净、混合来源**的数据重算 BN running stats。

    训练时输入永远是被仿射/加粗/加噪过的，running stats 会偏向"脏数据"分布；
    测试集是干净的，统计量不匹配会直接吃掉 5~10 个点。用干净数据做一次累积平均
    （momentum=None）即可抹平，代价约 1 秒。

    必须在**混合来源**上做：只拿数字校准会让字母/标点的统计量跑偏
    （实测 MNIST 反跌到 75%）。
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if not bns:
        return
    saved = [(b.momentum, b.num_batches_tracked.clone()) for b in bns]
    for b in bns:
        b.momentum = None
        b.num_batches_tracked.zero_()
    model.train()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), min(n_batches * bs, len(x)), replace=False)
    for i in range(0, len(idx), bs):
        part = idx[i : i + bs]
        xb = torch.from_numpy(x[part]).unsqueeze(1).to(device)
        model(xb)
    for b, (mom, cnt) in zip(bns, saved):
        b.momentum = mom
        b.num_batches_tracked.copy_(cnt)
    model.eval()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    x: np.ndarray,
    y1: np.ndarray,
    y2: Optional[np.ndarray],
    device: torch.device,
    batch_size: int = 512,
    n_classes: int = 100,
) -> Dict[str, float]:
    """返回 loss / identity 准确率 / exact 准确率 / top5(identity)。

    exact 只在硬标签样本上统计（软标签样本不知道真实大小写，统计 exact 没意义）。
    """
    model.eval()
    n = len(x)
    loss_sum = 0.0
    correct_id = 0
    correct_top5 = 0
    correct_exact = 0
    n_exact = 0
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(x[i : i + batch_size]).unsqueeze(1).to(device)
        with amp_context(device):
            logits = model(xb)
        logp = torch.log_softmax(logits.float(), dim=1)

        t1 = torch.from_numpy(y1[i : i + batch_size]).to(device)
        if y2 is not None:
            t2 = torch.from_numpy(y2[i : i + batch_size]).to(device)
            soft = torch.zeros_like(logp)
            hard = t2 < 0
            soft.scatter_(1, t1.unsqueeze(1), torch.where(hard, 1.0, 0.5).unsqueeze(1))
            if (~hard).any():
                soft.scatter_(1, t2[~hard].unsqueeze(1), 0.5)
            loss_sum += float(-(soft * logp).sum(1).sum())
        else:
            loss_sum += float(torch.nn.functional.nll_loss(logp, t1, reduction="sum"))
            soft = None

        pred = logits.argmax(1)
        t_id = _CLASS2IDENT[y1[i : i + batch_size]]
        pred_id = _CLASS2IDENT[pred.cpu().numpy()]
        correct_id += int((pred_id == t_id).sum())

        top5 = logits.topk(min(5, n_classes), dim=1).indices.cpu().numpy()
        top5_id = _CLASS2IDENT[top5]
        correct_top5 += int((top5_id == t_id[:, None]).any(1).sum())

        if y2 is not None:
            hard_np = y2[i : i + batch_size] < 0
            if hard_np.any():
                cur = pred.cpu().numpy()[hard_np]
                correct_exact += int((cur == y1[i : i + batch_size][hard_np]).sum())
                n_exact += int(hard_np.sum())
        else:
            correct_exact += int((pred.cpu().numpy() == y1[i : i + batch_size]).sum())
            n_exact += len(pred)

    out = {
        "loss": loss_sum / max(n, 1),
        "acc_id": correct_id / max(n, 1),
        "acc_top5_id": correct_top5 / max(n, 1),
    }
    if n_exact:
        out["acc_exact"] = correct_exact / n_exact
    model.train()
    return out
