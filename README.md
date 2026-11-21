# 手写识别输入法（MNIST 实验 + 100 类字符集）

一个能**用笔、触控或鼠标直接写字**的本地输入法：PyTorch 训练的字形分类模型 + 浏览器手写面板。
除课程要求的 MNIST 实验外，还支持数字、大小写字母、32 个半角符号与 `Æ æ Œ œ Ê ê`，共 **100 类**。

## 环境

| 项 | 版本 / 说明 |
|---|---|
| Python | 3.15.0 |
| PyTorch | 2.15.0+xpu（`pip install --index-url https://download.pytorch.org/whl/xpu torch`） |
| 其它 | numpy / Pillow / matplotlib / scipy（均已安装） |
| 设备 | 训练默认 **CPU**；`set MNIST_DEVICE=xpu` 可尝试核显（见"已知问题"） |

## 目录结构

```
mnist/
├─ src/                 核心代码
│  ├─ charset.py        100 类字符集 + 大小写配对 + EMNIST 映射
│  ├─ idx_reader.py     MNIST/EMNIST 的 IDX 读取（含 EMNIST 转置矫正）
│  ├─ canon.py          统一预处理管线（包围盒→保比例→质心居中→28×28）
│  ├─ synth.py          字体合成（8 字体系 × 3 风格 + 弹性/仿射/粗细形变）
│  ├─ data.py           四路数据装载、batch 增强、软标签
│  ├─ models.py         MnistNet（10 类）/ GlyphNet（100 类）
│  ├─ train_mnist.py    实验① 基线训练
│  ├─ train_ext.py      实验② 100 类模型训练
│  ├─ analyze_exp1.py   实验① 出图与误差分析
│  ├─ analyze_exp2.py   实验② 出图与误差分析
│  ├─ userdata.py       用户手写样本读取
│  └─ finetune.py       个性化微调（"越用越准"）
├─ app/
│  ├─ server.py         HTTP 服务 + 推理 + 采集落盘
│  └─ static/           面板前端（HTML/CSS/JS，无框架）
├─ data/
│  ├─ cache/            规范化后的 npz 缓存
│  └─ user/             用户手写样本（JSONL，按笔迹存储）
├─ runs/                训练日志、指标 CSV、模型、图
├─ mnist_data/          原始 IDX 数据集
├─ start_ime.cmd        一键启动面板
└─ 实验报告.md
```

## 快速开始

```bat
:: 1) 训练 MNIST 基线（约 6 分钟 / CPU）
python -m src.train_mnist
python -m src.analyze_exp1

:: 2) 训练 100 类模型（约 30 分钟 / CPU，首次会先生成字体合成数据）
python -m src.train_ext
python -m src.analyze_exp2

:: 3) 启动手写面板
start_ime.cmd
```

浏览器打开 <http://127.0.0.1:8765>，用笔/触控/鼠标在书写区写字，停顿 0.8 秒自动提交；
顶部候选条可点选改字，`Shift` 强制大写形态。

## 采集自己的手写数据

1. 打开面板 → 「采集」标签
2. 左侧显示目标字符与进度，写完停顿 0.8 秒自动保存
3. 「每类 N 个」可调；「存入」可切到**测试集**（测试集绝不可用于训练）
4. 数据存于 `data/user/train.jsonl` / `test.jsonl`

采集完成后：

```bat
python -m src.finetune      :: 用你自己的训练样本微调，并在你的测试集上对比前后准确率
```

## 已知问题

* **XPU 训练不稳定**：本机 Arc 140V + 驱动 32.0.101.8860 + torch 2.15.0+xpu 组合下，
  训练有 40%~60% 概率在首个 batch 抛 `level_zero 39/40`（显存/资源错误）；
  换 batch、bf16/fp32、去 channels_last、限制显存比例都只能降低概率。
  推理路径稳定（25~30ms/字），所以训练默认走 CPU（约 1/8 速度，但可复现）。
* **大小写不能只靠模型**：孤立字形做完包围盒归一化后 `c` 与 `C` 是同一张图。
  工程上用「本行字高中位数」几何启发式 + 候选条 + Shift 三件套解决。
* **标点与 Æ 系列靠合成数据打底**：没有公开手写数据集，准确率低于数字，属预期。
