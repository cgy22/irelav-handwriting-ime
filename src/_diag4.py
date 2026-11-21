import sys, torch
sys.path.insert(0, r"F:\mnist")
import numpy as np, torch.nn as nn
from torch.optim import AdamW
from src import data, idx_reader
from src.models import MnistNet
frac = float(sys.argv[1])
x, y = idx_reader.load_mnist("train"); x = x[:1536].astype(np.float32)/255.0; y = y[:1536]
try:
    torch.xpu.set_per_process_memory_fraction(frac)
    m = MnistNet(10).to("xpu"); opt = AdamW(m.parameters(), 1e-3)
    gen = torch.Generator().manual_seed(0); order = np.arange(len(x)); n = 0
    for xb, y1, _ in data.make_batches(x, y, np.full_like(y,-1), order, 256, torch.device("xpu"), True, gen):
        with torch.autocast("xpu", dtype=torch.bfloat16): logits = m(xb)
        loss = nn.functional.cross_entropy(logits.float(), y1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); n += 1
        if n >= 6: break
    torch.xpu.synchronize(); print("OK", flush=True)
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:60], flush=True)
