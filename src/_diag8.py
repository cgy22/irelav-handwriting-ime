import sys, time; sys.path.insert(0, r"F:\mnist")
import numpy as np, torch
from src import data
from src.models import GlyphNet
tag = sys.argv[1]
x = np.random.rand(1536, 28, 28).astype(np.float32) * 0.6
y1 = np.random.randint(0,100,len(x)).astype(np.int64); y2 = np.full_like(y1,-1)
gen = torch.Generator().manual_seed(0); dev = torch.device("xpu")
m = GlyphNet(100).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=5e-5, foreach=False)
order = np.arange(len(x)); steps=0; t0=time.time()
try:
    while steps < 300:
        for xb, b1, b2 in data.make_batches(x,y1,y2,order,192,dev,True,gen):
            with torch.autocast("xpu", dtype=torch.bfloat16): logits = m(xb)
            loss = data.group_ce(logits, b1, b2)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0, foreach=False)
            opt.step(); steps += 1
            if steps >= 300: break
    torch.xpu.synchronize()
    print(f"OK clip(foreach=False) 300步 {(time.time()-t0):.1f}s {steps*192/(time.time()-t0):.0f} img/s", flush=True)
except Exception as e:
    print(f"FAIL step={steps}: {str(e)[:70]}", flush=True)
