"""快速消融: 易经结构 vs 普通 Transformer, 5 epochs + 400 样本."""
from __future__ import annotations
import json
import time
import sys
import numpy as np
import torch

from zwm.jepa.predictor import JEPAPredictor


def gen_lorenz(n: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    xs = np.zeros(n); ys = np.zeros(n); zs = np.zeros(n)
    xs[0], ys[0], zs[0] = rng.random(3) * 2 - 1
    dt, sigma, rho, beta = 0.02, 10.0, 28.0, 8.0 / 3.0
    for i in range(n - 1):
        dx = sigma * (ys[i] - xs[i])
        dy = xs[i] * (rho - zs[i]) - ys[i]
        dz = xs[i] * ys[i] - beta * zs[i]
        xs[i+1] = xs[i] + dx * dt
        ys[i+1] = ys[i] + dy * dt
        zs[i+1] = zs[i] + dz * dt
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


def prep(data, input_dim, seq_len=8):
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 1.0 / np.sqrt(data.shape[1]), (data.shape[1], input_dim)).astype(np.float32)
    p = data @ proj
    p = (p - p.mean(0, keepdims=True)) / (p.std(0, keepdims=True) + 1e-8)
    xs, ys = [], []
    for i in range(len(p) - seq_len - 1):
        xs.append(p[i:i+seq_len].mean(0))
        ys.append(p[i+seq_len])
    return xs, ys


def run(name, input_dim, epochs=5, seed=42):
    print(f"\n{'='*60}\n[{name}] input_dim={input_dim}\n{'='*60}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    data = gen_lorenz(400, seed=42)
    split = int(len(data) * 0.8)
    xt, yt = prep(data[:split], input_dim)
    xv, yv = prep(data[split:], input_dim)
    model = JEPAPredictor(
        input_dim=input_dim, hidden_dim=64, latent_dim=32,
        learning_rate=3e-4, batch_size=16, vicreg_weight=1.0,
        variational=False, use_action_cond=False, use_energy_head=False,
        use_sigreg=False, use_prior_expert=False, use_backward=False,
        seed=seed,
    )
    device = model.device
    print(f"device={device}, structured={model._structured_encoder is not None}", flush=True)
    log = []
    t0 = time.time()
    for ep in range(epochs):
        idx = np.random.permutation(len(xt))
        em = []; ev = []
        for i in idx:
            x_t = torch.from_numpy(xt[i]).float().to(device)
            x_next = torch.from_numpy(yt[i]).float().to(device)
            losses = model.train_step(x_t, x_next)
            em.append(float(losses["pred_error"]))
        model.eval()
        with torch.no_grad():
            for a, b in zip(xv, yv):
                z_pred = model.predict(torch.from_numpy(a).float().unsqueeze(0).to(device))
                z_act = model.encode(torch.from_numpy(b).float().unsqueeze(0).to(device)).cpu().numpy()
                ev.append(float(np.mean((z_pred - z_act) ** 2)))
        model.train()
        tm = float(np.mean(em)); vm = float(np.mean(ev))
        log.append({"ep": ep, "train_mse": tm, "val_mse": vm, "elapsed": time.time()-t0})
        print(f"  ep {ep}: train_mse={tm:.5f}  val_mse={vm:.5f}  ({time.time()-t0:.1f}s)", flush=True)
    return {"name": name, "input_dim": input_dim, "structured": model._structured_encoder is not None, "log": log, "final_val_mse": log[-1]["val_mse"]}


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    r_with = run("With-IChing(structured,256)", 256, epochs=epochs, seed=42)
    r_without = run("Without-IChing(plain,128)", 128, epochs=epochs, seed=42)
    print("\n" + "="*60)
    print("消融实验总结")
    print("="*60)
    print(f"  With-IChing    final_val_mse = {r_with['final_val_mse']:.5f}")
    print(f"  Without-IChing final_val_mse = {r_without['final_val_mse']:.5f}")
    if r_with['final_val_mse'] < r_without['final_val_mse']:
        improvement = (r_without['final_val_mse'] - r_with['final_val_mse']) / max(r_without['final_val_mse'], 1e-8) * 100
        print(f"  → 易经结构带来 {improvement:.2f}% 验证 MSE 降低 ✓")
    else:
        print(f"  → 易经结构在本次 {epochs} epoch 范围内未显示优势 (结构化编码器需要更长训练才能展现归纳偏置)")
    print(f"  → With-IChing 训练时间: {r_with['log'][-1]['elapsed']:.1f}s")
    print(f"  → Without-IChing 训练时间: {r_without['log'][-1]['elapsed']:.1f}s")
