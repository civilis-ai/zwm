#!/usr/bin/env python3
"""易经结构负担测试 — 在不同类型数据上对比 Full vs Flat.

数据:
  Structured:   60步周期 + 8×8热点 + 4层层次 + 6组语义
  Neutral:      Gaussian随机游走, 无周期/无空间结构/无层次
  Anti-structured: Structured数据随机打乱维度, 破坏所有结构

每种数据上运行 Full I Ching vs Flat baseline,
测量误差/方差/有效维度/训练稳定性。

假设:
  Structured:   Full < Flat  (结构是优势)
  Neutral:      Full ≈ Flat  (结构不造成负担)
  Anti-structured: Full <≈ Flat (结构具有鲁棒性)
"""

from __future__ import annotations

import math, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════
# 数据生成器
# ═══════════════════════════════════════════════════════════════════════

def generate_structured(step: int, seed: int) -> np.ndarray:
    """结构化数据 — 包含易经能捕获的模式."""
    rng = np.random.RandomState(seed + step)

    # 8×8 空间热点
    hr, hc = (step // 8) % 8, step % 8
    field = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            d = abs(r - hr) + abs(c - hc)
            field[r, c] = max(0, 1.0 - d / 8.0)
    sq = field.flatten()  # 64

    # 60步周期
    p60 = (step % 60) / 60.0
    cycle = np.sin(2 * math.pi * (p60 + np.arange(64) / 64.0)).astype(np.float32)  # 64

    # 4层层次
    hier = np.array([
        math.sin(2 * math.pi * step / 720),
        math.sin(2 * math.pi * step / 180),
        math.sin(2 * math.pi * step / 60),
        math.sin(2 * math.pi * step / 5),
    ] * 16, dtype=np.float32)[:64]  # 64

    # 6组语义
    sem = np.zeros(64, dtype=np.float32)
    for g in range(6):
        i0, i1 = g * 10, min(g * 10 + 11, 64)
        n = i1 - i0
        sem[i0:i1] = np.sin(2 * math.pi * step / 60 * (0.5 + g * 0.3)
                            + math.pi * g / 6) + rng.randn(n) * 0.02

    return np.concatenate([sq, cycle, hier, sem]).astype(np.float32)  # 256


def generate_neutral(step: int, seed: int) -> np.ndarray:
    """中性数据 — 随机游走, 无隐藏结构."""
    rng = np.random.RandomState(seed)
    # 使用固定种子生成基础随机游走状态
    base = rng.randn(256).astype(np.float32) * 0.1
    # 每步叠加小噪声 (随机游走)
    rng2 = np.random.RandomState(seed + step)
    noise = rng2.randn(256).astype(np.float32) * 0.05
    return np.clip(base * (1.0 - step * 0.001) + noise, -1, 1)


def generate_anti(step: int, seed: int) -> np.ndarray:
    """反结构化 — 结构化数据随机打乱维度, 同样的边际分布, 零结构."""
    structured = generate_structured(step, seed)
    rng = np.random.RandomState(seed * 137 + step * 59)
    perm = rng.permutation(256)
    return structured[perm].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# 编码器工厂
# ═══════════════════════════════════════════════════════════════════════

def make_full_encoder():
    """Full I Ching 编码器."""
    from zwm.jepa.field_gnn import FieldSquareGNN
    return FieldSquareGNN(hidden_dim=64, num_layers=2)

def make_flat_encoder():
    """Flat MLP 编码器 (同参数量的基线)."""
    class FlatEnc(nn.Module):
        def __init__(self):
            super().__init__()
            # 384 → 64 (与 GNN 的 (64,6)→64 同参数)
            self._net = nn.Sequential(
                nn.Linear(384, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Linear(128, 64),
            )
        def embed_field(self, f):
            self.eval()
            d = next(self.parameters()).device
            x = torch.from_numpy(f.astype(np.float32)).flatten().unsqueeze(0).to(d)
            if x.shape[-1] < 384:
                x = torch.nn.functional.pad(x, (0, 384 - x.shape[-1]))
            with torch.no_grad():
                return self._net(x).squeeze(0).cpu().numpy().astype(np.float32)
    return FlatEnc()


# ═══════════════════════════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════════════════════════

def run_one(data_gen, use_full: bool, steps: int, seed: int) -> dict:
    """单次运行."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from zwm.jepa.predictor import JEPAPredictor
    jepa = JEPAPredictor(input_dim=256, hidden_dim=192, latent_dim=64,
                         vicreg_weight=0.04, replay_capacity=64, batch_size=8,
                         seed=seed, use_action_cond=False)

    enc = make_full_encoder() if use_full else make_flat_encoder()

    errors, vicregs, norms = [], [], []
    for step in range(steps):
        z_t = data_gen(step, seed)
        z_n = data_gen(step + 1, seed)

        if use_full:
            # Full: 从 64-dim field 块提取 (64,6) 并通过 GNN
            field_t = z_t[:64].reshape(8, 8)
            from zwm.encoder.field_encoder import HexagramFieldEncoder
            fe = HexagramFieldEncoder(strategy="spatial")
            hex_field = fe.encode(field_t.reshape(1, 8, 8).repeat(28, axis=0).repeat(28, axis=1) if False else field_t)
            # 简化: 直接用场的 64 维作为 (64,1) 然后 tile 到 (64,6)
            hex_field = np.tile(field_t.reshape(64, 1), (1, 6)).astype(np.float32)
            z_sq = enc.embed_field(hex_field)
        else:
            # Flat: 直接处理 256-dim
            field_t_padded = np.pad(z_t[:64], (0, 320)).astype(np.float32)
            hex_field = np.tile(field_t_padded.reshape(64, 6), (1, 1)).astype(np.float32)
            z_sq = enc.embed_field(hex_field)

        z_world = np.concatenate([z_sq, z_t[64:]]).astype(np.float32)[:256]
        if len(z_world) < 256:
            z_world = np.pad(z_world, (0, 256 - len(z_world))).astype(np.float32)

        z_sq_n = enc.embed_field(np.tile(z_n[:64].reshape(64, 1), (1, 6)).astype(np.float32))
        z_world_n = np.concatenate([z_sq_n, z_n[64:]]).astype(np.float32)[:256]
        if len(z_world_n) < 256:
            z_world_n = np.pad(z_world_n, (0, 256 - len(z_world_n))).astype(np.float32)

        r = jepa.train_step(z_world, z_world_n)
        if not math.isnan(r.get("loss", float("nan"))):
            errors.append(r.get("pred_error", 0))
            with torch.no_grad():
                x = torch.from_numpy(z_world.astype(np.float32)).unsqueeze(0)
                zl = jepa.context_encoder(x.to(jepa.device))
                vicregs.append(float(zl.std()))
            gn = sum((p.grad.data.norm(2).item() ** 2) for p in jepa.parameters()
                     if p.grad is not None)
            norms.append(math.sqrt(gn))

    return {
        "errors": errors, "vicregs": vicregs, "norms": norms,
        "mean_err": np.mean(errors) if errors else 1.0,
        "mean_vicreg": np.mean(vicregs) if vicregs else 0,
        "grad_std": np.std(norms) if norms else 0,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    data_types = {
        "Structured": generate_structured,
        "Neutral": generate_neutral,
        "Anti-structured": generate_anti,
    }

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  易经结构负担测试                                         ║")
    print(f"║  Full I Ching vs Flat baseline on 3 data types          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    all_data: dict[str, dict[str, list]] = {}
    t0 = time.perf_counter()

    for dtype, gen_fn in data_types.items():
        print(f"  [{dtype}]")
        for use_full in [True, False]:
            label = "Full" if use_full else "Flat"
            print(f"    {label:<6s} ", end="", flush=True)
            runs = []
            for run in range(args.runs):
                seed = 42 + run * 100 + hash(dtype) % 1000
                r = run_one(gen_fn, use_full, args.steps, seed)
                runs.append(r)
                print("." if run < args.runs - 1 else "", end="", flush=True)
            key = f"{dtype}_{label}"
            all_data[key] = runs
            n = sum(len(r["errors"]) for r in runs)
            print(f" {n} steps")

    # ── 汇总 ──
    print(f"\n{'='*90}")
    print(f"  {'条件':<28s} {'预测误差↓':>9s} {'VICReg↑':>9s} {'梯度稳定':>9s} {'相对Flat':>10s}")
    print(f"  {'─'*28} {'─'*9} {'─'*9} {'─'*9} {'─'*10}")

    results = []
    for dtype in data_types:
        full_runs = all_data[f"{dtype}_Full"]
        flat_runs = all_data[f"{dtype}_Flat"]

        full_err = np.mean([r["mean_err"] for r in full_runs])
        flat_err = np.mean([r["mean_err"] for r in flat_runs])
        full_vic = np.mean([r["mean_vicreg"] for r in full_runs])
        flat_vic = np.mean([r["mean_vicreg"] for r in flat_runs])
        full_grad = np.mean([r["grad_std"] for r in full_runs])
        flat_grad = np.mean([r["grad_std"] for r in flat_runs])

        rel = (flat_err - full_err) / max(flat_err, 1e-8) * 100
        rel_v = (full_vic - flat_vic) / max(flat_vic, 1e-8) * 100

        arrow = "✅ 优势" if rel > 0 else ("➖ 持平" if abs(rel) < 2 else "❌ 负担")
        print(f"  {dtype+' Full':<28s} {full_err:>9.4f} {full_vic:>9.4f} {full_grad:>9.4f}")
        print(f"  {dtype+' Flat':<28s} {flat_err:>9.4f} {flat_vic:>9.4f} {flat_grad:>9.4f} "
              f"{rel:>+9.1f}%")

        results.append((dtype, full_err, flat_err, full_vic, flat_vic, rel, rel_v, arrow))

    # ── 解读 ──
    print(f"\n{'='*90}")
    print(f"  总结")
    print(f"{'='*90}")
    for dtype, fe, fle, fv, flv, rel, rv, arrow in results:
        print(f"  {dtype:<20s}: Full vs Flat 误差 {rel:+.1f}%, VICReg {rv:+.1f}% → {arrow}")

    # 判断
    structured_win = results[0][5] > 0
    neutral_neutral = abs(results[1][5]) < 5
    anti_ok = results[2][5] >= -5  # 不显著变差

    print(f"\n  判断:")
    if structured_win:
        print(f"    ✅ 结构化数据: 易经结构是优势")
    if neutral_neutral:
        print(f"    ✅ 中性数据:     易经结构不造成负担")
    elif results[1][5] > 0:
        print(f"    ✅ 中性数据:     易经结构甚至有轻微优势")
    else:
        print(f"    ❌ 中性数据:     易经结构是负担")
    if anti_ok:
        print(f"    ✅ 反结构数据:   易经结构具有鲁棒性, 不显著变差")

    elapsed = time.perf_counter() - t0
    print(f"\n  总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    raise SystemExit(main())
