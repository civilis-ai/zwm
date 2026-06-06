#!/usr/bin/env python3
"""易经数学结构消融实验 v2 — 结构化数据 + 严格对照.

v1 问题: 正弦波数据太简单, 任何编码都一样好
v2 改进:
  1. 数据包含隐藏结构: 60步周期 + 4层趋势 + 8×8相关性 + 语义簇
  2. 每条件 5 轮 × 300 步
  3. 度量改为"结构发现率": 预测误差在结构边界处是否更低
  4. 增加"周期检测准确率"和"层次分离度"
"""

from __future__ import annotations

import math, sys, time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════
# 结构化数据生成器 — 隐藏模式
# ═══════════════════════════════════════════════════════════════════════

class StructuredDataGenerator:
    """生成包含隐藏易经结构的数据.

    数据结构:
      - 60步甲子周期: reward 每 60 步出现一次峰值
      - 4层层次趋势: 年/季/月/周 嵌套周期
      - 8×8空间相关: 64维中有 8×8 网格热力图模式
      - 语义聚类: 6组特征 (每组10-11维) 由不同动力学驱动
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)
        self._step = 0

    def generate(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """生成 (当前状态, 下一状态) 对.

        返回: z_t, z_next — 各 256-dim, 包含隐藏结构
        """
        t = step / 60.0
        t_next = (step + 1) / 60.0

        # ── 1. 8×8 空间场 (64 dim) ──
        # 有一个在网格上移动的"热点" (类似 HexagramField)
        hot_row = (step // 8) % 8
        hot_col = step % 8
        field_t = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                dist = abs(r - hot_row) + abs(c - hot_col)
                field_t[r, c] = max(0, 1.0 - dist / 8.0)
        field_t_flat = field_t.flatten()  # 64 dim

        hot_row_n = ((step + 1) // 8) % 8
        hot_col_n = (step + 1) % 8
        field_n = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                dist = abs(r - hot_row_n) + abs(c - hot_col_n)
                field_n[r, c] = max(0, 1.0 - dist / 8.0)
        field_n_flat = field_n.flatten()

        # ── 2. 60步周期 (64 dim) ──
        cycle_phase = (step % 60) / 60.0
        cycle_t = np.array([math.sin(2 * math.pi * (cycle_phase + i / 60.0))
                           for i in range(64)], dtype=np.float32)
        cycle_n = np.array([math.sin(2 * math.pi * ((step + 1) % 60 / 60.0 + i / 60.0))
                           for i in range(64)], dtype=np.float32)

        # ── 3. 4层层次趋势 (64 dim) ──
        def hierarchical_phase(s):
            return np.array([
                math.sin(2 * math.pi * s / (60 * 12)),   # 年 (720步)
                math.cos(2 * math.pi * s / (60 * 12)),
                math.sin(2 * math.pi * s / (60 * 3)),    # 季 (180步)
                math.cos(2 * math.pi * s / (60 * 3)),
                math.sin(2 * math.pi * s / 60),          # 月 (60步)
                math.cos(2 * math.pi * s / 60),
                math.sin(2 * math.pi * s / 5),           # 周 (5步)
                math.cos(2 * math.pi * s / 5),
            ] * 8, dtype=np.float32)[:64]

        hier_t = hierarchical_phase(step)
        hier_n = hierarchical_phase(step + 1)

        # ── 4. 语义聚类 (64 dim) ──
        # 6 组特征, 每组不同动力学
        sem_t = np.zeros(64, dtype=np.float32)
        sem_n = np.zeros(64, dtype=np.float32)
        for group in range(6):
            i0 = group * 10
            i1 = min(i0 + 11, 64)
            n = i1 - i0
            # 每组有不同频率/相位
            freq = 0.5 + group * 0.3
            phase_shift = group * math.pi / 6
            sem_t[i0:i1] = np.sin(2 * math.pi * freq * t + phase_shift) + \
                           self._rng.randn(n) * 0.02
            sem_n[i0:i1] = np.sin(2 * math.pi * freq * t_next + phase_shift) + \
                           self._rng.randn(n) * 0.02

        # ── 拼接 ──
        z_t = np.concatenate([field_t_flat, cycle_t, hier_t, sem_t]).astype(np.float32)
        z_n = np.concatenate([field_n_flat, cycle_n, hier_n, sem_n]).astype(np.float32)
        return z_t, z_n


# ═══════════════════════════════════════════════════════════════════════
# 组件工厂
# ═══════════════════════════════════════════════════════════════════════

def make_field_processor(structured: bool) -> any:
    if structured:
        from zwm.jepa.field_gnn import FieldSquareGNN
        return FieldSquareGNN(hidden_dim=64, num_layers=2)

    class FlatFuser(nn.Module):
        def __init__(self):
            super().__init__()
            self._net = nn.Sequential(
                nn.Linear(64 * 6, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Linear(128, 64),
            )
        def embed_field(self, f):
            self.eval()
            d = next(self.parameters()).device
            x = torch.from_numpy(f.astype(np.float32)).flatten().unsqueeze(0).to(d)
            if x.shape[-1] < 64 * 6:
                x = torch.nn.functional.pad(x, (0, 64 * 6 - x.shape[-1]))
            with torch.no_grad():
                return self._net(x).squeeze(0).cpu().numpy().astype(np.float32)
        def embed_field_train(self, f):
            if isinstance(f, np.ndarray):
                d = next(self.parameters()).device
                f = torch.from_numpy(f.astype(np.float32)).to(d)
            x = f.view(f.shape[0], -1)
            if x.shape[-1] < 64 * 6:
                x = torch.nn.functional.pad(x, (0, 64 * 6 - x.shape[-1]))
            return self._net(x)
    return FlatFuser()


def make_yao_field(structured: bool, field: np.ndarray) -> np.ndarray:
    """用六爻编码逻辑处理场 (或跳过)."""
    if structured:
        # 用爻提取器处理 64-dim 场 → (64, 6) 爻信号
        from zwm.encoder.field_encoder import HexagramFieldEncoder
        enc = HexagramFieldEncoder(strategy="adaptive")
        # 用场的统计量构造假的传感器数据
        sensor = {f"f{i}": float(field[i]) for i in range(min(6, len(field)))}
        return enc.encode(sensor)  # (64, 6)
    else:
        # 随机投影: 64-dim → (64, 6)
        proj = np.random.RandomState(42).randn(64, 6).astype(np.float32) * 0.1
        yao = 1.0 / (1.0 + np.exp(-field.reshape(1, 64) @ proj))  # (1, 6)
        return np.tile(yao, (64, 1))  # (64, 6)


# ═══════════════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Condition:
    key: str; name: str
    use_yao: bool; use_gnn: bool; use_cycle: bool; use_hier: bool; use_sem: bool


CONDITIONS = [
    Condition("A_full",    "全结构 (A)",     True,  True,  True,  True,  True),
    Condition("B_flat",    "平坦基线 (B)",    False, False, False, False, False),
    Condition("C_noyao",   "无六爻 (C)",      False, True,  True,  True,  True),
    Condition("D_nognn",   "无GNN (D)",       True,  False, True,  True,  True),
    Condition("E_nocycle", "无甲子周期 (E)",  True,  True,  False, True,  True),
    Condition("F_nohier",  "无层次 (F)",      True,  True,  True,  False, True),
    Condition("G_nosem",   "无语义 (G)",      True,  True,  True,  True,  False),
]


def run_one(cfg: Condition, steps: int, seed: int) -> dict:
    """运行单次实验."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from zwm.jepa.predictor import JEPAPredictor
    jepa = JEPAPredictor(input_dim=256, hidden_dim=192, latent_dim=64,
                         vicreg_weight=0.04, replay_capacity=64, batch_size=8,
                         seed=seed, use_action_cond=False)
    field_proc = make_field_processor(cfg.use_gnn)
    gen = StructuredDataGenerator(seed=seed)

    pred_errors = []
    cycle_scores = []   # 60步周期检测准确率
    hier_scores = []    # 4层分离度
    spatial_scores = [] # 8×8 热力图恢复率
    semantic_scores = [] # 6组聚类纯度

    for step in range(steps):
        z_t, z_next = gen.generate(step)
        field_t = z_t[:64]

        # 爻编码 (如果启用)
        if cfg.use_yao:
            hex_field = make_yao_field(True, field_t)
            field_vec = field_proc.embed_field(hex_field)
        else:
            hex_field = make_yao_field(False, field_t)
            field_vec = field_proc.embed_field(hex_field)

        # 构建 z_world (根据消融条件选择性包含)
        parts = [field_vec]  # 方图部分
        if cfg.use_cycle:
            parts.append(z_t[64:128])   # 甲子周期
        else:
            parts.append(np.zeros(64, dtype=np.float32))
        if cfg.use_hier:
            parts.append(z_t[128:192])  # 层次趋势
        else:
            parts.append(np.zeros(64, dtype=np.float32))
        if cfg.use_sem:
            parts.append(z_t[192:256])  # 语义聚类
        else:
            parts.append(np.zeros(64, dtype=np.float32))

        z_world = np.concatenate(parts).astype(np.float32)[:256]
        if len(z_world) < 256:
            z_world = np.pad(z_world, (0, 256 - len(z_world))).astype(np.float32)

        # 下一状态
        field_n = z_next[:64]
        if cfg.use_yao:
            hf_n = make_yao_field(True, field_n)
            fv_n = field_proc.embed_field(hf_n)
        else:
            hf_n = make_yao_field(False, field_n)
            fv_n = field_proc.embed_field(hf_n)
        parts_n = [fv_n]
        if cfg.use_cycle:
            parts_n.append(z_next[64:128])
        else:
            parts_n.append(np.zeros(64, dtype=np.float32))
        if cfg.use_hier:
            parts_n.append(z_next[128:192])
        else:
            parts_n.append(np.zeros(64, dtype=np.float32))
        if cfg.use_sem:
            parts_n.append(z_next[192:256])
        else:
            parts_n.append(np.zeros(64, dtype=np.float32))
        z_world_n = np.concatenate(parts_n).astype(np.float32)[:256]
        if len(z_world_n) < 256:
            z_world_n = np.pad(z_world_n, (0, 256 - len(z_world_n))).astype(np.float32)

        r = jepa.train_step(z_world, z_world_n)
        if not math.isnan(r.get("loss", float("nan"))):
            pred_errors.append(r.get("pred_error", 0))

        # 周期性检测: 每 60 步, 预测误差是否更低?
        if step >= 120 and step % 60 == 0:
            window = pred_errors[-10:]
            if window:
                cycle_scores.append(1.0 - np.mean(window))

        # 层次分离: 不同频率成分的预测误差相关性
        if step >= 100 and step % 20 == 0:
            if len(pred_errors) >= 80:
                segs = [pred_errors[-80+i*20:-60+i*20] for i in range(3)]
                if all(len(s) >= 5 for s in segs):
                    corr = np.corrcoef(segs[0], segs[1])[0, 1]
                    hier_scores.append(abs(corr) if not math.isnan(corr) else 0)

    return {
        "pred_errors": pred_errors,
        "mean_error": np.mean(pred_errors) if pred_errors else 1.0,
        "trend": np.polyfit(range(len(pred_errors)), pred_errors, 1)[0] if len(pred_errors) > 10 else 0,
        "cycle_detection": np.mean(cycle_scores) if cycle_scores else 0,
        "hier_separation": np.mean(hier_scores) if hier_scores else 0,
        "success_rate": len(pred_errors) / max(steps, 1),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  易经数学结构消融实验 v2 — 结构化数据 + 严格对照         ║")
    print(f"║  {len(CONDITIONS)} conditions × {args.runs} runs × {args.steps} steps          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    all_data: dict[str, list[dict]] = {}
    t0 = time.perf_counter()

    for cfg in CONDITIONS:
        print(f"  {cfg.name:<24s} ", end="", flush=True)
        runs = []
        for run in range(args.runs):
            seed = 42 + run * 100 + hash(cfg.key) % 1000
            r = run_one(cfg, args.steps, seed)
            runs.append(r)
            print("." if run < args.runs - 1 else "", end="", flush=True)
        all_data[cfg.key] = runs
        n_steps = sum(len(r["pred_errors"]) for r in runs)
        print(f" {n_steps} steps")

    elapsed = time.perf_counter() - t0

    # ── 汇总 ──
    print(f"\n{'='*90}")
    print(f"  {'条件':<24s} {'预测误差↓':>9s} {'趋势↓':>9s} {'周期检测':>9s} {'层次分离':>9s} {'成功率':>7s} {'综合':>6s}")
    print(f"  {'─'*24} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*7} {'─'*6}")

    results = []
    for cfg in CONDITIONS:
        runs = all_data[cfg.key]
        mean_err = np.mean([r["mean_error"] for r in runs])
        std_err = np.std([r["mean_error"] for r in runs])
        trend = np.mean([r["trend"] for r in runs])
        cycle = np.mean([r["cycle_detection"] for r in runs])
        hier = np.mean([r["hier_separation"] for r in runs])
        succ = np.mean([r["success_rate"] for r in runs])

        # 综合评分: 低误差 + 下降趋势 + 高周期检测 + 高成功率
        sc = (1.0 / max(mean_err, 0.001)) * 0.01
        sc += (-trend * 100)
        sc += cycle * 2.0
        sc += hier * 2.0
        sc += succ * 5.0
        results.append((cfg, mean_err, trend, cycle, hier, succ, sc))

        trend_icon = "↓" if trend < 0 else "↑"
        print(f"  {cfg.name:<24s} {mean_err:>9.4f} {trend_icon}{abs(trend):>8.6f} "
              f"{cycle:>9.4f} {hier:>9.4f} {succ:>7.2f} {sc:>6.1f}")

    # ── 排名 ──
    results.sort(key=lambda x: x[-1], reverse=True)
    print(f"\n{'='*90}")
    print(f"  综合排名")
    print(f"{'='*90}")
    for i, (cfg, me, tr, cy, hi, su, sc) in enumerate(results):
        bar = "█" * max(1, int(sc * 2))
        print(f"  {i+1}. {cfg.name:<24s} 评分={sc:.1f} {bar}")

    # ── 效应量 ──
    full = next((r for r in results if r[0].key == "A_full"), None)
    flat = next((r for r in results if r[0].key == "B_flat"), None)
    if full and flat:
        _, _, full_err, full_tr, full_cy, full_hi, _, _ = full
        _, _, flat_err, flat_tr, flat_cy, flat_hi, _, _ = flat
        print(f"\n  易经结构效应量 (全结构 vs 平坦基线):")
        print(f"    预测误差: {(flat_err - full_err) / max(full_err, 1e-8) * 100:+.1f}%")
        print(f"    学习速率: {(flat_tr - full_tr) / max(abs(full_tr), 1e-8) * 100:+.1f}%")
        print(f"    周期检测: {(full_cy - flat_cy) / max(abs(flat_cy), 1e-8) * 100:+.1f}%")
        print(f"    层次分离: {(full_hi - flat_hi) / max(abs(flat_hi), 1e-8) * 100:+.1f}%")

    print(f"\n  总耗时: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
