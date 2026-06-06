#!/usr/bin/env python3
"""易经数学结构消融实验 — 因果性验证.

对照条件 (controlled ablation):
  A. Full I Ching:       六爻 + 64卦场(GNN) + 圆图序 + 甲子周期 + 元会运世
  B. Random Yao:         六爻编码 → 随机6-dim投影
  C. Flat MLP:           64卦场(GNN) → 384-dim flat MLP
  D. Shuffle Circle:     圆图先天序 → 随机排列
  E. Random Ganzhi:      甲子周期 → 随机60映射
  F. Single Phase:       元会运世4层 → 单一年相位
  G. All Random:         全部随机替代

度量:
  1. JEPA prediction error (↓ = 更好)
  2. Surprise convergence rate (斜率 ↓ = 更快收敛)
  3. VICReg variance σ (↑ = 更少坍缩)
  4. Latent effective dim (↑ = 更丰富表示)
  5. Gradient norm stability (σ ↓ = 更稳定训练)

预期: 如果易经结构是装饰性的, Flat MLP (C) 应与 Full (A) 持平或更好。
      如果易经结构是因果性的, Full (A) 应在所有指标上优于 ablated 条件。

用法: python scripts/ablate_iching.py [--steps N] [--runs R] [--quick]
"""

from __future__ import annotations

import math, sys, time, json, random as py_random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════
# 实验配置
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AblationConfig:
    name: str
    use_yao_structure: bool = True      # 六爻语义编码
    use_field_gnn: bool = True          # 64卦场 + GNN
    use_circle_order: bool = True       # 圆图先天序
    use_ganzhi_cycle: bool = True       # 60甲子周期
    use_cosmic_layers: bool = True      # 元会运世4层
    seed: int = 42


# 所有实验条件
CONDITIONS = {
    "A_full": AblationConfig("易经全结构 (A)", True, True, True, True, True),
    "B_random_yao": AblationConfig("随机爻 (B)", False, True, True, True, True),
    "C_flat_mlp": AblationConfig("平坦MLP (C)", True, False, True, True, True),
    "D_shuffle_circle": AblationConfig("乱序圆图 (D)", True, True, False, True, True),
    "E_random_ganzhi": AblationConfig("随机干支 (E)", True, True, True, False, True),
    "F_single_phase": AblationConfig("单相位 (F)", True, True, True, True, False),
    "G_all_random": AblationConfig("全随机 (G)", False, False, False, False, False),
}


# ═══════════════════════════════════════════════════════════════════════
# 可替换的模块
# ═══════════════════════════════════════════════════════════════════════

def make_yao_encoder(structured: bool, seed: int) -> nn.Module:
    """六爻编码器 或 随机投影."""
    if structured:
        # 使用真正的 6-爻提取器 (mean/std/max/min/gradient/entropy)
        from zwm.encoder.field_encoder import HexagramFieldEncoder
        return HexagramFieldEncoder(strategy="adaptive")
    else:
        # 随机投影替代 (同维度, 随机权重)
        class RandomYaoEncoder:
            def __init__(self, seed):
                self._rng = np.random.RandomState(seed)
                self._proj = self._rng.randn(6, 6).astype(np.float32) * 0.1
            def encode(self, data):
                if isinstance(data, np.ndarray):
                    flat = data.flatten()[:6]
                else:
                    flat = np.array(list(data.values())[:6], dtype=np.float32) if isinstance(data, dict) else np.zeros(6, dtype=np.float32)
                if len(flat) < 6:
                    flat = np.pad(flat, (0, 6 - len(flat)))
                yao = 1.0 / (1.0 + np.exp(-flat @ self._proj.T))  # sigmoid
                return np.tile(yao.reshape(1, 6), (64, 1)).astype(np.float32)
        return RandomYaoEncoder(seed)


def make_field_processor(use_gnn: bool) -> nn.Module:
    """64卦场处理器: GNN 或 flat MLP."""
    if use_gnn:
        from zwm.jepa.field_gnn import FieldSquareGNN
        return FieldSquareGNN(hidden_dim=64, num_layers=2)
    else:
        # Flat MLP: 384-dim → 64-dim (同等表达能力)
        class FlatMLPField(nn.Module):
            def __init__(self):
                super().__init__()
                self._net = nn.Sequential(
                    nn.Linear(384, 128), nn.LayerNorm(128), nn.GELU(),
                    nn.Linear(128, 64),
                )
            def embed_field(self, field: np.ndarray) -> np.ndarray:
                self.eval()
                device = next(self.parameters()).device
                x = torch.from_numpy(field.astype(np.float32).flatten()).unsqueeze(0).to(device)
                with torch.no_grad():
                    return self._net(x).squeeze(0).cpu().numpy().astype(np.float32)
            def embed_field_train(self, field):
                if isinstance(field, np.ndarray):
                    device = next(self.parameters()).device
                    field = torch.from_numpy(field.astype(np.float32)).to(device)
                x = field.view(field.shape[0], -1) if field.dim() > 2 else field
                return self._net(x)
        return FlatMLPField()


def make_circle_order(shuffle: bool, seed: int) -> np.ndarray:
    """圆图64卦序: 先天序 或 随机排列."""
    from zwm.scene_field.time_field import _CIRCLE_ORDER
    if not shuffle:
        return np.array(_CIRCLE_ORDER, dtype=np.int32)
    rng = np.random.RandomState(seed)
    order = list(_CIRCLE_ORDER)
    rng.shuffle(order)
    return np.array(order, dtype=np.int32)


def make_ganzhi_mapping(random_map: bool, seed: int) -> dict[int, int]:
    """甲子→卦 映射: 周期映射 或 随机映射."""
    from zwm.scene_field.time_field import _GANZHI_TO_HEX
    if not random_map:
        return dict(_GANZHI_TO_HEX)
    rng = np.random.RandomState(seed)
    keys = list(range(60))
    vals = rng.permutation(64)[:60]
    return {k: int(v) for k, v in zip(keys, vals)}


def make_cosmic_phases(layered: bool, year: int) -> dict[str, float]:
    """元会运世: 4层 或 单层."""
    from zwm.core.constants import YUAN_HUI_YUN_SHI
    if layered:
        return {
            "元": 2 * math.pi * (year % YUAN_HUI_YUN_SHI["元"]) / YUAN_HUI_YUN_SHI["元"],
            "会": 2 * math.pi * (year % YUAN_HUI_YUN_SHI["会"]) / YUAN_HUI_YUN_SHI["会"],
            "运": 2 * math.pi * (year % YUAN_HUI_YUN_SHI["运"]) / YUAN_HUI_YUN_SHI["运"],
            "世": 2 * math.pi * (year % YUAN_HUI_YUN_SHI["世"]) / YUAN_HUI_YUN_SHI["世"],
        }
    else:
        p = 2 * math.pi * (year % 60) / 60.0
        return {"元": p, "会": p, "运": p, "世": p}  # 同相位=无层次


# ═══════════════════════════════════════════════════════════════════════
# 单次实验运行
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RunMetrics:
    """单次运行的度量."""
    jepa_losses: list[float] = field(default_factory=list)
    surprises: list[float] = field(default_factory=list)
    vicreg_vars: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    effective_dims: list[int] = field(default_factory=list)


def compute_effective_dim(latent: np.ndarray, threshold: float = 0.01) -> int:
    """PCA 有效维度: 占总方差 99% 的主成分数."""
    if latent.ndim == 1:
        latent = latent.reshape(1, -1)
    if latent.shape[0] < 2:
        return latent.shape[1]
    centered = latent - latent.mean(axis=0, keepdims=True)
    try:
        cov = centered.T @ centered / (latent.shape[0] - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        total = eigvals.sum()
        cumsum = np.cumsum(eigvals) / (total + 1e-8)
        return int(np.searchsorted(cumsum, 1.0 - threshold)) + 1
    except Exception:
        return latent.shape[1]


def run_experiment(
    cfg: AblationConfig,
    steps: int = 100,
    jepa_input_dim: int = 256,
    jepa_hidden: int = 192,
    jepa_latent: int = 64,
) -> RunMetrics:
    """运行一次消融实验."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── 构建可替换组件 ──
    yao_encoder = make_yao_encoder(cfg.use_yao_structure, cfg.seed)
    field_proc = make_field_processor(cfg.use_field_gnn)
    circle_order = make_circle_order(not cfg.use_circle_order, cfg.seed)
    ganzhi_map = make_ganzhi_mapping(not cfg.use_ganzhi_cycle, cfg.seed)

    # ── JEPA ──
    from zwm.jepa.predictor import JEPAPredictor
    jepa = JEPAPredictor(
        input_dim=jepa_input_dim, hidden_dim=jepa_hidden, latent_dim=jepa_latent,
        vicreg_weight=0.04, replay_capacity=64, batch_size=8,
        use_action_cond=False, seed=cfg.seed,
    )

    # ── 数据生成 ──
    rng = np.random.RandomState(cfg.seed)
    metrics = RunMetrics()
    latent_history: list[np.ndarray] = []

    for step in range(steps):
        # 生成结构化传感器数据 (模拟真实环境)
        t = step / 20.0
        sensor_data = {
            "temperature": 0.5 + 0.4 * math.sin(t),
            "terrain": 0.5 + 0.3 * math.cos(t * 1.3),
            "social_proximity": abs(math.sin(t * 0.7)),
            "resource_level": 0.5 + 0.2 * math.sin(t * 0.5 + 1.0),
            "momentum": 0.5 * math.cos(t * 1.1),
            "overall_favorability": 0.5 + 0.3 * math.sin(t * 0.9),
        }

        # ── 编码 → 卦象场 (或随机等效) ──
        if cfg.use_yao_structure:
            field = yao_encoder.encode(sensor_data)  # (64, 6)
        else:
            field = yao_encoder.encode(sensor_data)  # 随机投影也返回 (64,6)

        # ── 场处理 → z_sq ──
        z_sq = field_proc.embed_field(field)  # (64,)

        # ── 时间场 ──
        year = 2026 + step // 365
        cosmic = make_cosmic_phases(cfg.use_cosmic_layers, year)

        # 圆图: 用 order 生成 phase vector
        cp = np.array([
            math.cos(2 * math.pi * step / 64),
            math.sin(2 * math.pi * step / 64),
        ] + [math.cos(2 * math.pi * step * i / 64) for i in range(2, 7)]
          + [math.sin(2 * math.pi * step * i / 64) for i in range(2, 7)],
          dtype=np.float32)[:13]  # 截断到13

        # 干支: 用 ganzhi_map 生成 60-dim 特征
        gz_idx = step % 60
        gz_hex = ganzhi_map.get(gz_idx, 0)
        gz_vec = np.array([float((gz_hex >> i) & 1) for i in range(6)], dtype=np.float32)
        gz_vec = np.tile(gz_vec, 11)[:64].astype(np.float32)

        # 元会运世: 用 cosmic 生成
        cosmic_vec = np.array([
            math.cos(cosmic["元"]), math.sin(cosmic["元"]),
            math.cos(cosmic["会"]), math.sin(cosmic["会"]),
            math.cos(cosmic["运"]), math.sin(cosmic["运"]),
            math.cos(cosmic["世"]), math.sin(cosmic["世"]),
        ], dtype=np.float32)
        cosmic_vec = np.tile(cosmic_vec, 9)[:64].astype(np.float32)

        # ── 融合 → z_world ──
        z_world = np.concatenate([z_sq, cp, gz_vec, cosmic_vec]).astype(np.float32)
        if len(z_world) < jepa_input_dim:
            z_world = np.pad(z_world, (0, jepa_input_dim - len(z_world))).astype(np.float32)
        else:
            z_world = z_world[:jepa_input_dim]

        # ── 下一状态 (轻微扰动) ──
        field_next = field + rng.randn(64, 6).astype(np.float32) * 0.01
        field_next = np.clip(field_next, 0, 1)
        z_sq_next = field_proc.embed_field(field_next)
        z_world_next = np.concatenate([z_sq_next, cp * 1.01, gz_vec * 1.01, cosmic_vec * 1.01])
        z_world_next = z_world_next[:jepa_input_dim].astype(np.float32)
        if len(z_world_next) < jepa_input_dim:
            z_world_next = np.pad(z_world_next, (0, jepa_input_dim - len(z_world_next))).astype(np.float32)

        # ── JEPA 训练步 ──
        result = jepa.train_step(z_world, z_world_next)
        loss = result.get("loss", float("nan"))
        pred_err = result.get("pred_error", float("nan"))

        if not math.isnan(loss) and not math.isnan(pred_err):
            metrics.jepa_losses.append(pred_err)  # 用预测误差 (≥0)
            # Surprise: target vs prediction L2
            z_pred = jepa.predict(z_world)
            if isinstance(z_pred, dict):
                z_pred = z_pred["short"]
            z_target = jepa.target_latent(z_world_next)
            surprise = float(np.linalg.norm(np.asarray(z_pred).flatten() - np.asarray(z_target).flatten()))
            metrics.surprises.append(surprise)

            # VICReg variance
            with torch.no_grad():
                x_t = torch.from_numpy(z_world.astype(np.float32)).unsqueeze(0)
                z_lat = jepa.context_encoder(x_t.to(jepa.device))
                metrics.vicreg_vars.append(float(z_lat.std()))

            # Gradient norm
            total_norm = 0.0
            for p in jepa.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            metrics.grad_norms.append(math.sqrt(total_norm))

            # Effective dim
            latent_history.append(z_lat.squeeze(0).cpu().numpy())

    # 计算有效维度 (用最后 20 步)
    if len(latent_history) >= 20:
        metrics.effective_dims = [compute_effective_dim(np.stack(latent_history[-20:]))]
    elif latent_history:
        metrics.effective_dims = [compute_effective_dim(np.stack(latent_history))]

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# 汇总与报告
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AblationResult:
    condition: str
    name: str
    jepa_mean: float
    jepa_std: float
    jepa_trend: float          # 斜率 (负=下降)
    surprise_mean: float
    surprise_trend: float       # 收敛速率
    vicreg_mean: float
    grad_norm_std: float
    effective_dim: float
    success_rate: float         # 有效训练步比例

    def score(self) -> float:
        """综合评分 (越高越好).

        = + jepa_trend_sign (负=加分)
          + surprise_trend_sign (负=加分)
          + vicreg_mean (高=加分)
          - grad_norm_std (低=加分)
          + effective_dim (高=加分)
          + success_rate
        """
        sc = 0.0
        sc += 1.0 if self.jepa_trend < 0 else -1.0
        sc += 1.0 if self.surprise_trend < 0 else -1.0
        sc += min(self.vicreg_mean * 5.0, 2.0)
        sc += max(0.0, 2.0 - self.grad_norm_std * 0.1)
        sc += min(self.effective_dim / 20.0, 2.0)
        sc += self.success_rate * 2.0
        return sc


def analyze_run(name: str, cfg: AblationConfig, metrics: RunMetrics) -> AblationResult:
    """分析单次运行的度量."""
    n = len(metrics.jepa_losses)
    if n < 5:
        return AblationResult(name, cfg.name, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    losses = np.array(metrics.jepa_losses)
    surprises = np.array(metrics.surprises)
    vicregs = np.array(metrics.vicreg_vars)
    grad_norms = np.array(metrics.grad_norms)

    # Trend: 线性回归斜率
    x = np.arange(len(losses))
    jepa_slope = np.polyfit(x, losses, 1)[0]
    surp_slope = np.polyfit(x, surprises, 1)[0]

    return AblationResult(
        condition=name,
        name=cfg.name,
        jepa_mean=float(np.mean(losses)),
        jepa_std=float(np.std(losses)),
        jepa_trend=float(jepa_slope),
        surprise_mean=float(np.mean(surprises)),
        surprise_trend=float(surp_slope),
        vicreg_mean=float(np.mean(vicregs)),
        grad_norm_std=float(np.std(grad_norms)),
        effective_dim=float(metrics.effective_dims[0]) if metrics.effective_dims else 0,
        success_rate=n / max(len(metrics.jepa_losses), 1),
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80, help="每条件训练步数")
    parser.add_argument("--runs", type=int, default=3, help="每条件重复次数")
    parser.add_argument("--quick", action="store_true", help="仅全结构 vs 全随机")
    args = parser.parse_args()

    conditions = CONDITIONS
    if args.quick:
        conditions = {k: v for k, v in CONDITIONS.items() if k in ("A_full", "G_all_random")}

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  易经数学结构消融实验                                     ║")
    print(f"║  {len(conditions)} conditions × {args.runs} runs × {args.steps} steps          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    all_results: dict[str, list[RunMetrics]] = {}
    t0 = time.perf_counter()

    for cond_key, cfg in conditions.items():
        print(f"  {cfg.name:<24s} ", end="", flush=True)
        cond_results = []
        for run in range(args.runs):
            seed = 42 + run * 100
            cfg_copy = AblationConfig(
                name=cfg.name,
                use_yao_structure=cfg.use_yao_structure,
                use_field_gnn=cfg.use_field_gnn,
                use_circle_order=cfg.use_circle_order,
                use_ganzhi_cycle=cfg.use_ganzhi_cycle,
                use_cosmic_layers=cfg.use_cosmic_layers,
                seed=seed,
            )
            metrics = run_experiment(cfg_copy, steps=args.steps)
            cond_results.append(metrics)
            print("." if len(cond_results) < args.runs else "", end="", flush=True)
        all_results[cond_key] = cond_results
        print(f" {sum(len(r.jepa_losses) for r in cond_results)} steps")

    elapsed = time.perf_counter() - t0

    # ── 分析 ──
    print(f"\n{'='*80}")
    print(f"  结果汇总 (均值±标准差 over {args.runs} runs)")
    print(f"{'='*80}")
    print(f"  {'条件':<24s} {'JEPA↓':>8s} {'趋势':>6s} {'Surp↓':>8s} {'收敛':>6s} {'VICReg':>7s} {'有效D':>6s} {'评分':>6s}")
    print(f"  {'─'*24} {'─'*8} {'─'*6} {'─'*8} {'─'*6} {'─'*7} {'─'*6} {'─'*6}")

    analyses = []
    for cond_key, metrics_list in all_results.items():
        cfg = conditions[cond_key]
        results = [analyze_run(cond_key, cfg, m) for m in metrics_list]

        # Average across runs
        avg = AblationResult(
            condition=cond_key, name=cfg.name,
            jepa_mean=np.mean([r.jepa_mean for r in results]),
            jepa_std=np.mean([r.jepa_std for r in results]),
            jepa_trend=np.mean([r.jepa_trend for r in results]),
            surprise_mean=np.mean([r.surprise_mean for r in results]),
            surprise_trend=np.mean([r.surprise_trend for r in results]),
            vicreg_mean=np.mean([r.vicreg_mean for r in results]),
            grad_norm_std=np.mean([r.grad_norm_std for r in results]),
            effective_dim=np.mean([r.effective_dim for r in results]),
            success_rate=np.mean([r.success_rate for r in results]),
        )
        analyses.append(avg)
        sc = avg.score()

        trend_icon = "↓" if avg.jepa_trend < 0 else "↑"
        surp_icon = "↓" if avg.surprise_trend < 0 else "↑"
        print(f"  {cfg.name:<24s} {avg.jepa_mean:>8.4f} {trend_icon}{abs(avg.jepa_trend):>5.4f} "
              f"{avg.surprise_mean:>8.4f} {surp_icon}{abs(avg.surprise_trend):>5.4f} "
              f"{avg.vicreg_mean:>7.4f} {avg.effective_dim:>6.1f} {sc:>6.1f}")

    # ── 排序 ──
    analyses.sort(key=lambda a: a.score(), reverse=True)
    print(f"\n{'='*80}")
    print(f"  综合排名")
    print(f"{'='*80}")
    for i, a in enumerate(analyses):
        bar = "█" * max(1, int(a.score() - analyses[-1].score() + 1))
        print(f"  {i+1}. {a.name:<24s} 评分={a.score():.1f} {bar}")

    # ── 效应量 ──
    if len(analyses) >= 2:
        full = next((a for a in analyses if a.condition == "A_full"), None)
        random = next((a for a in analyses if a.condition == "G_all_random"), None)
        if full and random:
            effect_jepa = (random.jepa_mean - full.jepa_mean) / (abs(full.jepa_mean) + 1e-8) * 100
            effect_surp = (random.surprise_mean - full.surprise_mean) / (abs(full.surprise_mean) + 1e-8) * 100
            effect_vicreg = (full.vicreg_mean - random.vicreg_mean) / (abs(random.vicreg_mean) + 1e-8) * 100
            print(f"\n  易经结构效应量 (vs 全随机):")
            print(f"    JEPA误差降低:  {effect_jepa:+.1f}%")
            print(f"    Surprise降低:  {effect_surp:+.1f}%")
            print(f"    VICReg提升:    {effect_vicreg:+.1f}%")
            if effect_jepa > 0 and effect_vicreg > 0:
                print(f"    ✅ 易经结构显著改善世界模型性能")
            elif effect_jepa > 0:
                print(f"    🟡 易经结构部分改善性能")
            else:
                print(f"    ❌ 未观察到显著优势")

    print(f"\n  总耗时: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
