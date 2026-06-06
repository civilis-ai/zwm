#!/usr/bin/env python3
"""ZWM JEPA 真实数据训练 — Jena Climate 基线.

数据: Jena Climate (14 features, 10-min intervals, 2009-2016, 420K rows)
目标: 建立 JEPA 预测性能基线, 测试易经结构的真实效果

训练流程:
  1. 加载 + 预处理 (标准化, 滑动窗口)
  2. 数据 → 64卦场 (HexagramFieldEncoder)
  3. 场 → FieldSquareGNN → z_world
  4. JEPA 训练 (Structured encoder + EMA target + VICReg)
  5. 评估: 预测误差, VICReg方差, 有效维度

用法:
  python scripts/train_jena.py [--epochs 50] [--batch 32] [--full|--flat]
"""

from __future__ import annotations

import math, sys, time, os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════

def load_jena(data_dir: str = "data") -> tuple[np.ndarray, list[str]]:
    """加载 Jena Climate 数据集.

    Returns:
        data: (N, 14) — 标准化后的特征矩阵
        feature_names: 14 个特征名
    """
    csv_path = os.path.join(data_dir, "jena_climate_2009_2016.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Jena dataset not found at {csv_path}. "
            f"Download: scripts/train_jena.py"
        )

    print(f"Loading {csv_path}...")
    with open(csv_path) as f:
        header = f.readline().strip()
    feature_names = header.split(",")[1:]  # skip "Date Time"
    print(f"  Features ({len(feature_names)}): {feature_names}")

    # 加载数值数据
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=range(1, 15))
    print(f"  Shape: {raw.shape}")

    # 标准化 (z-score)
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1e-8
    normalized = (raw - mean) / std

    print(f"  Normalized: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
    return normalized.astype(np.float32), feature_names


# ═══════════════════════════════════════════════════════════════════════
# 数据 → ZWM 256-dim 世界向量
# ═══════════════════════════════════════════════════════════════════════

def row_to_zwm_world(row: np.ndarray, step: int, use_structure: bool = True) -> np.ndarray:
    """将 14-dim 传感器行 → 256-dim ZWM z_world.

    row: (14,) — 一帧传感器数据
    step: 全局步数 (用于时间编码)
    use_structure: 是否使用易经结构 (消融用)

    构建 256-dim z_world:
      [0:64]   = 方图空间场 (14传感器 → 64卦场 → GNN → 64)
      [64:128] = 圆图时间 (64步周期 + 相位)
      [128:192]= 干支历法 (60甲子周期)
      [192:256]= 元会运世 (4层嵌套周期)
    """
    n_features = len(row)

    if use_structure:
        # ── 1. 方图场: 14传感器 → 8×8 grid ──
        # 将 14 个传感器映射到 64 个方图位置
        # 策略: 每 2-3 个传感器组合为一个位置的特征
        square = np.zeros((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                pos = i * 8 + j
                # 循环取传感器值 (14个传感器不够填满64个, 用组合)
                f1 = row[pos % n_features]
                f2 = row[(pos + 7) % n_features]
                # 组合: 均值 + 交互
                square[i, j] = 0.6 * f1 + 0.4 * f2

        # 用 HexagramFieldEncoder 处理 (统计量 → 6爻)
        from zwm.encoder.field_encoder import HexagramFieldEncoder
        fe = HexagramFieldEncoder(strategy="spatial")
        # 将 8×8 当作"图像"编码
        sensor_dict = {f"f{i}": float(row[i % n_features]) for i in range(6)}
        hex_field = fe.encode(sensor_dict)  # (64, 6)

        # FieldGNN
        from zwm.jepa.field_gnn import FieldSquareGNN
        # 使用全局 GNN (懒加载)
        pass  # z_sq will be computed in training loop

        z_sq = _gnn_embed(hex_field)  # (64,)
    else:
        # 平坦基线: 14-dim → repeat → 64-dim
        z_sq = np.tile(row[:8], 8)[:64].astype(np.float32)
        z_sq = z_sq / (np.linalg.norm(z_sq) + 1e-8)

    # ── 2. 圆图时间 ──
    t_phase = (step % 64) / 64.0
    z_circ = np.array([
        math.sin(2 * math.pi * t_phase * (h + 1))
        for h in range(64)
    ], dtype=np.float32)

    # ── 3. 甲子周期 ──
    gz_step = step % 60
    z_ganzhi = np.array([
        math.sin(2 * math.pi * gz_step * (i + 1) / 60.0)
        for i in range(64)
    ], dtype=np.float32)

    # ── 4. 元会运世 ──
    z_cosmic = np.array([
        math.sin(2 * math.pi * step / 720),   # 年周期 (5天)
        math.cos(2 * math.pi * step / 720),
        math.sin(2 * math.pi * step / 180),   # 季周期
        math.cos(2 * math.pi * step / 180),
        math.sin(2 * math.pi * step / 60),    # 日周期 (10小时)
        math.cos(2 * math.pi * step / 60),
        math.sin(2 * math.pi * step / 6),     # 时周期 (1小时)
        math.cos(2 * math.pi * step / 6),
    ] * 8, dtype=np.float32)[:64]

    return np.concatenate([z_sq, z_circ, z_ganzhi, z_cosmic]).astype(np.float32)


# 全局 GNN (懒加载)
_GNN = None

def _gnn_embed(hex_field: np.ndarray) -> np.ndarray:
    global _GNN
    if _GNN is None:
        from zwm.jepa.field_gnn import FieldSquareGNN
        _GNN = FieldSquareGNN(hidden_dim=64, num_layers=2)
    return _GNN.embed_field(hex_field)


# ═══════════════════════════════════════════════════════════════════════
# 训练循环
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainMetrics:
    losses: list[float] = field(default_factory=list)
    pred_errors: list[float] = field(default_factory=list)
    vicreg_vars: list[float] = field(default_factory=list)
    surprises: list[float] = field(default_factory=list)
    effective_dims: list[int] = field(default_factory=list)


def train(
    data: np.ndarray,
    epochs: int = 30,
    steps_per_epoch: int = 500,
    use_structure: bool = True,
    seed: int = 42,
) -> TrainMetrics:
    """在 Jena 数据上训练 JEPA.

    Args:
        data: (N, 14) 标准化传感器数据
        epochs: 训练轮数
        steps_per_epoch: 每轮训练步数
        use_structure: True=易经全结构, False=平坦基线
        seed: 随机种子
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    from zwm.jepa.predictor import JEPAPredictor
    jepa = JEPAPredictor(
        input_dim=256, hidden_dim=192, latent_dim=64,
        vicreg_weight=0.04, replay_capacity=128, batch_size=16,
        seed=seed, use_action_cond=False,
    )

    N = len(data)
    metrics = TrainMetrics()
    latent_history: list[np.ndarray] = []

    t0 = time.perf_counter()
    total_steps = epochs * steps_per_epoch

    for epoch in range(epochs):
        epoch_losses = []
        epoch_errors = []

        for s in range(steps_per_epoch):
            step = epoch * steps_per_epoch + s
            idx_t = step % (N - 1)
            idx_n = (step + 1) % (N - 1)

            # 数据 → z_world
            z_world_t = row_to_zwm_world(data[idx_t], step, use_structure)
            z_world_n = row_to_zwm_world(data[idx_n], step + 1, use_structure)

            # JEPA 训练步
            result = jepa.train_step(z_world_t, z_world_n)
            loss = result.get("loss", float("nan"))
            pred_err = result.get("pred_error", float("nan"))

            if not math.isnan(pred_err):
                epoch_errors.append(pred_err)
                epoch_losses.append(loss)

                # VICReg
                with torch.no_grad():
                    x = torch.from_numpy(z_world_t.astype(np.float32)).unsqueeze(0)
                    zl = jepa.context_encoder(x.to(jepa.device))
                    metrics.vicreg_vars.append(float(zl.std()))

                # Surprise: z_pred vs z_target
                z_pred = jepa.predict(z_world_t)
                if isinstance(z_pred, dict):
                    z_pred = z_pred["short"]
                z_target = jepa.target_latent(z_world_n)
                surprise = float(np.linalg.norm(
                    np.asarray(z_pred).flatten() - np.asarray(z_target).flatten()
                ))
                metrics.surprises.append(surprise)

                latent_history.append(zl.squeeze(0).cpu().numpy())

        # Epoch summary
        if epoch_errors:
            avg_err = np.mean(epoch_errors)
            metrics.pred_errors.append(avg_err)
            metrics.losses.append(np.mean(epoch_losses))

            # Effective dim (every 5 epochs)
            if epoch % 5 == 0 and len(latent_history) >= 20:
                latents = np.stack(latent_history[-100:])
                eff_dim = _compute_eff_dim(latents)
                metrics.effective_dims.append(eff_dim)

            elapsed = time.perf_counter() - t0
            steps_done = (epoch + 1) * steps_per_epoch
            sps = steps_done / max(elapsed, 0.1)
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"error={avg_err:.4f} | "
                  f"VICReg σ={metrics.vicreg_vars[-1] if metrics.vicreg_vars else 0:.4f} | "
                  f"surprise={metrics.surprises[-1] if metrics.surprises else 0:.4f} | "
                  f"{sps:.0f} steps/s")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total: {elapsed:.0f}s, {total_steps/elapsed:.0f} steps/s")
    return metrics


def _compute_eff_dim(latents: np.ndarray, threshold: float = 0.01) -> int:
    """PCA 有效维度."""
    centered = latents - latents.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / (latents.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    total = eigvals.sum()
    cumsum = np.cumsum(eigvals) / (total + 1e-8)
    return int(np.searchsorted(cumsum, 1.0 - threshold)) + 1


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps", type=int, default=500, help="steps per epoch")
    p.add_argument("--full", action="store_true", default=True, help="Full I Ching (default)")
    p.add_argument("--flat", action="store_true", help="Flat baseline")
    p.add_argument("--data-dir", type=str, default="data")
    args = p.parse_args()

    use_structure = not args.flat
    mode = "Full I Ching" if use_structure else "Flat baseline"

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  ZWM JEPA 真实数据训练 — Jena Climate                   ║")
    print(f"║  Mode: {mode:<47s} ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # 加载数据
    try:
        data, feat_names = load_jena(args.data_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n请手动下载数据:")
        print("  https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip")
        print("  解压到 data/ 目录")
        return 1

    # 训练
    metrics = train(
        data, epochs=args.epochs, steps_per_epoch=args.steps,
        use_structure=use_structure,
    )

    # 结果
    print(f"\n{'='*60}")
    print(f"  训练结果 — {mode}")
    print(f"{'='*60}")
    if metrics.pred_errors:
        n = len(metrics.pred_errors)
        print(f"  Epochs: {n}")
        print(f"  Final pred error:  {metrics.pred_errors[-1]:.4f}")
        print(f"  Error trend:       {metrics.pred_errors[0]:.4f} → {metrics.pred_errors[-1]:.4f}")
        print(f"  VICReg σ (final):  {metrics.vicreg_vars[-1]:.4f}" if metrics.vicreg_vars else "")
        print(f"  Surprise μ:        {np.mean(metrics.surprises):.4f}" if metrics.surprises else "")
        if metrics.effective_dims:
            print(f"  Effective dim:     {metrics.effective_dims[-1]}")
    else:
        print("  ❌ No valid training steps")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
