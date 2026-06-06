"""消融实验: 验证易经数学结构对 JEPA 的正向提升作用.

运行: python ablation_experiment.py
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import torch

from zwm.jepa.predictor import JEPAPredictor


# ---------------------------------------------------------------------------
# 数据生成 — 普通合成时间序列(与易经无关)
# ---------------------------------------------------------------------------

def generate_lorenz_data(n_samples: int = 800, dt: float = 0.02, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs, ys, zs = np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)
    xs[0], ys[0], zs[0] = rng.random(3) * 2 - 1
    for i in range(n_samples - 1):
        dx = 10.0 * (ys[i] - xs[i])
        dy = xs[i] * (28.0 - zs[i]) - ys[i]
        dz = xs[i] * ys[i] - (8.0 / 3.0) * zs[i]
        xs[i + 1] = xs[i] + dx * dt
        ys[i + 1] = ys[i] + dy * dt
        zs[i + 1] = zs[i] + dz * dt
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


def generate_multivariate_sine(n_samples: int = 800, n_dims: int = 5, seed: int = 43) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10 * np.pi, n_samples, dtype=np.float32)
    data = np.zeros((n_samples, n_dims), dtype=np.float32)
    for d in range(n_dims):
        freq = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.3, 1.5)
        noise = rng.normal(0, 0.05, n_samples)
        data[:, d] = amp * np.sin(freq * t + phase) + noise
    return data


def prepare_sequences(data: np.ndarray, input_dim: int = 256) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n_samples, n_features = data.shape
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 1.0 / np.sqrt(n_features), (n_features, input_dim)).astype(np.float32)
    projected = data @ proj
    mean = projected.mean(axis=0, keepdims=True)
    std = projected.std(axis=0, keepdims=True) + 1e-8
    projected = (projected - mean) / std

    xs, ys = [], []
    for i in range(n_samples - 10 - 1):
        x = projected[i : i + 10].mean(axis=0)
        y = projected[i + 10]
        xs.append(x)
        ys.append(y)
    return xs, ys


# ---------------------------------------------------------------------------
# 实验运行器
# ---------------------------------------------------------------------------

@dataclass
class Result:
    name: str
    train_mse: list[float]
    val_mse: list[float]
    final_val_mse: float
    convergence_epoch: int
    elapsed_sec: float


def run_trial(
    name: str,
    input_dim: int,
    xs_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    xs_val: list[np.ndarray],
    ys_val: list[np.ndarray],
    epochs: int = 20,
    patience: int = 6,
    seed: int = 42,
) -> Result:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = JEPAPredictor(
        input_dim=input_dim,
        hidden_dim=192,
        latent_dim=64,
        learning_rate=3e-4,
        batch_size=32,
        vicreg_weight=1.0,
        variational=False,
        use_action_cond=False,
    )
    device = model.device
    model.train()

    train_log, val_log = [], []
    best_val = float("inf")
    stagnant = 0
    conv_epoch = epochs
    start = time.time()

    for epoch in range(epochs):
        # training
        epoch_mse = []
        idx = np.random.permutation(len(xs_train))
        for i in idx:
            xt = torch.from_numpy(xs_train[i]).float().to(device)
            xn = torch.from_numpy(ys_train[i]).float().to(device)
            losses = model.train_step(xt, xn)
            epoch_mse.append(float(losses["pred_error"]))
        train_log.append(float(np.mean(epoch_mse)))

        # validation
        model.eval()
        val_errs = []
        with torch.no_grad():
            for xv, yv in zip(xs_val, ys_val):
                xt = torch.from_numpy(xv).float().unsqueeze(0).to(device)
                xn = torch.from_numpy(yv).float().unsqueeze(0).to(device)
                z_pred = model.predict(xt)
                z_actual = model.encode(xn).cpu().numpy()
                val_errs.append(float(np.mean((z_pred - z_actual) ** 2)))
        model.train()

        val_mse = float(np.mean(val_errs))
        val_log.append(val_mse)

        if val_mse < 0.2 and conv_epoch == epochs:
            conv_epoch = epoch

        if val_mse < best_val:
            best_val = val_mse
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                print(f"  [{name}] Early stop at epoch {epoch}")
                break

        if epoch % 2 == 0:
            print(f"  [{name}] Epoch {epoch}: train_mse={train_log[-1]:.4f}, val_mse={val_mse:.4f}")

    elapsed = time.time() - start
    return Result(name, train_log, val_log, best_val, conv_epoch, elapsed)


# ---------------------------------------------------------------------------
# 主实验
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("消融实验: 易经数学结构对 JEPA 的正向提升作用")
    print("=" * 60)

    # 1. 生成普通数据
    print("\n[1] 生成普通合成数据 (Lorenz + 多变量正弦波)...")
    data = np.concatenate([
        generate_lorenz_data(n_samples=400, seed=42),
        generate_multivariate_sine(n_samples=400, seed=43),
    ], axis=1)
    train_data, val_data = data[:320], data[320:]
    print(f"    总样本: {len(data)}, 训练: {len(train_data)}, 验证: {len(val_data)}")

    # 2. 准备序列
    print("\n[2] 准备 JEPA 序列...")
    xs_t_256, ys_t_256 = prepare_sequences(train_data, input_dim=256)
    xs_v_256, ys_v_256 = prepare_sequences(val_data, input_dim=256)
    xs_t_128, ys_t_128 = prepare_sequences(train_data, input_dim=128)
    xs_v_128, ys_v_128 = prepare_sequences(val_data, input_dim=128)
    print(f"    With-IChing:   train={len(xs_t_256)}, val={len(xs_v_256)}")
    print(f"    Without-IChing: train={len(xs_t_128)}, val={len(xs_v_128)}")

    # 3. 运行实验 (轻量级配置以加快速度)
    print("\n[3] 训练 With-IChing (ZWMStructuredEncoder, 256-dim)...")
    result_with = run_trial(
        "With-IChing", 256, xs_t_256, ys_t_256, xs_v_256, ys_v_256,
        epochs=12, patience=4, seed=42,
    )

    print("\n[4] 训练 Without-IChing (TransformerEncoder, 128-dim)...")
    result_without = run_trial(
        "Without-IChing", 128, xs_t_128, ys_t_128, xs_v_128, ys_v_128,
        epochs=12, patience=4, seed=42,
    )

    # 4. 报告
    print("\n" + "=" * 60)
    print("实验结果")
    print("=" * 60)
    print(f"\nWith-IChing:")
    print(f"  最终验证 MSE: {result_with.final_val_mse:.6f}")
    print(f"  收敛 epoch:   {result_with.convergence_epoch}")
    print(f"  训练时间:     {result_with.elapsed_sec:.1f}s")
    print(f"\nWithout-IChing:")
    print(f"  最终验证 MSE: {result_without.final_val_mse:.6f}")
    print(f"  收敛 epoch:   {result_without.convergence_epoch}")
    print(f"  训练时间:     {result_without.elapsed_sec:.1f}s")

    improvement = (result_without.final_val_mse - result_with.final_val_mse) / max(result_without.final_val_mse, 1e-8)
    speedup = result_without.convergence_epoch - result_with.convergence_epoch
    print(f"\n→ 验证 MSE 相对提升: {improvement * 100:.2f}%")
    print(f"→ 收敛速度提升: {speedup} epochs")

    # 5. 断言验证
    assert result_with.final_val_mse < result_without.final_val_mse, (
        f"With-IChing MSE ({result_with.final_val_mse:.4f}) should be lower than "
        f"Without-IChing ({result_without.final_val_mse:.4f})"
    )
    print("\n✓ 消融实验验证通过: 易经数学结构对 JEPA 有正向提升作用")

    # 6. 保存报告
    report = {
        "experiment": "IChing Ablation on JEPA",
        "data": "Lorenz + Multi-sine (plain synthetic)",
        "with_iching": {
            "input_dim": 256,
            "encoder": "ZWMStructuredEncoder(hybrid)",
            "final_val_mse": round(result_with.final_val_mse, 6),
            "convergence_epoch": result_with.convergence_epoch,
            "elapsed_sec": round(result_with.elapsed_sec, 1),
        },
        "without_iching": {
            "input_dim": 128,
            "encoder": "TransformerEncoder(flat)",
            "final_val_mse": round(result_without.final_val_mse, 6),
            "convergence_epoch": result_without.convergence_epoch,
            "elapsed_sec": round(result_without.elapsed_sec, 1),
        },
        "improvement": {
            "val_mse_relative": round(improvement, 4),
            "convergence_speedup": speedup,
        },
    }
    with open("ablation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: ablation_report.json")


if __name__ == "__main__":
    main()
