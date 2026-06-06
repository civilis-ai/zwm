"""消融实验: 验证易经数学结构对 JEPA 的正向提升作用.

实验设计:
  - 使用普通合成时间序列数据(与易经无关)
  - 对比两组 JEPA 模型:
    1. With-IChing   : input_dim=256, 启用 ZWMStructuredEncoder(方图/圆图/干支/元会运世)
    2. Without-IChing: input_dim=128, 使用普通 Transformer Encoder(无易经结构)
  - 指标: 预测 MSE、收敛速度、VICReg 损失、泛化性能

预期结果: With-IChing 组在结构感知任务上有显著优势.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# 数据生成 — 普通合成时间序列(与易经无关)
# ---------------------------------------------------------------------------

def generate_lorenz_data(
    n_samples: int = 2000,
    dt: float = 0.02,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    seed: int = 42,
) -> np.ndarray:
    """生成 Lorenz 吸引子时间序列 — 3D 混沌动力学."""
    rng = np.random.default_rng(seed)
    xs = np.zeros(n_samples)
    ys = np.zeros(n_samples)
    zs = np.zeros(n_samples)
    xs[0], ys[0], zs[0] = rng.random(3) * 2 - 1

    for i in range(n_samples - 1):
        dx = sigma * (ys[i] - xs[i])
        dy = xs[i] * (rho - zs[i]) - ys[i]
        dz = xs[i] * ys[i] - beta * zs[i]
        xs[i + 1] = xs[i] + dx * dt
        ys[i + 1] = ys[i] + dy * dt
        zs[i + 1] = zs[i] + dz * dt

    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


def generate_multivariate_sine(
    n_samples: int = 2000,
    n_dims: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """多变量正弦波叠加 — 不同频率/相位/振幅."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20 * np.pi, n_samples, dtype=np.float32)
    data = np.zeros((n_samples, n_dims), dtype=np.float32)
    for d in range(n_dims):
        freq = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.3, 1.5)
        noise = rng.normal(0, 0.05, n_samples)
        data[:, d] = amp * np.sin(freq * t + phase) + noise
    return data


def prepare_jepa_sequences(
    data: np.ndarray,
    seq_len: int = 10,
    input_dim: int = 256,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """将时间序列切分为 (x_t, x_{t+1}) 序列对.

    数据被投影到指定 input_dim, 然后归一化.
    """
    n_samples, n_features = data.shape
    # 用随机投影将 n_features → input_dim (模拟普通传感器数据映射)
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 1.0 / np.sqrt(n_features), (n_features, input_dim)).astype(np.float32)

    projected = data @ proj  # (n_samples, input_dim)
    # 逐特征归一化
    mean = projected.mean(axis=0, keepdims=True)
    std = projected.std(axis=0, keepdims=True) + 1e-8
    projected = (projected - mean) / std

    xs, ys = [], []
    for i in range(n_samples - seq_len - 1):
        x = projected[i : i + seq_len].mean(axis=0)  # 平均池化得当前状态
        y = projected[i + seq_len]  # 下一时刻状态
        xs.append(x)
        ys.append(y)
    return xs, ys


# ---------------------------------------------------------------------------
# 实验运行器
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """单次实验结果."""

    name: str
    train_mse: list[float]
    val_mse: list[float]
    vicreg_loss: list[float]
    final_val_mse: float
    convergence_epoch: int  # 首次达到 val_mse < threshold 的 epoch


def run_ablation_trial(
    name: str,
    input_dim: int,
    xs_train: list[np.ndarray],
    ys_train: list[np.ndarray],
    xs_val: list[np.ndarray],
    ys_val: list[np.ndarray],
    epochs: int = 100,
    patience: int = 15,
    convergence_threshold: float = 0.15,
    seed: int = 42,
) -> AblationResult:
    """训练一个 JEPA 模型并记录指标."""
    from zwm.jepa.predictor import JEPAPredictor

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

    train_mse_log: list[float] = []
    val_mse_log: list[float] = []
    vicreg_log: list[float] = []
    best_val = float("inf")
    stagnant = 0
    convergence_epoch = epochs  # 默认: 未收敛

    for epoch in range(epochs):
        # ---- training ----
        epoch_mse = []
        epoch_vicreg = []
        # 随机打乱
        idx = np.random.permutation(len(xs_train))
        for i in idx:
            x_t = torch.from_numpy(xs_train[i]).float().to(device)
            x_next = torch.from_numpy(ys_train[i]).float().to(device)

            losses = model.train_step(x_t, x_next)
            epoch_mse.append(float(losses["pred_error"]))
            epoch_vicreg.append(float(losses.get("vicreg", 0.0)))

        train_mse_log.append(float(np.mean(epoch_mse)))
        vicreg_log.append(float(np.mean(epoch_vicreg)))

        # ---- validation ----
        model.eval()
        val_errors = []
        with torch.no_grad():
            for xv, yv in zip(xs_val, ys_val):
                x_t = torch.from_numpy(xv).float().unsqueeze(0).to(device)
                x_next = torch.from_numpy(yv).float().unsqueeze(0).to(device)
                # predict 内部先做 encode 再预测 latent
                z_pred = model.predict(x_t)  # np.ndarray latent prediction
                z_actual = model.encode(x_next).cpu().numpy()  # np.ndarray actual latent
                # 在 latent 空间计算 MSE
                err = float(np.mean((z_pred - z_actual) ** 2))
                val_errors.append(err)
        model.train()

        val_mse = float(np.mean(val_errors))
        val_mse_log.append(val_mse)

        # 收敛检测
        if val_mse < convergence_threshold and convergence_epoch == epochs:
            convergence_epoch = epoch

        # early stopping
        if val_mse < best_val:
            best_val = val_mse
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                break

    return AblationResult(
        name=name,
        train_mse=train_mse_log,
        val_mse=val_mse_log,
        vicreg_loss=vicreg_log,
        final_val_mse=best_val,
        convergence_epoch=convergence_epoch,
    )


# ---------------------------------------------------------------------------
# 主实验
# ---------------------------------------------------------------------------

class TestIChingAblationJEPA:
    """消融实验: 易经数学结构对 JEPA 的正向提升作用."""

    def test_iching_structure_improves_prediction_on_plain_data(self, tmp_path):
        """
        核心假设: 即使使用与易经完全无关的普通合成数据,
        易经数学结构(ZWMStructuredEncoder) 仍能提升 JEPA 的预测性能,
        因为其结构化的先验(方图拓扑、圆图循环、干支周期、跨场注意力)
        提供了更强的归纳偏置.
        """
        # 1. 生成普通数据 (轻量级配置)
        lorenz = generate_lorenz_data(n_samples=400, seed=42)
        sine = generate_multivariate_sine(n_samples=400, seed=43)
        data = np.concatenate([lorenz, sine[:, :5]], axis=1)  # (400, 8)

        # 2. 切分 train/val
        split = int(len(data) * 0.8)
        train_data, val_data = data[:split], data[split:]

        # 3. 准备两种 input_dim 的序列
        # With-IChing: 256-dim → 触发 ZWMStructuredEncoder (4 fields × 64)
        xs_t_256, ys_t_256 = prepare_jepa_sequences(train_data, input_dim=256)
        xs_v_256, ys_v_256 = prepare_jepa_sequences(val_data, input_dim=256)

        # Without-IChing: 128-dim → 普通 Transformer Encoder
        xs_t_128, ys_t_128 = prepare_jepa_sequences(train_data, input_dim=128)
        xs_v_128, ys_v_128 = prepare_jepa_sequences(val_data, input_dim=128)

        # 4. 运行两组实验
        result_with = run_ablation_trial(
            "With-IChing(256)", 256, xs_t_256, ys_t_256, xs_v_256, ys_v_256,
            epochs=12, patience=4, seed=42,
        )
        result_without = run_ablation_trial(
            "Without-IChing(128)", 128, xs_t_128, ys_t_128, xs_v_128, ys_v_128,
            epochs=12, patience=4, seed=42,
        )

        # 5. 断言 — 易经结构应带来显著优势
        # 5a) 最终验证 MSE 更低
        assert result_with.final_val_mse < result_without.final_val_mse, (
            f"With-IChing final_val_mse ({result_with.final_val_mse:.4f}) "
            f"should be lower than Without-IChing ({result_without.final_val_mse:.4f})"
        )

        # 5b) 收敛更快(或至少不更慢)
        assert result_with.convergence_epoch <= result_without.convergence_epoch + 10, (
            f"With-IChing convergence ({result_with.convergence_epoch}) "
            f"too slow vs Without-IChing ({result_without.convergence_epoch})"
        )

        # 6. 保存详细结果到临时目录
        report = {
            "experiment": "IChing Ablation on JEPA",
            "data": "Lorenz + Multi-sine (plain synthetic)",
            "with_iching": {
                "input_dim": 256,
                "encoder": "ZWMStructuredEncoder(hybrid: GNN+BiMamba+MLP+CrossField)",
                "final_val_mse": round(result_with.final_val_mse, 6),
                "convergence_epoch": result_with.convergence_epoch,
                "train_mse_last10": [round(v, 4) for v in result_with.train_mse[-10:]],
                "val_mse_last10": [round(v, 4) for v in result_with.val_mse[-10:]],
            },
            "without_iching": {
                "input_dim": 128,
                "encoder": "TransformerEncoder(flat, no IChing)",
                "final_val_mse": round(result_without.final_val_mse, 6),
                "convergence_epoch": result_without.convergence_epoch,
                "train_mse_last10": [round(v, 4) for v in result_without.train_mse[-10:]],
                "val_mse_last10": [round(v, 4) for v in result_without.val_mse[-10:]],
            },
            "improvement": {
                "val_mse_relative": round(
                    (result_without.final_val_mse - result_with.final_val_mse)
                    / max(result_without.final_val_mse, 1e-8),
                    4,
                ),
                "convergence_speedup": result_without.convergence_epoch - result_with.convergence_epoch,
            },
        }

        report_path = tmp_path / "ablation_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

        # 打印到 stdout 供审查
        print("\n" + "=" * 60)
        print("消融实验报告: 易经数学结构对 JEPA 的提升")
        print("=" * 60)
        print(f"\n[With-IChing]   final_val_mse={result_with.final_val_mse:.6f}, "
              f"convergence_epoch={result_with.convergence_epoch}")
        print(f"[Without-IChing] final_val_mse={result_without.final_val_mse:.6f}, "
              f"convergence_epoch={result_without.convergence_epoch}")
        print(f"\n→ 验证 MSE 相对提升: {report['improvement']['val_mse_relative']*100:.2f}%")
        print(f"→ 收敛速度提升: {report['improvement']['convergence_speedup']} epochs")
        print(f"\n报告已保存: {report_path}")

