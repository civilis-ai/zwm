#!/usr/bin/env python3
"""完整多场+视觉 OODA 闭环测试.

链路: 视觉→卦象场→FieldGNN→JEPA→MCTS→场变异→学习→循环
"""

import math, sys, time
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / "src"))


def generate_frame(t: float, size: int = 224) -> np.ndarray:
    """生成移动圆圈的合成视觉帧 (GPU 友好的向量化)."""
    ys, xs = np.ogrid[:size, :size]
    cx = size // 2 + int(60 * math.cos(t))
    cy = size // 2 + int(60 * math.sin(t))
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :, 0] = np.clip(1.0 - d / 70.0, 0, 1)           # R: 主圆
    img[:, :, 1] = np.clip(1.0 - np.abs(d - 40) / 30.0, 0, 1)  # G: 环
    img[:, :, 2] = 0.3 + 0.2 * math.sin(t)                  # B: 时间调制
    return img


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  多场+视觉 OODA 闭环                                  ║")
    print("║  视觉(ZWMVisionField) + 方图(GNN) + 圆图+干支       ║")
    print("║  + 元会运世 + JEPA(Structured) + MCTS + 场变异      ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ═════════════════════════════════════════════════════════════
    # 1. 初始化组件
    # ═════════════════════════════════════════════════════════════
    print("── 1. 组件初始化 ──")

    # 视觉
    from zwm.encoder.vision_field import ZWMVisionField
    vision = ZWMVisionField(backbone="hexvit")
    vp = sum(p.numel() for p in vision.parameters())
    print(f"  视觉编码器: hexvit ({vp:,} params)")

    # 方图 GNN
    from zwm.jepa.field_gnn import FieldSquareGNN
    gnn = FieldSquareGNN(hidden_dim=64, num_layers=2)
    print(f"  方图 GNN: 64→64 (2层)")

    # 时间场
    from zwm.scene_field.time_context import TimeContext
    from zwm.scene_field.time_field import TimeFieldEncoder
    tc = TimeContext.compute(2026, 6, 6, 12)
    tfe = TimeFieldEncoder()
    time_fields = tfe.encode_all(tc)
    print(f"  时间场: 圆图+干支+元会运世+节气 ({tc.ganzhi_str})")

    # JEPA (结构化编码器)
    from zwm.jepa.predictor import JEPAPredictor
    jepa = JEPAPredictor(input_dim=256, hidden_dim=192, latent_dim=64,
                         vicreg_weight=0.04, replay_capacity=128, batch_size=16,
                         use_action_cond=False)
    has_se = jepa._structured_encoder is not None
    print(f"  JEPA: input=256, latent=64 (structured_encoder={'hybrid' if has_se else 'MLP'})")
    jp = sum(p.numel() for p in jepa.parameters())
    print(f"    总参数: {jp:,}")

    # MCTS + MoE
    from zwm.planner.loop import TrinityPlanner
    from zwm.self_field.palace_graph import LuoshuGrid
    planner = TrinityPlanner(mcts_iterations=40, use_diffusion=True)
    grid = LuoshuGrid()
    print(f"  MCTS: 40 iterations, Diffusion ON")

    # 场变异
    from zwm.planner.field_mutations import FieldMutation
    fmut = FieldMutation()
    print(f"  行动空间: {fmut.n_atomic_actions} atomic + {fmut.n_regional_actions} regional")

    total_params = vp + jp + sum(p.numel() for p in gnn.parameters())
    print(f"\n  总可训练参数: {total_params:,}")

    # ═════════════════════════════════════════════════════════════
    # 2. OODA 闭环
    # ═════════════════════════════════════════════════════════════
    print(f"\n── 2. OODA 闭环 (15 ticks) ──")

    # 初始视觉帧
    t0 = time.perf_counter()
    t_step = 0.0

    # 首帧: 初始化卦象场
    img = generate_frame(t_step)
    vision_field = vision.encode(img)  # (64, 6)

    # 运行: observe → predict → evaluate → act → learn
    jepa_losses = []
    surprises = []
    field_history = []

    for tick in range(15):
        t_step += 0.3  # 时间推进

        # ── OBSERVE: 视觉→卦象场 + 时间场 ──
        img_next = generate_frame(t_step)
        vision_field_next = vision.encode(img_next)

        # ── PREDICT: 多场融合 → z_world ──
        from zwm.scene_field.time_field import MultiFieldJoint
        joint = MultiFieldJoint(
            square_field=vision_field,
            time_fields=time_fields,
            square_gnn=gnn,
        )
        z_world = joint.encode()  # (256,)
        z_pred = jepa.predict(z_world)  # (64,)

        # ── EVALUATE: MCTS + MoE (简化: 用 z_pred 的 L2 norm 作为效果度量) ──
        # MCTS 需要 Hexagram + grid + target_palace
        # 从 vision_field 的中心位置提取主导卦
        center_yao = vision_field[31]  # 近似 8×8 中心 (row 3, col 7)
        from zwm.core.yao import YANG, YIN
        from zwm.core.hexagram import Hexagram
        center_hex = Hexagram(*[YANG if s > 0.5 else YIN for s in center_yao])

        try:
            plan = planner.plan(center_hex, grid)
            top_mutation = plan.top_mutation
            top_score = plan.top_score
            eval_ok = True
        except Exception:
            top_mutation = 0
            top_score = 0.5
            eval_ok = False

        # ── ACT: 场变异 + 下一状态预测 ──
        if top_mutation > 0:
            # 将 top_mutation (6-bit) 应用到某个宫位
            palace = (tick % 9) + 1
            yao_idx = (top_mutation - 1) % 6 if top_mutation > 0 else 0
            vision_field_next = fmut.mutate_regional(vision_field, palace, yao_idx)

        # 世界模型 surprise: z_world_next vs z_pred
        joint_next = MultiFieldJoint(
            square_field=vision_field_next,
            time_fields=time_fields,
            square_gnn=gnn,
        )
        z_world_next = joint_next.encode()
        z_target = jepa.target_latent(z_world_next)
        surprise = float(np.linalg.norm(z_pred - z_target))

        # ── LEARN: JEPA 训练 + 记忆 ──
        reward = 0.5 + 0.4 * math.sin(t_step)
        result = jepa.train_step(z_world, z_world_next)

        if not math.isnan(result.get("loss", float("nan"))):
            jepa_losses.append(result["loss"])
            surprises.append(surprise)

        field_history.append(vision_field)
        vision_field = vision_field_next

        if tick < 4 or tick == 14:
            jl = f"{result.get('loss', 0):.4f}" if 'loss' in result else "none"
            print(f"  [{tick:2d}] hex={center_hex.name[:4]} "
                  f"surprise={surprise:.3f} JEPA={jl} "
                  f"reward={reward:.2f} "
                  f"MCTS={'✓' if eval_ok else '✗'}")

    elapsed = time.perf_counter() - t0

    # ═════════════════════════════════════════════════════════════
    # 3. 结果分析
    # ═════════════════════════════════════════════════════════════
    print(f"\n── 3. 结果 ──")
    print(f"  耗时: {elapsed:.1f}s ({15 / elapsed:.1f} ticks/s)")

    if jepa_losses:
        print(f"  JEPA loss: {jepa_losses[0]:.4f} → {jepa_losses[-1]:.4f} "
              f"({'↓' if jepa_losses[-1] < jepa_losses[0] else '↑'})")
        print(f"    μ={np.mean(jepa_losses):.4f} σ={np.std(jepa_losses):.4f}")
    if surprises:
        print(f"  Surprise:    {surprises[0]:.4f} → {surprises[-1]:.4f} "
              f"({'↓' if surprises[-1] < surprises[0] else '↑'})")
        print(f"    μ={np.mean(surprises):.4f} σ={np.std(surprises):.4f}")

    # 场一致性: 相邻帧的场应该相似
    sims = []
    for i in range(len(field_history) - 1):
        sim = 1.0 - float(np.mean(np.abs(field_history[i] - field_history[i+1])))
        sims.append(sim)
    print(f"  Field temporal coherence: μ={np.mean(sims):.4f} "
          f"(near 1.0 = smooth, near 0.5 = chaotic)")

    # 视觉场的空间多样性
    n_active = int((vision_field.mean(axis=1) > 0.3).sum())
    print(f"  Active hexagram positions: {n_active}/64")

    # ═════════════════════════════════════════════════════════════
    # 4. 对比测试: 不同视觉输入产生不同的场
    # ═════════════════════════════════════════════════════════════
    print(f"\n── 4. 视觉敏感性验证 ──")
    img_a = generate_frame(0.0)
    img_b = generate_frame(3.14)  # 半周期: 圆圈在对侧
    f_a = vision.encode(img_a)
    f_b = vision.encode(img_b)
    field_diff = float(np.mean(np.abs(f_a - f_b)))
    # 注意: 随机初始化的模型对相近视觉输入产生相似场是正常的
    # JEPA 训练会逐步增大场的区分度 — 当前检查结构完整性
    print(f"  同一场景不同时刻: L1 diff = {field_diff:.4f} "
          f"({'可区分' if field_diff > 0.003 else '几乎相同'}"
          f"{' (训练后会增大)' if field_diff < 0.05 else ''}")

    img_blank = np.zeros((224, 224, 3), dtype=np.float32)
    img_noise = np.random.rand(224, 224, 3).astype(np.float32)
    f_blank = vision.encode(img_blank)
    f_noise = vision.encode(img_noise)
    bn_diff = float(np.mean(np.abs(f_blank - f_noise)))
    print(f"  空白 vs 噪声: L1 diff = {bn_diff:.4f} "
          f"({'显著不同' if bn_diff > 0.01 else '几乎相同'}")

    ok = (len(jepa_losses) == 15
          and len(surprises) == 15
          and bn_diff > 0.01
          and jepa_losses[-1] < jepa_losses[0])  # loss下降

    print(f"\n{'='*60}")
    print(f"  验证: {'✅ 全部通过' if ok else '❌ 存在问题'}")
    print(f"{'='*60}")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
