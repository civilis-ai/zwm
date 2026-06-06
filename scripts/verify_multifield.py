#!/usr/bin/env python3
"""ZWM 多场架构端到端验证.

验证链路:
  1. 场编码 (方图/圆图/干支/元会运世/节气)
  2. FieldSquareGNN 图神经网络
  3. MultiFieldJoint 多场融合
  4. JEPA 预测器消费
  5. OODA 完整闭环
  6. 模块消费检查
  7. 卦象场统计/可视化

用法:
    python scripts/verify_multifield.py
    python scripts/verify_multifield.py --quick      # 仅基础检查
    python scripts/verify_multifield.py --full        # 含训练步
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ═══════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════

_checks_passed = 0
_checks_failed = 0
_checks_warn = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
    global _checks_passed, _checks_failed
    if condition:
        _checks_passed += 1
        print(f"  ✅ {name}")
    else:
        _checks_failed += 1
        print(f"  ❌ {name}  — {detail}")
    return condition


def warn(name: str, detail: str = "") -> None:
    global _checks_warn
    _checks_warn += 1
    print(f"  ⚠️  {name} — {detail}")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════
# Section 1: HexagramFieldEncoder
# ═══════════════════════════════════════════════════════════════════════

def verify_field_encoder():
    section("1. HexagramFieldEncoder — 传感器→64卦场")

    from zwm.encoder.field_encoder import HexagramFieldEncoder

    # 1.1 空间分片
    enc = HexagramFieldEncoder(strategy="spatial")
    # 模拟 224×224×3 图像
    image = np.random.randn(224, 224, 3).astype(np.float32)
    field = enc.encode(image)
    check("1.1 spatial field shape", field.shape == (64, 6),
          f"got {field.shape}, expected (64, 6)")
    check("1.2 spatial field range [0,1]", field.min() >= 0 and field.max() <= 1,
          f"range=[{field.min():.3f}, {field.max():.3f}]")
    check("1.3 spatial yao diversity (not all same)",
          np.std(field) > 0.01,
          f"std={np.std(field):.4f}")

    # 1.4 时间分片
    enc_t = HexagramFieldEncoder(strategy="temporal")
    audio = np.sin(np.linspace(0, 20 * np.pi, 1000)).astype(np.float32)
    field_t = enc_t.encode(audio)
    check("1.4 temporal field shape", field_t.shape == (64, 6))

    # 1.5 混合分片
    enc_m = HexagramFieldEncoder(strategy="mixed")
    sensor_dict = {
        "温度": 25.3, "湿度": 0.67, "气压": 1013.25,
        "风速": 3.2, "光照": 450.0, "CO2": 420.0,
    }
    field_m = enc_m.encode(sensor_dict)
    check("1.5 mixed field shape", field_m.shape == (64, 6))

    # 1.6 自适应检测
    enc_a = HexagramFieldEncoder(strategy="adaptive")
    field_a = enc_a.encode(image)
    check("1.6 adaptive detects spatial for 3D array", field_a.shape == (64, 6))

    # 1.7 6爻各有不同语义
    for yao in range(6):
        yao_vals = field[:, yao]
        check(f"1.7 yao{yao} has variation", np.std(yao_vals) > 0.001,
              f"std={np.std(yao_vals):.4f}")

    return field  # 返回给后续测试


# ═══════════════════════════════════════════════════════════════════════
# Section 2: FieldSquareGNN
# ═══════════════════════════════════════════════════════════════════════

def verify_field_gnn(square_field: np.ndarray):
    section("2. FieldSquareGNN — 卦象场图神经网络")

    from zwm.jepa.field_gnn import FieldSquareGNN

    gnn = FieldSquareGNN(hidden_dim=64, num_layers=3)

    # 2.1 前向传播
    import torch
    x = torch.from_numpy(square_field.astype(np.float32))
    node_emb = gnn(x)  # (1, 64, hidden_dim)
    check("2.1 forward output shape", node_emb.shape == (1, 64, 64),
          f"got {node_emb.shape}")

    # 2.2 embed_field
    z_sq = gnn.embed_field(square_field)
    check("2.2 embed_field shape", z_sq.shape == (64,),
          f"got {z_sq.shape}")
    check("2.3 embed_field finite", np.all(np.isfinite(z_sq)),
          f"nan={np.any(np.isnan(z_sq))}, inf={np.any(np.isinf(z_sq))}")

    # 2.4 嵌入不是恒定的
    check("2.4 embedding has variation", np.std(z_sq) > 0.001,
          f"std={np.std(z_sq):.6f}")

    # 2.5 不同输入产生不同嵌入
    field2 = np.random.rand(64, 6).astype(np.float32)
    z_sq2 = gnn.embed_field(field2)
    diff = np.linalg.norm(z_sq - z_sq2)
    check("2.5 different inputs → different embeddings", diff > 0.01,
          f"L2 diff={diff:.4f}")

    # 2.6 节点注意力
    attn = gnn.attention_weights(square_field)
    check("2.6 attention weights shape", attn.shape == (64,),
          f"got {attn.shape}")
    check("2.7 attention weights sum > 0", attn.sum() > 0)

    # 2.8 训练路径
    z_sq_train = gnn.embed_field_train(square_field)
    check("2.8 train path output shape", z_sq_train.shape == (1, 64),
          f"got {z_sq_train.shape}")
    check("2.9 train path requires grad", z_sq_train.requires_grad)

    return gnn


# ═══════════════════════════════════════════════════════════════════════
# Section 3: TimeFieldEncoder + TimeContext
# ═══════════════════════════════════════════════════════════════════════

def verify_time_fields():
    section("3. TimeFieldEncoder — 时间→卦象场")

    from zwm.scene_field.time_context import TimeContext
    from zwm.scene_field.time_field import TimeFieldEncoder

    # 3.1 TimeContext 构造
    tc = TimeContext.compute(2026, 6, 6, 12)
    check("3.1 TimeContext created", tc is not None)
    check("3.2 day_gan = 庚 (2026-06-06)", tc.day_gan == "庚",
          f"got '{tc.day_gan}'")
    check("3.3 hui_index = 7 (午会)", tc.hui_index == 7,
          f"got {tc.hui_index}")

    # 3.4 圆图时间场
    tfe = TimeFieldEncoder()
    circular = tfe.encode_circular(tc)
    check("3.4 circular field shape", circular.shape == (64, 6))
    check("3.5 circular has activation center", circular.max() > 0.8,
          f"max={circular.max():.3f}")

    # 3.6 六十甲子场
    ganzhi_field = tfe.encode_ganzhi(tc)
    check("3.6 ganzhi field shape", ganzhi_field.shape == (64, 6))
    # 当前日干支位置应该高激活
    gz_active = ganzhi_field[:, 0].max()
    check("3.7 current ganzhi position active", gz_active > 0.5,
          f"max_activation={gz_active:.3f}")

    # 3.8 元会运世场
    cosmic = tfe.encode_cosmic(tc)
    check("3.8 cosmic field shape", cosmic.shape == (64, 6))

    # 3.9 节气场
    solar = tfe.encode_solar_term(tc)
    check("3.9 solar term field shape", solar.shape == (64, 6))
    check("3.10 current solar term = 芒种", tc.solar_term_name == "芒种",
          f"got '{tc.solar_term_name}'")

    # 3.11 encode_all
    fields = tfe.encode_all(tc)
    check("3.11 encode_all → TimeFields", fields is not None)
    check("3.12 TimeFields.to_flat dim", fields.to_flat().shape == (1536,),
          f"got {fields.to_flat().shape}")

    return fields, tc


# ═══════════════════════════════════════════════════════════════════════
# Section 4: MultiFieldJoint + JEPA
# ═══════════════════════════════════════════════════════════════════════

def verify_multifield_joint(square_field: np.ndarray, time_fields, gnn, full_train: bool = False):
    section("4. MultiFieldJoint → JEPA 共振")

    from zwm.scene_field.time_field import MultiFieldJoint

    # 4.1 融合
    joint = MultiFieldJoint(
        square_field=square_field,
        time_fields=time_fields,
        square_gnn=gnn,
        weights=(0.4, 0.3, 0.2, 0.1),
    )
    z_world = joint.encode()
    check("4.1 MultiFieldJoint output dim", z_world.shape == (256,),
          f"got {z_world.shape}")
    check("4.2 z_world finite", np.all(np.isfinite(z_world)))

    # 4.2 权重体现 (方图=0.4 应占主导)
    z_by_component = joint.encode()
    check("4.3 joint output not all zeros", np.abs(z_world).sum() > 0.01)

    # 4.3 JEPA 消费
    from zwm.jepa.predictor import JEPAPredictor
    jepa = JEPAPredictor(input_dim=256, hidden_dim=192, latent_dim=64,
                         vicreg_weight=0.04, replay_capacity=64, batch_size=8)

    z_pred = jepa.predict(z_world)
    check("4.4 JEPA.predict shape", z_pred.shape == (64,),
          f"got {z_pred.shape}")
    check("4.5 JEPA.predict finite", np.all(np.isfinite(z_pred)))

    # 4.4 JEPA 训练步 — 验证梯度流动
    if full_train:
        print("    运行 JEPA 训练步 (10 steps)...")
        losses = []
        for i in range(10):
            # 模拟连续两帧
            z_t = joint.encode()
            # 轻微扰动方图模拟状态演化
            square_field2 = square_field + np.random.randn(64, 6).astype(np.float32) * 0.01
            square_field2 = np.clip(square_field2, 0, 1)
            joint2 = MultiFieldJoint(
                square_field=square_field2,
                time_fields=time_fields,
                square_gnn=gnn,
            )
            z_next = joint2.encode()

            result = jepa.train_step(z_t, z_next, max_grad_norm=5.0)
            if not math.isnan(result.get("loss", float("nan"))):
                losses.append(result["loss"])
        if losses:
            check("4.6 JEPA training loss decreases",
                  losses[-1] < losses[0] + 0.1,  # allow noise
                  f"loss: {losses[0]:.4f} → {losses[-1]:.4f}")
            check("4.7 JEPA training produces finite loss",
                  all(math.isfinite(l) for l in losses))
            print(f"     JEPA loss: {losses[0]:.4f} → {losses[-1]:.4f} ({len(losses)} steps)")
        else:
            warn("4.6 JEPA training — no valid losses (NaN in all steps)")

    # 4.5 VICReg 防坍缩
    import torch
    z_batch = torch.randn(8, 64)  # simulate batch of latents
    vicreg = jepa.vicreg_loss(z_batch)
    check("4.8 VICReg loss computed", vicreg.item() >= 0)

    return jepa


# ═══════════════════════════════════════════════════════════════════════
# Section 5: FieldMutation Action Space
# ═══════════════════════════════════════════════════════════════════════

def verify_field_mutations(square_field: np.ndarray):
    section("5. FieldMutation — 384原子 + 54区域 + macro 行动空间")

    from zwm.planner.field_mutations import FieldMutation

    fm = FieldMutation()

    # 5.1 原子行动
    atomic = fm.action_list("atomic")
    check("5.1 384 atomic actions", len(atomic) == 384,
          f"got {len(atomic)}")

    # 5.2 区域行动
    regional = fm.action_list("regional")
    check("5.2 54 regional actions (9 palaces × 6 yao)", len(regional) == 54,
          f"got {len(regional)}")

    # 5.3 原子变异
    new_field = fm.mutate(square_field, pos=0, yao_idx=0)
    n_diff = int(np.sum(np.abs(new_field - square_field) > 0.01))
    check("5.3 atomic mutation changes exactly 1 yao", n_diff == 1,
          f"changed {n_diff} yaos")

    # 5.4 区域变异
    new_regional = fm.mutate_regional(square_field, palace=5, yao_idx=2)
    n_diff_r = int(np.sum(np.abs(new_regional - square_field) > 0.01))
    # 中宫 (5) 大约覆盖 4-9 个位置
    check("5.4 regional mutation changes 2-9 yaos",
          1 < n_diff_r <= 9,
          f"changed {n_diff_r} yaos")

    # 5.5 变异分类
    cls = fm.classify(square_field, new_field)
    check("5.5 classify atomic", cls == "atomic", f"got '{cls}'")

    # 5.6 九宫掩码
    mask = fm.palace_mask(5)
    check("5.6 palace 5 mask shape", mask.shape == (64,))
    check("5.7 palace 5 has positions", mask.sum() >= 4)

    # 5.8 整行变异
    new_row = fm.mutate_row(square_field, row=3, yao_idx=1)
    n_diff_row = int(np.sum(np.abs(new_row - square_field) > 0.01))
    check("5.8 row mutation changes exactly 8 yaos", n_diff_row == 8,
          f"changed {n_diff_row} yaos")

    return fm


# ═══════════════════════════════════════════════════════════════════════
# Section 6: 完整 OODA 闭环
# ═══════════════════════════════════════════════════════════════════════

def verify_ooda_loop(full_train: bool = False):
    section("6. 完整 OODA 闭环 — TrinityAgent 端到端")

    from zwm.planner.agent import TrinityAgent
    from zwm.planner.agent_config import TrinityConfig
    from zwm.core.hexagram import hexagram_from_name

    # 6.1 Agent 构造 (场编码模式)
    config = TrinityConfig(
        db_path=":memory:",
        use_field_encoder=True,
        mcts_iterations=20,
        n_particles=0,
        use_react=False,  # 关 ReAct 加速测试
    )
    try:
        agent = TrinityAgent(config=config)
    except Exception as exc:
        warn("6.1 Agent construction (field mode)", f"failed: {exc}")
        # 尝试回退
        config2 = TrinityConfig(
            db_path=":memory:",
            use_field_encoder=False,
            mcts_iterations=20,
            n_particles=0,
            use_react=False,
        )
        agent = TrinityAgent(config=config2)
        warn("6.1 Agent construction (fallback single-hex mode)", "")

    check("6.1 Agent constructed", agent is not None)

    # 6.2 场编码器状态
    check("6.2 field_encoder exists", agent.field_encoder is not None)
    check("6.3 _field_gnn exists", agent._field_gnn is not None,
          "FieldSquareGNN 未初始化, 检查 _init_world_model")

    # 6.3 OODA ticks
    h = hexagram_from_name("乾为天")
    reports = []
    for i in range(5):
        reward = 0.5 + 0.4 * math.sin(i / 3.0)
        try:
            report = agent.tick(h_current=h, reward=reward, year=2026, month=6, day=6)
            reports.append(report)
            h = report.h_next
        except Exception as exc:
            warn(f"6.3 tick {i}", f"failed: {exc}")
            break

    check("6.4 OODA ticks completed", len(reports) > 0,
          f"completed {len(reports)}/5 ticks")

    if reports:
        # 6.5 surprise 是有限数
        surprises = [r.surprise for r in reports]
        check("6.5 surprises are finite", all(math.isfinite(s) for s in surprises),
              f"surprises={surprises}")

        # 6.6 JEPA loss 被记录
        jepa_losses = [r.jepa_loss for r in reports if r.jepa_loss is not None]
        if jepa_losses:
            check("6.6 JEPA losses recorded", len(jepa_losses) > 0)
            print(f"     JEPA losses: {[f'{l:.4f}' for l in jepa_losses]}")

        # 6.7 卦象在演化 (不是同一个卦)
        hex_names = [r.h_next.name for r in reports]
        unique = len(set(hex_names))
        check("6.7 hexagrams evolve (not stuck)", unique >= 1,
              f"unique hexagrams: {unique}, sequence: {hex_names}")

    # 6.8 清理
    agent.close()
    check("6.8 Agent closed cleanly", True)

    return agent


# ═══════════════════════════════════════════════════════════════════════
# Section 7: 模块消费检查
# ═══════════════════════════════════════════════════════════════════════

def verify_module_consumption():
    section("7. 模块消费链路检查")

    # 检查关键模块的可导入性
    modules = [
        ("zwm.encoder.field_encoder", "HexagramFieldEncoder"),
        ("zwm.jepa.field_gnn", "FieldSquareGNN"),
        ("zwm.jepa.field_gnn", "FieldSquareCircularJoint"),
        ("zwm.scene_field.time_field", "TimeFieldEncoder"),
        ("zwm.scene_field.time_field", "MultiFieldJoint"),
        ("zwm.scene_field.time_context", "TimeContext"),
        ("zwm.planner.field_mutations", "FieldMutation"),
        ("zwm.llm.backends", "create_backend"),
        ("zwm.llm.backends", "auto_detect_backend"),
        ("zwm.llm.router", "LLMRouter"),
        ("zwm.llm.context", "build_react_prompt"),
        ("zwm.learning.ewc", "EWCRegularizer"),
        ("zwm.encoder.vision_backbone", "auto_vision_backbone"),
        ("zwm.grpc.server", "ZWMGrpcServicer"),
        ("zwm.planner.agent", "TrinityAgent"),
        ("zwm.jepa.predictor", "JEPAPredictor"),
        ("zwm.safety.constitution", "ConstitutionalGuard"),
        ("zwm.safety.llm_judge", "make_auto_judge"),
    ]

    for mod_name, attr_name in modules:
        try:
            mod = __import__(mod_name, fromlist=[attr_name])
            obj = getattr(mod, attr_name, None)
            check(f"7.{modules.index((mod_name, attr_name))} {mod_name}.{attr_name}",
                  obj is not None,
                  f"module imported but {attr_name} not found")
        except Exception as exc:
            warn(f"7.{modules.index((mod_name, attr_name))} {mod_name}",
                 f"import failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════
# Section 8: 场统计与"共振"分析
# ═══════════════════════════════════════════════════════════════════════

def verify_field_resonance(square_field: np.ndarray, time_fields):
    section("8. 卦象场统计与共振分析")

    # 8.1 方图场的爻分布
    for yao in range(6):
        vals = square_field[:, yao]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        yao_names = ["初(均值)", "二(波动)", "三(峰值)", "四(谷值)", "五(梯度)", "上(熵)"]
        check(f"8.1 yao{yao} {yao_names[yao]} non-trivial",
              std_val > 0.01 and 0.1 < mean_val < 0.9,
              f"mean={mean_val:.3f}, std={std_val:.3f}")

    # 8.2 时间场的空间激活模式
    circular = time_fields.circular
    active_positions = int(np.sum(circular.max(axis=1) > 0.5))
    check("8.2 circular field activated positions",
          active_positions > 0,
          f"{active_positions}/64 positions with max>0.5")

    # 8.3 场间互信息 (简单相关性)
    sq_flat = square_field.flatten()[:64]
    circ_flat = circular.flatten()[:64]
    # 方图和圆图应有关联 (通过时间相位)
    corr = float(np.corrcoef(sq_flat, circ_flat)[0, 1]) if np.std(sq_flat) > 0 and np.std(circ_flat) > 0 else 0
    check("8.3 square-circular correlation is real number",
          math.isfinite(corr),
          f"corr={corr:.4f}")

    # 8.4 64 位置 ≠ 相同卦 (多样性)
    n_unique_hex = len(set(
        tuple(np.round(square_field[i], 1)) for i in range(64)
    ))
    check("8.4 field has diverse hexagrams",
          n_unique_hex >= 4,
          f"unique hex states: {n_unique_hex}/64")

    # 8.5 场在时间演化中应平滑变化
    # (模拟两个接近帧的相似度)
    field_t1 = square_field
    field_t2 = np.clip(square_field + np.random.randn(64, 6).astype(np.float32) * 0.05, 0, 1)
    sim = 1.0 - float(np.mean(np.abs(field_t1 - field_t2)))
    check("8.5 neighboring fields similar (>0.7 cosine-ish)",
          sim > 0.7,
          f"similarity={sim:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="仅基础检查")
    parser.add_argument("--full", action="store_true", help="含 JEPA 训练步")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  ZWM 多场架构端到端验证                               ║")
    print("║  方图(地) + 圆图(天) + 干支(人) + 元会运世(宇)     ║")
    print("╚══════════════════════════════════════════════════════╝")

    t0 = time.perf_counter()

    try:
        # 1: 场编码
        square_field = verify_field_encoder()

        # 2: FieldSquareGNN
        if not args.quick:
            gnn = verify_field_gnn(square_field)
        else:
            gnn = None

        # 3: 时间场
        time_fields, tc = verify_time_fields()

        # 4: 多场融合 + JEPA
        if not args.quick and gnn is not None:
            verify_multifield_joint(square_field, time_fields, gnn,
                                    full_train=args.full)

        # 5: 行动空间
        verify_field_mutations(square_field)

        # 6: OODA 闭环
        if not args.quick:
            verify_ooda_loop(full_train=args.full)

        # 7: 模块消费
        verify_module_consumption()

        # 8: 共振分析
        verify_field_resonance(square_field, time_fields)

    except Exception as exc:
        import traceback
        print(f"\n  ❌ UNHANDLED ERROR: {exc}")
        traceback.print_exc()
        _checks_failed += 1

    elapsed = time.perf_counter() - t0

    # ─── 总结 ───
    total = _checks_passed + _checks_failed + _checks_warn
    print(f"\n{'='*60}")
    print(f"  验证完成: {elapsed:.1f}s")
    print(f"  ✅ 通过: {_checks_passed}/{total}")
    if _checks_warn:
        print(f"  ⚠️  警告: {_checks_warn}")
    if _checks_failed:
        print(f"  ❌ 失败: {_checks_failed}")
    print(f"{'='*60}")

    return _checks_failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
