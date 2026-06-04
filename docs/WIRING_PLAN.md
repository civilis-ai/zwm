# ZWM 接线改造计划 (Wiring Remediation Plan)

> **状态:✅ 已全部实施(PR-1 ~ PR-6)。** 基于 2026-06-04 的自顶向下只读架构审计。
> 框架决策:可学习组件采用 **PyTorch**(真 autograd + EMA + VICReg)。
> 结果:107 passed,~86% cov,0 孤岛;OODA 闭环内 JEPA loss 实测 0.186→0.041。
> 配套:架构总览见 [BLUEPRINT.md](BLUEPRINT.md)。实现见 `src/zwm/planner/agent.py`。

## 0. 核心诊断:缺一个"持久学习态"载体

当前 `TrinityPlanner`(`src/zwm/planner/loop.py`)是**无状态**的 —— 每次
`plan()` 都从零构造,`_efe_score` 里 `visit_counts={}` 写死,学习产物无处
存活。所有"孤岛"断裂的**根因是同一个**:没有一个跨 tick 携带
`preference_weights / visit_counts / 记忆句柄 / JEPA 权重` 的对象。

**改造主轴:引入 `TrinityAgent`**(新文件 `planner/agent.py`),持有可变学习
态;`TrinityPlanner.plan()` 保持纯函数式评估,`Agent.tick()` 负责 OODA 编排
与状态回写。状态集中在 Agent 一处,`plan / PlanResult` 仍为纯函数 + frozen
dataclass。

```
TrinityAgent (持久态: 跨 tick 存活)
  ├─ OnlineLearner          (preference_weights, visit_counts)  ← 学习落地点
  ├─ HebbianAssociator      (卦象转移关联)
  ├─ EpisodicStore          (SQLite 情节)
  ├─ VSACodebook+MemoryBuffer (超维记忆)
  ├─ JEPAPredictor          (唯一真实可训练网络)
  ├─ SquareCircularJoint    (方圆编码器 → JEPA 输入)
  └─ TrinityPlanner         (无状态评估器: MCTS+EFE+MoE)
```

## 1. 审计基线(改造前的事实)

- 测试:93 passed,整体覆盖率 76%。
- 顶层入口 `TrinityPlanner.plan()` 只接通 13 个子系统中的 5 个
  (MoE / MCTS / EFE / FiveHexagramChain / core)。
- **0 个 src 文件引用的孤岛(仅测试供养):**
  `jepa.predictor`(0% 覆盖,纯死代码)、`learning.hebbian`、`learning.online`、
  `storage.episodic_db`、`encoder.base`、`topology.recursive`、
  `scene_field.unified_field`、`scene_field.calendar`。
- 已确认的断点:
  - `loop.py` `_efe_score`:`visit_counts={}` / `total_visits=1` 写死,EFE 认知项退化为常数。
  - `loop.py` plan():`LangevinSampler` 被构造但在 MCTS 分支下永不调用。
  - `observe_predict_evaluate_act`:实现仅 `return self.plan(...)`,OODA 名不副实。

## 2. 目标数据流(真正的 OODA 闭环)

★ = 当前断裂、本计划要接通的边。

```
OBSERVE
  sensor_data ─★─> RuleBasedEncoder.encode() ──> h_current
  wall_clock ──★─> MultiScaleCalendar.time_layers() ──> time_phase
  (h, grid, time_phase, day_gan) ─★─> UnifiedField.snapshot() ──> world_state
  vsa_vec = VSACodebook.encode_hexagram(h)
        ─★─> EpisodicStore.query_similar_vector() ──> priors
PREDICT
  world_state.to_tensor() ─★─> SquareCircularJoint.encode() ──> z_world
        ─★─> JEPAPredictor.predict(z_world) ──> z_pred
  HebbianAssociator.suggest_next(h) ─★─> mask_priors
EVALUATE (TrinityPlanner.plan)
  MCTS _expand 顺序 ─★◄─ mask_priors (Hebbian/Langevin 暖启动)
  _efe_score ─★◄─ self._visit_counts (来自 backprop)            [修复]
  MoE.evaluate ─★◄─ OnlineLearner.preference_weights            [修复]
ACT
  top_mutation ──> h_next = UnifiedField.evolve()
LEARN (tick 收尾, 回写持久态)
  reward ─★─> EpisodicStore.store(episode, vsa_vec, reward)
        ─★─> OnlineLearner.update_from_outcome() ──> pref_weights
        ─★─> HebbianAssociator.update_from_episode(traj, reward)
        ─★─> JEPAPredictor.train_step(z_world, z_next)   [真梯度]
```

## 3. 逐孤岛接入点

| # | 孤岛模块 | 当前状态 | 接入阶段 | 输入 ← | 输出 → 消费者 |
|---|---|---|---|---|---|
| 1 | `RuleBasedEncoder` | 0 src 引用 | OBSERVE | sensor_data | `h_current` → plan |
| 2 | `MultiScaleCalendar` | 仅测试 | OBSERVE | wall_clock | `time_phase` → plan / MoE |
| 3 | `UnifiedField` | 0 src 引用 | OBSERVE/ACT | h,grid,phase | `to_tensor()` → JEPA |
| 4 | `SquareCircularJoint` | 25% cov | PREDICT | `to_tensor()` | `z_world` → JEPA |
| 5 | `JEPAPredictor` | 0% 死代码 | PREDICT+LEARN | z_world,z_next | `z_pred`;train 损失 |
| 6 | `OnlineLearner` | 0 src 引用 | EVALUATE◄+LEARN | reward,moe_w | `preference_weights` → MoE/EFE |
| 7 | `HebbianAssociator` | 0 src 引用 | PREDICT+LEARN | trajectory | `mask_priors` → MCTS `_expand` |
| 8 | `EpisodicStore` | 0 src 引用 | OBSERVE+LEARN | episode | `priors` → plan 暖启动 |
| 9 | `VSACodebook/Buffer` | 72% 仅自测 | OBSERVE/LEARN | h_bits | `vsa_vec` → EpisodicStore |
| 10 | `RecursiveTopology` | 仅自测 | (可选/P4) | grid | 多尺度宫位上下文 |
| — | EFE 断点 | visit_counts={} | `_efe_score` | backprop | 真实 epistemic |
| — | Langevin 死枝 | plan() 不达 | `_expand` 提议器 | h_current | mask 排序 |

## 4. 分期实施(PR 粒度,TDD 顺序)

每期遵守:先写 RED 测试 → 接线 → code-reviewer 收尾。

### PR-1 ｜ 修两个断点(零新依赖,最高 ROI)
- 目标:让现有"已接通的 5 个子系统"先正确,不引入孤岛。
- 改动:
  - `TrinityPlanner` 增 `self._visit_counts: dict[int,int] = {}`;`_backpropagate`
    用 `node.hex_bits` 累加;`_efe_score` 读取它替换写死的 `{}`。
  - Langevin 死枝:让 `_expand` 用 `sampler.top_k_mutations` 的排序作为
    `untried_masks` 的弹出顺序(暖启动),而非随机 `pop()`。
- 测试:`test_efe_visit_counts_varies`(同一卦反复 plan,epistemic 项应单调下降)。

### PR-2 ｜ OnlineLearner 反馈环(让 MoE "真的学")
- 改动:
  - `SparseMoE.evaluate()` 增可选参 `preference_weights: dict|None`,用它缩放
    `active_weights`。
  - 新 `planner/agent.py`:`TrinityAgent` 持有 `OnlineLearner`;`tick()` 调 `plan`
    后 `update_from_outcome`,把 `preference_weights` 传回下一次 `evaluate`。
- 数据契约:`preference_weights` 键 = MoE 6 专家名
  (`time/space/social/element/risk/narrative`,已对齐)。
- 测试:`test_preference_feedback_shifts_routing`。

### PR-3 ｜ 记忆层接入(EpisodicStore + VSA)
- 改动:
  - LEARN 段:`VSACodebook.encode_hexagram(h)` →
    `EpisodicStore.store(..., encoded_vector=vsa_vec, reward=...)`。
  - OBSERVE 段:`query_similar_vector(vsa_vec)` → 取历史高 reward 卦作为 MCTS
    root 的 mask 先验。
  - `SemanticStore.increment_frequency` → 喂 `OnlineLearner.novelty_bonus`。
- 风险:`EpisodicStore.query_similar_vector` 用 `query_recent(1000)` 全表扫(O(n)),
  需加注释说明上限;`int8` BLOB 与 VSA `int8` 一致 ✓。
- 测试:`test_episode_roundtrip_biases_plan`。

### PR-4 ｜ JEPA 真训练(去掉"随机网络"指控)
- 目标:让全仓库**第一次有真实梯度在闭环中流动**。
- 改动:
  - PREDICT:`SquareCircularJoint.encode(world_state) → z_world →
    JEPAPredictor.predict`。
  - LEARN:缓存上一 tick 的 `z_world`,本 tick 拿到 `z_next` 后
    `JEPAPredictor.train_step(z_prev, z_next)`,损失入日志。
  - 维度对齐:`SquareCircularJoint.encode` 输出 `z_s`(64) + `z_t`(13) = 77,
    恰好匹配 `JEPAPredictor(input_dim=77)`;加 `assert z_world.shape[0]==77` 锁死。
- 测试:`test_jepa_loss_decreases`(喂重复转移,N 步后 loss 下降)。
- 决策点:若不做真训练,则诚实降级 —— 把 `JEPAPredictor` /
  `FixedWeightSquareGNN` 标注"非学习基线"并删 `train_step` 死代码。二选一,不留模糊。

### PR-5 ｜ Hebbian 先验 + OODA 正名
- 改动:
  - `HebbianAssociator.suggest_next(h)` → `_expand` 的 mask 先验(与 PR-1 的
    Langevin 排序加权融合)。
  - `observe_predict_evaluate_act` 从"`plan` 别名"重写为真正调用 `agent.tick()`
    的 O-P-E-A 四段,名实相符。
- 测试:`test_ooda_full_tick_persists_state`(连续 3 tick,记忆/权重/Hebbian 均增长)。

### PR-6 ｜ 文档落盘
- 写 `docs/BLUEPRINT.md`(修复 README 悬空链接)+ 本计划 + 数据流图。

## 5. 依赖排序与风险

```
PR-1 (断点) ──┬─> PR-2 (反馈环) ──> PR-3 (记忆) ──> PR-4 (JEPA训练) ──> PR-5 (Hebbian+正名)
              └─> 独立可并行                                            └─> PR-6 (文档)
```

| 风险 | 缓解 |
|---|---|
| `TrinityAgent` 引入可变状态 | 状态集中在 Agent 一处;`plan/PlanResult` 仍纯函数 + frozen dataclass |
| JEPA 77 维耦合脆弱 | PR-4 加 `assert z_world.shape[0]==77` fail-fast |
| `query_similar_vector` 全表扫 O(n) | 研究态可接受,`log` 标注上限,后续换 faiss |
| 手写 SGD 可能发散 | 已有 `sigreg_loss` 正则;加梯度裁剪 + loss NaN 守卫 |

## 6. 改造后预期

- `jepa/predictor.py`:0% → ~85%(PR-4 后真实执行)。
- 0-src-引用孤岛:8 个 → 0 个。
- `observe_predict_evaluate_act`:空壳 → 真 OODA。
- 全仓库首次有反向传播在规划闭环中运行(PR-4)。
