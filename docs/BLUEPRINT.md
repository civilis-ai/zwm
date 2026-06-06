# ZWM 蓝图 (BLUEPRINT)

> 天地人三才世界模型规划器 — Trinity World Model Planner based on I Ching mathematics.
> 本文件是架构总览。改造路线见 [WIRING_PLAN.md](WIRING_PLAN.md)。

## 1. 设计理念

将 **天地人三才** 易经框架与现代世界模型/规划技术对应起来:

- **复相位编码**:阴阳 ↔ φ∈{0,π},卦象 ↔ 6 次谐波叠加 F(t)=ΣAₙe^(i(nωt+φₙ))。
- **方圆图 JEPA**:方图(空间频谱)× 圆图(时间相位)→ 联合嵌入世界状态。
- **洛书九宫 Active Inference**:中宫 5 = 自我锚点,八方 = 频率滤波器组。
- **五卦叙事规划**:主卦 → 互卦 → 变卦 → 综卦 → 错卦 构成完整叙事弧。
- **VSA 超维计算**:bind / bundle / permute 精确对应卦象运算。
- **五行六亲社会场**:生克动力学 + 以我为中心的社会关系图。
- **四维自我定位**:空间(中宫)+ 属性(五行)+ 关系(六亲)+ 态势(世爻)。

## 2. 分层架构

```
core/        易经数学原语   YaoLine, Trigram, Hexagram, constants
spectrum/    复相位频谱     ComplexPhase, HexagramPhaseVector, FrequencySpectrum, 干涉
scene_field/ 天地场(场景)  五卦链, 五行, 六亲, 干支历, 统一场
self_field/  人场(自我)    洛书九宫图, 调和度, 期望自由能 EFE
moe/         专家混合       6 专家 + 路由器 + 稀疏激活
langevin/    采样           score 势函数 + Langevin 采样器
jepa/        联合嵌入预测   方图 GNN, 圆图编码, JEPA 预测器
hexaembed/   超维记忆       VSA 码本 / 情节 / 记忆缓冲
learning/    学习           在线学习, 好奇心调度, Hebbian 关联
storage/     持久化         SQLite 情节库 + JSON 语义库
topology/    递归宇宙       九宫递归展开
encoder/     感知入口       传感器 → 卦象规则编码
planner/     顶层规划器     TrinityPlanner (MCTS + EFE + MoE), 变爻, 密码子
```

## 3. 数据流(目标形态)

```
传感器 → Encoder → 当前卦 h
         ↓
   UnifiedField 快照(天地人统一场)
         ↓
   方圆 JEPA 编码 z_world ──> JEPAPredictor 预测 z_next
         ↓
   TrinityPlanner.plan:
     MCTS 搜索 × EFE 评估 × MoE 加权 → 最优变爻
         ↓
   五卦链演化 → 下一卦 h'
         ↓
   情节存储 + 在线学习 + Hebbian + JEPA 训练(回写学习态)
```

## 4. 顶层入口

- `TrinityPlanner.plan(h_current, grid, time_phase, target_palace, day_gan)`
  → 返回 `PlanResult`(五卦链、各卦得分、最优变爻、MoE 活跃专家、轨迹)。
- `TrinityPlanner.observe_predict_evaluate_act(...)` → OODA 闭环入口。

## 5. 实现状态(改造后)

> 改造前基线(2026-06-04 只读审计):93 passed,76% cov,8 个 0-引用孤岛。
> 改造后:**107 passed,~86% cov,0 个孤岛**。所有子系统接入 `TrinityAgent`
> 的 OODA 闭环并真实消费。

**OODA 闭环载体:`zwm.planner.agent.TrinityAgent`**
持有全部跨 tick 学习态(感知 / 时间 / 学习器 / 记忆 / 世界模型 / 拓扑),
`TrinityPlanner` 退为无状态单步评估器。

| 子系统 | 改造前 | 改造后 |
|---|---|---|
| `jepa/predictor` | 0% 死代码 | 真 torch JEPA:在线编码器 + **EMA 目标编码器** + 预测器 + **VICReg** 防坍缩,Adam + 梯度裁剪 + NaN 守卫。tick 内真实反向传播(loss 实测 0.186→0.041) |
| `jepa/square_encoder` | 25% | 接入 PREDICT,产出 z_world(64+13=77) |
| `moe/router` | 随机静态矩阵 | 真 torch `nn.Module` 学习门控,`train_toward` 按 reward 强化责任专家 |
| `learning/online` | 孤岛 | preference_weights 反馈进 MoE 评估 + EFE |
| `learning/hebbian` | 孤岛 | suggest_next → MCTS 扩展先验;每 tick update_from_episode |
| `storage/episodic_db` | 孤岛 | 每 tick store;query_similar_vector → 规划先验 |
| `hexaembed/vsa` | 仅自测 | 情节指纹 + VSA 记忆缓冲 + 巩固 |
| `encoder/base` | 孤岛 | OBSERVE 入口:sensor_data → 卦 |
| `scene_field/unified_field` | 孤岛 | 每 tick 世界状态快照 |
| `scene_field/calendar` | 孤岛 | 供给 time_phase |
| `topology/recursive` | 孤岛 | 多尺度宫位脚手架,驱动 EFE 宫位探索 |

**已修断点:**
- EFE `visit_counts` 写死 → 改为 planner 持久 `_visit_counts`,MCTS backprop 累积、跨 tick 存活。
- EFE 宫位/卦象键空间冲突(1-9 重叠)→ 拆分为独立 `palace_visit_counts`。
- `LangevinSampler` 死枝 → `_ordered_masks` 用其评分暖启动 MCTS 扩展。
- `observe_predict_evaluate_act` 空壳 → `TrinityAgent` 真 O-P-E-A-L 闭环;planner 的误导别名已删除。

**评审后加固(代码评审 APPROVE-WITH-FIXES,已全部修复):**
- VICReg 在单样本/tick 下梯度恒为 0(防坍缩形同虚设)→ 引入经验回放 minibatch,VICReg 跨样本方差/协方差项真实生效(实测潜变量 std≈0.40,未坍缩)。
- 感知边界 `RuleBasedEncoder.encode` 缺校验(缺失键静默置 YIN)→ 严格校验:缺键/非有限数 fail-fast。
- `reward` 边界无校验(NaN/越界污染所有学习器)→ `_validate_reward` 拒绝非有限值并 clamp 到 [-1,1]。
- JEPA 预测值 `_z_pred` 计算后丢弃("开销无消费")→ 改为真实消费:计算 world-model surprise(预测潜变量 vs EMA 目标潜变量的误差),实测 0.173→0.067 随训练下降。
- MCTS 先验顺序被 `_ordered_masks` 反转 → 修正,Hebbian/记忆 top 先验最先扩展。
- 宫位访问在规划前自增导致探索 bonus 自我抵消 → 改为规划后自增。
- SQLite 句柄构造期异常泄漏 → torch 组件先于句柄构造 + `__enter__/__exit__` 上下文管理。
- agent 穿透 `planner._moe.router` 私有内部 → planner 暴露公共 `reinforce_expert` / `expert_names`。
- `PlanResult` 事后可变赋值 → 移除,`TickReport.jepa_loss` 单一来源。

## 6. 路线图(已完成)

| 阶段 | 内容 | 状态 |
|---|---|---|
| PR-1 | 修 EFE / Langevin 两个断点 | ✅ |
| PR-2 | OnlineLearner 反馈环 + 可学习 MoE 路由 | ✅ |
| PR-3 | 记忆层接入(EpisodicStore + VSA) | ✅ |
| PR-4 | JEPA 真训练(EMA + VICReg,首次真实梯度) | ✅ |
| PR-5 | Hebbian 先验 + OODA 正名 + TrinityAgent | ✅ |
| PR-6 | torch 依赖 + 文档 + 全绿验证 | ✅ |
| P1-1 | 复频谱 (spectrum/) 接入 PREDICT/EVALUATE | ✅ |
| P1-2 | FAISS 增量更新 (vector_index.ivf_index) | ✅ |
| P1-3 | ReAct 反思日志 (textual chain-of-thought) | ✅ |
| P1-4 | vision↔language contrastive + CLI `zwm`/`zwm-serve` | ✅ |
| P2-1 | OpenTelemetry 兼容 tracing (`zwm.tracing`) | ✅ |
| P2-2 | Prometheus / W&B / MLflow 接线 (`zwm.observability`) | ✅ |
| P3-1 | MCP server (JSON-RPC 2.0 over stdio) | ✅ |
| P3-2 | A2A 多智能体协调 (`zwm.planner.a2a`) | ✅ |
| P3-3 | 文化具身场 (EmbodiedAgent) | ✅ |
| **P4-6** | **TrinityConfig 冻结 dataclass + 拓扑内联** | ✅ |
| **P4-7** | **CLI / API / MCP 表面参数统一** | ✅ |
| **P4-8** | **Constitutional AI 安全护栏** | ✅ |
| **P4-9** | **OpenTelemetry 阶段级追踪** | ✅ |
| **P4-10** | **251 测试全绿** | ✅ |
| **H1** | **OTLP/gRPC Exporter + Grafana 仪表盘** | ✅ |
| **H2** | **MCP Streamable-HTTP (2025-06-18) + capabilities** | ✅ |
| **H3** | **A2A HTTP/JSON 跨进程 transport** | ✅ |
| **H4** | **WebSocket 限流 (TokenBucket + SlidingWindow)** | ✅ |
| **H5** | **Constitutional LLM-as-Judge 规则** | ✅ |
| **M1** | **Multi-GPU FSDP2 集成测试** | ✅ |
| **M2** | **VQ tokens → policy head 混合** | ✅ |
| **M3** | **Checkpoint schema 版本 + 自动迁移** | ✅ |
| **M5** | **OpenInference 协议对齐 (LLM/Agent spans)** | ✅ |
| **L1** | **A2A `agent_card_url` 字段 + well-known 端点** | ✅ |
| **L2** | **CLI Bash/Zsh 补全脚本** | ✅ |
| **L3** | **Surface tracing 配置字段 (TrinityConfig)** | ✅ |
| **L4** | **FAISS IVF 自动调参 (nlist/nprobe)** | ✅ |
| **L5** | **Particle filter systematic resampling** | ✅ |

> 验证基线:**251 passed,~86% cov**(P4-10 收官)。50-tick 端到端演示:JEPA loss
> 0.213→0.104,world-model surprise 0.173→0.067,八宫均衡探索,偏好权重分化。
> 全部 28 个原审计问题 + P1/P2/P3 补全 + P4 配置/安全/追踪 收尾。

## 7. P4 阶段 (配置 / 安全 / 追踪)

> 2026-06-05 收尾阶段。统一了配置面、引入宪法式 AI 护栏和 OpenTelemetry 阶段追踪。

### 7.1 配置统一 — `TrinityConfig` (P4-6)

`TrinityAgent.__init__` 之前用 `self._cfg = {...}` 散字典,加字段要改 3 个入口(CLI/API/MCP)。
现在所有配置收敛到冻结 dataclass:

```python
@dataclass(frozen=True, slots=True)
class TrinityConfig:
    db_path: str = "zwm_episodes.db"
    mcts_iterations: int = 200
    n_particles: int = 16
    use_diffusion: bool = True
    learnable_encoder: bool = True
    hierarchical: bool = False
    use_fsdp2: bool = False
    quantize: str | None = None
    use_trainable_vsa: bool = True
    use_react: bool = True
    topology_max_depth: int = 2       # P4-6: 拓扑深度内联
    grid: "LuoshuGrid | None" = None
    enable_constitution: bool = True  # P4-8: 安全开关
```

便利方法:`to_dict()`、`from_dict()`(未知键静默丢弃)、`as_overrides()`(仅非默认值,用于遥测标签)。
`TrinityAgent` 同时支持 `TrinityConfig` 与旧 kwargs 形式,向后兼容。

### 7.2 表面统一 — CLI / API / MCP (P4-7)

新模块 `zwm.planner.surface` 单一来源:

| 入口 | 构建器 | 备注 |
|---|---|---|
| argparse CLI | `config_to_argparse(parser)` | bool 用 `--name/--no-name` 极性 |
| FastAPI / Pydantic | `build_config_overrides_model()` | 动态构造,新增字段自动出现 |
| MCP JSON-Schema | `config_to_mcp_schema(name, desc)` | bool→boolean, int→integer |

辅助函数:
- `apply_overrides(base, overrides)` — 在已有 config 上增量修改
- `build_config_from_args(args)` — argparse Namespace → `TrinityConfig`
- `build_config_from_mcp_args(args)` — 过滤掉工具特定参数(如 `hex_bits`)

### 7.3 Constitutional AI 护栏 (P4-8)

模块 `zwm.safety.constitution`:

- **6 条默认规则** (BLOCK 严重度):
  1. `finite-numbers` — 递归检查所有数字字段为有限数
  2. `reward-in-range` — `reward ∈ [-1, 1]`
  3. `hex-bits-in-range` — 卦象序数 ∈ [1, 64]
  4. `target-palace-in-range` — 目标宫位 ∈ [1, 9]
  5. `no-self-loop` (WARN) — 避免自循环变异
  6. `efe-bounds` (WARN) — 期望自由能合理范围
- **三级严重度**:`BLOCK` (抛 `ConstitutionalViolation`)、`WARN` (记录不抛)、`INFO` (仅日志)
- **可禁用**:`TrinityConfig(enable_constitution=False)` 关闭(仅研究态)
- **决策历史**:环形缓冲(`history_limit=1024`),可观测最近 N 次判定
- **运行时增删**:`guard.add_rule(rule)` / `guard.remove_rule(name)`
- 入口点:`tick()`、`observe_predict_evaluate_act()`、`SessionStartRequest` 验证

### 7.4 OpenTelemetry 阶段追踪 (P4-9)

模块 `zwm.tracing`:

- **InProcessTracer**(默认):O(1) push 到 ring buffer,无需 OTel SDK
- **OTel 桥接**:若安装 `opentelemetry-api`,span 同时导出到 OTel `TracerProvider`
- **零调用点修改**:`with tracer.start_as_current_span("ooda.observe") as span`
- **每 tick 5 个阶段 span**:
  - `ooda.observe` — target_palace, h_current
  - `ooda.predict` — z_world_dim, surprise
  - `ooda.evaluate` — efe_value, active_experts
  - `ooda.act` — hex_bits_out, surprise
  - `ooda.learn` — episode_id, mutation_class
- **错误自动标记**:异常时 span status="error",属性注入 `exception.type` / `exception.message`
- **辅助**: `get_tracer()`(单例)、`render_recent(n, tracer=None)`(pretty-print)、`configure_otel(service_name)`(绑定 OTel SDK)

### 7.5 收官基线

```
$ python -m pytest --no-cov -q
251 passed, 1 warning in 21.05s
```

| 模块 | 状态 | 关键测试 |
|---|---|---|
| `zwm.planner.surface` | ✅ 全绿 | `test_argparse_to_config_kwargs` / `test_pydantic_model_mirrors_dataclass` |
| `zwm.safety.constitution` | ✅ 全绿 | `test_block_severity_raises` / `test_disabled_constitution_allows_bad_input` |
| `zwm.tracing` | ✅ 全绿 | `test_span_status_marks_errors` / `test_render_recent_pretty` |
| `zwm.cli` | ✅ 全绿 | `test_cli_tick_uses_unified_surface` / `test_cli_eval_requires_checkpoint` |
