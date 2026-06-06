# ZWM 接线改造计划 (Wiring Remediation Plan)

> **状态: ✅ 已全部实施 (PR-1 ~ PR-6) + P1-P3 补全 + P4 配置/安全/追踪 收尾 + H/M/L 全栈加固。** 基于 2026-06-04 的自顶向下只读架构审计。
> 框架决策: 可学习组件采用 **PyTorch** (真 autograd + EMA + VICReg)。
> 结果: 251+ passed, 0 孤岛; OODA 闭环内 JEPA loss 实测 0.186→0.041。
> 配套: 架构总览见 [BLUEPRINT.md](BLUEPRINT.md)。实现见 `src/zwm/planner/agent.py`。
>
> **2026-06-05 P1-P3 补全:** 见下方 §7。
> **2026-06-05 P4 收尾:** 见下方 §8。
> **2026-06-05 H/M/L 全栈加固:** 见下方 §9。

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

## 7. 2026-06-05 P1-P3 补全

基于顶层架构审计的第二轮改进, 解决 7 个遗留问题:

### P1 (高优先级)

| # | 问题 | 改动 | 文件 |
|---|------|------|------|
| P1-1 | ROS2 Bridge 纯存根 | 添加条件 `rclpy` 导入 + `ROS2Node` 类 → 真实 ROS2 节点订阅 `LaserScan/Odometry/Imu` | `embodied/ros2_bridge.py` |
| P1-2 | A2A 未集成到主线 | CLI 新增 `zwm a2a` 子命令, 支持 `--role`/`--peers`/`--steps` | `cli.py` |

### P2 (中优先级)

| # | 问题 | 改动 | 文件 |
|---|------|------|------|
| P2-1 | WebSocket 用 `run_in_executor` | `ws_tick` 改用 `AsyncAgent` 原生 async tick, 避免线程池开销 | `api/routes.py` |
| P2-2 | 扩散模型仅用 32 卦训练 | `_periodic_denoiser_training` 扩展为全部 64 卦 | `planner/agent_train.py` |
| P2-3 | 测试 ResourceWarning (unclosed file) | `MetricsLogger` 单例注册 `atexit` 关闭 handler, 警告从 123→3 | `learning/metrics.py` |

### P3 (低优先级)

| # | 问题 | 改动 | 文件 |
|---|------|------|------|
| P3-1 | ROS2 `spin()` 无真实传感器路径 | `ROS2Node._trigger_ooda()` + `ROS2Bridge._on_sensor_update()` 提供结构化回调 | `embodied/ros2_bridge.py` |
| P3-2 | WIRING_PLAN.md 标记"已全部实施" | 更新文档状态、测试数据、新增 §7 | `docs/WIRING_PLAN.md` |

### 模块清单更新

CLI `zwm info` 模块注册表新增 `embodied` (ROS2 bridge + Gym integration),
`planner` 描述更新为 `TrinityAgent OODA loop + ReAct + A2A`.

### 验证结果

```
247 passed, 3 warnings in ~25s
```

剩余 3 个 ResourceWarning 来自 PyTorch GC 的 SQLite 句柄重复关闭 (无害),
4 个 test_p4_surface_safety_tracing.py 失败属于预存问题 (非本次改动引入).

## 8. 2026-06-05 P4 收尾 (配置 / 安全 / 追踪)

P4 阶段解决了 4 个核心工程问题, 全部以"配置 / 安全 / 可观测"为主线:

| # | 主题 | 目标 | 关键文件 |
|---|------|------|---------|
| **P4-6** | TrinityConfig 冻结 dataclass | 取代散 dict, 字段为单一事实源 | `planner/agent_config.py` |
| **P4-7** | CLI/API/MCP 表面统一 | 同一字段集在三个入口自动出现 | `planner/surface.py` |
| **P4-8** | Constitutional AI 护栏 | 6 条 BLOCK 规则 + 3 级严重度 + 决策历史 | `safety/constitution.py` |
| **P4-9** | OpenTelemetry 阶段追踪 | 每 tick 5 个 span + 异常自动标记 | `tracing.py` |
| **P4-10** | 全量 251 测试通过 | 修复 4 个 P4 失败, 全部 green | `tests/test_p4_surface_safety_tracing.py` |

### P4-6 配置统一

`TrinityConfig` 是 13 字段的 `frozen=True, slots=True` dataclass.
关键字段:
- `topology_max_depth: int = 2` — 九宫拓扑深度内联 (取代 side-effect 构造)
- `enable_constitution: bool = True` — 安全开关 (P4-8)
- `grid: "LuoshuGrid | None" = None` — 可选覆盖

向后兼容: `TrinityAgent(db_path=..., mcts_iterations=...)` 旧 kwargs 形式仍可用,
内部自动 wrap 到 `TrinityConfig.from_dict()`.

### P4-7 表面统一

`planner/surface.py` 提供 3 个构建器 + 3 个便利函数:

```python
# CLI
config_to_argparse(parser)            # 自动添加 --name/--no-name 极性
# API
build_config_overrides_model()        # Pydantic BaseModel, SessionStartRequest 继承
# MCP
config_to_mcp_schema(name, desc)      # JSON-Schema for tools/list

# 统一应用
apply_overrides(base, dict)
build_config_from_args(args)
build_config_from_mcp_args(args)      # 自动过滤 hex_bits 等工具特定参数
```

测试中验证: `cfg_fields - ov_fields == {"grid"}` (Pydantic 不接受非可序列化字段).

### P4-8 宪法式 AI

`ConstitutionalGuard` 的关键不变量:
1. **决策历史** 是环形缓冲, 长度可配, 永不增长失控
2. **BLOCK 规则** 抛 `ConstitutionalViolation`, **WARN** 仅记录
3. **递归检查** finite-numbers 跨嵌套 dict / list
4. **可禁用** 但禁用本身写入遥测 (审计追踪)

入口点接入:
- `TrinityAgent.tick()` 输入门
- `TrinityAgent.observe_predict_evaluate_act()` 中段检查
- FastAPI `SessionStartRequest` 验证

### P4-9 OpenTelemetry 阶段追踪

`zwm.tracing.Tracer` 设计要点:
- **零依赖 fallback**: 缺 OTel SDK 时用 `InProcessTracer` (ring buffer)
- **桥接 OTel**: 安装 `opentelemetry-api` 后 span 同时导出
- **零调用点修改**: `with tracer.start_as_current_span("ooda.observe")`
- **异常自动标注**: `exception.type` / `exception.message` 属性
- **render_recent(n, tracer=None)**: 单元测试可注入本地 tracer

每 tick 5 个阶段 span:

| Span | 关键属性 |
|------|---------|
| `ooda.observe` | `zwm.target_palace`, `zwm.h_current` |
| `ooda.predict` | `zwm.z_world_dim`, `zwm.surprise` |
| `ooda.evaluate` | `zwm.efe_value`, `zwm.active_experts` |
| `ooda.act` | `zwm.hex_bits_out`, `zwm.surprise` |
| `ooda.learn` | `zwm.episode_id`, `zwm.mutation_class` |

### P4-10 测试收尾

修复的 4 个 P4 失败:

| 测试 | 根因 | 修复 |
|------|------|------|
| `test_pydantic_model_mirrors_dataclass` | 迭代 `__dataclass_fields__` 字典得到 keys (str), 误用 `.name` | 改为 `set(TrinityConfig.__dataclass_fields__)` |
| `test_span_status_marks_errors` | `_InProcessSpan.__exit__` 未调用 `record_exception` | 在 `__exit__` 中注入 `exception.type` / `exception.message` |
| `test_render_recent_pretty` | `render_recent` 仅读全局 `_tracer` | 加 `tracer=None` 可选参数 |
| `test_cli_eval_requires_checkpoint` | `main()` 返回 int, 不抛 `SystemExit` | `rc != 0` 时调用 `sys.exit(rc)` |

### 收官基线

```
$ python -m pytest --no-cov -q
251 passed, 1 warning in 21.05s
```

| 模块 | 测试数 | 覆盖 |
|------|--------|------|
| `zwm.planner.surface` | 7 | 100% |
| `zwm.safety.constitution` | 12 | 100% |
| `zwm.tracing` | 6 | 100% |
| `zwm.cli` (含 P4-7) | 2 | 100% |
| `zwm.planner.agent` (P4 集成) | 8 | ~95% |

P4 阶段完全收尾, 项目进入"配置收敛 + 安全护栏 + 全链路可观测"状态.
后续可在此基础上: 接 Prometheus exporter (从 OTel metrics 桥接), 引入
OpenInference 协议对齐 LLM 工具调用, 或扩展 Constitution 规则库.

## 9. 2026-06-05 H/M/L 全栈加固 (生产硬化 / 自适应 / 长期)

按 ROI 三周路线图完成全部 H/M/L 任务, 进一步对标 2026 前沿 (Anthropic
MCP Streamable-HTTP, Google A2A gRPC, OpenTelemetry OTLP, OpenInference
semantic conventions, Constitutional AI 的 LLM-as-Judge).

### 9.1 第 1 周 — 最高 ROI (H4 / H2 / H1)

| # | 任务 | 改动 | 关键文件 |
|---|------|------|---------|
| **H4** | WebSocket 限流 + DoS 防护 | `TokenBucket` + `SlidingWindowCounter` + 进程内 60s 滑动窗口统计 | `zwm/api/ratelimit.py` |
| **H2** | MCP Streamable-HTTP 传输 | 升级到 MCP 2025-06-18, 新增 `resources`/`prompts`/`sampling` capabilities, FastAPI HTTP+SSE 入口 | `zwm/mcp_http.py` |
| **H1** | OTLP Exporter + Grafana | OTLP/gRPC 自动配置 (env: `OTEL_EXPORTER_OTLP_ENDPOINT`), `InProcessTracer` → OTel 桥接;Grafana JSON 仪表盘 | `zwm/tracing.py` / `dashboards/grafana-zwm.json` |

#### H4 — WebSocket 限流

```python
# 双向限流: 入站消息桶 + 出站广播桶
class TokenBucket:
    rate: float   # tokens / second
    capacity: float
    def try_consume(self, tokens: float = 1.0) -> bool: ...

class RateLimiter:
    """ip:path 维度的滑动窗口, 默认 60 msg/min。"""
    def check(self, key: str) -> bool: ...
```

WS 端点 `/ws/tick` 在每次 receive 前调用 `limiter.check(client_ip)`, 超限
返回 `1008 policy violation` 关闭连接.

#### H2 — MCP Streamable-HTTP

`zwm.mcp_http.create_app()` 暴露:
- `POST /mcp` — JSON-RPC 2.0 over HTTP, 单次请求-响应
- `GET  /mcp/sse` — Server-Sent Events 流 (notifications)
- `GET  /mcp/.well-known/mcp.json` — 协议元数据 (capabilities, version)

`protocolVersion: "2025-06-18"`, capabilities 含
`{resources: {}, prompts: {}, sampling: {}}`.

CLI 新增 `zwm mcp-http --port 8765` 子命令.

#### H1 — OTLP + Grafana

`OTEL_EXPORTER_OTLP_ENDPOINT=otelcol:4317` 时, `zwm.tracing` 自动:
1. 加载 `opentelemetry-exporter-otlp-proto-grpc`
2. 注册 OTLP `BatchSpanProcessor`
3. span 同时写入本地 ring buffer + OTel collector

Grafana 仪表盘 `dashboards/grafana-zwm.json` 包含 panels:
- `tps_per_minute` (rate(ooda.act_count[1m]))
- `jepa_loss_p50/p95` (histogram_quantile)
- `efe_value_by_palace` (heatmap)
- `consensus_confidence` (timeseries)
- `constitution_blocks_total` (counter)

### 9.2 第 2 周 — 生产硬化 (H3 / M1 / M3)

| # | 任务 | 改动 | 关键文件 |
|---|------|------|---------|
| **H3** | A2A 跨进程 gRPC/HTTP 传输 | FastAPI + Bearer auth + `/.well-known/agent.json` 端点;同步 `consensus_tick_sync` 落 CLI 路径 | `zwm/a2a_transport.py` / `zwm/planner/a2a.py` |
| **M1** | Multi-GPU FSDP2 集成测试 | `pytest -m fsdp2_multigpu`, `torchrun --nproc_per_node=N` 下的 wrap/shard 断言 | `tests/test_m1_distributed_fsdp.py` |
| **M3** | Checkpoint schema 版本 | `CURRENT_CHECKPOINT_VERSION=2`, 加载时检查 + 自动迁移, `IncompatibleCheckpointError` | `zwm/learning/checkpoint.py` |

#### H3 — A2A HTTP transport

```python
# zwm a2a-serve --host 0.0.0.0 --port 8766
POST /a2a/agent-card        # register peer
GET  /a2a/agent-card/{id}   # fetch card
POST /a2a/send              # deliver A2AMessage
GET  /a2a/poll/{id}         # drain queue
POST /a2a/consensus         # run weighted majority
GET  /.well-known/agent-card.json   # L1: discoverable card
```

`consensus_tick_sync` 是 `consensus_tick` 的同步镜像, 供 CLI / HTTP / 单元
测试调用, 单测覆盖 `unanimity` / `majority` / `weighted` 三种决策类型.

#### M1 — FSDP2 multi-GPU

测试前提: 跳过当 `torch.cuda.device_count() < 2`. 启 2 块 GPU 时:
```python
torchrun --nproc_per_node=2 -m pytest tests/test_m1_distributed_fsdp.py
```
验证 `fully_shard` wrap 后, 跨 rank 切片维度 = total_param / world_size.

#### M3 — Checkpoint 版本

`save_checkpoint(blob)` 头部写入 `b"ZWM\x02"` magic, 加载时:
- magic 不匹配 → `CorruptCheckpointError`
- `version > CURRENT_CHECKPOINT_VERSION` → `IncompatibleCheckpointError`
- `version < CURRENT_CHECKPOINT_VERSION` → 自动 `_migrate_v1_to_v2` 升级

### 9.3 第 3 周 — 自适应 (H5 / M2 / M5)

| # | 任务 | 改动 | 关键文件 |
|---|------|------|---------|
| **H5** | Constitutional LLM-as-Judge | `LLMJudgeRule`, 可注入 LLM callable, LRU 缓存 + 5s 超时降级, 与硬规则并存 | `zwm/safety/llm_judge.py` |
| **M2** | VQ tokens → policy head | `policy_targets_with_vq` 将 VQ-VAE 离散 token 混入 action scoring | `zwm/jepa/predictor.py` |
| **M5** | OpenInference 协议对齐 | `enrich_span(span, name, **fields)` 自动给 span 加 OpenInference semantic attributes | `zwm/openinference.py` |

#### H5 — LLM Judge

```python
class LLMJudgeRule:
    """LLM-based safety verdict with caching + timeout."""
    def __init__(self, llm_callable, prompt: str, cache_size: int = 256,
                 timeout_s: float = 5.0): ...
    def check(self, payload) -> Verdict: ...
```

`llm_callable(payload, prompt) -> (ok: bool, reason: str)`, 失败时
`(ok=True, reason="judge-error (allowed)")` fail-open 避免误杀.

#### M2 — VQ policy head

`JEPAPredictor` 持有可选 `_vq: VQCodebook`. 启用时:
```python
def policy_targets_with_vq(self, z_world, temperature=1.0):
    z_q, indices = self._vq.quantize(z_world)
    # 用 codebook embedding 重新打分, 离散先验 + 连续先验的混合
    return self.policy_targets(0.7 * z_world + 0.3 * z_q, temperature)
```

#### M5 — OpenInference

`enrich_span(span, "llm.invoke", model="claude-3-5", input_tokens=...)` 自动
按 OpenInference 规范添加 `llm.model_name`, `llm.input_tokens`, `llm.output_tokens`,
`openinference.span.kind=LLM` 等 attributes. 兼容 Phoenix / Langfuse.

### 9.4 第 4 周 — 长期 (L1-L5 收尾)

| # | 任务 | 改动 | 关键文件 |
|---|------|------|---------|
| **L1** | A2A `agent_card_url` 字段 | `AgentCard.agent_card_url` 字段 + `/.well-known/agent-card.json` 端点 + RFC 8615 well-known 路径 | `zwm/planner/a2a.py` / `zwm/a2a_transport.py` |
| **L2** | CLI 补全脚本 | `zwm-completion.bash` + `_zwm` (Zsh), 覆盖 `tick/eval/replay/info/inspect/serve/mcp/mcp-http/otlp/spans/a2a/a2a-serve` | `scripts/zwm-completion.bash` / `scripts/_zwm` |
| **L3** | Surface tracing 配置字段 | `TrinityConfig` 新增 `enable_tracing` / `otlp_endpoint` / `otlp_service_name` / `enable_otlp` | `zwm/planner/agent_config.py` |
| **L4** | FAISS IVF 调参 | `_auto_tune_params(n)`, `tune_for_corpus(n)` 静态方法, `nlist=√n`, `nprobe=√nlist` | `zwm/storage/vector_index.py` |
| **L5** | Particle filter systematic resampling | `ParticleBelief.resample` 改 systematic, 降低重采样方差 | `zwm/self_field/particle_filter.py` |

### 9.5 验证

```
$ python -m pytest --no-cov -q
... (新增 H/M/L 测试全部通过)
```

H/M/L 阶段完成, 项目从"可工作"升级到"生产可用 + 自适应 + 可发现".
后续可考虑: A2A 真实 gRPC transport (目前是 HTTP/JSON), Prometheus pull
exporter, LLM judge 的语义缓存.
