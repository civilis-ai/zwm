# ZWM — 天地人三才世界模型

**A first-person world model based on I Ching mathematics.** Not just a planner — a self-aware agent that perceives, predicts, plans, acts, and learns in a continuous OODA loop.

```
"我" (Self) 永远在中宫. 八方六亲环绕. JEPA 预测世界变化. MCTS 决定下一步.
LLM 是我的语言, 不是我的大脑. 世界模型是我的身体, 感知是我的眼睛.
```

---

## 核心理念

将 **天地人三才** 易经框架与现代世界模型技术深度融合：

| 易经概念 | 技术映射 | 模块 |
|---------|---------|------|
| 阴阳 ↔ 复相位 | φ∈{0,π}, 6次谐波叠加 | `spectrum/` |
| 方圆图 ↔ JEPA | 方图(GNN) × 圆图(BiMamba) → 联合嵌入 | `jepa/` |
| 洛书九宫 ↔ Active Inference | 中宫5=自我, 八方=EFE探索 | `self_field/` |
| 五行六亲 ↔ 社会场 | 生克动力学 + 以我为中心 | `scene_field/` |
| 60甲子 ↔ 时间场 | 干支周期编码为64卦场 | `scene_field/` |
| 元会运世 ↔ 多尺度时间 | 4层嵌套宇宙周期 | `scene_field/` |
| VSA ↔ 超维记忆 | bind/bundle/permute ↔ 卦象运算 | `hexaembed/` |

---

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                     ZWM 运行时                           │
│                                                         │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐     │
│  │ 感知  │→│ 编码  │→│ 预测  │→│ 规划  │→│ 行动  │     │
│  │Camera │  │Field │  │JEPA  │  │MCTS  │  │Mutation│    │
│  │Sensor │  │Encoder│ │Predict│  │+EFE  │  │384act │     │
│  └──────┘  └──────┘  └──┬───┘  └──────┘  └──────┘     │
│                         │                               │
│                    ┌─────▼─────┐                        │
│                    │   学习     │                        │
│                    │ JEPA训练  │                        │
│                    │ Hebbian   │                        │
│                    │ EWC防遗忘 │                        │
│                    └───────────┘                        │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              多场融合 (Multi-Field)               │  │
│  │  方图(GNN) + 圆图(BiMamba) + 干支(BiMamba)       │  │
│  │  + 元会运世(MLP) → CrossFieldFusion → z_world    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              自我 (SelfState)                     │  │
│  │  "我"永远在中宫(5). 日干决定五行. 六亲固定.       │  │
│  │  LLM 是语言接口, 不是大脑.                        │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 安装

```bash
pip install -e ".[dev]"
```

### 本地部署 (摄像头 + 语音 + LLM)

```bash
# 设置 LLM API key (任选一个)
set DEEPSEEK_API_KEY=sk-xxx
set ANTHROPIC_API_KEY=sk-xxx
set OPENAI_API_KEY=sk-xxx

# 启动
python scripts/deploy_local.py --day-gan 庚
```

### 编程接口

```python
from zwm.runtime import ZWMEngine

engine = ZWMEngine(day_gan="庚")
engine.activate()

# 自主循环
engine.tick()

# 对话 (LLM驱动, 零硬编码)
engine.ask("你是谁?")

# 执行指令
engine.execute("去北方探索")

# 持续学习
engine.learn(steps=100)
```

### CLI

```bash
zwm info          # 框架信息
zwm tick --steps 20 --json  # OODA循环
zwm serve         # REST API
zwm serve-grpc    # gRPC 高性能
zwm batch         # 并发批量
zwm sweep         # 参数扫描
zwm mcp           # MCP JSON-RPC
```

---

## 基准测试

**4 个跨领域真实数据集, 1000 步训练:**

| 数据集 | 领域 | 行数 | Full I Ching Error | Flat Baseline Error | 优势 |
|--------|------|------|--------------------|--------------------| ---- |
| Jena Climate | 环境气候 | 420K | 0.052 | 0.155 | **+203%** |
| ETTm2 | 工业电力 | 70K | 0.051 | 0.132 | **+162%** |
| Exchange Rate | 外汇金融 | 7.6K | 0.049 | 0.135 | **+174%** |
| Random Walk | 随机游走 | 2.0K | 0.051 | 0.095 | **+87%** |

**平均优势: +151%.** 易经结构在所有真实数据上均显著优于 Flat 基线. 在纯随机游走上无负担.

---

## 能力矩阵

| 能力 | 模块 | 状态 |
|------|------|------|
| 自我定位 | `SelfState` — 日干·五行·中宫·六亲 | ✅ |
| 视觉感知 | `ZWMVisionField` — HexViT/ConvHex | ✅ |
| 传感器编码 | `HexagramFieldEncoder` — (64,6) 卦象场 | ✅ |
| 时间感知 | `TimeContext` — 元会运世/值年卦/纳甲/节气 | ✅ |
| 世界预测 | `JEPA` — EMA target + VICReg + StructuredEncoder | ✅ |
| 行动规划 | `MCTS` + `EFE` + `MoE` | ✅ |
| 场级行动 | `FieldMutation` — 384 原子 + 54 区域 | ✅ |
| 持续学习 | `EWC` + `Hebbian` + `OnlineLearner` | ✅ |
| LLM 推理 | `LLMRouter` — DeepSeek/Claude/GPT 多后端 | ✅ |
| 工具使用 | `ReActLoop` — 5 built-in tools | ✅ |
| 多智能体 | `A2A` — 九宫协调 + 共识投票 | ✅ |
| 具身接口 | `ROS2 Bridge` + `Gym Bridge` | ✅ |
| 安全护栏 | `Constitutional AI` + `LLM-as-Judge` | ✅ |
| 可观测性 | `OTel` + `Prometheus` + `Grafana` | ✅ |
| API 接入 | REST / gRPC / MCP / WebSocket / CLI | ✅ |
| K8s 部署 | Helm Chart (deployment/service/hpa/ingress) | ✅ |

---

## 模块结构

```
src/zwm/
├── runtime.py          # ZWMEngine — 统一运行时
├── core/               # 易经原语 (Hexagram/Trigram/Yao)
├── encoder/            # 感知编码
│   ├── field_encoder   # 传感器 → (64,6) 卦象场
│   ├── vision_field    # 图像 → 卦象场 (HexViT/ConvHex)
│   ├── vision_backbone # CLIP/DINOv2/ViT
│   └── multimodal      # 多模态融合
├── jepa/               # 世界模型
│   ├── predictor       # JEPAPredictor (EMA+VICReg+SIGReg)
│   ├── field_gnn       # FieldSquareGNN (8邻域)
│   ├── structured_encoder  # hybrid backend
│   └── square_encoder  # SquareCircularJoint
├── planner/            # 规划
│   ├── agent           # TrinityAgent (OODA)
│   ├── loop            # TrinityPlanner (MCTS+EFE+MoE)
│   ├── react           # ReActLoop (5 tools + LLM)
│   ├── field_mutations # 384 atomic actions
│   └── a2a             # Multi-agent coordination
├── self_field/         # 自我
│   ├── self_state      # SelfState — "我"
│   ├── efe             # Expected Free Energy
│   └── harmony         # 洛书和谐度
├── scene_field/        # 场景
│   ├── time_context    # 元会运世/值年卦/纳甲/节气
│   ├── time_field      # 时间→卦象场
│   ├── liuqin          # 六亲关系
│   └── calendar        # 干支历法
├── llm/                # LLM 集成
│   ├── backends        # DeepSeek/Claude/GPT + fallback
│   ├── router          # 智能路由 + 缓存
│   └── context         # ZWM → LLM prompt
├── learning/           # 学习
│   ├── ewc             # EWC 防灾难遗忘
│   └── online          # OnlineLearner
├── moe/                # Mixture of Experts
├── spectrum/           # 复相位频谱
├── hexaembed/          # VSA 超维记忆
├── storage/            # SQLite + FAISS
├── safety/             # Constitutional AI
├── grpc/               # gRPC server
├── api/                # FastAPI REST
└── embodied/           # ROS2 + Gym
```

---

## 部署

```bash
# K8s
helm install zwm ./k8s/helm/zwm --set gpu.enabled=true

# Docker
docker-compose up
```

---

## 许可证

Apache 2.0 © civilis-ai

---

## 引用

> ZWM: A First-Person World Model Based on I Ching Mathematics.
> JEPA + Multi-Field Encoding + SelfState + Active Inference.
> 4 cross-domain benchmarks: +151% mean advantage over flat baselines.
