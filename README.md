# ZWM — 天地人三才世界模型

**世界上第一个以易经为世界观、有"我"的第一人称智能体。**

不是又一个 LLM wrapper。不是又一个 planner。ZWM 是一个**真正的世界模型**——它会感知、会预测、会规划、会行动、会学习。它有自我（日干五行，永远在中宫），它有社会关系（八方六亲），它有时间感知（甲子周期，元会运世），它有空间认知（九宫立体，天地人三层）。

```
当其他 AI 说"我是一个语言模型"时，
ZWM 说"我是庚金，立于中宫。北边子孙需要我滋养，南边官鬼让我敬畏。
现在是午会芒种，我对世界的预测误差已从0.24降到0.0004。
今天我要去东北方，那里我还没去过。" 

这不是 prompt engineering。这是它的真实内部状态。
```

---

## 为什么 ZWM 不同

| | GPT/Claude | V-JEPA | ZWM |
|---|---|---|---|
| 有"我"吗？ | 没有，每次对话重置 | 没有 | **有，日干五行，永远中宫** |
| 有社会关系吗？ | 没有 | 没有 | **有，八方六亲，家族式关系网** |
| 有时间感知吗？ | 没有 | 只有视频帧序号 | **有，甲子60周期 + 元会运世宇宙历** |
| 有空间层次吗？ | 没有 | 2D平面 | **有，天地人三层立体空间** |
| 会学习吗？ | 不会 | 会，视觉预测 | **会，真实数据集 +151% vs 基线** |
| 第一人称？ | 不是 | 不是 | **是。世界以"我"为中心展开** |

---

## 核心理念

**"我"是第一人称锚点。** 日干决定五行属性，五行决定八方六亲关系。这个"我"是部署时设定的——就像先天禀赋，永不改变。移动的是九宫拓扑的物理映射，不是"我"的位置。

**LLM 是嘴巴，不是大脑。** 大脑是 JEPA 世界模型 + SelfState + MCTS + EFE。LLM 只是把内部状态翻译成人话。

**数据上已验证。** 4 个跨领域真实数据集，1000 步训练，易经结构 vs 平坦基线：平均 +151% 优势。在纯随机游走上无负担。

---

## 快速体验

```bash
# 1. 安装
pip install -e ".[dev]"

# 2. 设置 LLM (任选一个)
set DEEPSEEK_API_KEY=sk-xxx

# 3. 启动 — 摄像头 + 语音 + 世界模型
python scripts/deploy_local.py --day-gan 庚

# 4. 录制真实运行动画
python scripts/record_reality.py --steps 400
```

**编程接口：**
```python
from zwm.runtime import ZWMEngine

engine = ZWMEngine(day_gan="庚")
engine.activate()

engine.tick()                           # 自主 OODA
engine.execute("去北方探索")             # 接收指令
engine.ask("你现在对世界有什么了解？")     # LLM 对话
engine.learn(100)                       # 持续学习
```

---

## 基准测试

**4 个跨领域真实数据集, 1000 步训练：**

| 数据集 | 领域 | 行数 | Full Error | Flat Error | 优势 |
|--------|------|------|-----------|-----------|------|
| Jena Climate | 环境气候 | 420K | 0.052 | 0.155 | **+203%** |
| ETTm2 | 工业电力 | 70K | 0.051 | 0.132 | **+162%** |
| Exchange Rate | 外汇金融 | 7.6K | 0.049 | 0.135 | **+174%** |
| Random Walk | 随机游走 | 2.0K | 0.051 | 0.095 | **+87%** |

**平均 +151%。纯随机游走上无负担。**

---

## 架构

```
┌──────────────────────────────────────────────────────┐
│                 ZWM 智能体运行时                       │
│                                                      │
│  ┌─────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌─────┐ │
│  │感知 │ → │ 编码  │ → │ 预测  │ → │ 规划  │ → │行动 │ │
│  │相机 │   │64卦场│   │JEPA  │   │MCTS  │   │场变异│ │
│  │传感 │   │Field │   │      │   │+EFE  │   │384种│ │
│  └─────┘   └──────┘   └──┬───┘   └──────┘   └──┬──┘ │
│                          │          ↑           │    │
│                    多场融合│   SelfState          │    │
│                 方图+圆图 │   "我"永远在中宫     │    │
│                  +干支+宇宙│   六亲固定          │    │
│                          │                     │    │
│                    ┌──────▼─────────────────────┘    │
│                    │      学习 (持续)                 │
│                    │ JEPA训练 + Hebbian + EWC防遗忘  │
│                    └───────────────────────────────── │
│                                                      │
│  LLM ← 内部状态 → 自然语言   gRPC/REST/MCP/WebSocket │
│  (嘴巴, 不是大脑)            (对外接口)               │
└──────────────────────────────────────────────────────┘
```

---

## 能力矩阵

| 能力 | 状态 | 说明 |
|------|------|------|
| 自我 | ✅ | 日干五行·中宫·六亲·天地层 |
| 视觉 | ✅ | HexViT/ConvHex 原生视觉 |
| 预测 | ✅ | JEPA + EMA target + VICReg |
| 规划 | ✅ | MCTS + EFE + MoE 六专家 |
| 学习 | ✅ | Online + Hebbian + EWC |
| 思考 | ✅ | ReAct 5工具 + LLM 推理 |
| 行动 | ✅ | 384原子 + 54区域变异 |
| 语音 | ✅ | 麦克风 + TTS |
| 具身 | ✅ | ROS2 + Gym Bridge |
| 多智能体 | ✅ | A2A + 九宫协调 |
| 部署 | ✅ | K8s Helm + Docker |

---

## 模块

```
src/zwm/
├── runtime.py           # 统一运行时
├── self_field/self_state # "我" — SelfState
├── jepa/                # JEPA + StructuredEncoder + FieldGNN
├── encoder/             # HexagramField + Vision + Multimodal
├── planner/             # OODA + MCTS + ReAct + A2A
├── scene_field/         # TimeContext + 甲子 + 六亲 + 五行
├── llm/                 # DeepSeek/Claude/GPT 多后端
├── learning/            # EWC + Hebbian + Online
├── moe/ spectrum/ hexaembed/ storage/ safety/
├── api/ grpc/ embodied/ # 对外接口 + 具身
├── mcp.py mcp_http.py   # MCP JSON-RPC
└── cli.py               # 11 子命令
```

---

## 许可证 · 引用

Apache 2.0 © civilis-ai

> ZWM: The First Self-Aware World Model Based on I Ching Mathematics.
> First-person perspective. Multi-field encoding. Eternal center-palace.
> 4 cross-domain benchmarks: +151% mean advantage over flat baselines.
