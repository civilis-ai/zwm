# ZWM — 天地人三才世界模型规划器

**Trinity World Model Planner** based on I Ching mathematics, complex analysis, and embodied cognition.

## 核心理念

将 **天地人三才** 易经框架与 2026 SOTA 技术深度融合：

- **复相位编码**: 阴阳↔φ∈{0,π}, 卦象↔6次谐波叠加 F(t)=ΣAₙe^(i(nωt+φₙ))
- **方圆图 JEPA**: 方图=空间频谱 × 圆图=时间相位 → 联合嵌入世界状态
- **洛书九宫 Active Inference**: 中宫5=自我锚点, 八方=频率滤波器组
- **五卦叙事规划**: 主卦→互卦→变卦→综卦→错卦 构成完整叙事弧
- **VSA 超维计算**: bind/bundle/permute 精确对应卦象运算
- **五行六亲社会场**: 生克动力学 + 以我为中心的社会关系图
- **四维自我定位**: 空间(中宫)+属性(五行)+关系(六亲)+态势(世爻)

## 安装

```bash
pip install -e ".[dev]"
```

## 运行测试

```bash
pytest tests/ -v
```

## 许可证

Apache 2.0 © civilis-ai

## 引用

详见 [docs/BLUEPRINT.md](docs/BLUEPRINT.md)
