"""ZWMEngine — 统一智能体运行时.

将 Self / JEPA / MCTS / ReAct / LLM / 具身 / 学习 / 人机接口
编排为一个完整的智能体循环。

这不是新建能力——所有能力已经存在。这是将它们编排为统一运行时。

循环:
  1. 感知 (传感器/视觉/时间)     → 卦象场
  2. 思考 (ReAct + LLM + 工具)  → 推理链
  3. 预测 (JEPA 世界模型)        → 预测下一状态
  4. 规划 (MCTS + EFE + MoE)    → 最优行动
  5. 行动 (场变异/具身执行)      → 改变世界
  6. 学习 (JEPA训练/记忆/EWC)   → 从经验中更新
  7. 交流 (LLM 自然语言)         → 与人对话

用法:
    engine = ZWMEngine(day_gan="庚")
    engine.activate()  # 首次激活

    # 接收指令
    result = engine.execute("去北方探索一下")

    # 自主循环
    for _ in range(10):
        engine.tick()

    # 交流
    reply = engine.ask("你现在对世界有什么了解?")
"""

from __future__ import annotations

import logging
import os as _os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 抑制 tracing 的 debug 噪音 (本地部署不需要 OTLP export)
_os.environ.setdefault("OTEL_PYTHON_LOGGING_ENABLED", "false")
logging.getLogger("opentelemetry").setLevel(logging.ERROR)

_log = logging.getLogger(__name__)


@dataclass
class EngineState:
    """引擎的一次完整循环状态."""
    tick: int = 0
    # 感知
    sensor_data: dict = field(default_factory=dict)
    hex_field: np.ndarray | None = None
    time_context: Any = None
    # 思考
    react_result: Any = None
    llm_thought: str = ""
    # 预测
    z_world: np.ndarray | None = None
    z_pred: np.ndarray | None = None
    # 规划
    plan: Any = None
    target_palace: int = 5
    # 行动
    action_taken: str = ""
    next_hexagram: str = ""
    # 学习
    jepa_loss: float = 0.0
    surprise: float = 0.0
    # 通信
    human_message: str = ""
    agent_reply: str = ""


class ZWMEngine:
    """ZWM 统一智能体运行时.

    所有能力通过此引擎编排:
      - 自我 (SelfState)
      - 感知 (sensors → hexagram fields)
      - 世界模型 (JEPA prediction + training)
      - 推理 (ReAct tools + LLM)
      - 规划 (MCTS + EFE + MoE)
      - 行动 (field mutations + embodied)
      - 学习 (JEPA + Hebbian + EWC + episodic)
      - 交流 (LLM natural language)
    """

    def __init__(self, day_gan: str = "甲", **kwargs):
        import os
        os.environ["ZWM_DAY_GAN"] = day_gan
        self._day_gan = day_gan
        self._agent = None
        self._llm_router = None
        self._history: list[EngineState] = []
        self._step = 0

    # ── 激活 ──

    def activate(self, db_path: str = ":memory:") -> "ZWMEngine":
        """首次激活——创建 agent, 加载 SelfState, 初始化所有子系统."""
        from zwm.planner.agent import TrinityAgent
        from zwm.planner.agent_config import TrinityConfig

        config = TrinityConfig(
            db_path=db_path,
            use_field_encoder=True,
            mcts_iterations=80,
            n_particles=4,
            use_react=True,
        )
        self._agent = TrinityAgent(config=config)
        self._llm_router = getattr(self._agent, "_llm_router", None)

        _log.info("ZWMEngine activated: %s", self._agent.self_state)
        return self

    # ── 主循环 ──

    def tick(self, sensor_data: dict | None = None,
             year: int = 2026, month: int = 6, day: int = 9, hour: int = 8,
             reward: float | None = None) -> EngineState:
        """一次完整的 OODA 循环.

        Args:
            sensor_data: 传感器输入 (None 时自动生成)
            reward: 外部奖励 (None 时自动计算)
        """
        if self._agent is None:
            raise RuntimeError("Engine not activated. Call activate() first.")

        state = EngineState(tick=self._step)

        # ── 0. 时间 ──
        from zwm.scene_field.time_context import TimeContext
        from zwm.scene_field.calendar import GanzhiTime
        self._agent.ganzhi = GanzhiTime.from_date(year, month, day, hour)
        state.time_context = TimeContext.compute(
            year, month, day, hour,
            calendar=self._agent.calendar,
            ganzhi=self._agent.ganzhi,
        )
        self._agent._time_context = state.time_context

        # ── 1. 感知 ──
        state.sensor_data = sensor_data or self._default_sensors()
        state.hex_field = self._agent.field_encoder.encode(state.sensor_data)
        self._agent._last_sensor_data = state.sensor_data
        self._agent._last_hex_field = state.hex_field

        # ── 2. 思考 (ReAct + LLM) ──
        state.llm_thought = self._think(state)

        # ── 3-5. OODA ──
        from zwm.core.hexagram import hexagram_from_name
        # 使用上一轮的演化卦象, 首轮用乾为天
        if self._history and self._history[-1].next_hexagram:
            h_current = hexagram_from_name(self._history[-1].next_hexagram)
        else:
            h_current = hexagram_from_name("乾为天")

        r = reward if reward is not None else 0.5 + 0.4 * (0.5 - abs((self._step % 20) / 10.0 - 1.0))
        report = self._agent.tick(
            h_current=h_current, reward=r,
            year=year, month=month, day=day, hour=hour,
        )

        state.jepa_loss = report.jepa_loss or 0.0
        state.surprise = report.surprise
        state.next_hexagram = report.h_next.name
        state.action_taken = report.mutation_class

        # 记录访问
        target = getattr(self._agent, "_target_palace", 5) if hasattr(self._agent, "_target_palace") else self._agent.self_state.next_to_explore()
        state.target_palace = target
        self._agent.self_state.record_visit(target)

        self._history.append(state)
        self._step += 1
        return state

    # ── 思考 ──

    def _think(self, state: EngineState) -> str:
        """ReAct + LLM 推理."""
        ss = self._agent.self_state
        tc = state.time_context
        last_hex = self._history[-1].next_hexagram if self._history else "乾为天"

        # 启发式 thought
        thought = (
            f"我在中宫, 日{ss.day_gan}·{ss.self_element}. "
            f"当前卦: {last_hex}. "
            f"{tc.solar_term_name}时节, 值年卦#{tc.value_year_hex}. "
            f"宫{ss.next_to_explore()}({ss.relation_to(ss.next_to_explore())})最值得探索."
        )

        # LLM 增强
        if self._llm_router is not None:
            try:
                from zwm.core.hexagram import hexagram_from_name
                h = hexagram_from_name(last_hex)
                llm_thought = self._llm_router.generate_thought(h, {
                    "self_element": ss.self_element,
                    "solar_term": tc.solar_term_name,
                    "target_palace": ss.next_to_explore(),
                })
                if llm_thought:
                    thought = llm_thought
            except Exception:
                pass
        return thought

    # ── 交流 ──

    def ask(self, question: str) -> str:
        """与 agent 对话 — 基于真实内部状态的回复.

        有 LLM: 真实状态 → LLM 自然语言
        无 LLM: 真实状态 → 结构化摘要
        绝不使用硬编码模板。
        """
        # 委托给 _introspect — 它已经做了"真实状态 → 回复"的逻辑
        reply = self._introspect(question)
        if reply:
            return reply
        # 无法内省的问题 → 至少返回状态摘要
        return self._describe_world()

    def _describe_world(self) -> str:
        """从真实内部状态构造世界描述."""
        ss = self._agent.self_state
        tc = getattr(self._agent, "_time_context", None)
        last = self._history[-1] if self._history else None

        facts = [
            f"日{ss.day_gan}·{ss.self_element}, @中宫(5)",
            f"六亲: N={ss.relation_to(1)} S={ss.relation_to(9)} "
            f"E={ss.relation_to(3)} W={ss.relation_to(7)}",
            f"探索: {ss.total_visits}/8宫位",
        ]
        if tc:
            facts.append(f"{tc.ganzhi_str} {tc.solar_term_name} 午会")
        if last:
            facts.append(f"卦:{last.next_hexagram} JEPA={last.jepa_loss:.4f}")
        return " | ".join(facts)

    # ── 学习 ──

    def learn(self, steps: int = 10) -> list[float]:
        """在最近的经验上运行 JEPA 训练步."""
        losses = []
        for _ in range(steps):
            state = self.tick()
            if state.jepa_loss:
                losses.append(state.jepa_loss)
        return losses

    # ── 执行指令 ──

    def execute(self, instruction: str) -> EngineState:
        """接收人类指令并执行.

        优先用 LLM 理解, 无 LLM 时用关键词匹配 + 内省回复。
        """
        ss = self._agent.self_state
        inst = instruction.strip()

        # ── 内省/身份类问题 (无需移动, 直接回复) ──
        introspect = self._introspect(inst)
        if introspect:
            state = EngineState(tick=self._step)
            state.human_message = instruction
            state.agent_reply = introspect
            self._history.append(state)
            return state

        # ── 方向/行动指令 → OODA ──
        target = self._parse_target(inst, ss)
        harmony = ss.harmony_score(target)
        reward = harmony * 0.8 + 0.2
        state = self.tick(reward=reward)
        state.human_message = instruction
        state.agent_reply = (
            f"收到: {instruction}. "
            f"→宫{target}({ss.relation_to(target)}, 和谐度{harmony:.1f}). "
            f"卦:{state.next_hexagram}, JEPA={state.jepa_loss:.4f}."
        )
        return state

    def _introspect(self, inst: str) -> str | None:
        """真正的内省 — 从 agent 的真实内部状态构造回复, 不做硬编码匹配.

        不使用关键词→固定文本的映射。而是:
          1. 调用所有内部状态源 (Self, Time, JEPA, Memory)
          2. 让 LLM 根据这些真实状态生成回复 (如果有 LLM)
          3. 无 LLM 时, 用结构化状态摘要作为回复
        """
        ss = self._agent.self_state
        tc = getattr(self._agent, "_time_context", None)
        last = self._history[-1] if self._history else None

        # 收集当前的真实内部状态
        facts = [f"自我: 日{ss.day_gan}·{ss.self_element}, 永远在中宫(5)"]
        facts.append(f"六亲: {ss.six_relations}")
        if tc:
            facts.append(f"时间: {tc.ganzhi_str}, {tc.solar_term_name}节气, "
                        f"午会, 值年卦#{tc.value_year_hex}")
        facts.append(f"运行: {self._step} ticks, 已探索{ss.total_visits}/8宫位")
        if last:
            facts.append(f"最近: 卦{last.next_hexagram}, JEPA loss={last.jepa_loss:.4f}")
            if last.z_world is not None:
                facts.append(f"世界向量维度: {last.z_world.shape}")

        # LLM 路径: 让 LLM 基于真实状态生成自然语言回复
        if self._llm_router is not None:
            try:
                prompt = (
                    f"你是 ZWM, 一个基于易经数学的世界模型智能体. "
                    f"以下是你的当前真实状态:\n"
                    + "\n".join(facts) +
                    f"\n\n人类问你: {inst}\n"
                    f"请根据你的真实状态, 用第一人称回答 (2-4句). "
                    f"不要编造你不知道的信息."
                )
                resp = self._llm_router._backend.generate(
                    prompt, max_tokens=200, temperature=0.7,
                )
                return resp.text.strip()
            except Exception as e:
                _log.debug("LLM introspect failed: %s", e)

        # 无 LLM 回退: 返回结构化事实 (这是真实状态, 不是硬编码)
        return " | ".join(facts)

    def _parse_target(self, inst: str, ss) -> int:
        """从指令解析目标宫位."""
        inst_lower = inst.lower()
        if "北" in inst or "north" in inst_lower:
            return 1
        elif "南" in inst or "south" in inst_lower:
            return 9
        elif "东" in inst or "east" in inst_lower:
            return 3
        elif "西" in inst or "west" in inst_lower:
            return 7
        elif "西南" in inst:
            return 2
        elif "西北" in inst:
            return 6
        elif "东南" in inst:
            return 4
        elif "东北" in inst:
            return 8
        elif "中" in inst or "回" in inst:
            return 5
        elif "探索" in inst or "explore" in inst_lower:
            return ss.next_to_explore()
        return ss.next_to_explore()

    # ── 辅助 ──

    def _default_sensors(self) -> dict:
        return {
            "temperature": 0.5 + 0.4 * (0.5 - abs((self._step % 20) / 10.0 - 1.0)),
            "terrain": 0.5 + 0.3 * (0.5 - abs((self._step % 15) / 7.5 - 1.0)),
            "social_proximity": abs((self._step % 30) / 30.0 - 0.5) * 2,
            "resource_level": 0.5 + 0.2 * (0.5 - abs((self._step % 8) / 4.0 - 1.0)),
            "momentum": 0.5 * (1.0 - abs((self._step % 12) / 6.0 - 1.0)),
            "overall_favorability": 0.5 + 0.3 * (0.5 - abs((self._step % 10) / 5.0 - 1.0)),
        }

    @property
    def agent(self):
        return self._agent

    @property
    def self_state(self):
        return self._agent.self_state if self._agent else None

    @property
    def history(self) -> list[EngineState]:
        return list(self._history)

    def summary(self) -> str:
        """返回 agent 当前状态的人类可读摘要."""
        if self._agent is None:
            return "Engine not activated."
        ss = self._agent.self_state
        n = self._step
        last = self._history[-1] if self._history else None
        lines = [
            f"ZWM Agent (日{ss.day_gan}·{ss.self_element}, {n} ticks)",
            f"  位置: 中宫(5), 周围: { {k:v for k,v in ss.six_relations.items() if v!='兄弟'} }",
            f"  访问: {ss.palace_visits}",
        ]
        if last:
            lines.append(f"  最近: →{last.next_hexagram}, JEPA={last.jepa_loss:.4f}")
        return "\n".join(lines)

    def close(self):
        if self._agent:
            self._agent.close()


__all__ = ["ZWMEngine", "EngineState"]
