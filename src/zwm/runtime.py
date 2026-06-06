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
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
        """与 agent 进行自然语言对话.

        使用 LLM 路由器, 将 agent 的当前状态作为上下文。
        """
        if self._llm_router is None:
            return self._describe_world()

        ss = self._agent.self_state
        tc = getattr(self._agent, "_time_context", None)
        last = self._history[-1] if self._history else None

        context = (
            f"You are ZWM, a world model agent based on I Ching mathematics.\n"
            f"Your self: 日{ss.day_gan}·{ss.self_element}, always at central palace 5.\n"
            f"Six relations: {ss.six_relations}\n"
        )
        if tc:
            context += (
                f"Time: {tc.ganzhi_str}, {tc.solar_term_name}, "
                f"hui={tc.hui_index}, value year hex=#{tc.value_year_hex}\n"
            )
        if last:
            context += (
                f"Last action: {last.next_hexagram}, "
                f"JEPA loss={last.jepa_loss:.4f}, surprise={last.surprise:.3f}\n"
            )

        prompt = f"{context}\nHuman asks: {question}\nRespond in 1-3 sentences as ZWM."
        try:
            resp = self._llm_router._backend.generate(prompt, max_tokens=200)
            return resp.text.strip()
        except Exception:
            return self._describe_world()

    def _describe_world(self) -> str:
        """回退: 用结构化数据描述世界."""
        ss = self._agent.self_state
        tc = getattr(self._agent, "_time_context", None)
        parts = [
            f"我是日{ss.day_gan}·{ss.self_element}, 永远在中宫.",
            f"八方关系: { {k:v for k,v in ss.six_relations.items() if v!='兄弟'} }",
        ]
        if tc:
            parts.append(f"时间: {tc.ganzhi_str}, {tc.solar_term_name}, 午会.")
        if self._history:
            last = self._history[-1]
            parts.append(f"上次: →{last.next_hexagram}, JEPA loss={last.jepa_loss:.4f}.")
        return " | ".join(parts)

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

        解析自然语言指令, 映射到 OODA 行动。
        """
        ss = self._agent.self_state
        inst_lower = instruction.lower()

        # 简单指令映射
        if "北" in instruction or "north" in inst_lower:
            target = 1
        elif "南" in instruction or "south" in inst_lower:
            target = 9
        elif "东" in instruction or "east" in inst_lower:
            target = 3
        elif "西" in instruction or "west" in inst_lower:
            target = 7
        elif "探索" in instruction or "explore" in inst_lower:
            target = ss.next_to_explore()
        elif "停" in instruction or "stop" in inst_lower:
            target = 5
        else:
            target = ss.next_to_explore()

        # 根据目标计算 reward (朝向和谐方向 = 正奖励)
        harmony = ss.harmony_score(target)
        reward = harmony * 0.8 + 0.2

        state = self.tick(reward=reward)
        state.human_message = instruction
        state.agent_reply = (
            f"收到指令: {instruction}. "
            f"我决定去宫{target}({ss.relation_to(target)}, 和谐度{harmony:.1f}). "
            f"执行: →{state.next_hexagram}, JEPA loss={state.jepa_loss:.4f}."
        )
        return state

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
