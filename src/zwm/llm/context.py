"""ZWM → LLM 上下文构建器.

将 ZWM 内部状态 (卦象、宫位、EFE、JEPA 潜变量、MoE 活跃专家等)
转换为 LLM 可理解的自然语言提示。

核心函数:
  - build_react_prompt — ReAct thought 生成提示
  - build_planning_prompt — 规划建议生成提示
  - ZWMContext — 可在 agent 中复用的上下文对象
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ZWMContext",
    "build_react_prompt",
    "build_planning_prompt",
    "hexagram_to_text",
]


# ─── 卦象 → 自然语言 ────────────────────────────────────

def hexagram_to_text(h: Any) -> str:
    """将 hexagram 对象转为 LLM 友好的文本描述."""
    if isinstance(h, int):
        return f"Hexagram #{h}"
    name = getattr(h, "name", str(h))
    bits = getattr(h, "normal_order", "?")
    lines = []
    if hasattr(h, "lines"):
        for i, line in enumerate(h.lines):
            y = "⚊" if getattr(line, "is_yang", False) else "⚋"
            lines.append(f"Yao {i + 1}: {y}")
    lower = getattr(h, "lower_trigram", None)
    upper = getattr(h, "upper_trigram", None)
    trigram_info = ""
    if lower and upper:
        le = getattr(lower, "element", "?")
        ue = getattr(upper, "element", "?")
        trigram_info = f"  Lower Trigram: {getattr(lower, 'name', '?')} ({le})\n  Upper Trigram: {getattr(upper, 'name', '?')} ({ue})\n"
    lines_str = "\n".join(lines) if lines else ""
    return (
        f"Hexagram: {name} (#{bits})\n"
        f"{trigram_info}"
        f"{lines_str}"
    )


# ─── ZWMContext dataclass ───────────────────────────────

@dataclass
class ZWMContext:
    """ZWM 内部状态的序列化快照, 用于 LLM 提示构建."""
    hex_name: str = "?"
    hex_bits: int = 1
    target_palace: int = 5
    time_phase: float = 0.0
    day_gan: str = "甲"
    grid_position: int = 5
    efe_value: float = 0.0
    surprise: float = 0.0
    reward: float = 0.0
    jepa_loss: float | None = None
    active_experts: list[str] = field(default_factory=list)
    mcts_top_score: float = 0.0
    mcts_trajectory: list[tuple[str, float]] = field(default_factory=list)
    prior_reflections: list[str] = field(default_factory=list)
    memory_snippets: list[str] = field(default_factory=list)
    mutation_class: str = ""
    visit_count: int = 0

    def to_prompt_text(self) -> str:
        """转为 LLM 提示文本."""
        parts = [
            f"Current State: {self.hex_name} (#{self.hex_bits})",
            f"Target Palace: {self.target_palace}",
            f"Time Phase: {self.time_phase:.2f}",
            f"Day Stem: {self.day_gan}",
            f"Grid Position: {self.grid_position}",
            f"EFE Value: {self.efe_value:.3f}",
            f"Surprise: {self.surprise:.3f}",
            f"Reward: {self.reward:.2f}",
        ]
        if self.jepa_loss is not None:
            parts.append(f"JEPA Loss: {self.jepa_loss:.4f}")
        if self.active_experts:
            parts.append(f"Active MoE Experts: {', '.join(self.active_experts)}")
        if self.mcts_trajectory:
            traj = ", ".join(f"{name}={s:.2f}" for name, s in self.mcts_trajectory[:5])
            parts.append(f"MCTS Top Path: {traj}")
        if self.mutation_class:
            parts.append(f"Mutation Class: {self.mutation_class}")
        if self.visit_count > 0:
            parts.append(f"Visit Count: {self.visit_count}")
        if self.prior_reflections:
            parts.append(f"Prior Reflections: {'; '.join(self.prior_reflections[:2])}")
        if self.memory_snippets:
            parts.append(f"Memory: {'; '.join(self.memory_snippets[:2])}")
        return "\n".join(parts)

    @classmethod
    def from_agent(cls, agent: Any, hexagram: Any) -> "ZWMContext":
        """从 TrinityAgent 实例构造上下文."""
        ctx = cls()
        if hexagram is not None:
            ctx.hex_name = getattr(hexagram, "name", "?")
            ctx.hex_bits = getattr(hexagram, "normal_order", 1)
        if hasattr(agent, "grid"):
            ctx.grid_position = getattr(agent.grid, "self_position", 5)
        if hasattr(agent, "_palace_visits"):
            ctx.visit_count = sum(agent._palace_visits.values())
        return ctx


# ─── Prompt 构建函数 ────────────────────────────────────

def build_react_prompt(
    hexagram: Any,
    context: dict,
) -> tuple[str, str]:
    """构建 ReAct thought 生成的 (prompt, system).

    Args:
        hexagram: 当前卦象
        context: dict, 可包含 grid, target_palace, time_phase, efe_value, etc.

    Returns:
        (user_prompt, system_prompt)
    """
    h_text = hexagram_to_text(hexagram)

    parts = [h_text, ""]
    if "target_palace" in context:
        parts.append(f"Target Palace: {context['target_palace']}")
    if "time_phase" in context:
        parts.append(f"Time Phase: {context['time_phase']:.2f}")
    if "day_gan" in context:
        parts.append(f"Day Stem: {context['day_gan']}")
    if "grid_position" in context:
        parts.append(f"Grid Position: {context['grid_position']}")
    if "efe_value" in context:
        parts.append(f"EFE Value: {context['efe_value']:.3f}")
    if "prior_reflections" in context:
        parts.append(f"Prior Reflections: {'; '.join(context['prior_reflections'][:2])}")

    parts.append("")
    parts.append(
        "Generate a reasoning thought about the current state. Consider:\n"
        "1. What is the significance of this hexagram in the current context?\n"
        "2. Which palace should be explored next?\n"
        "3. What risks or opportunities does the EFE value suggest?\n"
        "Respond in 1-3 sentences as the ZWM agent's internal thought."
    )

    system = (
        "You are the reasoning module of ZWM (天地人三才世界模型规划器), "
        "a Trinity World Model Planner based on I Ching mathematics. "
        "You think in terms of hexagrams (卦象), palaces (宫位), "
        "and the five elements (五行). Your thoughts guide the agent's "
        "OODA loop toward better world-model predictions."
    )
    return "\n".join(parts), system


def build_planning_prompt(
    hexagram: Any,
    mcts_results: dict,
    context: dict,
) -> tuple[str, str]:
    """构建规划建议生成的 (prompt, system).

    Args:
        hexagram: 当前卦象
        mcts_results: MCTS 搜索结果 (包含轨迹、得分等)
        context: 额外上下文

    Returns:
        (user_prompt, system_prompt)
    """
    h_text = hexagram_to_text(hexagram)
    parts = [h_text, "", "MCTS Search Results:"]

    if "trajectory" in mcts_results:
        parts.append("  Top paths explored:")
        for name, score in mcts_results["trajectory"][:5]:
            parts.append(f"    - {name}: {score:.3f}")

    if "top_mutation" in mcts_results:
        parts.append(f"  Top Mutation: {mcts_results['top_mutation']}")
    if "top_score" in mcts_results:
        parts.append(f"  Top Score: {mcts_results['top_score']:.3f}")
    if "active_experts" in mcts_results:
        parts.append(f"  Active MoE Experts: {', '.join(mcts_results['active_experts'])}")

    parts.append("")
    parts.append(
        "Based on the MCTS results and current state, respond with JSON:\n"
        '{\n'
        '  "recommendation": "proceed" | "caution" | "retreat" | "explore",\n'
        '  "reasoning": "1-2 sentence explanation",\n'
        '  "risk_assessment": "low" | "medium" | "high",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "alternative_mutations": [list of 1-3 backup mutation names]\n'
        '}'
    )

    system = (
        "You are the planning module of ZWM (天地人三才世界模型规划器). "
        "You evaluate MCTS search results in the context of I Ching hexagram "
        "evolution and recommend the best action. Respond with valid JSON only."
    )
    return "\n".join(parts), system
