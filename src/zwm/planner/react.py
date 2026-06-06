"""ReAct / Tool-Use Agent architecture (2026 SOTA).

Implements the ReAct (Reasoning + Acting) loop for the ZWM agent.
The 2026 SOTA for autonomous agents is a **tool-augmented reasoning
loop** where the LLM/world-model can:

1. **Reason** — generate a thought about the current state
2. **Act** — invoke a tool (search, calculate, query memory, etc.)
3. **Observe** — process the tool's output and update beliefs

This replaces the pure OODA loop with a richer *reason-act-observe*
cycle that supports multi-step reasoning chains, tool composition,
and external API calls.

Key design:
  * ``Tool`` — base class for all tools (search, memory, calculator, etc.)
  * ``ReActLoop`` — the reasoning-acting loop that selects and invokes tools
  * Built-in tools: MemoryQuery, HarmonyCalculator, RiskAssessor,
    TopologyExplorer, TimePhaseCalculator
  * The loop integrates with ``TrinityAgent`` via ``react_tick()``

P0 — LLM 推理集成:
  * ``ReActLoop`` 现在接受可选的 ``llm_router`` (从 ``zwm.llm`` 模块)
  * 当注入 LLM 路由器时, thought 生成使用真实 LLM 推理
  * 当没有 LLM 路由器时, 回退到原有的启发式模板
  * 支持多提供商: Claude / GPT / DeepSeek
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from zwm.core.hexagram import Hexagram
    from zwm.self_field.palace_graph import LuoshuGrid
    from zwm.llm.router import LLMRouter

_log = logging.getLogger(__name__)


# ================= Tool Protocol =================

@dataclass
class ToolResult:
    """Result from a tool invocation."""
    tool_name: str
    output: str
    score: float = 0.0  # relevance / confidence score
    data: dict | None = None  # structured output


class Tool:
    """Base class for ReAct tools."""

    name: str = "base"
    description: str = "Base tool (override in subclass)"

    def run(self, query: str, context: dict) -> ToolResult:
        raise NotImplementedError


# ================= Built-in Tools =================

class MemoryQueryTool(Tool):
    """Query the episodic memory for similar past experiences."""

    name = "memory_query"
    description = "Search episodic memory for similar hexagram outcomes"

    def __init__(self, agent) -> None:
        self._agent = agent

    def run(self, query: str, context: dict) -> ToolResult:
        h = context.get("hexagram")
        if h is None:
            return ToolResult(self.name, "No hexagram in context", 0.0)
        try:
            query_vec = self._agent.vsa.encode_hexagram(h.normal_order)
            similar = self._agent.store.query_similar_vector(
                query_vec.astype(np.float32), limit=5,
            )
            if not similar:
                return ToolResult(self.name, "No similar episodes found", 0.1)
            outcomes = [ep.get("outcome", "?") for ep in similar]
            avg_reward = sum(
                ep.get("reward", 0.0) for ep in similar
            ) / len(similar)
            desc = f"Found {len(similar)} similar episodes: {outcomes}, avg_reward={avg_reward:.2f}"
            return ToolResult(self.name, desc, min(avg_reward, 1.0), {"episodes": similar})
        except Exception as exc:
            return ToolResult(self.name, f"Memory query failed: {exc}", 0.0)


class HarmonyCalculatorTool(Tool):
    """Calculate the luoshu harmony score for a hexagram-palace pair."""

    name = "harmony_calculator"
    description = "Calculate luoshu harmony between a hexagram and target palace"

    def run(self, query: str, context: dict) -> ToolResult:
        h = context.get("hexagram")
        grid = context.get("grid")
        target = context.get("target_palace", 5)
        if h is None or grid is None:
            return ToolResult(self.name, "Missing hexagram or grid", 0.0)
        try:
            from zwm.self_field.harmony import luoshu_harmony
            harmony = luoshu_harmony(h, grid, target)
            desc = f"Harmony score for palace {target}: {harmony:.3f}"
            return ToolResult(self.name, desc, float(harmony), {"harmony": harmony})
        except Exception as exc:
            return ToolResult(self.name, f"Harmony calc failed: {exc}", 0.0)


class RiskAssessorTool(Tool):
    """Assess the risk level of a hexagram using the inter-hexagram analysis."""

    name = "risk_assessor"
    description = "Assess risk via inter-hexagram (互卦) analysis"

    def run(self, query: str, context: dict) -> ToolResult:
        h = context.get("hexagram")
        if h is None:
            return ToolResult(self.name, "No hexagram in context", 0.0)
        try:
            from zwm.moe.experts import risk_expert
            risk = risk_expert(h)
            desc = f"Risk level: {risk:.3f} ({'HIGH' if risk > 0.6 else 'MEDIUM' if risk > 0.3 else 'LOW'})"
            return ToolResult(self.name, desc, float(risk), {"risk": risk})
        except Exception as exc:
            return ToolResult(self.name, f"Risk assessment failed: {exc}", 0.0)


class TopologyExplorerTool(Tool):
    """Explore the recursive topology for unexplored palaces."""

    name = "topology_explorer"
    description = "Find least-visited palaces in the recursive topology"

    def run(self, query: str, context: dict) -> ToolResult:
        agent = context.get("agent")
        if agent is None:
            return ToolResult(self.name, "No agent in context", 0.0)
        try:
            from zwm.planner.agent_priors import _next_palace_to_explore
            grid = context.get("grid", agent.grid)
            next_palace = _next_palace_to_explore(agent, grid)
            visits = agent._palace_visits.get(next_palace, 0)
            desc = f"Next unexplored palace: {next_palace} (visits: {visits})"
            return ToolResult(
                self.name, desc,
                1.0 / (1.0 + visits),
                {"next_palace": next_palace, "visits": visits},
            )
        except Exception as exc:
            return ToolResult(self.name, f"Topology exploration failed: {exc}", 0.0)


class TimePhaseCalculatorTool(Tool):
    """Calculate the current time phase and calendar context."""

    name = "time_phase"
    description = "Calculate time phase and calendar context for the current moment"

    def run(self, query: str, context: dict) -> ToolResult:
        agent = context.get("agent")
        if agent is None:
            return ToolResult(self.name, "No agent in context", 0.0)
        try:
            ganzhi = agent.ganzhi
            cosmic = getattr(agent, "_cosmic_phases", {})
            desc = (
                f"Ganzhi: year={ganzhi.year_index}, "
                f"month={ganzhi.month_index}, "
                f"day={ganzhi.day_index}, "
                f"hour={ganzhi.hour_index}"
            )
            if cosmic:
                desc += f", cosmic_phases={cosmic}"
            return ToolResult(self.name, desc, 0.5, {"ganzhi": ganzhi, "cosmic": cosmic})
        except Exception as exc:
            return ToolResult(self.name, f"Time phase calc failed: {exc}", 0.0)


# ================= ReAct Loop =================

@dataclass
class ReActStep:
    """One step in a ReAct reasoning chain."""
    thought: str
    tool_name: str | None = None
    tool_input: str = ""
    observation: str = ""
    score: float = 0.0


@dataclass
class ReActResult:
    """Result of a complete ReAct reasoning chain."""
    steps: list[ReActStep] = field(default_factory=list)
    final_thought: str = ""
    recommendation: str = ""
    confidence: float = 0.0
    tool_scores: dict[str, float] = field(default_factory=dict)


class ReActLoop:
    """ReAct (Reasoning + Acting) loop for the ZWM agent.

    The 2026 SOTA for autonomous agents is a tool-augmented reasoning
    loop.  This implementation:

    1. Generates a *thought* about the current state (using the JEPA
       world model's prediction and the analytical experts).
    2. Selects a *tool* to invoke (memory search, harmony calculation,
       risk assessment, topology exploration, time phase).
    3. *Observes* the tool output and updates the reasoning chain.
    4. Repeats for up to ``max_steps`` iterations.
    5. Produces a final recommendation with confidence score.

    The loop is called from ``TrinityAgent.react_tick()`` and its
    output is fed back into the OODA planner as additional priors.
    """

    def __init__(
        self,
        agent,
        max_steps: int = 3,
        tool_timeout: float = 1.0,
        llm_router: "LLMRouter | None" = None,
    ) -> None:
        self._agent = agent
        self._max_steps = max_steps
        self._tool_timeout = tool_timeout
        # P0: LLM 路由器 — None 时使用启发式模板, 注入后使用真实 LLM 推理
        self._llm = llm_router
        # Register built-in tools.
        self._tools: dict[str, Tool] = {
            "memory_query": MemoryQueryTool(agent),
            "harmony_calculator": HarmonyCalculatorTool(),
            "risk_assessor": RiskAssessorTool(),
            "topology_explorer": TopologyExplorerTool(),
            "time_phase": TimePhaseCalculatorTool(),
        }
        # Tool selection policy: score each tool by relevance to the
        # current context, then pick the highest-scoring untried tool.
        self._tool_policy: dict[str, Callable] = {
            "memory_query": self._score_memory_query,
            "harmony_calculator": self._score_harmony,
            "risk_assessor": self._score_risk,
            "topology_explorer": self._score_topology,
            "time_phase": self._score_time,
        }

    @property
    def has_llm(self) -> bool:
        """P0: 是否已注入 LLM 推理后端."""
        return self._llm is not None

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def register_tool(self, tool: Tool, score_fn: Callable | None = None) -> None:
        """Register a custom tool (for extensibility).

        ``score_fn`` is an optional callable ``(context: dict) -> float``
        that returns a relevance score for the tool in the current context.
        When omitted, the tool gets a default score of 0.5 (neutral) and
        participates in the tool selection loop.
        """
        self._tools[tool.name] = tool
        if score_fn is not None:
            self._tool_policy[tool.name] = score_fn

    def run(
        self,
        hexagram: "Hexagram",
        grid: "LuoshuGrid",
        target_palace: int = 5,
        time_phase: float = 0.0,
    ) -> ReActResult:
        """Execute the ReAct loop and return a reasoning chain."""
        context: dict = {
            "hexagram": hexagram,
            "grid": grid,
            "target_palace": target_palace,
            "time_phase": time_phase,
            "agent": self._agent,
        }
        result = ReActResult()
        used_tools: set[str] = set()

        for step_idx in range(self._max_steps):
            # Step 1: Generate a thought about the current state.
            thought = self._generate_thought(context, result.steps)
            step = ReActStep(thought=thought)

            # Step 2: Select the most relevant untried tool.
            tool_name = self._select_tool(context, used_tools)
            if tool_name is None:
                # All tools tried or none relevant — conclude.
                step.observation = "No more relevant tools."
                result.steps.append(step)
                break

            step.tool_name = tool_name
            step.tool_input = thought  # use the thought as the query
            used_tools.add(tool_name)

            # Step 3: Invoke the tool.
            tool = self._tools[tool_name]
            tool_result = tool.run(thought, context)
            step.observation = tool_result.output
            step.score = tool_result.score

            # Step 4: Update context with tool output.
            if tool_result.data:
                context.update(tool_result.data)
            result.tool_scores[tool_name] = tool_result.score

            result.steps.append(step)

            # AUDIT-F4 (Reflexion): generate a verbal self-critique
            # after each tool call.  The score-based template below is
            # the 2026 SOTA pattern (Shinn et al. 2023, "Reflexion"):
            # low-score observations are explicitly tagged so the
            # next iteration can re-think the action.  The critique
            # text is appended to the step but does not block — it's
            # a textual signal consumed downstream by the LLM-based
            # reflection synthesis (when one is plugged in).
            if tool_result.score is not None and tool_result.score < 0.4:
                step.observation += " [self-critique: low-score observation — reconsider]"

        # Final synthesis.
        result.final_thought = self._synthesize(result.steps)
        result.recommendation = self._make_recommendation(context, result)
        result.confidence = self._compute_confidence(result)
        return result

    def _generate_thought(
        self,
        context: dict,
        prev_steps: list[ReActStep],
    ) -> str:
        """Generate a reasoning thought about the current state.

        P0 — LLM 集成: 当 ``self._llm`` 可用时, 使用真实 LLM
        生成 thought; 否则回退到原有的启发式模板。

        AUDIT-I8 / F4 (Reflexion): the thought now incorporates the
        most-recent ReAct reflections stored in the agent's
        episodic DB.  Without this, every tick was reasoning
        *tabula rasa* — the agent had no idea what it had previously
        concluded about similar states, and the ``store_react_reflection``
        writes were never consumed.
        """
        h = context.get("hexagram")
        target = context.get("target_palace", 5)

        # P0 — 使用 LLM 生成 thought
        if self._llm is not None and h is not None:
            try:
                # 收集先前的反思
                prior = []
                if self._agent is not None and getattr(self._agent, "store", None) is not None:
                    try:
                        reflections = self._agent.store.query_react_reflections(limit=2)
                        for r in reflections:
                            t = str(r.get("thought", "")).strip()
                            if t:
                                prior.append(t[:120])
                    except Exception as exc:
                        logging.getLogger(__name__).debug("ReAct reflection recall failed: %s", exc)
                # 丰富上下文
                llm_context = dict(context)
                llm_context["prior_reflections"] = prior
                if prev_steps:
                    llm_context["last_observation"] = prev_steps[-1].observation[:120]

                thought = self._llm.generate_thought(h, llm_context)
                if thought:
                    return thought
            except Exception as exc:
                _log.warning("LLM thought generation failed: %s; falling back to heuristic", exc)
                # 失败后回退到启发式

        # ─── 启发式回退 (原有逻辑) ───
        grid = context.get("grid")

        parts = [f"Hexagram {h.normal_order if h else '?'} at palace {target}"]
        if grid:
            parts.append(f"position={grid.self_position}")
        if prev_steps:
            parts.append(f"previous_observation={prev_steps[-1].observation[:80]}")

        if self._agent is not None and getattr(self._agent, "store", None) is not None:
            try:
                if h is not None:
                    reflections = self._agent.store.query_react_reflections(limit=3)
                else:
                    reflections = self._agent.store.query_react_reflections(limit=3)
                if reflections:
                    prior = reflections[0]
                    thought_txt = str(prior.get("thought", "")).strip()
                    rec = str(prior.get("recommendation", "")).strip()
                    if thought_txt:
                        parts.append(f"prior_thought={thought_txt[:120]}")
                    if rec:
                        parts.append(f"prior_recommendation={rec}")
            except Exception as exc:
                import logging
                logging.getLogger(__name__).debug(
                    "ReAct reflection recall failed: %s", exc,
                )
        return "; ".join(parts)

    def _select_tool(
        self,
        context: dict,
        used_tools: set[str],
    ) -> str | None:
        """Select the highest-scoring untried tool.

        Tools with a registered scoring policy use that function;
        tools without one get a default neutral score of 0.5.
        """
        scores: list[tuple[str, float]] = []
        for name in self._tools:
            if name in used_tools:
                continue
            if name in self._tool_policy:
                score = self._tool_policy[name](context)
            else:
                score = 0.5  # neutral default for custom tools
            scores.append((name, score))
        if not scores:
            return None
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    def _score_memory_query(self, context: dict) -> float:
        """Score memory_query tool relevance — higher when visits are low."""
        h = context.get("hexagram")
        if h is None:
            return 0.0
        visits = self._agent.learner.get_visit_count(h)
        # Exponential decay: very relevant for unseen states, less for familiar.
        return min(1.0, 2.0 / (1.0 + visits))

    def _score_harmony(self, context: dict) -> float:
        """Score harmony_calculator tool relevance — higher when target palace is set."""
        target = context.get("target_palace")
        if target is None:
            return 0.3
        # More relevant for central palaces (5) and edge palaces (1,9).
        dist_from_center = abs(target - 5)
        return 0.5 + 0.1 * (4 - dist_from_center) / 4

    def _score_risk(self, context: dict) -> float:
        """Score risk_assessor tool relevance — higher for complex hexagrams."""
        h = context.get("hexagram")
        if h is None:
            return 0.0
        # Count yang lines — more yang = more active = higher risk relevance.
        yang_count = sum(1 for line in h.lines if line.is_yang)
        return 0.3 + 0.1 * yang_count / 6

    def _score_topology(self, context: dict) -> float:
        """Score topology_explorer tool relevance — higher early in exploration."""
        visits = sum(self._agent._palace_visits.values())
        # Exponential decay: very relevant early, less as exploration matures.
        return max(0.1, math.exp(-visits / 50.0))

    def _score_time(self, context: dict) -> float:
        """Score time_phase tool relevance — moderate baseline."""
        return 0.4

    def _synthesize(self, steps: list[ReActStep]) -> str:
        """Synthesize the reasoning chain into a final thought.

        P0 — 当 LLM 可用时, 让 LLM 综合整个推理链; 否则拼接观察。
        """
        if not steps:
            return "No reasoning steps taken."
        # P0: LLM-based synthesis
        if self._llm is not None:
            try:
                reflection = self._llm.generate_reflection(steps)
                if reflection:
                    return reflection
            except Exception as exc:
                _log.debug("LLM synthesis failed: %s", exc)
        observations = [s.observation for s in steps if s.observation]
        return " | ".join(observations)

    def _make_recommendation(self, context: dict, result: ReActResult) -> str:
        """Generate a recommendation based on the reasoning chain.

        P0 — LLM 注入时, 使用 LLM 生成建议; 否则使用启发式规则。
        """
        scores = result.tool_scores
        if not scores:
            return "proceed_with_current_plan"

        # P0: LLM-based recommendation
        if self._llm is not None:
            try:
                resp = self._llm.classify_outcome(
                    context.get("hexagram"),
                    reward=scores.get("risk_assessor", 0.5),
                    surprise=1.0 - scores.get("harmony_calculator", 0.5),
                )
                rec = resp.get("outcome", "")
                if rec == "吉":
                    return "proceed_high_harmony"
                elif rec == "凶":
                    return "caution_high_risk"
                elif rec in ("悔", "吝"):
                    return "explore_cautiously"
            except Exception as exc:
                _log.debug("LLM recommendation failed: %s", exc)

        # 启发式回退
        risk = scores.get("risk_assessor", 0.0)
        if risk > 0.6:
            return "caution_high_risk"
        harmony = scores.get("harmony_calculator", 0.0)
        if harmony > 0.5:
            return "proceed_high_harmony"
        memory = scores.get("memory_query", 0.0)
        if memory > 0.5:
            return "follow_precedent"
        return "explore_cautiously"

    def _compute_confidence(self, result: ReActResult) -> float:
        """Compute confidence from tool scores."""
        if not result.tool_scores:
            return 0.3
        return float(sum(result.tool_scores.values()) / len(result.tool_scores))

    def self_reflect(self, result: ReActResult) -> str:
        """AUDIT-F4 + P0: verbal self-reflection after a ReAct run.

        P0 — 当 LLM 路由器可用时, 使用 LLM 生成丰富的自我反思。
        否则回退到启发式模板。

        Implements the "verbal" half of the 2026 SOTA Reflexion
        pattern (Shinn et al. 2023).  Returns a short textual
        critique that the host agent (or, in production, an LLM)
        can use as the next step's prompt prefix.

        The result is *also* persisted to the episodic store by
        the calling dispatcher (see ``agent_phases._evaluate``),
        so this method stays side-effect-free and testable.
        """
        if not result.tool_scores:
            return "no_tools_evaluated"

        # P0: LLM-based reflection
        if self._llm is not None and result.steps:
            try:
                reflection = self._llm.generate_reflection(result.steps)
                if reflection:
                    return reflection
            except Exception as exc:
                _log.debug("LLM self_reflect failed: %s", exc)

        # 启发式回退
        worst = min(result.tool_scores.items(), key=lambda kv: kv[1])
        worst_name, worst_score = worst
        if worst_score < 0.3:
            return (
                f"The {worst_name} tool returned a low score "
                f"({worst_score:.2f}). On the next pass, consider "
                f"gathering more context before invoking it again."
            )
        if worst_score < 0.6:
            return (
                f"The {worst_name} tool's score ({worst_score:.2f}) "
                f"is uncertain — verify with a second tool before "
                f"committing to the recommendation."
            )
        return (
            f"All tools scored confidently (min={worst_score:.2f} "
            f"on {worst_name}); proceed with the recommendation."
        )
