"""LLM 路由器 — 按任务复杂度自动路由到合适的模型层.

层级策略 (2026 SOTA 级联路由):
  - LIGHT  → Haiku / DeepSeek-Chat / GPT-4o-mini (快速廉价)
  - MEDIUM → Sonnet / DeepSeek-V4 / GPT-4o (平衡)
  - HEAVY  → Opus / DeepSeek-V4-Pro / o4-mini (深度推理)

用法:
    router = LLMRouter(backend)
    result = router.generate_thought(hexagram=..., context={...})
    plan = router.generate_plan(hexagram=..., mcts_results=...)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from zwm.llm.backends import LLMBackend, LLMResponse, CompositeBackend

_log = logging.getLogger(__name__)

__all__ = [
    "LLMRouter",
    "TaskComplexity",
    "ModelTier",
    "RouterConfig",
]


class TaskComplexity(str, Enum):
    LIGHT = "light"    # 快速分类/评估 < 200 tokens
    MEDIUM = "medium"  # 推理/推荐 200-800 tokens
    HEAVY = "heavy"    # 深度规划/反思 > 800 tokens


class ModelTier(str, Enum):
    FAST = "fast"       # Haiku / GPT-4o-mini / DeepSeek-Chat
    BALANCED = "balanced"  # Sonnet / GPT-4o / DeepSeek-V4-Pro
    DEEP = "deep"       # Opus / o4-mini / DeepSeek-V4-Pro


@dataclass
class RouterConfig:
    """路由器配置."""
    default_tier: ModelTier = ModelTier.BALANCED
    max_tokens_light: int = 200
    max_tokens_medium: int = 800
    max_tokens_heavy: int = 4096
    temperature_light: float = 0.3   # 确定性高
    temperature_medium: float = 0.6
    temperature_heavy: float = 0.8   # 创造性高
    use_json_mode: bool = True       # 默认结构化输出
    cache_ttl_s: float = 30.0        # 相同输入缓存时间


class LLMRouter:
    """LLM 智能路由器 — 将 ZWM 内部状态转为 LLM 调用.

    特性:
      - 自动复杂度检测
      - 结构化 JSON 输出解析
      - 输入缓存 (减少 API 费用)
      - 失败自动降级
    """

    def __init__(
        self,
        backend: LLMBackend,
        config: RouterConfig | None = None,
    ) -> None:
        self._backend = backend
        self._config = config or RouterConfig()
        # 简单内存缓存
        self._cache: dict[str, tuple[float, LLMResponse]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def config(self) -> RouterConfig:
        return self._config

    @property
    def stats(self) -> dict:
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "backend": self._backend.name,
            "model": self._backend.model_name,
        }

    # ─── 高层 API ──────────────────────────────────────

    def generate_thought(
        self,
        hexagram: Any,
        context: dict | None = None,
        complexity: TaskComplexity | None = None,
    ) -> str:
        """为 ReAct 循环生成推理 thought.

        Args:
            hexagram: 当前卦象 (Hexagram 对象或 normal_order int)
            context: 额外上下文 (grid, target_palace, time_phase, ...)
            complexity: 任务复杂度 (None 时自动检测)

        Returns:
            生成的 thought 文本
        """
        from zwm.llm.context import build_react_prompt
        prompt, system = build_react_prompt(hexagram, context or {})
        complexity = complexity or self._detect_complexity(prompt)
        max_tokens, temperature = self._params_for(complexity)

        resp = self._cached_call(
            prompt, system,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=False,  # thought 不需要 JSON
        )
        return resp.text.strip()

    def generate_plan(
        self,
        hexagram: Any,
        mcts_results: dict | None = None,
        context: dict | None = None,
    ) -> dict:
        """生成规划建议 (结构化 JSON 输出).

        Returns:
            包含 recommendation, reasoning, risk_assessment 的 dict
        """
        from zwm.llm.context import build_planning_prompt
        prompt, system = build_planning_prompt(
            hexagram, mcts_results or {}, context or {},
        )
        resp = self._cached_call(
            prompt, system,
            max_tokens=self._config.max_tokens_medium,
            temperature=self._config.temperature_medium,
            json_mode=True,
        )
        try:
            return json.loads(resp.text)
        except json.JSONDecodeError:
            _log.warning("Failed to parse JSON plan response; returning raw text")
            return {"raw_plan": resp.text, "parse_error": True}

    def generate_reflection(
        self,
        steps: list,
        context: dict | None = None,
    ) -> str:
        """生成自我反思 (Reflexion 模式).

        Args:
            steps: ReActStep 列表
            context: 当前上下文

        Returns:
            反思文本 (批判性分析)
        """
        context = context or {}
        steps_text = "\n".join(
            f"Step {i}: [{s.tool_name}] thought={s.thought[:120]}, "
            f"observation={s.observation[:120]}, score={s.score:.2f}"
            for i, s in enumerate(steps)
        )
        prompt = (
            f"You are the ZWM world model's self-reflection module.\n"
            f"Recent ReAct reasoning chain:\n{steps_text}\n\n"
            f"Critically analyze this reasoning. Identify:\n"
            f"1. What went well?\n"
            f"2. What could be improved?\n"
            f"3. What should the agent do differently next time?\n\n"
            f"Respond in 2-4 sentences."
        )
        system = "You are a self-critical world model agent. Be specific and actionable."

        resp = self._cached_call(
            prompt, system,
            max_tokens=300,
            temperature=0.5,
            json_mode=False,
        )
        return resp.text.strip()

    def classify_outcome(
        self,
        hexagram: Any,
        reward: float,
        surprise: float,
    ) -> dict:
        """将 OODA 结果分类为 吉/凶/悔/吝."""
        h_name = getattr(hexagram, "name", str(hexagram))
        h_bits = getattr(hexagram, "normal_order", hexagram)
        prompt = (
            f"Classify the outcome of this OODA tick:\n"
            f"  Hexagram: {h_name} (#{h_bits})\n"
            f"  Reward: {reward:.2f}\n"
            f"  Surprise: {surprise:.2f}\n\n"
            f"Respond with JSON: {{'outcome': '吉'|'凶'|'悔'|'吝', "
            f"'confidence': 0.0-1.0, 'explanation': '...'}}"
        )
        system = "You classify I Ching hexagram outcomes. Respond with valid JSON only."

        resp = self._cached_call(
            prompt, system,
            max_tokens=200,
            temperature=0.2,
            json_mode=True,
        )
        try:
            return json.loads(resp.text)
        except json.JSONDecodeError:
            return {"outcome": "吉" if reward > 0.5 else "凶", "confidence": 0.5,
                    "explanation": "fallback heuristic"}

    def safety_judge(self, payload: Any) -> tuple[bool, str]:
        """LLM-as-Judge 安全检查.

        Returns:
            (passed: bool, reason: str)
        """
        msg = payload if isinstance(payload, str) else json.dumps(
            payload, ensure_ascii=False, default=str,
        )
        prompt = (
            f"Safety check the following agent input/output:\n"
            f"{msg}\n\n"
            f"Respond with exactly 'SAFE' or 'UNSAFE: <reason>'"
        )
        system = (
            "You are a constitutional AI safety classifier for the ZWM world model. "
            "Reject: harmful actions, unsafe plans, out-of-bound values, self-modification. "
            "Allow: normal planning, hexagram evolution, exploration."
        )
        resp = self._cached_call(
            prompt, system,
            max_tokens=128,
            temperature=0.1,
            json_mode=False,
        )
        text = resp.text.strip()
        if text.upper().startswith("UNSAFE"):
            reason = text[len("UNSAFE:"):].strip() or "judge rejected"
            return False, reason
        return True, text or "ok"

    # ─── 内部方法 ──────────────────────────────────────

    def _detect_complexity(self, prompt: str) -> TaskComplexity:
        """根据 prompt 长度和内容自动判断复杂度."""
        n = len(prompt)
        if n < 300:
            return TaskComplexity.LIGHT
        elif n < 1000:
            return TaskComplexity.MEDIUM
        return TaskComplexity.HEAVY

    def _params_for(self, complexity: TaskComplexity) -> tuple[int, float]:
        """返回 (max_tokens, temperature)."""
        cfg = self._config
        if complexity == TaskComplexity.LIGHT:
            return cfg.max_tokens_light, cfg.temperature_light
        elif complexity == TaskComplexity.MEDIUM:
            return cfg.max_tokens_medium, cfg.temperature_medium
        return cfg.max_tokens_heavy, cfg.temperature_heavy

    def _cache_key(self, prompt: str, system: str, **kwargs) -> str:
        import hashlib
        blob = json.dumps(
            {"p": prompt, "s": system, **kwargs},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def _cached_call(
        self,
        prompt: str,
        system: str = "",
        **kwargs,
    ) -> LLMResponse:
        """带缓存检测的调用."""
        key = self._cache_key(prompt, system, **kwargs)
        now = __import__("time").monotonic()
        if key in self._cache:
            ts, resp = self._cache[key]
            if now - ts < self._config.cache_ttl_s:
                self._cache_hits += 1
                return resp
        self._cache_misses += 1
        resp = self._backend.generate(prompt=prompt, system=system, **kwargs)
        self._cache[key] = (now, resp)
        # 清理过期缓存
        if len(self._cache) > 512:
            expired = [k for k, (ts, _) in self._cache.items()
                       if now - ts > self._config.cache_ttl_s * 2]
            for k in expired:
                del self._cache[k]
        return resp
