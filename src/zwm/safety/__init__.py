"""P4-8 (audit) — Constitutional AI safety guardrails.

The :mod:`zwm.safety.constitution` module provides a 2026 SOTA
constitutional-AI safety layer that gates every input/output of the
OODA loop.  See :class:`zwm.safety.constitution.ConstitutionalGuard`.

The :mod:`zwm.safety.llm_judge` module provides LLM-as-Judge safety
checks that can be plugged into the constitutional guard or used
standalone.  See :class:`zwm.safety.llm_judge.LLMJudgeRule`.
"""
from __future__ import annotations

from zwm.safety.constitution import (
    DEFAULT_CONSTITUTION,
    ConstitutionalGuard,
    ConstitutionalViolation,
    Rule,
    Severity,
    Verdict,
    rule_max_field,
)
from zwm.safety.llm_judge import (
    LLMJudgeRule,
    JudgeFn,
    make_anthropic_judge,
    make_auto_judge,
    make_deepseek_judge,
    make_openai_judge,
)

__all__ = [
    "ConstitutionalGuard",
    "ConstitutionalViolation",
    "DEFAULT_CONSTITUTION",
    "JudgeFn",
    "LLMJudgeRule",
    "Rule",
    "Severity",
    "Verdict",
    "make_anthropic_judge",
    "make_auto_judge",
    "make_deepseek_judge",
    "make_openai_judge",
    "rule_max_field",
]
