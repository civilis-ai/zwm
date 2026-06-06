"""P4-8 (audit) — Constitutional AI safety guardrails.

The :mod:`zwm.safety.constitution` module provides a 2026 SOTA
constitutional-AI safety layer that gates every input/output of the
OODA loop.  See :class:`zwm.safety.constitution.ConstitutionalGuard`.
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

__all__ = [
    "ConstitutionalGuard",
    "ConstitutionalViolation",
    "DEFAULT_CONSTITUTION",
    "Rule",
    "Severity",
    "Verdict",
    "rule_max_field",
]
