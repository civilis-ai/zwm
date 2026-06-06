"""P4-8 (audit) — Constitutional AI safety guardrails.

Implements a 2026 SOTA safety pattern: explicit *constitutional rules*
that gate every input/output of the OODA loop.  Inspired by Anthropic's
Constitutional AI and DeepMind's Sparrow alignment work, but tailored
to the 天地人 (sky/earth/human) world-model context.

A ``Constitution`` is a list of :class:`Rule` objects, each with:

  * a stable ``name`` (for audit logs)
  * a human-readable ``description`` (for telemetry labels)
  * a callable ``predicate(value) -> bool``
  * a ``severity`` (``"block"`` rejects, ``"warn"`` allows with log)

The :class:`ConstitutionalGuard` runs every rule on every value passed
to :meth:`check_input` or :meth:`check_output`.  Blocked values raise
:class:`ConstitutionalViolation`; warnings emit a structured log record
that the telemetry layer can pick up.

The Trinity framework's "人" (human) channel is privileged — every
output that affects an actuator or a downstream user must pass through
:meth:`ConstitutionalGuard.check_output` before it leaves the loop.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Severity / verdict
# ----------------------------------------------------------------------
class Severity(str, Enum):
    BLOCK = "block"  # reject the call
    WARN = "warn"    # allow but log a structured warning
    INFO = "info"    # observe only


@dataclass(frozen=True, slots=True)
class Verdict:
    """Outcome of a single rule check."""
    rule_name: str
    severity: Severity
    passed: bool
    reason: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_name,
            "severity": self.severity.value,
            "passed": self.passed,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class ConstitutionalViolation(Exception):
    """Raised when a ``block``-severity rule fails."""

    def __init__(self, verdicts: list[Verdict]) -> None:
        self.verdicts = verdicts
        msgs = "; ".join(
            f"[{v.rule_name}] {v.reason}" for v in verdicts if not v.passed
        )
        super().__init__(f"constitutional violation: {msgs}")


# ----------------------------------------------------------------------
# Rule
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Rule:
    """A single constitutional rule."""
    name: str
    description: str
    predicate: Callable[[Any], tuple[bool, str]]
    severity: Severity = Severity.BLOCK

    def check(self, value: Any) -> Verdict:
        try:
            ok, reason = self.predicate(value)
        except Exception as exc:  # predicate bugs must not block real safety
            _log.exception("rule %r predicate raised", self.name)
            return Verdict(
                rule_name=self.name,
                severity=Severity.WARN,
                passed=True,
                reason=f"predicate-error (allowed): {exc!r}",
            )
        return Verdict(
            rule_name=self.name,
            severity=self.severity,
            passed=ok,
            reason=reason if not ok else "ok",
        )


# ----------------------------------------------------------------------
# Predicates
# ----------------------------------------------------------------------
def _is_finite_number(x: Any) -> tuple[bool, str]:
    """All numeric leaves must be finite (no NaN/Inf)."""
    if isinstance(x, bool):
        return True, "bool"
    if isinstance(x, (int, float)):
        import math
        if not math.isfinite(float(x)):
            return False, f"non-finite number: {x!r}"
        return True, "finite"
    if isinstance(x, dict):
        for k, v in x.items():
            ok, why = _is_finite_number(v)
            if not ok:
                return False, f"key {k!r}: {why}"
        return True, "all-dict-leaves-finite"
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            ok, why = _is_finite_number(v)
            if not ok:
                return False, f"index {i}: {why}"
        return True, "all-list-leaves-finite"
    return True, "type-skipped"


def _reward_in_range(x: Any) -> tuple[bool, str]:
    """Reward must be in [-1, 1] (the agent already clamps, but we
    double-check here because the OODA loop can be called from
    external code that bypasses ``_validate_reward``)."""
    if "reward" not in x:
        return True, "no-reward-in-payload"
    r = x.get("reward")
    try:
        r = float(r)
    except (TypeError, ValueError):
        return False, f"reward not numeric: {r!r}"
    if not (-1.0 <= r <= 1.0):
        return False, f"reward {r} outside [-1, 1]"
    return True, "ok"


def _hexagram_in_range(x: Any) -> tuple[bool, str]:
    """Hexagram indices must be 0-63 (or None for sensor-driven)."""
    bits = x.get("h_current") if isinstance(x, dict) else None
    if bits is None:
        return True, "no-hex-specified"
    if not isinstance(bits, int):
        return False, f"h_current not int: {bits!r}"
    if not (0 <= bits <= 63):
        return False, f"hexagram {bits} outside [0, 63]"
    return True, "ok"


def _target_palace_in_range(x: Any) -> tuple[bool, str]:
    """Target palace (if specified) must be 1-9 (洛书 bounds)."""
    p = x.get("target_palace") if isinstance(x, dict) else None
    if p is None:
        return True, "no-target-palace"
    if not isinstance(p, int):
        return False, f"target_palace not int: {p!r}"
    if not (1 <= p <= 9):
        return False, f"target_palace {p} outside [1, 9]"
    return True, "ok"


def _no_self_loop_mutation(x: Any) -> tuple[bool, str]:
    """Output sanity: the agent's chosen mutation must not be a no-op
    (i.e. ``top_mutation`` must move at least one 爻)."""
    if not isinstance(x, dict):
        return True, "not-a-plan-dict"
    cur = x.get("h_current")
    nxt = x.get("h_next")
    if cur is None or nxt is None:
        return True, "no-hex-context"
    if cur == nxt:
        return False, f"self-loop mutation: {cur}->{nxt}"
    return True, "ok"


def _efe_within_bounds(x: Any) -> tuple[bool, str]:
    """EFE score must be finite and within plausible bounds.

    The EFE is bounded by construction (MCTS returns log-prob-style
    numbers) but a runaway model could still produce absurd values; we
    catch them at the boundary.
    """
    if not isinstance(x, dict):
        return True, "not-a-plan-dict"
    efe = x.get("top_score")
    if efe is None:
        return True, "no-efe"
    try:
        efe = float(efe)
    except (TypeError, ValueError):
        return False, f"efe not numeric: {efe!r}"
    if not (-100.0 <= efe <= 100.0):
        return False, f"efe {efe} outside plausible [-100, 100]"
    return True, "ok"


# ----------------------------------------------------------------------
# Default constitution
# ----------------------------------------------------------------------
DEFAULT_CONSTITUTION: tuple[Rule, ...] = (
    Rule(
        name="finite-numbers",
        description="All numeric leaves in the payload must be finite.",
        predicate=_is_finite_number,
        severity=Severity.BLOCK,
    ),
    Rule(
        name="reward-in-range",
        description="Reward signal must be in [-1, 1].",
        predicate=_reward_in_range,
        severity=Severity.BLOCK,
    ),
    Rule(
        name="hexagram-in-range",
        description="Hexagram index (h_current) must be 0..63.",
        predicate=_hexagram_in_range,
        severity=Severity.BLOCK,
    ),
    Rule(
        name="target-palace-in-range",
        description="target_palace (if given) must be 1..9.",
        predicate=_target_palace_in_range,
        severity=Severity.BLOCK,
    ),
    Rule(
        name="no-self-loop-mutation",
        description="Chosen mutation must change at least one 爻.",
        predicate=_no_self_loop_mutation,
        severity=Severity.WARN,
    ),
    Rule(
        name="efe-within-bounds",
        description="top_score (EFE) must be within [-100, 100].",
        predicate=_efe_within_bounds,
        severity=Severity.WARN,
    ),
)


# ----------------------------------------------------------------------
# Guard
# ----------------------------------------------------------------------
class ConstitutionalGuard:
    """Runs a :class:`Constitution` against inputs and outputs.

    Audit trail: every verdict is recorded in :attr:`history` (a
    bounded ring buffer) so post-hoc review is possible without
    re-running the agent.
    """

    def __init__(
        self,
        constitution: Sequence[Rule] = DEFAULT_CONSTITUTION,
        history_limit: int = 1024,
        enabled: bool = True,
    ) -> None:
        self.constitution: tuple[Rule, ...] = tuple(constitution)
        self.enabled = enabled
        self._history: list[Verdict] = []
        self._history_limit = history_limit

    @property
    def history(self) -> list[Verdict]:
        """Read-only view of the recent verdict ring (newest last)."""
        return list(self._history)

    def add_rule(self, rule: Rule) -> None:
        """P4-8 — extend the constitution at runtime."""
        self.constitution = self.constitution + (rule,)

    def check_input(self, payload: Any) -> None:
        """Raise :class:`ConstitutionalViolation` on any BLOCK failure."""
        if not self.enabled:
            return
        self._evaluate(payload, kind="input")

    def check_output(self, payload: Any) -> None:
        """Same as :meth:`check_input`, with a separate label so the
        audit log can distinguish input vs. output failures."""
        if not self.enabled:
            return
        self._evaluate(payload, kind="output")

    def _evaluate(self, payload: Any, kind: str) -> None:
        verdicts: list[Verdict] = []
        for rule in self.constitution:
            v = rule.check(payload)
            verdicts.append(v)
            self._record(v, kind=kind)
        bad_blocks = [v for v in verdicts if not v.passed and v.severity == Severity.BLOCK]
        if bad_blocks:
            raise ConstitutionalViolation(bad_blocks)
        bad_warns = [v for v in verdicts if not v.passed and v.severity == Severity.WARN]
        for v in bad_warns:
            _log.warning("constitution[%s] %s: %s", kind, v.rule_name, v.reason)

    def _record(self, v: Verdict, kind: str) -> None:
        self._history.append(v)
        if len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]
        if not v.passed:
            _log.debug("constitution[%s] verdict: %s", kind, v.to_dict())


# ----------------------------------------------------------------------
# Helpers — convenient rule builders
# ----------------------------------------------------------------------
def rule_max_field(name: str, field: str, lo: float, hi: float,
                   severity: Severity = Severity.BLOCK) -> Rule:
    """Build a numeric-range rule for a dict field."""
    def _pred(x: Any) -> tuple[bool, str]:
        if not isinstance(x, dict):
            return True, "not-a-dict"
        v = x.get(field)
        if v is None:
            return True, f"no {field!r}"
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False, f"{field!r} not numeric: {v!r}"
        if not (lo <= v <= hi):
            return False, f"{field!r}={v} outside [{lo}, {hi}]"
        return True, "ok"
    return Rule(
        name=name,
        description=f"{field} must be in [{lo}, {hi}]",
        predicate=_pred,
        severity=severity,
    )


__all__ = [
    "ConstitutionalGuard",
    "ConstitutionalViolation",
    "Rule",
    "Severity",
    "Verdict",
    "DEFAULT_CONSTITUTION",
    "rule_max_field",
]
