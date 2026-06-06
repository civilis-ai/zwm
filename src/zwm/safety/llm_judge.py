"""H5 — Constitutional LLM-as-Judge 通路 (Anthropic 2024 风格).

A :class:`LLMJudgeRule` plugs an LLM-based evaluator into the
:class:`ConstitutionalGuard`.  The host registers a *judge function* at
construction time; the function takes a payload and returns a
``(passed, reason)`` tuple.  The default implementation is a no-op
deterministic stub — users wire their own judge (Anthropic Claude,
OpenAI, local model, etc.) by passing ``judge_fn=``.

The judge is **opt-in and cached**: identical inputs within a short
window (default 60 s) return the cached verdict so we don't burn
tokens on every tick.  Failures inside the judge function are
**fail-open** (allow with WARN) so a flaky LLM provider doesn't bring
the agent down.

Usage::

    from zwm.safety.llm_judge import LLMJudgeRule, make_anthropic_judge

    # 1) Use the stub (always passes)
    rule = LLMJudgeRule(name="harm-check")

    # 2) Use a real Anthropic Claude judge
    judge = make_anthropic_judge(
        api_key="sk-...",
        model="claude-haiku-4-5-20251001",
        system="Reject if the request asks to harm a person.",
    )
    rule = LLMJudgeRule(name="harm-check", judge_fn=judge)

    # 3) Plug into the constitutional guard
    guard = ConstitutionalGuard(constitution=[rule])
    guard.check_input({"text": "hello"})  # → either pass or BLOCK
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from zwm.safety.constitution import Rule, Severity, Verdict

_log = logging.getLogger(__name__)


# Type alias for a judge function.  Receives the payload, returns
# ``(passed, reason)``.  ``reason`` is shown in the audit log.
JudgeFn = Callable[[Any], tuple[bool, str]]


# ----------------------------------------------------------------------
# Caching — saves tokens on identical inputs.
# ----------------------------------------------------------------------
@dataclass
class _CacheEntry:
    passed: bool
    reason: str
    expires_at: float


class _JudgeCache:
    """Tiny TTL cache — 256 entries max, 60s default TTL."""

    def __init__(self, max_size: int = 256, ttl_s: float = 60.0) -> None:
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl_s = ttl_s

    @staticmethod
    def _key(payload: Any) -> str:
        try:
            blob = json.dumps(payload, sort_keys=True, default=str)
        except (TypeError, ValueError):
            blob = repr(payload)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def get(self, payload: Any) -> _CacheEntry | None:
        key = self._key(payload)
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at < now:
                del self._entries[key]
                return None
            return entry

    def put(self, payload: Any, passed: bool, reason: str) -> None:
        key = self._key(payload)
        with self._lock:
            # Evict oldest if we're at the cap.
            if len(self._entries) >= self._max_size:
                oldest = min(self._entries.items(), key=lambda kv: kv[1].expires_at)
                self._entries.pop(oldest[0], None)
            self._entries[key] = _CacheEntry(
                passed=passed, reason=reason,
                expires_at=time.monotonic() + self._ttl_s,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# ----------------------------------------------------------------------
# LLMJudgeRule
# ----------------------------------------------------------------------
class LLMJudgeRule:
    """A constitutional rule whose predicate is an LLM call.

    Wraps an internal :class:`Rule` (frozen dataclass) and exposes the
    same interface (``check``) so it can be plugged into a
    :class:`ConstitutionalGuard` alongside plain :class:`Rule` objects.

    Parameters
    ----------
    name, description
        As in :class:`Rule`.
    judge_fn
        A :data:`JudgeFn` (payload) -> (passed, reason).  When
        ``None``, a no-op (always passes) judge is installed —
        useful for unit tests and for running without an LLM
        provider.
    severity
        BLOCK or WARN.
    cache_ttl_s
        Cached verdicts expire after this many seconds.  Set
        to 0 to disable caching.
    timeout_s
        Soft timeout for the judge call.  When the judge blocks
        longer than this, the call is treated as fail-open (WARN).
    """

    def __init__(
        self,
        name: str,
        description: str = "LLM-as-Judge safety check",
        judge_fn: JudgeFn | None = None,
        severity: Severity = Severity.BLOCK,
        cache_ttl_s: float = 60.0,
        timeout_s: float = 5.0,
        fail_open: bool | None = None,
    ) -> None:
        self._judge_fn = judge_fn or _noop_judge
        self._cache = _JudgeCache(ttl_s=cache_ttl_s) if cache_ttl_s > 0 else None
        self._timeout_s = timeout_s
        self._severity = severity
        # R3: rule-level fail_open mirrors the factory-level flag.  When
        # the caller constructs an LLMJudgeRule explicitly (e.g. with
        # ``make_anthropic_judge(fail_open=False)``), the rule should
        # respect that decision.  ``None`` falls back to the env var
        # ZWM_LLM_JUDGE_FAIL_OPEN for backward compatibility.
        if fail_open is None:
            fail_open = os.environ.get("ZWM_LLM_JUDGE_FAIL_OPEN", "1").strip() != "0"
        self._fail_open = fail_open
        # Stats for observability
        self._n_calls = 0
        self._n_cached = 0
        self._n_fail_open = 0
        self._lock = threading.Lock()
        # We expose the same attributes as Rule so the guard's
        # ``for rule in self.constitution`` loop can read them.
        self.name = name
        self.description = description
        self.severity = severity

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "calls": self._n_calls,
                "cached": self._n_cached,
                "fail_open": self._n_fail_open,
            }

    def check(self, payload: Any) -> Verdict:
        """Run the LLM judge on ``payload`` and return a :class:`Verdict`.

        R3: previously this method swallowed every exception and
        returned ``passed=True`` — making the factory-level
        ``fail_open=False`` a no-op for the rule layer.  We now honour
        ``self._fail_open``: when it's ``False`` the exception is
        re-raised so the constitutional guard fails loud.
        """
        try:
            ok, reason = self._evaluate(payload)
        except Exception as exc:
            with self._lock:
                self._n_fail_open += 1
            if not self._fail_open:
                _log.error("LLM judge %r failed and fail_open=False: %s",
                           self.name, exc)
                raise
            _log.warning("LLM judge %r failed (fail-open): %s", self.name, exc)
            return Verdict(
                rule_name=self.name,
                severity=self._severity,
                passed=True,
                reason=f"judge-error (allowed): {exc!r}",
            )
        return Verdict(
            rule_name=self.name,
            severity=self._severity,
            passed=ok,
            reason=reason,
        )

    def _evaluate(self, payload: Any) -> tuple[bool, str]:
        # 1) Cache fast-path
        if self._cache is not None:
            entry = self._cache.get(payload)
            if entry is not None:
                with self._lock:
                    self._n_cached += 1
                return entry.passed, entry.reason
        # 2) Call the judge.  Time out via a background thread to
        # avoid pulling in ``concurrent.futures``.
        with self._lock:
            self._n_calls += 1
        try:
            passed, reason = self._call_with_timeout(payload)
        except Exception as exc:
            with self._lock:
                self._n_fail_open += 1
            _log.warning("LLM judge %r failed (fail-open): %s", self.name, exc)
            return True, f"judge-error (allowed): {exc!r}"
        if self._cache is not None:
            self._cache.put(payload, passed, reason)
        return passed, reason

    def _call_with_timeout(self, payload: Any) -> tuple[bool, str]:
        """Run the judge in a daemon thread; return its result or
        fail-open if it doesn't return in time."""
        result: dict[str, Any] = {}

        def runner():
            try:
                result["v"] = self._judge_fn(payload)
            except Exception as exc:
                result["exc"] = exc
        th = threading.Thread(target=runner, daemon=True, name=f"judge-{self.name}")
        th.start()
        th.join(self._timeout_s)
        if th.is_alive():
            raise TimeoutError(f"judge timed out after {self._timeout_s}s")
        if "exc" in result:
            raise result["exc"]
        return result["v"]


# ----------------------------------------------------------------------
# Built-in judges
# ----------------------------------------------------------------------
def _noop_judge(payload: Any) -> tuple[bool, str]:
    """Default judge — always passes.  Used when no LLM is wired."""
    return True, "noop-judge"


def make_anthropic_judge(
    api_key: str | None = None,
    model: str = "claude-haiku-4-5-20251001",
    system: str = (
        "You are a safety classifier. If the user's request is safe, "
        "respond with exactly the word 'SAFE'.  Otherwise respond with "
        "'UNSAFE: <reason>'."
    ),
    max_tokens: int = 64,
    fail_open: bool | None = None,
) -> JudgeFn:
    """Build an Anthropic Claude judge function.

    The returned function takes any payload (dict, str, list) and
    coerces it to a user-message string before calling the API.

    Requires the ``anthropic`` SDK (optional dependency).

    F2: ``fail_open`` controls behaviour when the SDK or the API key
    is missing.  ``None`` (default) consults the env var
    ``ZWM_LLM_JUDGE_FAIL_OPEN`` (defaults to ``"1"`` for backward
    compat).  Set ``fail_open=False`` (or the env var to ``"0"``) to
    raise instead of silently downgrading to the no-op judge — a
    much safer default for production deployments.
    """
    if fail_open is None:
        fail_open = os.environ.get("ZWM_LLM_JUDGE_FAIL_OPEN", "1").strip() != "0"
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError as exc:
        msg = "anthropic SDK not installed"
        if fail_open:
            _log.warning("%s; judge will no-op", msg)
            return _noop_judge
        raise RuntimeError(msg) from exc
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        msg = "ANTHROPIC_API_KEY unset"
        if fail_open:
            _log.warning("%s; judge will no-op", msg)
            return _noop_judge
        raise RuntimeError(msg)
    client = Anthropic(api_key=key)

    def _judge(payload: Any) -> tuple[bool, str]:
        msg = payload if isinstance(payload, str) else json.dumps(
            payload, ensure_ascii=False, default=str,
        )
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": msg}],
            )
            text = ""
            for blk in resp.content:
                if getattr(blk, "type", "") == "text":
                    text += blk.text
            text = text.strip()
            if text.upper().startswith("UNSAFE"):
                reason = text[len("UNSAFE:"):].strip() or "judge rejected"
                return False, reason
            return True, text or "ok"
        except Exception as exc:
            raise RuntimeError(f"anthropic judge failed: {exc}") from exc

    return _judge


def make_openai_judge(
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    system: str = (
        "You are a safety classifier. If the user's request is safe, "
        "respond with exactly the word 'SAFE'.  Otherwise respond with "
        "'UNSAFE: <reason>'."
    ),
    max_tokens: int = 64,
    fail_open: bool | None = None,
) -> JudgeFn:
    """Build an OpenAI judge function.  Mirror of :func:`make_anthropic_judge`.

    F2: ``fail_open`` mirrors the Anthropic factory's behaviour.  Set
    to ``False`` to raise on missing SDK / API key instead of silently
    downgrading to the no-op judge.
    """
    if fail_open is None:
        fail_open = os.environ.get("ZWM_LLM_JUDGE_FAIL_OPEN", "1").strip() != "0"
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        msg = "openai SDK not installed"
        if fail_open:
            _log.warning("%s; judge will no-op", msg)
            return _noop_judge
        raise RuntimeError(msg) from exc
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        msg = "OPENAI_API_KEY unset"
        if fail_open:
            _log.warning("%s; judge will no-op", msg)
            return _noop_judge
        raise RuntimeError(msg)
    client = OpenAI(api_key=key)

    def _judge(payload: Any) -> tuple[bool, str]:
        msg = payload if isinstance(payload, str) else json.dumps(
            payload, ensure_ascii=False, default=str,
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": msg},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.upper().startswith("UNSAFE"):
                reason = text[len("UNSAFE:"):].strip() or "judge rejected"
                return False, reason
            return True, text or "ok"
        except Exception as exc:
            raise RuntimeError(f"openai judge failed: {exc}") from exc

    return _judge


def make_deepseek_judge(
    api_key: str | None = None,
    model: str = "deepseek-chat",
    system: str = (
        "You are a safety classifier. If the user's request is safe, "
        "respond with exactly the word 'SAFE'.  Otherwise respond with "
        "'UNSAFE: <reason>'."
    ),
    max_tokens: int = 64,
    fail_open: bool | None = None,
) -> JudgeFn:
    """Build a DeepSeek judge function.

    Uses the OpenAI-compatible API interface.  Mirrors the
    ``make_openai_judge`` factory with DeepSeek defaults.

    F2: ``fail_open`` controls behaviour when the SDK or API key
    is missing.  ``None`` (default) consults
    ``ZWM_LLM_JUDGE_FAIL_OPEN``.
    """
    if fail_open is None:
        fail_open = os.environ.get("ZWM_LLM_JUDGE_FAIL_OPEN", "1").strip() != "0"
    try:
        from openai import OpenAI
    except ImportError as exc:
        msg = "openai SDK not installed (needed for DeepSeek compatibility)"
        if fail_open:
            _log.warning("%s; judge will no-op", msg)
            return _noop_judge
        raise RuntimeError(msg) from exc
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        msg = "DEEPSEEK_API_KEY unset"
        if fail_open:
            _log.warning("%s; judge will no-op", msg)
            return _noop_judge
        raise RuntimeError(msg)
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    client = OpenAI(api_key=key, base_url=base_url)

    def _judge(payload: Any) -> tuple[bool, str]:
        msg = payload if isinstance(payload, str) else json.dumps(
            payload, ensure_ascii=False, default=str,
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": msg},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.upper().startswith("UNSAFE"):
                reason = text[len("UNSAFE:"):].strip() or "judge rejected"
                return False, reason
            return True, text or "ok"
        except Exception as exc:
            raise RuntimeError(f"deepseek judge failed: {exc}") from exc

    return _judge


def make_auto_judge(
    fail_open: bool | None = None,
) -> JudgeFn:
    """Auto-detect the best available LLM judge.

    Detection order: DEEPSEEK_API_KEY → ANTHROPIC_API_KEY → OPENAI_API_KEY.
    Returns a no-op judge if no API key is found (and fail_open is True).
    """
    if fail_open is None:
        fail_open = os.environ.get("ZWM_LLM_JUDGE_FAIL_OPEN", "1").strip() != "0"

    if os.environ.get("DEEPSEEK_API_KEY"):
        _log.info("auto_judge: using DeepSeek")
        return make_deepseek_judge(fail_open=fail_open)
    if os.environ.get("ANTHROPIC_API_KEY"):
        _log.info("auto_judge: using Anthropic")
        return make_anthropic_judge(fail_open=fail_open)
    if os.environ.get("OPENAI_API_KEY"):
        _log.info("auto_judge: using OpenAI")
        return make_openai_judge(fail_open=fail_open)
    if fail_open:
        _log.debug("auto_judge: no API key found, using no-op judge")
        return _noop_judge
    raise RuntimeError(
        "No LLM API key for judge. Set DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, "
        "or OPENAI_API_KEY, or set ZWM_LLM_JUDGE_FAIL_OPEN=1"
    )


__all__ = [
    "LLMJudgeRule",
    "make_anthropic_judge",
    "make_openai_judge",
    "make_deepseek_judge",
    "make_auto_judge",
    "JudgeFn",
    "_noop_judge",
]
