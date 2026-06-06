"""M5 — OpenInference semantic conventions for the agent.

The 2025/2026 SOTA for LLM/agent observability is the OpenInference
spec (https://github.com/Arize-ai/openinference), which defines
attribute namespaces like ``openinference.span.kind``,
``llm.*``, ``tool.*``, ``retrieval.*`` etc.  Tools like
``phoenix`` (Arize), ``langfuse``, and ``traceloop`` already
ingest these.

This module exposes:

  * ``openinference_attr(kind, ...)`` — returns a dict of
    OpenInference-style attributes for the given ``kind`` (one
    of ``"agent"``, ``"chain"``, ``"tool"``, ``"retriever"``,
    ``"llm"``).
  * ``apply_openinference(span, kind, **fields)`` — convenience
    wrapper that calls ``set_attribute`` for each key.
  * ``OPENINFERENCE_KINDS`` — the supported kinds (for validation
    in tests).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

# The OpenInference semantic-convention span kinds (subset relevant
# to ZWM).  See https://github.com/Arize-ai/openinference/blob/main/
# docs/spec/semantic_conventions.md.
OPENINFERENCE_KINDS: frozenset[str] = frozenset({
    "agent", "chain", "tool", "retriever", "llm", "embedding", "reranker",
})

# Per-kind default attributes.  We populate these whenever the
# agent starts a span, so backend tools (Phoenix, Langfuse) get
# a recognisable trace even without the agent having to set them
# by hand.
_DEFAULTS: dict[str, dict[str, str]] = {
    "agent": {
        "openinference.span.kind": "AGENT",
        "zwm.agent.role": "trinity",
    },
    "chain": {
        "openinference.span.kind": "CHAIN",
    },
    "tool": {
        "openinference.span.kind": "TOOL",
        "openinference.tool.name": "",
    },
    "retriever": {
        "openinference.span.kind": "RETRIEVER",
    },
    "llm": {
        "openinference.span.kind": "LLM",
    },
    "embedding": {
        "openinference.span.kind": "EMBEDDING",
    },
    "reranker": {
        "openinference.span.kind": "RERANKER",
    },
}

# Mapping from ZWM span names to OpenInference kinds — the agent
# uses these defaults so the OTel export is meaningful without
# any per-call configuration.
_SPAN_KIND_AUTODETECT: dict[str, str] = {
    "ooda.tick": "chain",
    "ooda.observe": "agent",
    "ooda.predict": "llm",
    "ooda.evaluate": "chain",
    "ooda.act": "tool",
    "ooda.learn": "chain",
    "react.run": "agent",
    "react.tool": "tool",
    "react.reflect": "llm",
    "a2a.consensus": "agent",
    "a2a.send": "tool",
    "constitution.check": "chain",
    "mcp.dispatch": "tool",
    "jepa.predict": "llm",
}


def openinference_attr(kind: str, **fields: Any) -> dict[str, Any]:
    """Return a dict of OpenInference-style attributes for ``kind``.

    Unknown ``kind`` falls back to a plain ``openinference.span.kind``
    attribute (stringified).  ``fields`` are merged on top of the
    defaults so callers can override ``openinference.tool.name`` etc.
    """
    if kind not in OPENINFERENCE_KINDS:
        attrs: dict[str, Any] = {"openinference.span.kind": str(kind).upper()}
    else:
        attrs = dict(_DEFAULTS.get(kind, {}))
    # If the caller provided ``tool_name=``, map it to the canonical
    # ``openinference.tool.name``.
    if "tool_name" in fields:
        attrs["openinference.tool.name"] = str(fields.pop("tool_name"))
    if "model_name" in fields:
        attrs["llm.model_name"] = str(fields.pop("model_name"))
    if "input_value" in fields:
        attrs["input.value"] = str(fields.pop("input_value"))
    if "output_value" in fields:
        attrs["output.value"] = str(fields.pop("output_value"))
    for k, v in fields.items():
        attrs[k] = v
    return attrs


def apply_openinference(span: Any, kind: str, **fields: Any) -> None:
    """Apply OpenInference attributes to ``span``.

    ``span`` is any object that exposes ``set_attribute(key, value)``
    (matches :class:`_InProcessSpan`, :class:`_OtelBridgedSpan`, and
    the OTel SDK ``Span``)."""
    if span is None:
        return
    attrs = openinference_attr(kind, **fields)
    for k, v in attrs.items():
        try:
            span.set_attribute(k, v)
        except Exception:
            pass


def autodetect_kind(span_name: str) -> str:
    """Return the OpenInference kind for a known ZWM span name.

    Falls back to ``"chain"`` for unknown names — the conservative
    default that doesn't mislead downstream tools."""
    return _SPAN_KIND_AUTODETECT.get(span_name, "chain")


def enrich_span(span: Any, span_name: str, **fields: Any) -> None:
    """One-call helper: detect the kind from ``span_name`` and apply
    OpenInference attributes.  Used by the OODA loop to make every
    span OpenInference-compliant without per-call boilerplate."""
    kind = autodetect_kind(span_name)
    apply_openinference(span, kind, **fields)


__all__ = [
    "OPENINFERENCE_KINDS",
    "openinference_attr",
    "apply_openinference",
    "autodetect_kind",
    "enrich_span",
]
