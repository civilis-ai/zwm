"""P4-9 (audit) — OpenTelemetry-compatible tracing.

The framework is built around the OODA loop, so observability of *what
phase did what and how long it took* is critical.  We provide a thin
facade that:

  1. **Always works** — a small in-process tracer that records spans
     in a ring buffer, even when the OTel SDK is not installed.
  2. **Bridges to OTel** — if the real OpenTelemetry packages are
     installed (``opentelemetry-api`` / ``opentelemetry-sdk``), spans
     are exported through the standard ``TracerProvider`` so a Jaeger
     / Tempo / Honeycomb / OTLP backend can pick them up unchanged.
  3. **Zero call-site changes** — replace ``PhaseStopwatch(...)`` (the
     existing context manager in :mod:`zwm.observability`) with
     ``tracer.start_as_current_span("observe")`` and the rest of the
     framework continues to work.

The 2026 SOTA pattern is "tracing, not just metrics": we record the
hexagram index, the EFE score, the reward, the constitutional verdict
as span attributes, so a backend can slice the agent's behaviour
without re-instrumenting the code.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

_log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Real OpenTelemetry import — optional.  Falls back silently.
# ----------------------------------------------------------------------
try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.span import Span as _OtelSpan
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    _otel_trace = None  # type: ignore
    Status = None        # type: ignore
    StatusCode = None    # type: ignore
    _OtelSpan = None     # type: ignore


# ----------------------------------------------------------------------
# In-process span record (for when OTel is not installed, and for tests)
# ----------------------------------------------------------------------
@dataclass
class SpanRecord:
    """A single finished span kept in the in-process buffer."""
    name: str
    start: float
    end: float
    duration: float
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # "ok" | "error"
    parent_id: int | None = None
    span_id: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "attributes": dict(self.attributes),
            "status": self.status,
            "parent_id": self.parent_id,
            "span_id": self.span_id,
        }


# ----------------------------------------------------------------------
# In-process Tracer
# ----------------------------------------------------------------------
class InProcessTracer:
    """Records spans in a ring buffer; the cheap default for tests
    and for environments without the OTel SDK.
    """

    def __init__(self, ring_size: int = 1024) -> None:
        self._ring: list[SpanRecord] = []
        self._ring_size = ring_size
        self._next_id = 1
        self._stack: list[int] = []  # parent chain
        self._use_otel = False  # auto-set by ``Tracer`` when SDK is present

    @property
    def spans(self) -> list[SpanRecord]:
        return list(self._ring)

    def clear(self) -> None:
        self._ring.clear()
        self._stack.clear()

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def start(self, name: str, attributes: dict | None = None) -> "_InProcessSpan":
        sid = self._alloc_id()
        parent = self._stack[-1] if self._stack else None
        rec = SpanRecord(
            name=name,
            start=time.perf_counter(),
            end=0.0,
            duration=0.0,
            attributes=dict(attributes or {}),
            parent_id=parent,
            span_id=sid,
        )
        return _InProcessSpan(self, rec)

    def finish(self, rec: SpanRecord, status: str) -> None:
        rec.end = time.perf_counter()
        rec.duration = rec.end - rec.start
        rec.status = status
        self._ring.append(rec)
        if len(self._ring) > self._ring_size:
            del self._ring[: len(self._ring) - self._ring_size]


class _InProcessSpan:
    def __init__(self, tracer: InProcessTracer, rec: SpanRecord) -> None:
        self._tracer = tracer
        self._rec = rec
        self._tracer._stack.append(rec.span_id)

    def set_attribute(self, key: str, value: Any) -> None:
        # JSON-safe coercion: keep numbers / strs / bools as-is, str() the rest.
        if isinstance(value, (int, float, str, bool)) or value is None:
            self._rec.attributes[key] = value
        else:
            self._rec.attributes[key] = str(value)

    def set_status(self, status: str) -> None:
        self._rec.status = status

    def record_exception(self, exc: BaseException) -> None:
        self._rec.attributes["exception.type"] = type(exc).__name__
        self._rec.attributes["exception.message"] = str(exc)

    def __enter__(self) -> "_InProcessSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Pop ourselves off the parent stack.
        if self._tracer._stack and self._tracer._stack[-1] == self._rec.span_id:
            self._tracer._stack.pop()
        if exc_type is not None and exc is not None:
            # Mirror the OTel convention: mark the span as failed and
            # tag it with ``exception.type`` / ``exception.message``.
            self._rec.status = "error"
            self._rec.attributes["exception.type"] = exc_type.__name__
            self._rec.attributes["exception.message"] = str(exc)
        else:
            self._rec.status = "ok"
        self._tracer.finish(self._rec, self._rec.status)
        # Don't swallow exceptions.
        return False


# ----------------------------------------------------------------------
# OTel-bridged span
# ----------------------------------------------------------------------
class _OtelBridgedSpan:
    """Adapts an OTel span to the tiny surface we use."""

    def __init__(self, otel_span: Any, inproc: _InProcessSpan, rec: SpanRecord,
                 tracer: InProcessTracer) -> None:
        self._otel = otel_span
        self._inproc = inproc
        self._rec = rec
        self._tracer = tracer

    def set_attribute(self, key: str, value: Any) -> None:
        self._inproc.set_attribute(key, value)
        try:
            self._otel.set_attribute(key, value)
        except Exception as exc:
            _log.debug("otel set_attribute failed: %s", exc)

    def set_status(self, status: str) -> None:
        self._inproc.set_status(status)
        if _OTEL_AVAILABLE and Status is not None:
            try:
                if status == "error":
                    self._otel.set_status(Status(StatusCode.ERROR))
                else:
                    self._otel.set_status(Status(StatusCode.OK))
            except Exception as exc:
                _log.debug("otel set_status failed: %s", exc)

    def record_exception(self, exc: BaseException) -> None:
        self._inproc.record_exception(exc)
        try:
            self._otel.record_exception(exc)
        except Exception as exc2:
            _log.debug("otel record_exception failed: %s", exc2)

    def __enter__(self) -> "_OtelBridgedSpan":
        self._otel.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._otel.__exit__(exc_type, exc, tb)
        except Exception as exc2:
            _log.debug("otel __exit__ failed: %s", exc2)
        # Pop ourselves off the parent stack.
        if self._tracer._stack and self._tracer._stack[-1] == self._rec.span_id:
            self._tracer._stack.pop()
        if exc_type is not None and exc is not None:
            self._rec.status = "error"
            self._rec.attributes.setdefault(
                "exception.type", exc_type.__name__,
            )
            self._rec.attributes.setdefault("exception.message", str(exc))
        else:
            self._rec.status = "ok"
        self._tracer.finish(self._rec, self._rec.status)
        return False


# ----------------------------------------------------------------------
# Tracer — the public entry point
# ----------------------------------------------------------------------
class Tracer:
    """P4-9 — the agent's tracer.

    The agent calls :meth:`start_as_current_span` to wrap each OODA
    phase.  When the real OTel SDK is installed and a
    ``TracerProvider`` has been configured (see :func:`configure_otel`),
    the spans are exported to the OTel backend in parallel to being
    recorded in the in-process ring.
    """

    def __init__(self, ring_size: int = 1024) -> None:
        self._inproc = InProcessTracer(ring_size=ring_size)
        self._otel_tracer: Any = None

    @property
    def spans(self) -> list[SpanRecord]:
        return self._inproc.spans

    def clear(self) -> None:
        self._inproc.clear()

    def use_otel(self) -> bool:
        """``True`` if spans are being exported to OTel as well."""
        return self._otel_tracer is not None

    def start_as_current_span(
        self,
        name: str,
        attributes: dict | None = None,
    ):
        """Start a span, return a context-manager-like object.

        Usage::

            with tracer.start_as_current_span("evaluate") as span:
                span.set_attribute("efe", 0.83)
                ...
        """
        rec_sid = self._inproc._alloc_id()
        parent = self._inproc._stack[-1] if self._inproc._stack else None
        rec = SpanRecord(
            name=name,
            start=time.perf_counter(),
            end=0.0,
            duration=0.0,
            attributes=dict(attributes or {}),
            parent_id=parent,
            span_id=rec_sid,
        )
        self._inproc._stack.append(rec_sid)
        if self._otel_tracer is not None:
            try:
                otel_cm = self._otel_tracer.start_as_current_span(name)
                otel_span = otel_cm.__enter__()
                for k, v in rec.attributes.items():
                    try:
                        otel_span.set_attribute(k, v)
                    except Exception as exc:
                        _log.debug("OTel set_attribute(%s) failed: %s", k, exc)
                # We need to defer the in-process __exit__ until both
                # the OTel __exit__ has run and we've popped the stack.
                return _OtelBridgedSpan(otel_span, _InProcessSpan(self._inproc, rec), rec, self._inproc)
            except Exception as exc:
                _log.debug("otel start_as_current_span failed: %s", exc)
        return _InProcessSpan(self._inproc, rec)


# ----------------------------------------------------------------------
# Module-level singleton + helpers
# ----------------------------------------------------------------------
_tracer = Tracer()


def get_tracer() -> Tracer:
    """P4-9 — return the module-level tracer (cheap singleton)."""
    return _tracer


@contextmanager
def start_span(name: str, attributes: dict | None = None) -> Iterator[Any]:
    """Sugar: ``with start_span("observe"): ...``."""
    with _tracer.start_as_current_span(name, attributes=attributes) as span:
        yield span


class _span_guard:
    """P4-9 — minimal context manager that mirrors the API the OODA
    tick needs without forcing the whole function to be indented under
    a ``with`` block.

    Usage::

        guard = _span_guard(tracer, "ooda.tick", {"x": 1})
        span = guard.__enter__()
        try:
            ... do work, may raise ...
        finally:
            guard.__exit__(None, None, None)
    """

    def __init__(self, tracer: "Tracer", name: str,
                 attributes: dict | None = None) -> None:
        self._cm = tracer.start_as_current_span(name, attributes=attributes)
        self._span: Any = None

    def __enter__(self) -> Any:
        self._span = self._cm.__enter__()
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Delegate to the underlying context manager.  Returning False
        # so any exception raised in the body propagates.
        return self._cm.__exit__(exc_type, exc, tb)


def configure_otel(
    service_name: str = "zwm-agent",
    endpoint: str | None = None,
) -> bool:
    """P4-9 — wire the OTel SDK to the in-process tracer.

    Returns ``True`` on success, ``False`` if the OTel SDK is not
    installed.  The ``endpoint`` is a no-op for now (the SDK exporter
    is configured by the *caller* via the standard ``OTEL_*`` env vars
    or a custom ``TracerProvider``); we just install the global tracer.
    """
    if not _OTEL_AVAILABLE:
        _log.info("configure_otel: opentelemetry not installed (skipping)")
        return False
    try:
        _tracer._otel_tracer = _otel_trace.get_tracer(service_name)
        _log.info("configure_otel: tracer bound to service %r", service_name)
        return True
    except Exception as exc:
        _log.warning("configure_otel: failed to bind tracer: %s", exc)
        return False


# ----------------------------------------------------------------------
# H1 — OTLP Exporter auto-configuration
# ----------------------------------------------------------------------
def configure_otlp(
    endpoint: str | None = None,
    service_name: str = "zwm-agent",
    insecure: bool = True,
    headers: dict | None = None,
    timeout_s: float = 10.0,
) -> bool:
    """H1 — install an OTLP gRPC exporter and bind it to the tracer.

    This is the *automatic* counterpart of :func:`configure_otel`:
    instead of requiring the user to wire the SDK exporter by hand, we
    install a sensible default (OTLP gRPC over the endpoint derived
    from ``ZWM_OTLP_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_ENDPOINT`` /
    the function argument).

    Falls back to ``False`` if the OTel SDK or the OTLP exporter
    package is not installed; the in-process tracer keeps working in
    that case.

    Parameters
    ----------
    endpoint : str | None
        ``host:port`` of the OTLP collector.  Defaults to
        ``ZWM_OTLP_ENDPOINT`` or ``OTEL_EXPORTER_OTLP_ENDPOINT``
        env var, or ``localhost:4317`` (the standard gRPC OTLP port).
    service_name : str
        ``service.name`` resource attribute.
    insecure : bool
        If True, use an unencrypted gRPC channel.  Set False in prod
        and provide TLS credentials via ``headers`` / env.
    headers : dict | None
        Extra gRPC metadata (e.g. ``{"x-honeycomb-team": "..."}``).
    """
    if not _OTEL_AVAILABLE:
        _log.info("configure_otlp: opentelemetry not installed (skipping)")
        return False
    ep = (
        endpoint
        or os.environ.get("ZWM_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or "localhost:4317"
    )
    try:
        # Resource: service.name, deployment.environment
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        _log.info("configure_otlp: opentelemetry-sdk not installed (skipping)")
        return False
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError:
        _log.info("configure_otlp: OTLP gRPC exporter not installed (skipping)")
        return False
    try:
        resource = Resource.create({
            "service.name": service_name,
            "service.version": os.environ.get("ZWM_VERSION", "0.1.0"),
            "deployment.environment":
                os.environ.get("ZWM_ENV", "dev"),
        })
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=ep,
            insecure=insecure,
            headers=headers or {},
            timeout=int(timeout_s),
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        _otel_trace.set_tracer_provider(provider)
        _tracer._otel_tracer = _otel_trace.get_tracer(service_name)
        _log.info(
            "configure_otlp: OTLP gRPC exporter bound to %s (service=%s)",
            ep, service_name,
        )
        return True
    except Exception as exc:
        _log.warning("configure_otlp: failed to install OTLP exporter: %s", exc)
        return False


def configure_otlp_from_env() -> bool:
    """H1 — convenience wrapper that reads ``ZWM_OTLP_*`` env vars.

    Honours the following env vars (all optional):

    * ``ZWM_OTLP_ENDPOINT``            — gRPC ``host:port``
    * ``ZWM_OTLP_SERVICE_NAME``        — defaults to ``zwm-agent``
    * ``ZWM_OTLP_INSECURE``            — ``1`` (default) / ``0``
    * ``ZWM_OTLP_TIMEOUT``             — seconds (default 10)

    Returns ``True`` on success, ``False`` if the SDK is not installed
    or the env vars do not explicitly enable the exporter
    (``ZWM_OTLP_ENABLED=1``).
    """
    if os.environ.get("ZWM_OTLP_ENABLED", "0") != "1":
        _log.info("configure_otlp_from_env: disabled (set ZWM_OTLP_ENABLED=1 to enable)")
        return False
    return configure_otlp(
        endpoint=None,
        service_name=os.environ.get("ZWM_OTLP_SERVICE_NAME", "zwm-agent"),
        insecure=os.environ.get("ZWM_OTLP_INSECURE", "1") == "1",
        timeout_s=float(os.environ.get("ZWM_OTLP_TIMEOUT", "10")),
    )


def render_recent(n: int = 20, tracer: "Tracer | None" = None) -> str:
    """P4-9 — pretty-print the most recent N spans, useful for the
    ``/debug/spans`` API endpoint or a CLI smoke test.

    ``tracer`` defaults to the module-level singleton but can be
    overridden (e.g. in unit tests) to render a local tracer's buffer.
    """
    src = tracer if tracer is not None else _tracer
    spans = src.spans[-n:]
    out: list[str] = []
    out.append(f"=== last {len(spans)} spans ===")
    for s in spans:
        out.append(
            f"  [{s.span_id}] {s.name:<24s} {s.duration*1000:7.2f} ms  "
            f"status={s.status}  attrs={len(s.attributes)}"
        )
    return "\n".join(out)


__all__ = [
    "Tracer",
    "SpanRecord",
    "get_tracer",
    "start_span",
    "configure_otel",
    "render_recent",
    "_span_guard",
]
