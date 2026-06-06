"""P2-1 (audit) — Prometheus metrics for ZWM.

Implements a minimal, dependency-free Prometheus exposition format endpoint
with the metrics that actually matter for the OODA loop:

  * ``zwm_ticks_total``            — counter, number of OODA ticks executed
  * ``zwm_tick_duration_seconds``  — histogram, wall time per tick
  * ``zwm_jepa_loss``              — gauge, latest JEPA prediction loss
  * ``zwm_surprise``               — gauge, latest world-model surprise
  * ``zwm_reward``                 — gauge, latest reward
  * ``zwm_episodes_stored``        — gauge, episodes in the SQLite store
  * ``zwm_react_reflections``      — gauge, reflections logged so far
  * ``zwm_mcts_iterations``        — gauge, MCTS budget per plan
  * ``zwm_active_experts``         — gauge, count of MoE experts active
  * ``zwm_particles``              — gauge, ensemble size of particle filter
  * ``zwm_efe_value``              — gauge, latest EFE score
  * ``zwm_hex_bits``               — gauge, current hexagram identity (1..64)

The module is deliberately small (no ``prometheus_client`` dependency): we
emit the standard ``# HELP`` / ``# TYPE`` lines and ``metric{label="..."} value``
rows that any Prometheus / OpenMetrics scraper can ingest.  Counters and
histograms are kept as in-memory dicts; state is per-process, which is what
Prometheus expects.

Usage::

    from zwm.observability import metrics
    metrics.inc_ticks()
    metrics.observe_tick_duration(0.123)
    print(metrics.render())  # Prometheus text exposition

Or via the FastAPI ``/metrics`` endpoint — see ``zwm.api.routes``.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import defaultdict
from typing import Iterable

# Histogram buckets in seconds, covering the full OODA budget from
# 0.5 ms (warm path) to 30 s (cold path / MCTS).
_HIST_BUCKETS: tuple[float, ...] = (
    0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
    2.5, 5.0, 10.0, 30.0,
)

# Standard Prometheus label sets — keep them tiny, label cardinality
# blows up the storage engine.
_JA_LABEL_NAMES: tuple[str, ...] = ("phase",)  # observe/predict/...


class _Counter:
    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        return self._value


class _Gauge:
    __slots__ = ("_value", "_lock")

    def __init__(self, default: float = 0.0) -> None:
        self._value: float = default
        self._lock = threading.Lock()

    def set(self, v: float) -> None:
        with self._lock:
            self._value = float(v)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value


class _Histogram:
    __slots__ = ("_counts", "_sum", "_lock")

    def __init__(self, buckets: Iterable[float]) -> None:
        self._counts: dict[float, int] = {b: 0 for b in buckets}
        self._sum: float = 0.0
        self._lock = threading.Lock()

    def observe(self, v: float) -> None:
        with self._lock:
            self._sum += v
            for b in self._counts:
                if v <= b:
                    self._counts[b] += 1

    def render(self, name: str, help_text: str, label_pairs: dict[str, str] | None = None) -> list[str]:
        """Render a histogram in Prometheus text-exposition format.

        ``label_pairs`` (e.g. ``{"phase": "observe"}``) is appended to
        the histogram's lines so that two histograms with the same
        ``name`` but different labels can coexist — the standard
        Prometheus pattern for "one metric, multiple label sets"."""
        with self._lock:
            lines = [
                f"# HELP {name} {help_text}",
                f"# TYPE {name} histogram",
            ]
            label_str = ""
            if label_pairs:
                label_str = "{" + ",".join(
                    f'{k}="{v}"' for k, v in label_pairs.items()
                ) + "}"
            cumulative = 0
            for b in sorted(self._counts):
                cumulative = self._counts[b]
                lines.append(f'{name}_bucket{label_str}{{le="{b}"}} {cumulative}')
            total = cumulative
            lines.append(f'{name}_bucket{label_str}{{le="+Inf"}} {total}')
            lines.append(f"{name}_sum{label_str} {self._sum}")
            lines.append(f"{name}_count{label_str} {total}")
        return lines


class MetricsRegistry:
    """Process-wide metrics store with Prometheus text rendering.

    Thread-safe (the agent's OODA loop is single-threaded but FastAPI
    request handlers may run in parallel).  Designed to be cheap enough
    to call on every tick (~hundreds of nanoseconds per call)."""

    def __init__(self) -> None:
        # Counters
        self.ticks_total = _Counter()
        self.errors_total = _Counter()
        # H4-限流: per-(scope,reason) rejection counter
        self.rate_limit_rejected: dict[tuple[str, str], _Counter] = defaultdict(_Counter)
        # Gauges
        self.jepa_loss = _Gauge()
        self.router_loss = _Gauge()
        self.surprise = _Gauge()
        self.reward = _Gauge()
        self.episodes_stored = _Gauge()
        self.react_reflections = _Gauge()
        self.mcts_iterations = _Gauge()
        self.active_experts = _Gauge()
        self.particles = _Gauge()
        self.efe_value = _Gauge()
        self.hex_bits = _Gauge()
        # P2-3: 复频谱 (spectrum) gauges
        self.interference_resonance = _Gauge()
        self.interference_phase_coherence = _Gauge()
        self.dominant_harmonic = _Gauge()
        # Histogram
        self.tick_duration = _Histogram(_HIST_BUCKETS)
        # Phase-scoped tick histogram (observe / predict / evaluate / act / learn)
        self.phase_duration: dict[str, _Histogram] = {
            phase: _Histogram(_HIST_BUCKETS)
            for phase in ("observe", "predict", "evaluate", "act", "learn")
        }
        # Lock for the text render path only.
        self._render_lock = threading.Lock()
        self._process_start = time.time()

    # ----- lifecycle -----
    def reset(self) -> None:
        """Re-initialise every metric to its zero state.

        Used by the test harness to guarantee determinism: the registry
        is a process-wide singleton, so without an explicit reset the
        counters/gauges/histograms accumulate across tests and make
        assertions order-dependent. Re-running ``__init__`` rebuilds all
        fields atomically while preserving the singleton identity.
        """
        self.__init__()

    # ----- public API -----
    def inc_ticks(self, n: int = 1) -> None:
        self.ticks_total.inc(n)

    def inc_errors(self, n: int = 1) -> None:
        self.errors_total.inc(n)

    def inc_rate_limit_rejected(self, scope: str, reason: str, n: int = 1) -> None:
        """H4-限流: record a rate-limit rejection for Prometheus scraping."""
        self.rate_limit_rejected[(scope, reason)].inc(n)

    def observe_tick_duration(self, seconds: float) -> None:
        self.tick_duration.observe(seconds)

    def observe_phase(self, phase: str, seconds: float) -> None:
        h = self.phase_duration.get(phase)
        if h is not None:
            h.observe(seconds)

    def set_jepa_loss(self, v: float | None) -> None:
        if v is not None and math.isfinite(v):
            self.jepa_loss.set(v)

    def set_router_loss(self, v: float | None) -> None:
        if v is not None and math.isfinite(v):
            self.router_loss.set(v)

    def set_surprise(self, v: float) -> None:
        if math.isfinite(v):
            self.surprise.set(v)

    def set_reward(self, v: float) -> None:
        if math.isfinite(v):
            self.reward.set(v)

    def set_episodes_stored(self, n: int) -> None:
        self.episodes_stored.set(int(n))

    def set_react_reflections(self, n: int) -> None:
        self.react_reflections.set(int(n))

    def set_mcts_iterations(self, n: int) -> None:
        self.mcts_iterations.set(int(n))

    def set_active_experts(self, n: int) -> None:
        self.active_experts.set(int(n))

    def set_particles(self, n: int) -> None:
        self.particles.set(int(n))

    def set_efe_value(self, v: float) -> None:
        if math.isfinite(v):
            self.efe_value.set(v)

    def set_hex_bits(self, n: int) -> None:
        self.hex_bits.set(int(n))

    def set_interference_resonance(self, v: float) -> None:
        if math.isfinite(v):
            self.interference_resonance.set(v)

    def set_interference_phase_coherence(self, v: float) -> None:
        if math.isfinite(v):
            self.interference_phase_coherence.set(v)

    def set_dominant_harmonic(self, n: int) -> None:
        self.dominant_harmonic.set(int(n))

    # ----- rendering -----
    def render(self) -> str:
        with self._render_lock:
            lines: list[str] = []
            # Counters
            lines.append("# HELP zwm_ticks_total Total OODA ticks executed")
            lines.append("# TYPE zwm_ticks_total counter")
            lines.append(f"zwm_ticks_total {self.ticks_total.value}")
            lines.append("# HELP zwm_errors_total Total errors during OODA")
            lines.append("# TYPE zwm_errors_total counter")
            lines.append(f"zwm_errors_total {self.errors_total.value}")
            # Gauges
            for name, gauge, help_text in (
                ("zwm_jepa_loss", self.jepa_loss, "Latest JEPA prediction loss"),
                ("zwm_router_loss", self.router_loss, "Latest MoE router loss"),
                ("zwm_surprise", self.surprise, "Latest world-model surprise"),
                ("zwm_reward", self.reward, "Latest reward"),
                ("zwm_episodes_stored", self.episodes_stored,
                 "Episodes in the SQLite store"),
                ("zwm_react_reflections", self.react_reflections,
                 "ReAct reflections logged so far"),
                ("zwm_mcts_iterations", self.mcts_iterations,
                 "MCTS budget per plan"),
                ("zwm_active_experts", self.active_experts,
                 "Count of MoE experts active per tick"),
                ("zwm_particles", self.particles,
                 "Ensemble size of the particle filter"),
                ("zwm_efe_value", self.efe_value, "Latest Expected Free Energy"),
                ("zwm_hex_bits", self.hex_bits, "Current hexagram (1..64)"),
                # P2-3: 复频谱 — resonance / phase coherence of the action's
                # complex-phase spectrum.  Useful for spotting ticks where
                # the chosen mutation is destructively interfering with
                # the trinity field.
                ("zwm_interference_resonance", self.interference_resonance,
                 "Resonance of the chosen action's complex-phase spectrum"),
                ("zwm_interference_phase_coherence", self.interference_phase_coherence,
                 "Phase coherence of the chosen action's spectrum"),
                ("zwm_dominant_harmonic", self.dominant_harmonic,
                 "Dominant harmonic (1..6) of the chosen action"),
                ("zwm_uptime_seconds", _Gauge(time.time() - self._process_start),
                 "Process uptime in seconds"),
            ):
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {gauge.value if isinstance(gauge, _Gauge) else gauge}")
            # Tick-duration histogram
            lines.extend(self.tick_duration.render(
                "zwm_tick_duration_seconds",
                "Wall time per OODA tick, in seconds",
            ))
            # Per-phase histograms
            for phase, h in self.phase_duration.items():
                lines.extend(h.render(
                    "zwm_phase_duration_seconds",
                    f"Wall time per OODA phase ({phase}), in seconds",
                    label_pairs={"phase": phase},
                ))
            # H4-限流: render rate-limit rejection counter
            if self.rate_limit_rejected:
                lines.append("# HELP zwm_rate_limit_rejected_total Rate-limit rejections")
                lines.append("# TYPE zwm_rate_limit_rejected_total counter")
                for (scope, reason), c in sorted(self.rate_limit_rejected.items()):
                    lines.append(
                        f'zwm_rate_limit_rejected_total{{scope="{scope}",reason="{reason}"}} {c.value}'
                    )
            return "\n".join(lines) + "\n"


# Singleton — imported wherever the agent loop lives.
metrics = MetricsRegistry()


# ----------------------------------------------------------------------
# P2-1 — structured logging
# ----------------------------------------------------------------------
def configure_logging(level: str | int = "INFO") -> None:
    """Configure the ``zwm`` logger with a JSON formatter.

    Honours the ``ZWM_LOG_FORMAT`` env var: ``"json"`` for production
    (machine-readable), ``"text"`` for development.  ``INFO`` is the
    default level.  Idempotent — calling twice won't double up handlers.
    """
    import os
    import sys

    fmt = os.environ.get("ZWM_LOG_FORMAT", "text").strip().lower()
    root = logging.getLogger("zwm")
    # Remove existing handlers to avoid double-emit on re-config.
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream=sys.stderr)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        ))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


class _JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter — emits one JSON object per line.

    Compatible with the structured log pipelines used by 2026 cloud
    providers (Cloud Logging, CloudWatch, Datadog, etc.).  Avoids any
    external dependency."""

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        from datetime import datetime, timezone
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k in self._RESERVED or k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except TypeError:
                payload[k] = str(v)
        return json.dumps(payload, ensure_ascii=False)


# ----------------------------------------------------------------------
# Convenience: a per-tick stopwatch that records phase duration.
# ----------------------------------------------------------------------
class PhaseStopwatch:
    """Tiny context manager for recording per-phase wall time."""

    __slots__ = ("_phase", "_t0", "_observer")

    def __init__(self, phase: str) -> None:
        self._phase = phase
        self._t0 = 0.0
        self._observer = metrics.observe_phase

    def __enter__(self) -> "PhaseStopwatch":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self._t0
        try:
            self._observer(self._phase, elapsed)
        except Exception:
            pass
