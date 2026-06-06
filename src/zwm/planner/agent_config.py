"""P4-6 (audit) — TrinityAgent configuration dataclass.

Replaces the loose ``self._cfg = {...}`` dict in ``TrinityAgent.__init__``
with a typed, frozen dataclass.  Benefits:

* **Type safety** — every field has a real type the static checker
  understands.
* **Discoverability** — ``TrinityConfig.__dataclass_fields__`` is the
  authoritative list of knobs; CLI / API / MCP layers can introspect it
  (see :pymod:`zwm.cli.serve`).
* **Immutability** — ``frozen=True`` prevents accidental mid-loop mutation
  that previously caused subtle bugs when one phase method patched
  ``self._cfg["..."]``.
* **Topology inlining** — the topology expansion is configured here, not
  as a side-effect of ``__init__``, so callers can pre-compute / share a
  topology between agents if they want.

The dataclass deliberately omits *runtime* attributes (the planner, the
encoder, the SQLite store, …) — those remain plain instance attributes
on ``TrinityAgent`` because they are not configuration, they are state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zwm.self_field.palace_graph import LuoshuGrid


@dataclass(frozen=True, slots=True)
class TrinityConfig:
    """Static configuration for :class:`TrinityAgent`.

    All defaults preserve the behaviour of the previous dict-based
    configuration; the audit renamed a few fields to drop the leading
    ``use_`` prefix where it was obvious (e.g. ``quantize`` is
    self-describing) and inlined the topology depth.
    """

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    db_path: str = "zwm_episodes.db"
    """SQLite path for the episodic store."""
    semantic_path: str | None = None
    """Optional SQLite path for the semantic store; ``None`` disables it."""
    checkpoint_path: str | None = None
    """Optional ``.pt`` path for save/load of all learning state."""

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    mcts_iterations: int = 200
    """Number of MCTS iterations per tick (recommended 100-500)."""
    n_particles: int = 16
    """Particle-filter ensemble size; ``0`` disables the filter."""
    use_diffusion: bool = True
    """Use Langevin-dynamics mutation sampling inside MCTS."""

    # ------------------------------------------------------------------
    # World model
    # ------------------------------------------------------------------
    learnable_encoder: bool = True
    """Use :class:`LearnableSquareGNN` for the square-circular joint."""
    use_field_encoder: bool = True
    """Use :class:`HexagramFieldEncoder` + :class:`FieldSquareGNN`.

    When True (default), the agent encodes sensor data as a 64-hexagram
    field (384-bit state) instead of a single hexagram (6-bit state).
    Set to False for backward-compatible single-hexagram encoding.
    """
    hierarchical: bool = False
    """Use :class:`HierarchicalJEPAPredictor` instead of the flat one."""
    use_fsdp2: bool = False
    """Wrap the JEPA predictor in FSDP2 (multi-GPU)."""
    quantize: str | None = None
    """One of ``None``, ``"4bit"``, ``"lora"``, ``"qlora"``."""

    # ------------------------------------------------------------------
    # VSA / memory
    # ------------------------------------------------------------------
    use_trainable_vsa: bool = True
    """Use :class:`TrainableVSACodebook` (gradient-trained) vs static."""

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------
    use_react: bool = True
    """Enable the ReAct tool-use loop in front of the OODA planner."""
    topology_max_depth: int = 2
    """Recursive nine-palace topology depth; ``0`` = 1 node, ``3`` = 729."""

    # ------------------------------------------------------------------
    # Grid (optional override)
    # ------------------------------------------------------------------
    grid: "LuoshuGrid | None" = None
    """Caller can pin a specific :class:`LuoshuGrid`; ``None`` builds one."""

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------
    enable_constitution: bool = True
    """P4-8 — enable the constitutional-AI safety guardrails.

    When ``True``, every input (sensor data, tick request) and output
    (plan, mutation) is checked against
    :data:`zwm.safety.constitution.DEFAULT_CONSTITUTION` and BLOCK-severity
    failures raise :class:`ConstitutionalViolation`.

    Set to ``False`` to bypass the guardrails — useful for offline
    research but should never be done in production.
    """

    # ------------------------------------------------------------------
    # L3 — Observability
    # ------------------------------------------------------------------
    enable_tracing: bool = True
    """L3 — record per-phase spans in the in-process ring buffer.

    Disable for the absolute lowest-latency OODA path (saves ~5 µs/tick
    on a 2026 CPU).  When disabled, the
    :mod:`zwm.tracing` module still works but produces no spans.
    """
    otlp_endpoint: str | None = None
    """L3 — gRPC OTLP collector endpoint (``host:port``).

    ``None`` (default) defers to the ``ZWM_OTLP_ENDPOINT`` env var or
    the standard ``OTEL_EXPORTER_OTLP_ENDPOINT``.  When unset on both
    fronts the tracer stays in-process (no network call).
    """
    otlp_service_name: str = "zwm-agent"
    """L3 — ``service.name`` resource attribute reported to the OTLP
    collector."""
    enable_otlp: bool = False
    """L3 — explicitly enable OTLP export.

    Defaults to ``False`` so a vanilla ``TrinityConfig()`` stays
    self-contained; users opt in via ``TrinityConfig(enable_otlp=True)``
    or the ``ZWM_OTLP_ENABLED=1`` env var.
    """

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "TrinityConfig":
        """Backwards-compat: build a config from the legacy ``_cfg`` dict.

        Unknown keys are silently dropped (with a debug log) so legacy
        callers that pass extra options do not crash.
        """
        import logging as _logging
        _log = _logging.getLogger(__name__)
        valid = {f for f in cls.__dataclass_fields__}
        clean: dict = {}
        for k, v in data.items():
            if k in valid:
                clean[k] = v
            else:
                _log.debug("TrinityConfig.from_dict: dropped unknown key %r", k)
        return cls(**clean)

    def to_dict(self) -> dict:
        """Round-trip with :meth:`from_dict`; omits ``None`` grid."""
        return {
            f: getattr(self, f)
            for f in self.__dataclass_fields__
            if f != "grid" or getattr(self, f) is not None
        }

    def as_overrides(self) -> dict:
        """Return only the non-default values, useful for telemetry labels."""
        from dataclasses import fields, MISSING
        out: dict = {}
        for f in fields(self):
            default = f.default
            value = getattr(self, f.name)
            if value != default and not (default is MISSING and value is None):
                out[f.name] = value
        return out


__all__ = ["TrinityConfig"]
