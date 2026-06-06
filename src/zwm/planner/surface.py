"""P4-7 (audit) — Unified config surface for CLI / API / MCP.

The three entry points used to each maintain their own subset of
``TrinityConfig`` knobs, which meant:

* ``zwm tick --mcts-iters`` accepted an int but ``SessionStartRequest``
  had a different default.
* The MCP ``zwm/plan`` tool silently ignored the agent's
  ``n_particles`` / ``use_diffusion`` / ``quantize`` settings — the
  caller could not override them either.
* Adding a new config field required touching three files plus tests.

This module fixes that.  Each surface now has *one* builder that
maps ``TrinityConfig`` to the surface-specific schema:

  * ``config_to_argparse(parser)``            → argparse CLI
  * ``ConfigOverrides(BaseModel)``            → pydantic FastAPI
  * ``config_to_mcp_schema(field_subset)``   → JSON-Schema for MCP

The field set is the authoritative ``TrinityConfig.__dataclass_fields__``
— no field can drift between the three surfaces any more.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import MISSING, fields as _dc_fields
from typing import Any, Callable, get_args, get_origin

from pydantic import BaseModel, Field, create_model

_log = logging.getLogger(__name__)


# Public surface mapping — one place to change a CLI flag, API field
# name, or MCP arg.
FIELD_NAMES: tuple[str, ...] = tuple(
    f.name for f in _dc_fields(__import__("zwm.planner.agent_config", fromlist=["TrinityConfig"]).TrinityConfig)
)


def _flag_for(name: str) -> str:
    """Convert a snake_case dataclass field to a CLI --flag-name."""
    return "--" + name.replace("_", "-")


_TYPE_TO_ARGPARSE: dict[type, tuple[Callable, dict[str, Any]]] = {
    bool: (argparse.BooleanOptionalAction, {}),
    int: (int, {}),
    float: (float, {}),
    str: (str, {}),
}


def _resolved_fields() -> list:
    """P4-7 — return dataclass fields with *real* types resolved.

    ``from __future__ import annotations`` in ``agent_config.py``
    means ``f.type`` is a string until we call ``typing.get_type_hints``.
    Without resolution, all flags degrade to ``--name str`` and the
    BooleanOptionalAction never gets installed.
    """
    import typing
    from zwm.planner.agent_config import TrinityConfig
    # The LuoshuGrid type is behind a TYPE_CHECKING guard in
    # ``agent_config.py``; we import it lazily here so runtime
    # annotation evaluation succeeds.
    try:
        from zwm.self_field.palace_graph import LuoshuGrid  # noqa: F401
    except Exception:
        LuoshuGrid = None  # type: ignore[assignment]
    try:
        hints = typing.get_type_hints(TrinityConfig, localns=locals())
    except Exception:
        # Fallback: a partial hint map (without LuoshuGrid) still
        # gives us correct types for every other field.
        hints = typing.get_type_hints(TrinityConfig, include_extras=False)
    out = []
    for f in _dc_fields(TrinityConfig):
        real = hints.get(f.name, f.type)
        out.append((f, real))
    return out


def config_to_argparse(parser: argparse.ArgumentParser) -> None:
    """P4-7 — attach ``--<name>`` flags to ``parser`` for every config
    field.  Bool fields use ``--<name> / --no-<name>`` so the user can
    pick the polarity.
    """
    from zwm.planner.agent_config import TrinityConfig
    for f, real in _resolved_fields():
        if f.name == "grid":
            # Grid is a complex object — caller passes a path or
            # in-process reference; not a CLI flag.
            continue
        if real is bool:
            parser.add_argument(
                _flag_for(f.name),
                action=argparse.BooleanOptionalAction,
                default=f.default,
                help=f"(TrinityConfig.{f.name})",
            )
            continue
        if real in _TYPE_TO_ARGPARSE:
            ctor, kwargs = _TYPE_TO_ARGPARSE[real]
            parser.add_argument(
                _flag_for(f.name),
                type=ctor,
                default=f.default,
                help=f"(TrinityConfig.{f.name})",
                **kwargs,
            )
            continue
        # Optional[...], list, tuple, etc. — accept as a string and
        # let ``build_config`` coerce.
        parser.add_argument(
            _flag_for(f.name),
            type=str,
            default=None,
            help=f"(TrinityConfig.{f.name}) — JSON literal",
        )


def argparse_to_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """P4-7 — pull only the recognised config flags out of an argparse
    Namespace and return them as a kwargs dict.  Unknown attributes are
    silently dropped (with a debug log).
    """
    from zwm.planner.agent_config import TrinityConfig
    valid = {f.name for f in _dc_fields(TrinityConfig)}
    out: dict[str, Any] = {}
    for f in _dc_fields(TrinityConfig):
        if f.name == "grid":
            continue
        if hasattr(args, f.name):
            v = getattr(args, f.name)
            if v is None and f.default is None:
                continue
            out[f.name] = v
    # Also sweep extras defensively.
    for name in list(vars(args).keys()):
        if name in valid and name not in out:
            out[name] = getattr(args, name)
    return out


def _py_type_to_pydantic(field_type: Any) -> Any:
    """Map a Python annotation to a pydantic-compatible type.

    Strips ``Optional[X]``/``X | None`` wrappers and emits ``X | None``
    instead.  For everything else, returns the type unchanged.
    """
    origin = get_origin(field_type)
    args = get_args(field_type)
    if origin is None:
        return field_type
    # Optional[X] or X | None
    if type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0] | None
    return field_type


def build_config_overrides_model() -> type[BaseModel]:
    """P4-7 — dynamically build a pydantic ``BaseModel`` exposing the
    *overridable* subset of :class:`TrinityConfig`.

    The FastAPI ``SessionStartRequest`` now inherits from this model
    (or, more precisely, embeds it) so any new field added to
    ``TrinityConfig`` automatically appears in the API surface.
    """
    from zwm.planner.agent_config import TrinityConfig
    model_fields: dict[str, tuple[Any, Any]] = {}
    for f, real in _resolved_fields():
        if f.name == "grid":
            # Grid is not serialisable over HTTP.
            continue
        py_type = _py_type_to_pydantic(real)
        default = f.default if f.default is not MISSING else None
        model_fields[f.name] = (py_type, Field(default=default))
    Model: type[BaseModel] = create_model(  # type: ignore[call-overload]
        "ConfigOverrides",
        **model_fields,
    )
    return Model


def config_to_mcp_schema(name: str, description: str) -> dict:
    """P4-7 — convert :class:`TrinityConfig` to an MCP JSON Schema.

    Bool becomes ``{"type": "boolean"}``; numbers / ints are
    ``{"type": "integer" | "number"}``; strings / Optional are handled
    accordingly.
    """
    from zwm.planner.agent_config import TrinityConfig
    properties: dict[str, dict] = {}
    for f, real in _resolved_fields():
        if f.name == "grid":
            continue
        py_type = _py_type_to_pydantic(real)
        if py_type is bool:
            properties[f.name] = {"type": "boolean"}
        elif py_type is int:
            properties[f.name] = {"type": "integer"}
        elif py_type is float:
            properties[f.name] = {"type": "number"}
        elif py_type is str:
            properties[f.name] = {"type": "string"}
        else:
            properties[f.name] = {"type": "string"}  # fall-through
        if f.default is not MISSING and f.default is not None:
            properties[f.name]["default"] = f.default
    return {
        "type": "object",
        "properties": properties,
        "description": description,
    }


def apply_overrides(
    base: "Any | None",
    overrides: dict[str, Any],
) -> "Any":
    """P4-7 — return a new ``TrinityConfig`` (or the original, if no
    overrides apply) with the given fields replaced.  ``None`` values
    are dropped, so the caller can pass a partial dict.
    """
    from zwm.planner.agent_config import TrinityConfig
    if not overrides:
        return base if base is not None else TrinityConfig()
    if base is None:
        base = TrinityConfig()
    clean = {k: v for k, v in overrides.items() if v is not None}
    if not clean:
        return base
    return TrinityConfig(**{**base.to_dict(), **clean})


def build_config_from_args(
    args: argparse.Namespace,
    base: "Any | None" = None,
) -> "Any":
    """P4-7 — convenience: parse the recognised config flags out of
    ``args`` and apply them on top of ``base`` (a :class:`TrinityConfig`
    or ``None``).
    """
    return apply_overrides(base, argparse_to_config_kwargs(args))


def build_config_from_overrides(
    overrides: "BaseModel | dict | None",
    base: "Any | None" = None,
) -> "Any":
    """P4-7 — apply a pydantic ``ConfigOverrides`` model (or a plain
    dict) on top of ``base``."""
    if overrides is None:
        return base
    if isinstance(overrides, BaseModel):
        d = overrides.model_dump(exclude_none=True)
    else:
        d = dict(overrides)
    return apply_overrides(base, d)


def build_config_from_mcp_args(
    args: dict,
    base: "Any | None" = None,
) -> "Any":
    """P4-7 — apply MCP ``zwm/plan`` arguments on top of ``base``.

    The MCP client may pass a mix of ``hex_bits`` (tool-specific) and
    ``mcts_iterations`` (config) arguments.  Only the config fields
    are forwarded to :class:`TrinityConfig`; the rest are dropped so
    we never hit ``unexpected keyword argument``.
    """
    from zwm.planner.agent_config import TrinityConfig
    valid = {f.name for f in _dc_fields(TrinityConfig)}
    cfg_args = {k: v for k, v in args.items() if k in valid}
    return apply_overrides(base, cfg_args)


__all__ = [
    "FIELD_NAMES",
    "config_to_argparse",
    "argparse_to_config_kwargs",
    "build_config_overrides_model",
    "config_to_mcp_schema",
    "apply_overrides",
    "build_config_from_args",
    "build_config_from_overrides",
    "build_config_from_mcp_args",
]
