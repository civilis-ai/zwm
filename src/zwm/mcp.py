"""P3-1 (audit) — MCP (Model Context Protocol) server for ZWM.

Implements a minimal MCP-compliant JSON-RPC 2.0 server over stdio that
exposes the ZWM agent's three primary capabilities as discrete tools:

  * ``zwm/observe``   — 天 (sky): encode sensor/observation data into a hexagram
  * ``zwm/plan``      — 地 (earth): MCTS + EFE search for the next best action
  * ``zwm/reflect``   — 人 (human): retrieve ReAct chain-of-thought reflections

The protocol follows the 2026 MCP spec (https://modelcontextprotocol.io/):
each request is a JSON object with ``jsonrpc: "2.0"``, ``id``, ``method``,
and (optionally) ``params``.  We support three methods:

  * ``initialize``       — handshake, returns server capabilities
  * ``tools/list``       — returns the registered tool catalogue
  * ``tools/call``       — invokes a tool by name with JSON arguments

Usage::

    # stdio transport (the standard MCP transport)
    python -m zwm.mcp

    # OR via the CLI
    zwm mcp

This module is deliberately dependency-free — no ``mcp`` SDK, no
``fastapi`` — so it can be embedded in any agent runtime that already
ships a JSON parser.  A roundtrip looks like::

    $ echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \\
        | python -m zwm.mcp
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from zwm import __version__
from zwm.observability import configure_logging

_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# MCP protocol version (P5-2)
# ----------------------------------------------------------------------
# The 2025-06-18 release adds the Streamable-HTTP transport, structured
# content, and ``resources`` / ``prompts`` / ``sampling`` capabilities.
# Older clients (2024-11-05) are still served; we always advertise the
# highest version both sides support.
MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_LEGACY_VERSIONS = ("2024-11-05", "2025-03-26")


# ----------------------------------------------------------------------
# JSON-RPC 2.0 error codes (subset of the spec)
# ----------------------------------------------------------------------
ERR_PARSE = -32700          # Invalid JSON
ERR_INVALID_REQUEST = -32600  # Not a valid JSON-RPC request object
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603


@dataclass
class JsonRpcError(Exception):
    code: int
    message: str
    data: Any = None

    def to_dict(self) -> dict:
        d: dict = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


# ----------------------------------------------------------------------
# Tool registry — three canonical tools mapping to 天地人
# ----------------------------------------------------------------------
@dataclass
class ToolDef:
    name: str
    description: str
    # JSON-Schema-like description of the parameters (informational; we
    # don't actually validate against it — the runtime is small enough
    # that hand-rolled coercion is clearer than a schema engine).
    input_schema: dict = field(default_factory=dict)
    handler: Callable[[dict, Any], dict] | None = None


def _tool_observe(args: dict, agent: Any) -> dict:
    """天 — observe: encode sensor data into a hexagram.

    ``args``: ``{"sensor_data": {key: float}}`` or
              ``{"hex_name": "乾为天"}`` (one of the two is required).
    Returns ``{"hex_bits": int, "hex_name": str, "yao_signals": [...]}``.
    """
    from zwm.core.hexagram import hexagram_from_name, hexagram_from_bits
    if "hex_name" in args:
        h = hexagram_from_name(args["hex_name"])
    elif "hex_bits" in args:
        h = hexagram_from_bits(int(args["hex_bits"]))
    elif "sensor_data" in args:
        from zwm.encoder.base import RuleBasedEncoder
        h = RuleBasedEncoder().encode(args["sensor_data"])
    else:
        raise JsonRpcError(
            ERR_INVALID_PARAMS,
            "observe requires sensor_data, hex_name, or hex_bits",
        )
    return {
        "hex_bits": h.normal_order,
        "hex_name": h.name,
        "binary_str": h.binary_str,
        "yao_signals": [1.0 if line.is_yang else 0.0 for line in h.lines],
    }


def _tool_plan(args: dict, agent: Any) -> dict:
    """地 — plan: run MCTS + EFE for the next best action.

    ``args``: ``{"hex_bits": int, ...}`` (current state, default 乾=1).
    P4-7 (audit): every ``TrinityConfig`` field is now a valid override
    — callers can tweak ``mcts_iterations``, ``n_particles``,
    ``use_diffusion``, ``quantize`` from the MCP client.  The agent is
    built (or re-used) honouring the live config.

    Returns ``{"top_mutation": int, "top_score": float,
              "moe_active_experts": [..], "trajectory": [...]}``.
    """
    from zwm.core.hexagram import hexagram_from_bits
    from zwm.planner.surface import build_config_from_mcp_args
    from zwm.self_field.palace_graph import LuoshuGrid

    bits = int(args.get("hex_bits", 1))
    h = hexagram_from_bits(bits)

    # If the host passed in an agent, we *must* respect its config
    # (the caller already paid the cost of constructing it).  When the
    # host did not, we build a fresh agent using the MCP-supplied
    # overrides — the previous code hard-coded ``mcts_iters=50`` and
    # silently dropped every other knob.
    if agent is not None and hasattr(agent, "config"):
        config = build_config_from_mcp_args(args, base=agent.config)
        if config != agent.config:
            # Only spin up a temporary planner if the caller asked for
            # a different mcts_iters / n_particles.
            from zwm.planner.loop import TrinityPlanner
            planner = TrinityPlanner(
                mcts_iterations=config.mcts_iterations,
                use_diffusion=config.use_diffusion,
            )
        else:
            planner = agent.planner
    else:
        from zwm.planner.agent import TrinityAgent
        config = build_config_from_mcp_args(args)
        tmp = TrinityAgent(config=config)
        try:
            planner = tmp.planner
            plan = planner.plan(h, LuoshuGrid())
        finally:
            tmp.close()
        return {
            "top_mutation": plan.top_mutation,
            "top_score": plan.top_score,
            "moe_active_experts": plan.moe_active_experts,
            "moe_weight": plan.moe_weight,
            "trajectory": [
                {"name": name, "score": score} for name, score in plan.trajectory
            ],
        }
    plan = planner.plan(h, LuoshuGrid())
    return {
        "top_mutation": plan.top_mutation,
        "top_score": plan.top_score,
        "moe_active_experts": plan.moe_active_experts,
        "moe_weight": plan.moe_weight,
        "trajectory": [
            {"name": name, "score": score} for name, score in plan.trajectory
        ],
    }


def _tool_reflect(args: dict, agent: Any) -> dict:
    """人 — reflect: query the most recent ReAct chain-of-thought.

    ``args``: ``{"limit": int}`` (default 5).
    Returns ``{"reflections": [...], "count": int}``.
    """
    # ``agent`` is the optional TrinityAgent (set by the host).  When
    # absent (e.g. test harness), instantiate a temporary store.
    limit = int(args.get("limit", 5))
    if agent is not None and hasattr(agent, "store"):
        store = agent.store
    else:
        from zwm.storage.episodic_db import EpisodicStore
        store = EpisodicStore(
            db_path=args.get("db_path", "zwm_mcp.db"),
            use_index=False,
        )
        try:
            reflections = store.query_react_reflections(limit=limit)
            count = store.count_react_reflections()
        finally:
            store.close()
        return {"reflections": reflections, "count": count}
    return {
        "reflections": store.query_react_reflections(limit=limit),
        "count": store.count_react_reflections(),
    }


def _tool_preference(args: dict, agent: Any) -> dict:
    """P0-1: DPO preference feedback — record a human preference pair.

    ``args``: ``{"chosen_experts": [...], "rejected_experts": [...], "reward_diff": float}``
    Returns ``{"recorded": True, "pair_count": int}``.
    """
    chosen = args.get("chosen_experts", [])
    rejected = args.get("rejected_experts", [])
    reward_diff = float(args.get("reward_diff", 1.0))
    if not chosen or not rejected:
        return {"recorded": False, "reason": "both chosen_experts and rejected_experts required"}
    if agent is not None and hasattr(agent, "learner"):
        agent.learner.record_preference_pair(
            chosen_experts=chosen,
            rejected_experts=rejected,
            reward_diff=reward_diff,
        )
        return {"recorded": True, "pair_count": agent.learner.preference_pair_count}
    return {"recorded": False, "reason": "no agent available"}


TOOLS: list[ToolDef] = [
    ToolDef(
        name="zwm/observe",
        description=(
            "天 (sky): encode sensor data, hexagram name, or hexagram bits "
            "into a structured hexagram observation. Returns bits, name, and "
            "yao-level binary signals."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "sensor_data": {"type": "object",
                                "description": "key→float sensor readings"},
                "hex_name": {"type": "string",
                             "description": "Chinese hexagram name (e.g. 乾为天)"},
                "hex_bits": {"type": "integer",
                             "description": "normal_order index 1..64"},
            },
            "anyOf": [
                {"required": ["sensor_data"]},
                {"required": ["hex_name"]},
                {"required": ["hex_bits"]},
            ],
        },
        handler=_tool_observe,
    ),
    ToolDef(
        name="zwm/plan",
        description=(
            "地 (earth): run the Trinity planner (MCTS + EFE) to pick the "
            "next best mutation from a current hexagram state. P4-7: "
            "any TrinityConfig field is also accepted as an override "
            "(mcts_iterations, n_particles, use_diffusion, ...)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "hex_bits": {"type": "integer", "default": 1,
                             "description": "current hexagram normal_order"},
                # P4-7 — config-overrides surface.  We only advertise
                # the high-impact fields to keep the schema concise;
                # the surface module is the authoritative list.
                "mcts_iterations": {"type": "integer", "default": 50,
                                    "description": "MCTS budget"},
                "n_particles": {"type": "integer", "default": 0,
                                "description": "particle-filter ensemble size"},
                "use_diffusion": {"type": "boolean", "default": True,
                                  "description": "Langevin mutation sampling"},
                "use_react": {"type": "boolean", "default": True,
                              "description": "ReAct tool-use loop"},
            },
        },
        handler=_tool_plan,
    ),
    ToolDef(
        name="zwm/reflect",
        description=(
            "人 (human): retrieve the most recent ReAct chain-of-thought "
            "reflections stored in the agent's episodic DB. The reflections "
            "are the agent's *textual* reasoning, not the latent vector — "
            "they are the durable, queryable memory of every step the "
            "agent has reasoned about."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 5},
                "db_path": {"type": "string", "default": "zwm_mcp.db"},
            },
        },
        handler=_tool_reflect,
    ),
    ToolDef(
        name="zwm/preference",
        description=(
            "DPO alignment: record a human preference pair for preference "
            "optimisation.  Provide chosen_experts (list of expert names the "
            "user preferred) and rejected_experts (list of expert names the "
            "user rejected).  The pair is stored in the OnlineLearner's "
            "preference-pairs buffer and consumed by the periodic DPO step "
            "during agent training.  This is the standard DPO (Direct "
            "Preference Optimisation) alignment mechanism from 2024-2026 SOTA."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "chosen_experts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Expert names the user preferred",
                },
                "rejected_experts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Expert names the user rejected",
                },
                "reward_diff": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Strength of preference signal",
                },
            },
            "required": ["chosen_experts", "rejected_experts"],
        },
        handler=_tool_preference,
    ),
]


TOOL_BY_NAME: dict[str, ToolDef] = {t.name: t for t in TOOLS}


# ----------------------------------------------------------------------
# JSON-RPC dispatcher
# ----------------------------------------------------------------------
def _ok(id_: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _err(id_: Any, err: JsonRpcError) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": id_,
        "error": err.to_dict(),
    }


def dispatch(req: dict, agent: Any | None = None) -> dict | None:
    """Public dispatcher — catches ``JsonRpcError`` and converts it to
    a JSON-RPC error response.  Used by ``serve_stdio`` and the
    in-process test suite."""
    try:
        return _handle_request(req, agent=agent)
    except JsonRpcError as exc:
        return _err(req.get("id") if isinstance(req, dict) else None, exc)


def _handle_request(req: dict, agent: Any | None = None) -> dict | None:
    """Dispatch a single JSON-RPC request.  Returns the response object
    (or ``None`` for notifications, per the spec)."""
    if not isinstance(req, dict):
        raise JsonRpcError(ERR_INVALID_REQUEST, "request must be a JSON object")
    if req.get("jsonrpc") != "2.0":
        raise JsonRpcError(ERR_INVALID_REQUEST, "jsonrpc must be '2.0'")
    method = req.get("method")
    params = req.get("params") or {}
    req_id = req.get("id")
    is_notification = "id" not in req

    if method == "initialize":
        return _ok(req_id, {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "serverInfo": {
                "name": "zwm-mcp",
                "version": __version__,
            },
            # P5-2: advertise resources / prompts / sampling alongside
            # tools so 2025-06-18 clients (Cursor 1.5+, Claude 4.5+,
            # Cline 3.10+) accept the handshake.
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False},
                "prompts": {"listChanged": False},
                # Sampling is OPT-IN: the host must opt in by passing a
                # ``sampling`` callback to ``serve_stdio``.  We still
                # advertise the capability so the client knows we
                # *could* call back, but the runtime no-ops if no
                # callback is registered.
                "sampling": {},
            },
            "instructions": (
                "ZWM (天地人三才世界模型规划器) is a hexagram-based "
                "agent. Use the three tools zwm/observe, zwm/plan, "
                "zwm/reflect for the full OODA loop. Resources expose "
                "episodic memory (episodes://recent) and ReAct "
                "reflections (react://recent). Prompts provide "
                "ready-to-use planning templates."
            ),
        })

    if method == "tools/list":
        return _ok(req_id, {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in TOOLS
            ],
        })

    if method == "tools/call":
        if not isinstance(params, dict):
            raise JsonRpcError(ERR_INVALID_PARAMS, "params must be an object")
        tool_name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            raise JsonRpcError(ERR_INVALID_PARAMS, "arguments must be an object")
        tool = TOOL_BY_NAME.get(tool_name)
        if tool is None or tool.handler is None:
            raise JsonRpcError(
                ERR_METHOD_NOT_FOUND,
                f"unknown tool: {tool_name!r}",
            )
        try:
            result = tool.handler(args, agent)
        except JsonRpcError as exc:
            # Argument-validation errors should be reported as tool
            # results (isError=True) per the MCP spec, not as
            # transport-level errors — the call was well-formed, only
            # the params were bad.
            return _ok(req_id, {
                "content": [
                    {"type": "text", "text": f"{exc.message} ({exc.data!r})"
                                              if exc.data is not None else exc.message},
                ],
                "isError": True,
            })
        except Exception as exc:
            _log.exception("tool %s failed", tool_name)
            return _ok(req_id, {
                "content": [
                    {"type": "text", "text": f"error: {exc!r}"},
                ],
                "isError": True,
            })
        return _ok(req_id, {
            "content": [
                {"type": "text",
                 "text": json.dumps(result, ensure_ascii=False, default=str)},
            ],
            "isError": False,
        })

    if method == "ping":
        return _ok(req_id, {"pong": True})

    # P5-2: resources / prompts / sampling handlers.  These are
    # read-only handlers that do *not* mutate agent state — they
    # expose persistent data for the client to render / inject.
    if method == "resources/list":
        return _ok(req_id, {
            "resources": [
                {
                    "uri": "episodes://recent",
                    "name": "Recent Episodes",
                    "description": (
                        "Most recent episodes from the agent's "
                        "episodic SQLite store. Each row has "
                        "main_hex_bits, evolved_hex_bits, reward, "
                        "outcome_label."
                    ),
                    "mimeType": "application/json",
                },
                {
                    "uri": "react://recent",
                    "name": "Recent ReAct Reflections",
                    "description": (
                        "Most recent ReAct chain-of-thought "
                        "reflections (textual reasoning log)."
                    ),
                    "mimeType": "application/json",
                },
                {
                    "uri": "config://current",
                    "name": "Current TrinityConfig",
                    "description": (
                        "Live TrinityConfig of the active agent "
                        "(or defaults if no agent is hosted)."
                    ),
                    "mimeType": "application/json",
                },
            ],
        })

    if method == "resources/read":
        uri = params.get("uri") if isinstance(params, dict) else None
        if not isinstance(uri, str):
            raise JsonRpcError(ERR_INVALID_PARAMS, "uri must be a string")
        # Build a transient store if no agent is hosted.  We don't
        # mutate, so a temporary store is safe (P5-1 makes it thread
        # safe too).
        from zwm.storage.episodic_db import EpisodicStore
        if agent is not None and hasattr(agent, "store"):
            store = agent.store
            own_store = False
        else:
            store = EpisodicStore(use_index=False)
            own_store = True
        try:
            if uri == "episodes://recent":
                rows = store.query_recent(limit=20)
                text = json.dumps(rows, ensure_ascii=False, default=str)
            elif uri == "react://recent":
                rows = store.query_react_reflections(limit=20)
                text = json.dumps(rows, ensure_ascii=False, default=str)
            elif uri == "config://current":
                if agent is not None and hasattr(agent, "config"):
                    cfg = agent.config.to_dict()
                else:
                    from zwm.planner.agent_config import TrinityConfig
                    cfg = TrinityConfig().to_dict()
                text = json.dumps(cfg, ensure_ascii=False, default=str)
            else:
                raise JsonRpcError(
                    ERR_METHOD_NOT_FOUND, f"unknown resource: {uri!r}"
                )
        finally:
            if own_store:
                store.close()
        return _ok(req_id, {
            "contents": [
                {"uri": uri, "mimeType": "application/json", "text": text},
            ],
        })

    if method == "prompts/list":
        return _ok(req_id, {
            "prompts": [
                {
                    "name": "zwm-plan-from-state",
                    "description": (
                        "Plan the next best action from a current "
                        "hexagram state — returns a single mutation "
                        "candidate and the active MoE experts."
                    ),
                    "arguments": [
                        {
                            "name": "hex_name",
                            "description": "Current hexagram name",
                            "required": False,
                        },
                        {
                            "name": "mcts_iterations",
                            "description": "MCTS budget (10-500)",
                            "required": False,
                        },
                    ],
                },
                {
                    "name": "zwm-reflect-on-outcome",
                    "description": (
                        "Reflect on the most recent outcome and "
                        "produce a textual chain-of-thought for the "
                        "next tick (Reflexion-style)."
                    ),
                    "arguments": [
                        {
                            "name": "limit",
                            "description": "Number of past reflections to include",
                            "required": False,
                        },
                    ],
                },
            ],
        })

    if method == "prompts/get":
        name = params.get("name") if isinstance(params, dict) else None
        arguments = (params.get("arguments") or {}) if isinstance(params, dict) else {}
        if name == "zwm-plan-from-state":
            hex_name = arguments.get("hex_name", "乾为天")
            mcts = arguments.get("mcts_iterations", 50)
            text = (
                f"Plan the next best mutation from the current "
                f"hexagram '{hex_name}'. Use the zwm/plan tool with "
                f"hex_bits=<resolve from name>, mcts_iterations={mcts}. "
                f"Explain your interpretation of the top_score and "
                f"the active MoE experts in 2-3 sentences."
            )
        elif name == "zwm-reflect-on-outcome":
            limit = arguments.get("limit", 5)
            text = (
                f"Call zwm/reflect with limit={limit} and write a "
                f"concise textual chain-of-thought explaining the "
                f"patterns you see. Then call zwm/plan with the "
                f"current hex to suggest the next step."
            )
        else:
            raise JsonRpcError(
                ERR_METHOD_NOT_FOUND, f"unknown prompt: {name!r}"
            )
        return _ok(req_id, {
            "description": f"ZWM prompt: {name}",
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": text},
                },
            ],
        })

    if method == "sampling/createMessage":
        # Sampling is opt-in: the host (e.g. Claude) registers a
        # callback at ``serve_stdio(agent, sampling=fn)``.  When no
        # callback is registered we return a structured "not
        # available" response so the client falls back gracefully.
        cb = globals().get("_SAMPLING_CALLBACK")
        if cb is None:
            return _ok(req_id, {
                "error": {
                    "code": -32603,
                    "message": "sampling not available: host did not register a callback",
                },
            })
        try:
            result = cb(params)
        except Exception as exc:
            _log.exception("sampling callback failed")
            return _ok(req_id, {
                "error": {"code": -32603, "message": f"sampling failed: {exc}"},
            })
        return _ok(req_id, result)

    raise JsonRpcError(ERR_METHOD_NOT_FOUND, f"unknown method: {method!r}")


# ----------------------------------------------------------------------
# Stdio transport — one JSON object per line in / out
# ----------------------------------------------------------------------
def serve_stdio(
    agent: Any | None = None,
    sampling: Callable[[dict], dict] | None = None,
) -> int:
    """Read JSON-RPC requests from stdin, write responses to stdout.

    ``agent`` (optional) is passed to every tool handler so the tools can
    read from / write to the live agent's state instead of building a
    fresh store on every call.

    ``sampling`` (optional, P5-2) is the host-supplied callback invoked
    by the ``sampling/createMessage`` MCP method.  When supplied, the
    MCP server can ask the host LLM to score / rephrase a thought
    before persisting it.  When absent, sampling no-ops gracefully.
    """
    global _SAMPLING_CALLBACK
    _SAMPLING_CALLBACK = sampling
    configure_logging(level="INFO")
    _log.info("zwm-mcp %s starting (stdio transport, protocol %s)",
              __version__, MCP_PROTOCOL_VERSION)
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as exc:
            resp = _err(None, JsonRpcError(ERR_PARSE, f"invalid JSON: {exc}"))
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue
        resp = dispatch(req, agent=agent)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console entry point — ``python -m zwm.mcp`` or ``zwm mcp``."""
    return serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
