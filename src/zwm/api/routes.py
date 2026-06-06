"""P2-3 (audit) — FastAPI 路由 (REST + WebSocket)。

所有端点共享一个 ``TrinityAgent`` 单例 (lifespan 管理)。
同步 OODA 循环通过线程池执行,避免阻塞 asyncio 事件循环。
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import Request

from .schemas import (
    HealthResponse,
    InfoResponse,
    SessionHistory,
    SessionInfo,
    SessionStartRequest,
    TickReportResponse,
    TickRequest,
    WSTickCommand,
    WSTickResult,
)

# ------------------------------------------------------------------
# 单例 agent (由 lifespan 注入)
# ------------------------------------------------------------------
_agent: object = None         # TrinityAgent
_sessions: dict[str, object] = {}  # session_id → TrinityAgent


def set_agent(agent) -> None:
    global _agent
    _agent = agent


def get_agent():
    if _agent is None:
        raise RuntimeError("Agent not initialised — call set_agent() before serving")
    return _agent


# ------------------------------------------------------------------
# REST 端点
# ------------------------------------------------------------------
async def health():
    from zwm import __version__
    return HealthResponse(version=__version__)


async def info():
    from zwm import __version__
    from zwm.cli import MODULES
    return InfoResponse(
        name="ZWM",
        version=__version__,
        modules=list(MODULES.keys()),
        description="天地人三才世界模型规划器",
    )


async def metrics_endpoint():
    """P2-1 — Prometheus text-exposition endpoint.

    Returns ``text/plain; version=0.0.4`` (the standard Prometheus
    content-type) with the current process-wide metrics.  Intentionally
    unauthenticated so Prometheus / OpenMetrics scrapers can poll it
    without a Bearer token.  When the ``prometheus_client`` library is
    not installed, the lightweight in-process registry is used
    (see ``zwm.observability``)."""
    from fastapi.responses import PlainTextResponse
    from zwm.observability import metrics as _m
    # Refresh process-wide gauges from the live agent before rendering.
    try:
        agent = get_agent()
        if agent is not None and hasattr(agent, "store"):
            try:
                _m.set_episodes_stored(agent.store.count())
            except Exception:
                pass
            if hasattr(agent.store, "count_react_reflections"):
                try:
                    _m.set_react_reflections(agent.store.count_react_reflections())
                except Exception:
                    pass
        if agent is not None and hasattr(agent, "_particle_filter"):
            try:
                _m.set_particles(agent._particle_filter.belief.n)
            except Exception:
                pass
        if agent is not None and hasattr(agent, "planner"):
            try:
                _m.set_mcts_iterations(agent.planner._mcts_iterations)
            except Exception:
                pass
    except RuntimeError:
        # Agent not initialised — render with whatever state we have.
        pass
    return PlainTextResponse(
        content=_m.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


async def do_tick(req: TickRequest):
    """单步 OODA 循环 — 无状态, 每次创建临时 agent。

    P4-8 — wraps the synchronous tick in a constitutional-violation
    trap so a malicious or malformed payload yields a clean HTTP 422
    rather than a 500.
    """
    from zwm.safety.constitution import ConstitutionalViolation
    from fastapi import HTTPException
    try:
        agent = get_agent()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _tick_sync, agent, req)
    except ConstitutionalViolation as exc:
        # Bubble up as a 422 — the request was syntactically valid
        # (it parsed) but failed the safety policy.
        raise HTTPException(
            status_code=422,
            detail=f"constitutional violation: {exc}",
        ) from exc


def _tick_sync(agent, req: TickRequest):
    from zwm.core.hexagram import hexagram_from_bits
    h_current = None
    if req.h_current is not None:
        h_current = hexagram_from_bits(req.h_current)
    report = agent.observe_predict_evaluate_act(
        sensor_data=req.sensor_data,
        h_current=h_current,
        year=req.year,
        month=req.month,
        day=req.day,
        hour=req.hour,
        time_phase=req.time_phase,
        target_palace=req.target_palace,
        day_gan=req.day_gan,
        reward=req.reward,
        language_text=req.language_text,
        vision_features=np.array(req.vision_features, dtype=np.float32) if req.vision_features else None,
    )
    return _report_to_response(report)


async def session_start(req: SessionStartRequest):
    """P4-7 — create a persistent agent session.

    The request now embeds the dynamic ``ConfigOverrides`` model, so
    any ``TrinityConfig`` field is accepted (with its declared
    default).  We no longer enumerate ``db_path=`` / ``mcts_iterations=``
    / etc. by hand — the surface module is the single source of truth.
    """
    session_id = uuid.uuid4().hex[:12]
    def _build():
        from zwm.planner.agent import TrinityAgent
        from zwm.planner.surface import build_config_from_overrides
        config = build_config_from_overrides(req)
        return TrinityAgent(config=config)
    loop = asyncio.get_running_loop()
    agent = await loop.run_in_executor(None, _build)
    _sessions[session_id] = agent
    # AUDIT-I2: read the real step_count from the freshly-built agent.
    step = int(getattr(agent, "_step_count", 0))
    return SessionInfo(
        session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        step_count=step,
        db_path=str(req.db_path) if req.db_path else "zwm_sessions.db",
    )


async def session_tick(session_id: str, req: TickRequest):
    agent = _sessions.get(session_id)
    if agent is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _tick_sync, agent, req)


async def session_history(session_id: str, limit: int = 50):
    agent = _sessions.get(session_id)
    if agent is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    eps = agent.store.query_recent(limit=limit)
    ticks = []
    for ep in eps:
        # P0-1: read the real metrics from the context_json (not 0.0).
        # The agent's _learn() phase writes jepa_loss / surprise / router_loss
        # / moe_active_experts / top_mutation into context on every store().
        ctx = ep.get("context") or {}
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except Exception:
                ctx = {}
        # AUDIT-I1: column names from ``EpisodicStore._row_to_dict`` are
        # ``main_hex_bits`` / ``evolved_hex_bits`` — the route used to
        # read ``main_bits`` / ``evolved_bits`` (the add_episode() kwarg
        # names) and silently got 0 for every episode, which meant the
        # /session/{id}/history endpoint was reporting the wrong
        # hexagrams for the entire history.
        main = int(ep.get("main_hex_bits", 0))
        evolved = int(ep.get("evolved_hex_bits", 0))
        ticks.append(TickReportResponse(
            episode_id=ep.get("id", 0),
            # AUDIT-I9: use the persisted DB timestamp, not datetime.now().
            # The latter is identical for every tick in the same response,
            # which destroys the ordering semantics of /history.
            timestamp=str(ep.get("timestamp", "")),
            h_current={"normal_order": main},
            h_next={"normal_order": evolved},
            top_mutation=int(ctx.get("top_mutation", main ^ evolved)),
            top_score=float(ctx.get("top_score", 0.0)),
            reward=float(ep.get("reward", 0.0)),
            jepa_loss=ctx.get("jepa_loss"),
            router_loss=ctx.get("router_loss"),
            surprise=float(ctx.get("surprise", 0.0)),
            mutation_class=str(ctx.get("mutation_class", "")),
            codon=str(ctx.get("codon", "")),
            codon_aa=str(ctx.get("codon_aa", "")),
            moe_active_experts=list(ctx.get("moe_active_experts", [])),
            trajectory=list(ctx.get("trajectory", [])),
        ))
    return SessionHistory(session_id=session_id, ticks=ticks, total=len(ticks))


async def session_delete(session_id: str):
    agent = _sessions.pop(session_id, None)
    if agent is not None:
        agent.close()
    return {"deleted": session_id}


# ------------------------------------------------------------------
# P0-1: DPO preference feedback
# ------------------------------------------------------------------
# P0-3: Use Request body for JSON parsing (not query params).
async def preference_feedback(request: Request):
    """Record a human preference pair for DPO alignment.

    POST /preference
    Body: {"chosen_experts": ["time", "space"], "rejected_experts": ["risk"],
           "reward_diff": 1.0}

    Stored in the OnlineLearner's preference-pairs buffer and consumed
    by the periodic DPO step (every 4 ticks).
    """
    body = await request.json()
    chosen_experts = body.get("chosen_experts")
    rejected_experts = body.get("rejected_experts")
    reward_diff = body.get("reward_diff", 1.0)
    agent = get_agent()
    if chosen_experts and rejected_experts:
        agent.learner.record_preference_pair(
            chosen_experts=chosen_experts,
            rejected_experts=rejected_experts,
            reward_diff=reward_diff,
        )
        return {
            "recorded": True,
            "pair_count": agent.learner.preference_pair_count,
        }
    return {"recorded": False, "reason": "both chosen_experts and rejected_experts required"}


# ------------------------------------------------------------------
# WebSocket
# ------------------------------------------------------------------
async def ws_tick(websocket):
    """实时 WebSocket 流式 OODA — 客户端发送 WSTickCommand, 服务端推送 WSTickResult。

    P2-1 (audit): 使用 AsyncAgent 包装器替代 ``run_in_executor``,
    原生 async 接口无需线程池, 不阻塞事件循环。

    AUDIT-I3: Bearer-token gate.  The HTTP routes are protected by
    ``_verify_bearer`` (see ``zwm.api.app``), but the WebSocket was
    open to anyone because the FastAPI ``Depends`` mechanism does not
    apply to ``websocket.endpoint`` handlers.  We now perform the same
    constant-time check inline.  When ``ZWM_API_TOKEN`` is unset (or
    ``ZWM_REQUIRE_AUTH=0``), auth is skipped — same convention as the
    REST surface.

    Clients authenticate by passing a ``token`` field in the very first
    JSON message, or via the ``Sec-WebSocket-Protocol: bearer.<token>``
    sub-protocol header (the standard WebSocket pattern for auth).

    H4-限流: 集成令牌桶 + 滑动窗口限流器, 防止 DoS.
    每个 WebSocket 连接 / 客户端 IP 独立计数.
    超限时不关闭连接, 而是发送 429 帧并等待, 保护正常流量.
    """
    import os
    import secrets as _secrets
    expected = os.environ.get("ZWM_API_TOKEN", "").strip()
    require_auth = os.environ.get("ZWM_REQUIRE_AUTH", "1").strip() != "0"
    authenticated = not (expected and require_auth)

    # 1) Sub-protocol auth (preferred — happens during the handshake).
    if not authenticated:
        try:
            for proto in (websocket.headers.get("sec-websocket-protocol") or "").split(","):
                proto = proto.strip()
                if proto.startswith("bearer."):
                    token = proto[len("bearer."):]
                    if _secrets.compare_digest(token, expected):
                        authenticated = True
                        break
        except Exception:
            pass

    # 2) ``auth_token`` query-param fallback (handy for ``wscat`` / CLI
    #    smoke tests; not as secret-safe as the sub-protocol).
    if not authenticated:
        token = websocket.query_params.get("token", "")
        if token and _secrets.compare_digest(token, expected):
            authenticated = True

    if not authenticated:
        # Reject the connection *before* accepting it.
        from fastapi import WebSocketDisconnect
        await websocket.close(code=4401, reason="Unauthorized")
        return

    # MAJOR-3 FIX: Accept once with the negotiated sub-protocol.
    # The previous code called accept() twice — once without subprotocol
    # and once with "zwm.v1" — which caused a double-accept error on
    # any compliant WebSocket implementation.
    subproto = "zwm.v1"
    await websocket.accept(subprotocol=subproto)

    # H4-限流: Identify the client for rate-limiting.
    # Prefer authenticated token, fall back to client IP.
    from zwm.api.ratelimit import RateLimiterRegistry
    from zwm.observability import metrics as _obs_metrics
    client_ip = websocket.client.host if websocket.client else "unknown"
    rl_key = f"tok:{expected[:8]}" if authenticated else f"ip:{client_ip}"
    rl = RateLimiterRegistry.instance()

    # P2-1: Create an AsyncAgent wrapper for native async OODA.
    # This avoids the thread-pool overhead of ``run_in_executor``.
    from zwm.planner.async_agent import AsyncAgent, AsyncTickRequest
    async_agent: AsyncAgent | None = None
    try:
        async_agent = AsyncAgent(db_path="zwm_ws.db")
        await async_agent.start()
        try:
            async for raw in websocket.iter_text():
                try:
                    data = json.loads(raw)
                    # 3) First-message auth fallback — the client may have
                    #    opted to authenticate by including ``auth_token`` in
                    #    the very first payload.  This is less secure than
                    #    the handshake pattern but is the only option for
                    #    some WebSocket clients.
                    if not authenticated and isinstance(data, dict):
                        token = data.pop("auth_token", None)
                        if token and _secrets.compare_digest(str(token), expected):
                            authenticated = True
                    if not authenticated:
                        await websocket.send_text(json.dumps({
                            "error": "auth_required",
                            "message": "send auth_token in first message "
                                       "or use ?token=... query param",
                        }))
                        continue
                    # H4-限流: Check the rate limit *before* running the tick.
                    allowed, retry_after_s, reason = rl.check_and_record("ws", rl_key)
                    if not allowed:
                        # Record metric, notify client, but don't drop the connection
                        try:
                            _obs_metrics.inc_rate_limit_rejected(scope="ws", reason=reason)
                        except Exception:
                            pass
                        await websocket.send_text(json.dumps({
                            "error": "rate_limited",
                            "reason": reason,
                            "retry_after_s": round(retry_after_s, 3),
                            "message": f"too many ticks, retry after {retry_after_s:.2f}s",
                        }))
                        # If burst is severe, sleep to back off
                        if retry_after_s > 0:
                            await asyncio.sleep(min(retry_after_s, 2.0))
                        continue
                    cmd = WSTickCommand.model_validate(data)
                    req = cmd.payload
                    # P2-1: Use AsyncAgent's native async tick instead of
                    # run_in_executor.  The tensor computation still runs
                    # in a thread pool internally, but the async interface
                    # is cleaner and avoids blocking the event loop.
                    # CRITICAL FIX: req.h_current is int | None (not dict)
                    async_req = AsyncTickRequest(
                        sensor_data=req.sensor_data,
                        h_current=req.h_current,
                        year=req.year,
                        month=req.month,
                        day=req.day,
                        hour=req.hour,
                        time_phase=req.time_phase,
                        target_palace=req.target_palace,
                        day_gan=req.day_gan,
                        reward=req.reward,
                        language_text=req.language_text,
                        vision_features=req.vision_features,
                    )
                    report = await async_agent.tick(async_req)
                    resp = _report_to_response(report)
                    result = WSTickResult(payload=resp)
                    await websocket.send_text(result.model_dump_json())
                except Exception as exc:
                    await websocket.send_text(
                        WSTickResult(
                            payload=TickReportResponse(
                                episode_id=-1,
                                h_current={},
                                h_next={},
                                top_mutation=0,
                                top_score=0.0,
                                reward=0.0,
                                jepa_loss=None,
                                router_loss=None,
                                surprise=0.0,
                                mutation_class="",
                                codon="",
                                codon_aa="",
                                moe_active_experts=[],
                                trajectory=[],
                            )
                        ).model_dump_json()
                    )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("WebSocket tick loop error: %s", exc)
    finally:
        if async_agent is not None:
            await async_agent.close()


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _report_to_response(report) -> TickReportResponse:
    """Convert a TickReport to a TickReportResponse."""
    return TickReportResponse(
        episode_id=report.episode_id,
        h_current={
            "normal_order": report.h_current.normal_order,
            "name": report.h_current.name,
            "binary": report.h_current.binary,
        },
        h_next={
            "normal_order": report.h_next.normal_order,
            "name": report.h_next.name,
            "binary": report.h_next.binary,
        },
        top_mutation=report.top_mutation,
        top_score=report.top_score,
        reward=report.reward,
        jepa_loss=report.jepa_loss,
        router_loss=report.router_loss,
        surprise=report.surprise,
        mutation_class=report.mutation_class,
        codon=report.codon,
        codon_aa=report.codon_aa,
        moe_active_experts=report.plan.moe_active_experts,
        trajectory=[
            {"name": name, "score": score}
            for name, score in report.plan.trajectory
        ],
    )


# ------------------------------------------------------------------
# MCP proxy — forward JSON-RPC 2.0 requests to zwm.mcp
# ------------------------------------------------------------------
async def mcp_dispatch(req: dict):
    """Forward a JSON-RPC 2.0 request to the MCP dispatch layer.

    This bridges the REST API with the MCP protocol so external tools
    (Claude, Cursor, etc.) can call ZWM via either transport.
    """
    from zwm.mcp import dispatch
    agent = get_agent() if _agent is not None else None
    return dispatch(req, agent=agent)


# ------------------------------------------------------------------
# A2A endpoints — multi-agent coordination
# ------------------------------------------------------------------
_a2a_coordinator = None


def _get_a2a():
    global _a2a_coordinator
    if _a2a_coordinator is None:
        from zwm.planner.a2a import A2ACoordinator
        _a2a_coordinator = A2ACoordinator()
    return _a2a_coordinator


async def a2a_register(agent_id: str, palace: int, capabilities: list[str] | None = None):
    """Register an agent in the A2A coordinator."""
    coord = _get_a2a()
    agent = get_agent() if _agent is not None else None
    card = coord.register(agent_id, agent, palace=palace, capabilities=capabilities)
    return {"agent_id": card.agent_id, "palace": card.palace, "status": card.status}


async def a2a_send(sender_id: str, recipient_id: str, msg_type: str, payload: dict):
    """Send an A2A message."""
    from zwm.planner.a2a import A2AMessage
    coord = _get_a2a()
    msg = A2AMessage(
        sender_id=sender_id,
        recipient_id=recipient_id,
        msg_type=msg_type,
        payload=payload,
    )
    coord.send(msg)
    return {"msg_id": msg.msg_id, "status": "sent"}


async def a2a_poll(agent_id: str, limit: int = 10):
    """Poll messages for an agent."""
    coord = _get_a2a()
    msgs = coord.poll(agent_id, limit=limit)
    return {"messages": [m.to_dict() for m in msgs], "count": len(msgs)}


async def a2a_heartbeat():
    """Check all agent statuses."""
    coord = _get_a2a()
    return coord.heartbeat()