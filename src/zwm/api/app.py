"""P2-3 (audit) — FastAPI 应用工厂。

提供 ``create_app()`` 工厂函数, 返回配置完整的 FastAPI 实例,
包含:
  * lifespan 管理 — 启动/关闭 TrinityAgent 单例
  * CORS 中间件 (P0-6: 收紧到白名单 + Bearer token 鉴权)
  * REST 路由注册
  * WebSocket 端点
  * OpenAPI 文档 (自动生成)

用法:
  uv run uvicorn zwm.api.app:app --reload
  uv run zwm-serve

环境变量:
  ZWM_CORS_ORIGINS  — 逗号分隔的允许 Origin (默认 ``http://localhost:3000,http://127.0.0.1:3000``)
  ZWM_API_TOKEN     — Bearer token; 设空则禁用鉴权 (开发用)
  ZWM_REQUIRE_AUTH  — ``"1"`` 强制鉴权, ``"0"`` 允许匿名 (默认 ``"1"``)
"""
from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .ratelimit import require_rate_limit
from .routes import (
    a2a_heartbeat,
    a2a_poll,
    a2a_register,
    a2a_send,
    do_tick,
    health,
    info,
    mcp_dispatch,
    metrics_endpoint,
    preference_feedback,
    session_delete,
    session_history,
    session_start,
    session_tick,
    set_agent,
    ws_tick,
)
from .schemas import (
    SessionHistory,
    SessionInfo,
    SessionStartRequest,
    TickReportResponse,
    TickRequest,
)


def _default_cors_origins() -> list[str]:
    """Return the default CORS allow-list (dev-friendly, not ``*``)."""
    raw = os.environ.get("ZWM_CORS_ORIGINS", "")
    if raw.strip():
        return [o.strip() for o in raw.split(",") if o.strip()]
    # Local dev defaults — explicitly NOT wildcard.
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


def _verify_bearer(
    credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
) -> str | None:
    """P0-6: Bearer token gate.

    Compares the supplied ``Authorization: Bearer <token>`` against the
    ``ZWM_API_TOKEN`` environment variable.  When ``ZWM_API_TOKEN`` is unset,
    the dependency is a no-op (development convenience).  When it is set,
    the request is rejected with 401 unless the token matches.

    Returns the verified token (or ``None`` if auth is disabled).
    """
    expected = os.environ.get("ZWM_API_TOKEN", "").strip()
    require_auth = os.environ.get("ZWM_REQUIRE_AUTH", "1").strip() != "0"
    if not expected or not require_auth:
        return None
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Constant-time compare to avoid timing leaks.
    if not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用程序生命周期:

    startup:  创建 TrinityAgent 单例
    shutdown: 关闭 agent 并清理所有会话
    """
    import threading
    from zwm.planner.agent import TrinityAgent

    # 在后台线程构造 agent (torch 初始化可能耗时)
    agent = None
    error = [None]

    def _build():
        nonlocal agent
        try:
            agent = TrinityAgent(
                db_path=os.environ.get("ZWM_DB_PATH", "zwm_api.db"),
                checkpoint_path=os.environ.get("ZWM_CHECKPOINT_PATH"),
                mcts_iterations=int(os.environ.get("ZWM_MCTS_ITERATIONS", "200")),
            )
        except Exception as exc:
            error[0] = exc

    t = threading.Thread(target=_build, daemon=True)
    t.start()
    t.join(timeout=30)

    if error[0] is not None:
        raise error[0] from None
    if agent is None:
        raise RuntimeError("Agent construction timed out")

    set_agent(agent)
    yield
    # shutdown
    agent.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="ZWM API",
        description="天地人三才世界模型规划器 — REST + WebSocket 接入层",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — P0-6: explicit allow-list (no wildcard), reads from
    # ZWM_CORS_ORIGINS env var.  Default is a small dev-friendly list.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_default_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )

    # ---- REST 路由 ----
    # /health, /info, /metrics are intentionally unauthenticated so load
    # balancers, uptime monitors, and Prometheus scrapers can poll them
    # without a token.
    app.get("/health", response_model=dict)(health)
    app.get("/info", response_model=dict)(info)
    # P2-1: Prometheus exposition — exposes OODA-loop metrics in the
    # standard text format (counters, gauges, histograms).  No auth so
    # scrapers work out-of-the-box; lock down the port with a firewall
    # if you don't want this exposed.
    app.get("/metrics")(metrics_endpoint)
    # P0-6: every other route requires a valid Bearer token (when
    # ZWM_API_TOKEN is set).  ``_verify_bearer`` is a no-op when the
    # token is unset, preserving the dev-mode behaviour.
    # P0-RL: all mutating REST endpoints are also rate-limited via
    # ``require_rate_limit`` (token bucket + sliding window).
    _auth_rl = [Depends(_verify_bearer), Depends(require_rate_limit)]
    _auth_only = [Depends(_verify_bearer)]

    app.post("/tick", response_model=TickReportResponse, dependencies=_auth_rl)(do_tick)
    app.post("/session/start", response_model=SessionInfo, dependencies=_auth_rl)(session_start)
    app.post("/session/{session_id}/tick", response_model=TickReportResponse, dependencies=_auth_rl)(session_tick)
    app.get("/session/{session_id}/history", response_model=SessionHistory, dependencies=_auth_only)(session_history)
    app.delete("/session/{session_id}", dependencies=_auth_only)(session_delete)

    # ---- P0-1: DPO preference feedback ----
    app.post("/preference", dependencies=_auth_rl)(preference_feedback)

    # ---- MCP proxy ----
    # Forward JSON-RPC 2.0 requests to the MCP dispatch layer.
    app.post("/mcp", dependencies=_auth_rl)(mcp_dispatch)

    # ---- A2A multi-agent ----
    app.post("/a2a/register", dependencies=_auth_rl)(a2a_register)
    app.post("/a2a/send", dependencies=_auth_rl)(a2a_send)
    app.get("/a2a/poll/{agent_id}", dependencies=_auth_only)(a2a_poll)
    app.get("/a2a/heartbeat")(a2a_heartbeat)

    # ---- WebSocket ----
    # /ws/tick keeps the same auth model; the bearer token is supplied
    # in the WSTickCommand payload (the WS protocol cannot carry the
    # standard Authorization header).  The token check is performed
    # inside ws_tick() in routes.py.
    app.websocket("/ws/tick")(ws_tick)

    return app


# 模块级 app 实例 (uvicorn 入口)
app = create_app()