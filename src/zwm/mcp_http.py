"""H2 — MCP Streamable-HTTP transport for ZWM.

Implements the 2025-06-18 MCP "Streamable-HTTP" transport — the modern
replacement for the legacy HTTP+SSE transport.  Key properties:

* **Single endpoint** — ``POST /mcp`` (with optional ``GET /mcp`` for SSE
  upgrade when the client requests it via ``Accept: text/event-stream``)
* **Stateless mode** — every request carries a session id in the
  ``Mcp-Session-Id`` header; the server keeps an in-process map from
  session id → ``TrinityAgent`` (created on first ``initialize`` call).
* **Bearer-token auth** — reuses ``ZWM_API_TOKEN`` / ``ZWM_REQUIRE_AUTH``
  env vars (same as the REST surface).
* **Backward compatible** — the old stdio transport (``serve_stdio``)
  is untouched; this file only *adds* the HTTP path.

Reference: https://modelcontextprotocol.io/specification/2025-06-18/
            transports/streamable-http
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable

from zwm.mcp import (
    JsonRpcError,
    MCP_PROTOCOL_VERSION,
    dispatch,
)
from zwm.observability import configure_logging, metrics as _obs_metrics

_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Session registry — Mcp-Session-Id → (agent, created_at)
# ----------------------------------------------------------------------
class _SessionStore:
    """Thread-safe in-process session registry.

    A real deployment would put Redis here.  For the reference
    implementation we keep it in-process; each FastAPI worker has its own
    registry.  Sessions expire after ``ZWM_MCP_SESSION_TTL`` seconds of
    inactivity (default 1 h)."""

    def __init__(self) -> None:
        self._sessions: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._ttl = float(os.environ.get("ZWM_MCP_SESSION_TTL", "3600"))

    def get_or_create(self, sid: str | None) -> tuple[str, Any]:
        with self._lock:
            now = time.monotonic()
            self._gc_locked(now)
            if sid is None:
                sid = uuid.uuid4().hex
                self._sessions[sid] = (None, now)
                return sid, None
            entry = self._sessions.get(sid)
            if entry is None:
                # New session id provided by client — register it.
                self._sessions[sid] = (None, now)
                return sid, None
            agent, _ = entry
            self._sessions[sid] = (agent, now)
            return sid, agent

    def attach(self, sid: str, agent: Any) -> None:
        with self._lock:
            self._sessions[sid] = (agent, time.monotonic())

    def drop(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)

    def _gc_locked(self, now: float) -> None:
        threshold = now - self._ttl
        for sid in list(self._sessions.keys()):
            _, ts = self._sessions[sid]
            if ts < threshold:
                self._sessions.pop(sid, None)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"sessions": len(self._sessions)}


_SESSIONS = _SessionStore()


# ----------------------------------------------------------------------
# FastAPI app factory
# ----------------------------------------------------------------------
def create_app() -> Any:
    """Build and return a FastAPI app exposing the MCP HTTP transport.

    Mounts:

    * ``POST /mcp``    — primary endpoint (Streamable-HTTP POST)
    * ``GET  /mcp``    — SSE upgrade (for clients that send
                          ``Accept: text/event-stream``)
    * ``GET  /healthz``— liveness probe
    """
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse, PlainTextResponse
    try:
        from fastapi.responses import StreamingResponse
    except Exception:  # pragma: no cover - very old FastAPI
        StreamingResponse = None  # type: ignore[assignment]

    app = FastAPI(
        title="zwm-mcp",
        version=MCP_PROTOCOL_VERSION,
        description="ZWM MCP server — Streamable-HTTP transport (2025-06-18)",
    )

    # ----- auth dependency (same shape as REST) -----
    def _check_auth(req: Request) -> None:
        import secrets
        expected = os.environ.get("ZWM_API_TOKEN", "").strip()
        require = os.environ.get("ZWM_REQUIRE_AUTH", "1").strip() != "0"
        if not (expected and require):
            return
        # Accept via Authorization header or ?token= query param
        auth = req.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            if token and secrets.compare_digest(token, expected):
                return
        token = req.query_params.get("token", "")
        if token and secrets.compare_digest(token, expected):
            return
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ----- POST /mcp -----
    @app.post("/mcp")
    async def mcp_post(request: Request) -> Response:
        _check_auth(request)
        sid = request.headers.get("mcp-session-id")
        body_bytes = await request.body()
        # Determine content type — clients can send single JSON, JSON
        # array (batch), or NDJSON streams.
        ctype = (request.headers.get("content-type") or "").lower()
        if "application/x-ndjson" in ctype:
            try:
                lines = [json.loads(ln) for ln in body_bytes.decode("utf-8").splitlines() if ln.strip()]
            except json.JSONDecodeError as exc:
                raise HTTPException(400, f"invalid ndjson: {exc}")
        else:
            try:
                parsed = json.loads(body_bytes)
            except json.JSONDecodeError as exc:
                raise HTTPException(400, f"invalid json: {exc}")
            lines = parsed if isinstance(parsed, list) else [parsed]

        sid, agent = _SESSIONS.get_or_create(sid)
        responses: list[dict] = []
        for req_obj in lines:
            resp = dispatch(req_obj, agent=agent)
            # On initialize, build a fresh per-session agent.
            if isinstance(req_obj, dict) and req_obj.get("method") == "initialize" and resp is not None:
                try:
                    from zwm.planner.agent import TrinityAgent
                    from zwm.planner.surface import build_config_from_mcp_args
                    cfg = build_config_from_mcp_args(req_obj.get("params") or {})
                    new_agent = TrinityAgent(config=cfg)
                    _SESSIONS.attach(sid, new_agent)
                    agent = new_agent
                    _log.info("mcp-http session %s initialised (config=%s)", sid[:8], cfg)
                except Exception as exc:
                    _log.warning("mcp-http session initialise agent failed: %s", exc)
            if resp is not None:
                responses.append(resp)
        try:
            _obs_metrics.inc_ticks(n=0)  # touch singleton
        except Exception:
            pass

        # Streamable-HTTP supports two response modes:
        # 1) application/json — single or batch response
        # 2) text/event-stream — when client requested it
        accept = (request.headers.get("accept") or "").lower()
        if "text/event-stream" in accept and StreamingResponse is not None:
            async def _gen():
                for r in responses:
                    yield f"data: {json.dumps(r, ensure_ascii=False)}\n\n"
            return StreamingResponse(
                _gen(),
                media_type="text/event-stream",
                headers={"Mcp-Session-Id": sid, "Cache-Control": "no-cache"},
            )
        payload = responses[0] if len(responses) == 1 else responses
        return JSONResponse(
            content=payload,
            headers={"Mcp-Session-Id": sid},
        )

    # ----- GET /mcp (SSE stream) -----
    @app.get("/mcp")
    async def mcp_get(request: Request) -> Response:
        _check_auth(request)
        accept = (request.headers.get("accept") or "").lower()
        if "text/event-stream" not in accept or StreamingResponse is None:
            raise HTTPException(400, "GET /mcp requires Accept: text/event-stream")
        sid = request.headers.get("mcp-session-id") or uuid.uuid4().hex
        _SESSIONS.get_or_create(sid)

        async def _event_source():
            # Initial endpoint event — the Streamable-HTTP "endpoint"
            # event tells the client where to POST its requests.  We
            # point it at the same /mcp path on this server.
            yield f"event: endpoint\ndata: /mcp\n\n"
            # Heartbeat every 15 s to keep the connection alive.
            while True:
                await __import__("asyncio").sleep(15.0)
                yield ": heartbeat\n\n"
        return StreamingResponse(
            _event_source(),
            media_type="text/event-stream",
            headers={"Mcp-Session-Id": sid, "Cache-Control": "no-cache"},
        )

    # ----- DELETE /mcp (close session) -----
    @app.delete("/mcp")
    async def mcp_delete(request: Request) -> Response:
        _check_auth(request)
        sid = request.headers.get("mcp-session-id")
        if sid:
            _SESSIONS.drop(sid)
        return JSONResponse({"closed": bool(sid)})

    @app.get("/healthz")
    async def healthz() -> Response:
        return PlainTextResponse("ok")

    return app


# ----------------------------------------------------------------------
# CLI entry — ``zwm mcp-http`` (added in H2)
# ----------------------------------------------------------------------
def serve_http(
    host: str = "127.0.0.1",
    port: int = 8765,
    log_level: str = "info",
) -> None:
    """Run the Streamable-HTTP MCP transport using uvicorn.

    Falls back to a graceful error if uvicorn is not installed.
    """
    configure_logging(level=log_level.upper() if isinstance(log_level, str) else "INFO")
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required for serve_http; install via `pip install uvicorn[standard]`"
        ) from exc
    app = create_app()
    _log.info("zwm-mcp-http listening on http://%s:%d (protocol %s)",
              host, port, MCP_PROTOCOL_VERSION)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


__all__ = ["create_app", "serve_http", "_SessionStore"]
