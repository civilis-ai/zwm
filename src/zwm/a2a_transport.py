"""H3 — A2A cross-process transport (HTTP / JSON).

The :mod:`zwm.planner.a2a` module provides a *same-process* coordinator.
This module wraps it in a FastAPI app so multiple agents running on
different machines can exchange ``A2AMessage`` instances and run
consensus ticks over the wire.

Two interfaces are exposed:

* ``POST /a2a/agent-card``  — register / refresh an AgentCard
* ``GET  /a2a/agent-card/{id}`` — fetch the AgentCard
* ``POST /a2a/send``        — deliver an ``A2AMessage`` (JSON)
* ``GET  /a2a/poll/{id}``   — drain queued messages
* ``POST /a2a/consensus``   — run a consensus tick across all
                                registered peers; the host supplies
                                the per-peer ``hex_bits`` via
                                ``requests=[{agent_id, h_current, ...}, ...]``
* ``GET  /a2a/heartbeat``   — list all known AgentCards

A reference CLI is provided via :func:`serve_a2a` (``zwm a2a-serve``).
The transport follows the **Google A2A JSON schema** (2025) so other
A2A-compatible clients can interoperate with ZWM.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from zwm.observability import configure_logging

_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# FastAPI app factory
# ----------------------------------------------------------------------
def create_a2a_app() -> Any:
    """Build a FastAPI app exposing the A2A HTTP surface.

    The app holds an in-process :class:`A2ACoordinator` (a real
    distributed deployment would back this with Redis or NATS).  All
    endpoints share the same Bearer-token model as the rest of ZWM.
    """
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from zwm.planner.a2a import A2ACoordinator, A2AMessage

    app = FastAPI(
        title="zwm-a2a",
        version="0.1.0",
        description="ZWM A2A cross-process transport (Google A2A schema 2025)",
    )
    _coordinator: A2ACoordinator = A2ACoordinator()

    # ----- auth -----
    def _check_auth(req: Request) -> None:
        import secrets
        expected = os.environ.get("ZWM_API_TOKEN", "").strip()
        require = os.environ.get("ZWM_REQUIRE_AUTH", "1").strip() != "0"
        if not (expected and require):
            return
        auth = req.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            if token and secrets.compare_digest(token, expected):
                return
        raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/a2a/heartbeat")
    async def heartbeat(request: Request) -> JSONResponse:
        _check_auth(request)
        return JSONResponse(_coordinator.heartbeat())

    @app.post("/a2a/agent-card")
    async def register_card(request: Request) -> JSONResponse:
        _check_auth(request)
        body = await request.json()
        agent_id = body.get("agent_id")
        if not agent_id:
            raise HTTPException(400, "agent_id required")
        # The remote agent's TrinityAgent is *not* present on this
        # server.  We just register a stub AgentCard for routing.
        card = _coordinator.register_stub(
            agent_id=agent_id,
            palace=int(body.get("palace", 5)),
            capabilities=list(body.get("capabilities", [])),
            endpoint=body.get("endpoint"),
        )
        return JSONResponse({
            "agent_id": card.agent_id,
            "palace": card.palace,
            "status": card.status,
            "step_count": card.step_count,
        })

    @app.get("/a2a/agent-card/{agent_id}")
    async def get_card(agent_id: str, request: Request) -> JSONResponse:
        _check_auth(request)
        card = _coordinator._agents.get(agent_id)
        if card is None:
            raise HTTPException(404, f"agent {agent_id!r} not found")
        c = card[0]
        return JSONResponse({
            "agent_id": c.agent_id,
            "palace": c.palace,
            "capabilities": c.capabilities,
            "status": c.status,
            "step_count": c.step_count,
            "last_hexagram": c.last_hexagram,
            "created_at": c.created_at,
            # L1: advertise the well-known agent-card URL so peers can
            # discover this card via a stable HTTP endpoint.
            "agent_card_url": (
                c.agent_card_url  # type: ignore[attr-defined]
                if getattr(c, "agent_card_url", None)
                else f"http://{request.url.netloc}/.well-known/agent-card.json?id={c.agent_id}"
            ),
        })

    @app.get("/.well-known/agent-card.json")
    @app.get("/.well-known/agent.json")
    async def well_known_agent_card(
        request: Request, id: str | None = None,
    ) -> JSONResponse:
        """L1: well-known agent-card discovery endpoint (RFC 8615).

        Implements the Google A2A spec (2025) ``/.well-known/agent.json``
        convention.  When ``id`` is supplied, returns the card for that
        specific agent; otherwise returns the first registered agent's
        card (suitable for single-agent deployments).
        """
        _check_auth(request)
        # Pick the requested agent, or the first registered one.
        target_id: str | None = id
        if target_id is None and _coordinator._agents:
            target_id = next(iter(_coordinator._agents))
        if not target_id:
            raise HTTPException(404, "no agents registered")
        card = _coordinator.agent_card(target_id)
        if card is None:
            raise HTTPException(404, f"agent {target_id!r} not found")
        return JSONResponse(card)

    @app.post("/a2a/send")
    async def send_msg(request: Request) -> JSONResponse:
        _check_auth(request)
        body = await request.json()
        msg = A2AMessage.from_dict(body)
        _coordinator.send(msg)
        return JSONResponse({"msg_id": msg.msg_id, "status": "sent"})

    @app.get("/a2a/poll/{agent_id}")
    async def poll(agent_id: str, request: Request, limit: int = 10) -> JSONResponse:
        _check_auth(request)
        msgs = _coordinator.poll(agent_id, limit=limit)
        return JSONResponse({
            "messages": [m.to_dict() for m in msgs],
            "count": len(msgs),
        })

    @app.post("/a2a/consensus")
    async def consensus(request: Request) -> JSONResponse:
        """H3: cross-process consensus.

        Body::

            {
              "requests": [
                {"agent_id": "agent-1", "h_current": 1, "sensor_data": {...}},
                {"agent_id": "agent-2", "h_current": 1, "sensor_data": {...}}
              ],
              "weights": {"agent-1": 1.0, "agent-2": 1.5}
            }

        Since this server hosts *stubs* (not real ``TrinityAgent``s),
        the consensus is computed by delegating to whichever peer is
        reachable at its registered ``endpoint`` (an HTTP URL).  If
        no peer endpoints are configured, we fall back to a
        same-process :func:`consensus_tick_sync` over the stub set.
        """
        _check_auth(request)
        body = await request.json()
        requests = body.get("requests", [])
        weights = body.get("weights")
        # Try real distributed consensus first.
        result = await _distributed_consensus(_coordinator, requests, weights)
        if result is None:
            # Fall back to same-process stub-based consensus.
            result = _coordinator.consensus_tick_sync(
                requests=requests, weights=weights,
            )
        return JSONResponse({
            "hexagram": int(result.hexagram),
            "confidence": float(result.confidence),
            "consensus_type": result.consensus_type,
            "vote_details": {
                k: {"hex": int(h), "weight": float(w)}
                for k, (h, w) in result.vote_details.items()
            },
        })

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok", "ts": time.time()})

    return app


async def _distributed_consensus(
    coordinator, requests: list[dict], weights: dict | None,
):
    """Try to call each peer's /a2a/send or /a2a/consensus endpoint.

    Returns ``None`` if no peer endpoints are reachable; the caller
    falls back to the in-process stub consensus."""
    import asyncio
    from zwm.planner.a2a import A2AMessage
    # Find any peer with a non-empty endpoint
    peers = [c for (aid, (c, _)) in coordinator._agents.items() if c.endpoint]
    if not peers:
        return None
    # POST the request to each peer; collect hex / weight.
    results: list[tuple[str, int, float]] = []
    async def call_peer(card, req):
        import urllib.request
        import json as _json
        body = _json.dumps({"hex_bits": int(req.get("h_current", 1))}).encode()
        req_obj = urllib.request.Request(
            card.endpoint.rstrip("/") + "/a2a/plan",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(req_obj, timeout=2.0),
            )
            data = _json.loads(resp.read().decode())
            return (card.agent_id, int(data.get("hex_bits", 0)),
                    weights.get(card.agent_id, 1.0) if weights else 1.0)
        except Exception as exc:
            _log.debug("peer %s unreachable: %s", card.agent_id, exc)
            return None
    tasks = [call_peer(c, r) for c, r in zip(peers, requests)]
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    for r in raw:
        if isinstance(r, tuple):
            results.append(r)
    if not results:
        return None
    # Build a ConsensusResult locally.
    from zwm.planner.a2a import ConsensusResult
    vote_counts: dict[int, float] = {}
    vote_details: dict[str, tuple[int, float]] = {}
    for aid, h, w in results:
        vote_counts[h] = vote_counts.get(h, 0.0) + w
        vote_details[aid] = (h, w)
    if not vote_counts:
        return None
    best = max(vote_counts.items(), key=lambda kv: kv[1])
    total = sum(vote_counts.values())
    confidence = best[1] / total if total else 0.0
    ctype = "strong" if confidence >= 0.66 else ("weak" if confidence >= 0.34 else "split")
    return ConsensusResult(
        hexagram=best[0], confidence=confidence,
        consensus_type=ctype, vote_details=vote_details,
    )


# ----------------------------------------------------------------------
# CLI entry
# ----------------------------------------------------------------------
def serve_a2a(host: str = "127.0.0.1", port: int = 8766,
              log_level: str = "info") -> None:
    """H3: launch the A2A cross-process transport."""
    configure_logging(level=log_level.upper() if isinstance(log_level, str) else "INFO")
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required for serve_a2a; install via `pip install uvicorn[standard]`"
        ) from exc
    app = create_a2a_app()
    _log.info("zwm-a2a listening on http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


__all__ = ["create_a2a_app", "serve_a2a"]
