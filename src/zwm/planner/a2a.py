"""P3-4 (audit) — 多智能体协调协议 A2A (Agent-to-Agent)。

灵感来自 Google A2A 协议 (2025), 针对 ZWM 的 hexagram 空间定制:
  * 每个 agent 是一个 TrinityAgent 实例, 在特定宫位 (palace) 运行
  * 消息通过 hexagram 编码 (VSA) 传输, 天然支持模糊匹配
  * 协调器 (Coordinator) 负责消息路由 + 共识汇聚

核心概念:
  * AgentCard          — 智能体身份卡 (宫位、能力、状态)
  * A2AMessage         — 六爻编码消息 (VSA 向量 + 元数据)
  * ConsensusResult    — 多智能体共识结果 (加权投票)
  * A2ACoordinator     — 协调器 (注册、路由、共识)
  * TaskState / A2ATask — 任务生命周期 (submitted→working→completed/failed)

用法:
  coord = A2ACoordinator()
  coord.register("agent-1", agent1, palace=1)
  coord.register("agent-2", agent2, palace=4)
  result = await coord.consensus_tick(requests)
"""
from __future__ import annotations

import asyncio
import enum
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Task lifecycle (Google A2A specification)
# ------------------------------------------------------------------
class TaskState(enum.Enum):
    """A2A Task states per Google A2A specification.

    Lifecycle: SUBMITTED → WORKING → COMPLETED | FAILED
    """
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


# ------------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------------
@dataclass
class AgentCard:
    """智能体身份卡 — 描述一个 agent 的能力和当前状态。"""
    agent_id: str
    palace: int  # 1-9 洛书宫位
    capabilities: list[str] = field(default_factory=lambda: ["planning", "prediction"])
    status: str = "idle"  # idle | running | error
    step_count: int = 0
    last_hexagram: int | None = None  # normal_order
    created_at: float = field(default_factory=time.time)
    # H3: HTTP endpoint for cross-process agents (A2A transport).
    endpoint: str | None = None  # type: ignore[misc]
    # L1: well-known ``/.well-known/agent-card.json`` URL (RFC 8615
    # + Google A2A §1.0).  When set, the agent advertises a
    # discoverable public URL that other A2A peers can fetch to
    # retrieve its full card (name, description, version, skills,
    # state) without prior knowledge.  Falls back to a synthesized
    # ``a2a://`` URL when no HTTP transport is registered.
    agent_card_url: str | None = None  # type: ignore[misc]


@dataclass
class A2AMessage:
    """A2A 消息 — 六爻编码, 在 hexagram 空间传输。"""
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender_id: str = ""
    recipient_id: str = ""  # "" = broadcast
    msg_type: str = "tick"  # tick | query | consensus | heartbeat
    payload: dict[str, Any] = field(default_factory=dict)
    # VSA 编码 (128-dim binary vector, 用于相似度路由)
    vsa_vector: np.ndarray | None = None
    timestamp: float = field(default_factory=time.time)
    task_id: str | None = None  # optional link to an A2ATask

    def to_dict(self) -> dict:
        d = {
            "msg_id": self.msg_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "msg_type": self.msg_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
        }
        if self.vsa_vector is not None:
            d["vsa_vector"] = self.vsa_vector.astype(np.int8).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "A2AMessage":
        vsa = None
        if "vsa_vector" in d and d["vsa_vector"] is not None:
            vsa = np.array(d["vsa_vector"], dtype=np.int8)
        return cls(
            msg_id=d["msg_id"],
            sender_id=d["sender_id"],
            recipient_id=d["recipient_id"],
            msg_type=d["msg_type"],
            payload=d["payload"],
            vsa_vector=vsa,
            timestamp=d["timestamp"],
            task_id=d.get("task_id"),
        )


@dataclass
class ConsensusResult:
    """多智能体共识 — 加权投票结果。"""
    hexagram: int  # 共识 hexagram (normal_order)
    confidence: float  # 共识置信度 [0, 1]
    votes: dict[str, tuple[int, float]]  # agent_id → (hexagram, weight)
    num_agents: int
    consensus_type: str = "majority"  # majority | weighted | unanimity


@dataclass
class A2ATask:
    """A2A Task — represents a unit of work between two agents.

    Follows the Google A2A specification Task object with states:
    submitted → working → completed / failed.
    """
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender_id: str = ""
    receiver_id: str = ""
    description: str = ""
    state: TaskState = TaskState.SUBMITTED
    artifacts: list[Any] = field(default_factory=list)
    history: list[A2AMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "description": self.description,
            "state": self.state.value,
            "artifacts": self.artifacts,
            "history": [m.to_dict() for m in self.history],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }


# ------------------------------------------------------------------
# 协调器
# ------------------------------------------------------------------
class A2ACoordinator:
    """多智能体协调器 — 注册、路由、共识。

    每个 agent 绑定到一个洛书宫位 (1-9), 形成空间分布。
    协调器管理消息路由, 并支持基于投票的共识汇聚。

    P2 FIX: optional SQLite persistence for message log so
    multi-agent coordination survives process restarts.
    """

    # P2 FIX: message persistence schema
    _MSG_DDL = """
    CREATE TABLE IF NOT EXISTS a2a_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_id TEXT NOT NULL,
        sender_id TEXT NOT NULL,
        recipient_id TEXT NOT NULL DEFAULT '',
        msg_type TEXT NOT NULL,
        payload TEXT NOT NULL DEFAULT '{}',
        timestamp REAL NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_a2a_recipient
        ON a2a_messages(recipient_id, timestamp);
    """

    def __init__(self, max_log_size: int = 10000, db_path: str | None = None) -> None:
        self._agents: dict[str, tuple[AgentCard, Any]] = {}  # id → (card, agent)
        self._message_log: list[A2AMessage] = []
        self._max_log_size = max_log_size
        self._palace_agents: dict[int, list[str]] = {i: [] for i in range(1, 10)}
        # Task lifecycle (Google A2A spec)
        self._tasks: dict[str, A2ATask] = {}  # task_id → A2ATask
        # P2 FIX: SQLite persistence for message log
        self._db_path: str | None = db_path
        self._db_conn: sqlite3.Connection | None = None
        if db_path is not None:
            self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite persistence layer."""
        import json
        try:
            self._db_conn = sqlite3.connect(self._db_path)
            self._db_conn.execute("PRAGMA journal_mode=WAL")
            self._db_conn.executescript(self._MSG_DDL)
            self._db_conn.commit()
            # Restore messages from previous runs.
            self._load_messages()
        except Exception as exc:
            _log.warning("A2A SQLite persistence init failed: %s — using in-memory only", exc)
            self._db_conn = None

    def _persist_message(self, msg: A2AMessage) -> None:
        """Persist a single message to SQLite."""
        import json
        if self._db_conn is None:
            return
        try:
            self._db_conn.execute(
                "INSERT INTO a2a_messages (msg_id, sender_id, recipient_id, msg_type, payload, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (msg.msg_id, msg.sender_id, msg.recipient_id, msg.msg_type,
                 json.dumps(msg.payload, ensure_ascii=False, default=str), msg.timestamp),
            )
            self._db_conn.commit()
        except Exception as exc:
            _log.debug("A2A message persist failed: %s", exc)

    def _load_messages(self) -> None:
        """Load recent messages from SQLite into the in-memory log."""
        import json
        if self._db_conn is None:
            return
        try:
            rows = self._db_conn.execute(
                "SELECT msg_id, sender_id, recipient_id, msg_type, payload, timestamp "
                "FROM a2a_messages ORDER BY id DESC LIMIT ?",
                (self._max_log_size,),
            ).fetchall()
            for row in reversed(rows):
                payload = json.loads(row[4]) if row[4] else {}
                self._message_log.append(A2AMessage(
                    msg_id=row[0], sender_id=row[1], recipient_id=row[2],
                    msg_type=row[3], payload=payload, timestamp=row[5],
                ))
        except Exception as exc:
            _log.debug("A2A message load failed: %s", exc)

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception as exc:
                _log.debug("A2A SQLite close failed: %s", exc)
            self._db_conn = None

    def register(
        self, agent_id: str, agent, palace: int, capabilities: list[str] | None = None,
    ) -> AgentCard:
        """注册一个 agent 到协调器。"""
        card = AgentCard(
            agent_id=agent_id,
            palace=palace,
            capabilities=capabilities or ["planning", "prediction"],
        )
        self._agents[agent_id] = (card, agent)
        self._palace_agents[palace].append(agent_id)
        return card

    def register_stub(
        self, agent_id: str, palace: int,
        capabilities: list[str] | None = None,
        endpoint: str | None = None,
        agent_card_url: str | None = None,
    ) -> AgentCard:
        """H3: register a *remote* agent (no in-process ``TrinityAgent``).

        Used by the A2A HTTP transport to track a peer running on
        another machine.  The remote agent's messages are routed
        via :attr:`AgentCard.endpoint` (an HTTP URL) instead of
        through in-process queues.

        ``agent_card_url`` (L1) points at the peer's well-known
        ``/.well-known/agent-card.json`` so discovery works without
        prior knowledge of the peer.
        """
        card = AgentCard(
            agent_id=agent_id,
            palace=palace,
            capabilities=capabilities or ["planning", "prediction"],
        )
        card.endpoint = endpoint  # type: ignore[attr-defined]
        card.agent_card_url = agent_card_url  # type: ignore[attr-defined]
        self._agents[agent_id] = (card, None)
        self._palace_agents[palace].append(agent_id)
        return card

    def unregister(self, agent_id: str) -> None:
        card, _ = self._agents.pop(agent_id, (None, None))
        if card is not None:
            self._palace_agents[card.palace].remove(agent_id)

    def get_card(self, agent_id: str) -> AgentCard | None:
        entry = self._agents.get(agent_id)
        return entry[0] if entry else None

    def list_agents(self) -> list[AgentCard]:
        return [card for card, _ in self._agents.values()]

    def agents_in_palace(self, palace: int) -> list[str]:
        return self._palace_agents.get(palace, [])

    # ------------------------------------------------------------------
    # 消息路由
    # ------------------------------------------------------------------
    def send(self, msg: A2AMessage) -> None:
        """发送消息 (记录到日志, 不立即投递 — 由接收方 poll)。

        日志容量限制: 超过 ``max_log_size`` 时丢弃最旧的消息。
        """
        self._message_log.append(msg)
        if len(self._message_log) > self._max_log_size:
            self._message_log = self._message_log[-self._max_log_size:]
        # P2 FIX: persist to SQLite for crash recovery
        self._persist_message(msg)

    def poll(self, agent_id: str, limit: int = 10) -> list[A2AMessage]:
        """拉取发给指定 agent 的消息 (含广播)。

        消费语义: 返回的消息会从日志中移除, 不会被重复拉取。
        """
        msgs = [
            m for m in self._message_log
            if m.recipient_id in ("", agent_id) and m.sender_id != agent_id
        ]
        msgs = msgs[-limit:]
        # Remove consumed messages so they are not returned again.
        consumed_ids = {m.msg_id for m in msgs}
        self._message_log = [
            m for m in self._message_log if m.msg_id not in consumed_ids
        ]
        return msgs

    def broadcast(self, sender_id: str, msg_type: str, payload: dict) -> A2AMessage:
        """广播消息到所有 agent。"""
        msg = A2AMessage(
            sender_id=sender_id,
            msg_type=msg_type,
            payload=payload,
        )
        self.send(msg)
        return msg

    # ------------------------------------------------------------------
    # Task lifecycle (Google A2A specification)
    # ------------------------------------------------------------------
    def submit_task(
        self, sender_id: str, receiver_id: str, description: str = "",
    ) -> A2ATask:
        """Create a new task in SUBMITTED state.

        The sender delegates work to the receiver.  The task starts
        in :attr:`TaskState.SUBMITTED` and must be transitioned to
        :attr:`TaskState.WORKING` via :meth:`start_task`.
        """
        task = A2ATask(
            sender_id=sender_id,
            receiver_id=receiver_id,
            description=description,
            state=TaskState.SUBMITTED,
        )
        self._tasks[task.task_id] = task
        return task

    def start_task(self, task_id: str) -> A2ATask:
        """Transition a task from SUBMITTED to WORKING.

        Raises :class:`KeyError` if the task does not exist.
        Raises :class:`ValueError` if the task is not in SUBMITTED state.
        """
        task = self._tasks[task_id]
        if task.state != TaskState.SUBMITTED:
            raise ValueError(
                f"Task {task_id} is {task.state.value}, expected SUBMITTED"
            )
        task.state = TaskState.WORKING
        task.updated_at = time.time()
        return task

    def complete_task(
        self, task_id: str, artifacts: list[Any] | None = None,
    ) -> A2ATask:
        """Transition a task from WORKING to COMPLETED.

        Optionally attach artifacts (results) produced by the task.
        """
        task = self._tasks[task_id]
        if task.state != TaskState.WORKING:
            raise ValueError(
                f"Task {task_id} is {task.state.value}, expected WORKING"
            )
        task.state = TaskState.COMPLETED
        if artifacts is not None:
            task.artifacts = artifacts
        task.updated_at = time.time()
        return task

    def fail_task(self, task_id: str, error: str = "") -> A2ATask:
        """Transition a task from WORKING to FAILED.

        Optionally record an error description.
        """
        task = self._tasks[task_id]
        if task.state != TaskState.WORKING:
            raise ValueError(
                f"Task {task_id} is {task.state.value}, expected WORKING"
            )
        task.state = TaskState.FAILED
        task.error = error
        task.updated_at = time.time()
        return task

    def get_task(self, task_id: str) -> A2ATask | None:
        """Retrieve a task by ID, or ``None`` if not found."""
        return self._tasks.get(task_id)

    def list_tasks(self, agent_id: str | None = None) -> list[A2ATask]:
        """List all tasks, optionally filtered by agent involvement.

        When *agent_id* is given, returns tasks where the agent is
        either the sender or the receiver.
        """
        if agent_id is None:
            return list(self._tasks.values())
        return [
            t for t in self._tasks.values()
            if t.sender_id == agent_id or t.receiver_id == agent_id
        ]

    # ------------------------------------------------------------------
    # 共识
    # ------------------------------------------------------------------
    async def consensus_tick(
        self,
        requests: list[dict] | None = None,
        agent_ids: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> ConsensusResult:
        """并行执行 OODA tick, 并汇聚共识。

        每个 agent 运行一步 OODA, 然后对 hexagram 结果进行加权投票。
        权重默认基于 agent 的宫位距离 (离中心越近权越高)。
        """
        ids = agent_ids or list(self._agents.keys())
        if not ids:
            raise ValueError("No agents registered")

        # 默认权重: 宫位 5 (中宫) 最高, 其他按距离衰减
        if weights is None:
            weights = {}
            for aid in ids:
                card, _ = self._agents.get(aid, (None, None))
                if card is None:
                    continue
                dist = abs(card.palace - 5)
                weights[aid] = 1.0 / (1.0 + dist)

        # 并行执行
        async def _agent_tick(aid: str):
            card, agent = self._agents.get(aid, (None, None))
            if card is None:
                return aid, None
            try:
                loop = asyncio.get_running_loop()
                if requests and aid in [r.get("agent_id", "") for r in requests]:
                    req = next(r for r in requests if r.get("agent_id") == aid)
                else:
                    req = {}
                report = await loop.run_in_executor(
                    None,
                    lambda: agent.observe_predict_evaluate_act(
                        sensor_data=req.get("sensor_data"),
                        h_current=req.get("h_current"),
                    ),
                )
                return aid, report
            except Exception:
                return aid, None

        tasks = [_agent_tick(aid) for aid in ids]
        results = await asyncio.gather(*tasks)

        # 加权投票
        vote_counts: dict[int, float] = {}
        vote_details: dict[str, tuple[int, float]] = {}
        for aid, report in results:
            if report is None:
                continue
            h = report.h_next.normal_order
            w = weights.get(aid, 1.0)
            vote_counts[h] = vote_counts.get(h, 0.0) + w
            vote_details[aid] = (h, w)

        if not vote_counts:
            return ConsensusResult(
                hexagram=0, confidence=0.0, votes={},
                num_agents=len(ids), consensus_type="majority",
            )

        # 选出最高票
        top_hex = max(vote_counts, key=vote_counts.get)
        total_weight = sum(weights.get(aid, 1.0) for aid in ids)
        confidence = vote_counts[top_hex] / total_weight if total_weight > 0 else 0.0

        # 确定共识类型
        if confidence >= 0.9:
            ctype = "unanimity"
        elif confidence >= 0.5:
            ctype = "majority"
        else:
            ctype = "weighted"

        return ConsensusResult(
            hexagram=top_hex,
            confidence=confidence,
            votes=vote_details,
            num_agents=len(ids),
            consensus_type=ctype,
        )

    def heartbeat(self) -> dict[str, str]:
        """检查所有 agent 状态。"""
        return {
            aid: card.status
            for aid, (card, _) in self._agents.items()
        }

    # ------------------------------------------------------------------
    # P5-3 (audit): synchronous variant of ``consensus_tick`` for
    # non-asyncio hosts (CLI, REST, smoke-tests).  Behaviour mirrors
    # the async version 1:1 — every agent runs one OODA step, then
    # we tally a weighted majority over the resulting ``h_next``.
    # ------------------------------------------------------------------
    def consensus_tick_sync(
        self,
        requests: list[dict] | None = None,
        agent_ids: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> "ConsensusResult":
        """Synchronous consensus — runs each agent's tick in turn.

        Used by ``zwm a2a`` and the A2A HTTP endpoint when the host
        is not running an asyncio loop.  The async variant
        (``consensus_tick``) remains the canonical path for
        high-throughput streaming hosts.
        """
        ids = agent_ids or list(self._agents.keys())
        if not ids:
            raise ValueError("No agents registered")
        if weights is None:
            weights = {}
            for aid in ids:
                card, _ = self._agents.get(aid, (None, None))
                if card is None:
                    continue
                dist = abs(card.palace - 5)
                weights[aid] = 1.0 / (1.0 + dist)

        req_by_id = {r.get("agent_id", ""): r for r in (requests or [])}
        # Per-agent tick — guarded so one failure does not poison the
        # whole consensus run.
        results: list[tuple[str, Any]] = []
        for aid in ids:
            card, agent = self._agents.get(aid, (None, None))
            if card is None or agent is None:
                continue
            try:
                req = req_by_id.get(aid, {})
                if hasattr(agent, "observe_predict_evaluate_act"):
                    report = agent.observe_predict_evaluate_act(
                        sensor_data=req.get("sensor_data"),
                        h_current=req.get("h_current"),
                    )
                else:
                    report = None
                card.status = "running" if report else "error"
                card.step_count += 1
                if report is not None and getattr(report, "h_next", None):
                    card.last_hexagram = report.h_next.normal_order
                results.append((aid, report))
            except Exception:
                card.status = "error"
                results.append((aid, None))
            finally:
                if card.status == "running":
                    card.status = "idle"

        # Weighted majority
        vote_counts: dict[int, float] = {}
        vote_details: dict[str, tuple[int, float]] = {}
        for aid, report in results:
            if report is None:
                continue
            h = report.h_next.normal_order
            w = weights.get(aid, 1.0)
            vote_counts[h] = vote_counts.get(h, 0.0) + w
            vote_details[aid] = (h, w)

        if not vote_counts:
            return ConsensusResult(
                hexagram=0, confidence=0.0, votes={},
                num_agents=len(ids), consensus_type="majority",
            )

        top_hex = max(vote_counts, key=vote_counts.get)
        total_weight = sum(weights.get(aid, 1.0) for aid in ids)
        confidence = vote_counts[top_hex] / total_weight if total_weight > 0 else 0.0
        if confidence >= 0.9:
            ctype = "unanimity"
        elif confidence >= 0.5:
            ctype = "majority"
        else:
            ctype = "weighted"

        return ConsensusResult(
            hexagram=top_hex,
            confidence=confidence,
            votes=vote_details,
            num_agents=len(ids),
            consensus_type=ctype,
        )

    # ------------------------------------------------------------------
    # P5-3: AgentCard / 消息对外暴露 (兼容 Google A2A 协议 §1.0 的
    # 关键字段: name / description / url / version / skills / state)
    # ------------------------------------------------------------------
    def agent_card(self, agent_id: str) -> dict | None:
        """Return a Google-A2A-style AgentCard for the given agent.

        Format mirrors the A2A spec (2025) ``/.well-known/agent.json``
        schema: ``name``, ``description``, ``url``, ``version``,
        ``skills`` (list of {id, name, description}), and a custom
        ``state`` extension carrying ZWM-specific fields (palace,
        step_count, last_hexagram, status).

        The ``url`` field reflects :attr:`AgentCard.agent_card_url`
        when the card is hosted on a real A2A transport, otherwise
        falls back to a synthesised ``a2a://`` identifier.
        """
        entry = self._agents.get(agent_id)
        if entry is None:
            return None
        card, agent = entry
        skills = [
            {
                "id": cap,
                "name": cap.replace("_", " ").title(),
                "description": f"ZWM capability: {cap}",
            }
            for cap in card.capabilities
        ]
        # L1: prefer the registered well-known URL (RFC 8615 +
        # Google A2A spec) so peers can discover the card via a
        # stable HTTP endpoint.  Synthesise a fallback that still
        # uniquely identifies the agent.
        agent_card_url = (
            card.agent_card_url  # type: ignore[attr-defined]
            if getattr(card, "agent_card_url", None)
            else (
                card.endpoint.rstrip("/") + "/.well-known/agent-card.json"  # type: ignore[union-attr]
                if getattr(card, "endpoint", None)
                else f"a2a://zwm/agents/{agent_id}"
            )
        )
        return {
            "name": f"zwm-agent-{agent_id}",
            "description": (
                "A ZWM Trinity Agent specialised on a Lo Shu palace. "
                "Exposes plan / observe / reflect via the A2A protocol."
            ),
            "url": agent_card_url,
            "version": "1.0.0",
            "skills": skills,
            "state": {
                "palace": card.palace,
                "status": card.status,
                "step_count": card.step_count,
                "last_hexagram": card.last_hexagram,
                "created_at": card.created_at,
            },
        }