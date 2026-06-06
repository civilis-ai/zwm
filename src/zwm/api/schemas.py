"""P2-3 (audit) — FastAPI 接入层 (REST + WebSocket)。

提供:
  * ``POST /tick``          — 单步 OODA 循环 (同步)
  * ``POST /session/start`` — 创建持久化 agent 会话
  * ``POST /session/{id}/tick`` — 在会话中执行 OODA 步
  * ``GET  /session/{id}/history`` — 查询会话历史
  * ``DELETE /session/{id}`` — 销毁会话
  * ``WS   /ws/tick``        — 实时 WebSocket 流式 OODA
  * ``GET  /health``         — 健康检查
  * ``GET  /info``           — 框架信息

Pydantic 数据模型 (请求/响应),无外部依赖 (除 pydantic 本身)。

P4-7 (audit): ``SessionStartRequest`` now embeds the dynamically-built
``ConfigOverrides`` pydantic model, so every field in
:class:`zwm.planner.agent_config.TrinityConfig` is exposed to the API
*without* having to be hand-listed here.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# P4-7 — dynamic overrides model is built once at import time.
# See :pymod:`zwm.planner.surface` for the build helper.
from zwm.planner.surface import build_config_overrides_model  # noqa: E402

# Build the pydantic model that mirrors TrinityConfig (minus non-serialisable fields).
ConfigOverrides: type[BaseModel] = build_config_overrides_model()


# ------------------------------------------------------------------
# 枚举
# ------------------------------------------------------------------
class Outcome(str, Enum):
    ji = "吉"
    xiong = "凶"


# ------------------------------------------------------------------
# 请求
# ------------------------------------------------------------------
class TickRequest(BaseModel):
    """单步 OODA 请求 — 传感器数据或直接指定 hexagram。"""
    sensor_data: dict[str, float] | None = Field(
        default=None,
        description="传感器读数 (key→value), 走 RuleBasedEncoder 编码为 hexagram",
    )
    h_current: int | None = Field(
        default=None,
        ge=0, le=63,
        description="直接指定当前 hexagram 的 normal_order (0-63)",
    )
    year: int = Field(default=2026, ge=1, description="年份")
    month: int = Field(default=1, ge=1, le=12, description="月份")
    day: int = Field(default=1, ge=1, le=31, description="日")
    hour: int = Field(default=0, ge=0, le=23, description="小时")
    time_phase: float | None = Field(default=None, description="时间相位 (None=自动计算)")
    target_palace: int | None = Field(default=None, ge=1, le=9, description="目标宫位 (None=自动探索)")
    day_gan: str | None = Field(default=None, pattern=r"^[甲乙丙丁戊己庚辛壬癸]$", description="日干")
    reward: float = Field(default=0.0, ge=-1.0, le=1.0, description="奖励信号 [-1, 1]")
    # P0-2: multimodal input fields
    language_text: str | None = Field(default=None, description="自然语言输入")
    vision_features: list[float] | None = Field(default=None, description="视觉特征向量")

    model_config = {"extra": "forbid"}


class SessionStartRequest(ConfigOverrides):
    """P4-7 — create an agent session.

    Inherits every ``TrinityConfig`` field dynamically.  New fields
    added to ``TrinityConfig`` automatically appear here with their
    declared defaults, so the API surface stays in sync with the CLI
    and MCP surfaces.
    """

    model_config = {"extra": "forbid"}


# ------------------------------------------------------------------
# 响应
# ------------------------------------------------------------------
class TickReportResponse(BaseModel):
    """OODA 单步结果, 对齐 TickReport dataclass。"""
    episode_id: int
    h_current: dict[str, Any]
    h_next: dict[str, Any]
    top_mutation: int
    top_score: float
    reward: float
    jepa_loss: float | None
    router_loss: float | None
    surprise: float
    mutation_class: str
    codon: str
    codon_aa: str
    moe_active_experts: list[str]
    trajectory: list[dict[str, Any]]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    step_count: int
    db_path: str


class SessionHistory(BaseModel):
    session_id: str
    ticks: list[TickReportResponse]
    total: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class InfoResponse(BaseModel):
    name: str
    version: str
    modules: list[str]
    description: str


# ------------------------------------------------------------------
# WebSocket 消息
# ------------------------------------------------------------------
class WSTickCommand(BaseModel):
    """WebSocket 客户端发送的 OODA 指令。"""
    type: str = "tick"
    payload: TickRequest = Field(default_factory=TickRequest)


class WSTickResult(BaseModel):
    """WebSocket 服务端推送的 OODA 结果。"""
    type: str = "tick_result"
    payload: TickReportResponse