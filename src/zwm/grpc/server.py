"""ZWM gRPC 服务器 — 高性能 OODA 循环接入.

实现以下 RPC:
  - Tick — 单步 OODA (unary)
  - StreamTick — 流式 OODA (server-streaming, 持续推送每步结果)
  - GetInfo — 获取 agent 信息
  - HealthCheck — 健康检查

使用 grpcio + protobuf (无需外部 .proto 文件, 使用 proto-plus 内联定义).
当 grpcio 不可用时降级为存根.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

_log = logging.getLogger(__name__)

# 条件导入 grpcio
try:
    import grpc
    from grpc import ServicerContext, server
    from concurrent import futures
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False

__all__ = ["serve_grpc", "ZWMGrpcServicer", "TickRequest", "TickResponse"]


# ─── 数据模型 (proto 等价) ─────────────────────────────

@dataclass(frozen=True, slots=True)
class TickRequest:
    """gRPC Tick 请求."""
    hex_bits: int = 1
    hex_name: str = "乾为天"
    target_palace: int = 5
    time_phase: float = 0.0
    reward: float = 0.0
    year: int = 2026
    month: int = 1
    day: int = 1
    hour: int = 0
    sensor_data: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "TickRequest":
        return cls(
            hex_bits=int(d.get("hex_bits", 1)),
            hex_name=str(d.get("hex_name", "乾为天")),
            target_palace=int(d.get("target_palace", 5)),
            time_phase=float(d.get("time_phase", 0.0)),
            reward=float(d.get("reward", 0.0)),
            year=int(d.get("year", 2026)),
            month=int(d.get("month", 1)),
            day=int(d.get("day", 1)),
            hour=int(d.get("hour", 0)),
            sensor_data=d.get("sensor_data", {}),
        )


@dataclass(frozen=True, slots=True)
class TickResponse:
    """gRPC Tick 响应."""
    hex_bits_out: int
    hex_name_out: str
    surprise: float
    reward: float
    jepa_loss: float
    router_loss: float
    efe_value: float
    active_experts: list[str] = field(default_factory=list)
    mutation_class: str = ""
    episode_id: int = 0
    step_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "hex_bits_out": self.hex_bits_out,
            "hex_name_out": self.hex_name_out,
            "surprise": self.surprise,
            "reward": self.reward,
            "jepa_loss": self.jepa_loss,
            "router_loss": self.router_loss,
            "efe_value": self.efe_value,
            "active_experts": list(self.active_experts),
            "mutation_class": self.mutation_class,
            "episode_id": self.episode_id,
            "step_ms": self.step_ms,
        }


# ─── gRPC Servicer ─────────────────────────────────────

class ZWMGrpcServicer:
    """ZWM gRPC 服务实现.

    当 grpcio 不可用时, 也可以直接用于本地调用 (无网络开销).
    """

    def __init__(self, agent: Any = None) -> None:
        self._agent = agent
        self._tick_count = 0
        self._total_time_ms = 0.0
        self._lock = threading.Lock()

    @property
    def agent(self) -> Any:
        return self._agent

    @agent.setter
    def agent(self, ag: Any) -> None:
        self._agent = ag

    @property
    def stats(self) -> dict:
        with self._lock:
            avg_ms = self._total_time_ms / max(self._tick_count, 1)
            return {
                "tick_count": self._tick_count,
                "avg_latency_ms": round(avg_ms, 2),
            }

    def tick(self, request: TickRequest) -> TickResponse:
        """执行单步 OODA 并返回结果."""
        if self._agent is None:
            raise RuntimeError("No agent registered")
        t0 = time.perf_counter()

        from zwm.core.hexagram import hexagram_from_bits, hexagram_from_name
        try:
            h = hexagram_from_bits(request.hex_bits)
        except Exception:
            h = hexagram_from_name(request.hex_name)

        report = self._agent.tick(
            h_current=h,
            target_palace=request.target_palace,
            time_phase=request.time_phase,
            reward=request.reward,
            year=request.year,
            month=request.month,
            day=request.day,
            hour=request.hour,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        with self._lock:
            self._tick_count += 1
            self._total_time_ms += elapsed

        return TickResponse(
            hex_bits_out=int(report.h_next.normal_order),
            hex_name_out=str(report.h_next.name),
            surprise=float(report.surprise),
            reward=float(report.reward),
            jepa_loss=float(report.jepa_loss) if report.jepa_loss is not None else 0.0,
            router_loss=float(report.router_loss) if report.router_loss is not None else 0.0,
            efe_value=float(report.top_score),
            active_experts=list(getattr(report, "moe_active_experts", [])),
            mutation_class=str(report.mutation_class),
            episode_id=int(report.episode_id),
            step_ms=round(elapsed, 2),
        )

    def stream_tick(self, requests: list[TickRequest]) -> list[TickResponse]:
        """批量执行多步 OODA (流式等价)."""
        return [self.tick(req) for req in requests]

    def get_info(self) -> dict:
        """获取 agent 和服务器信息."""
        from zwm import __version__
        info = {
            "version": __version__,
            "grpc_available": _GRPC_AVAILABLE,
            "agent_ready": self._agent is not None,
        }
        info.update(self.stats)
        if self._agent is not None and hasattr(self._agent, "config"):
            info["config"] = self._agent.config.to_dict()
        return info


# ─── 服务器启动 ────────────────────────────────────────

def serve_grpc(
    agent: Any = None,
    port: int = 50051,
    max_workers: int = 10,
) -> Any:
    """启动 gRPC 服务器.

    Args:
        agent: TrinityAgent 实例 (可选, 可通过 servicer.agent 后设置)
        port: 监听端口
        max_workers: 线程池大小

    Returns:
        grpc.Server 对象 (可调用 .stop() 停止)
    """
    if not _GRPC_AVAILABLE:
        _log.warning(
            "grpcio not installed; gRPC server unavailable. "
            "Install with: pip install grpcio"
        )
        return None

    servicer = ZWMGrpcServicer(agent=agent)
    grpc_server = server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # 注册服务 — 使用通用 RPC handler (无 .proto 编译)
    _register_handlers(grpc_server, servicer)

    addr = f"[::]:{port}"
    grpc_server.add_insecure_port(addr)
    grpc_server.start()
    _log.info("ZWM gRPC server listening on %s", addr)
    return grpc_server


def _register_handlers(grpc_server: Any, servicer: ZWMGrpcServicer) -> None:
    """注册通用 gRPC handlers (无需 .proto 编译).

    使用 grpc 的 generic handler API 支持动态方法调用。
    在生产环境中应使用编译好的 protobuf stubs。
    """
    if not _GRPC_AVAILABLE:
        return

    # 使用 generic service handler
    handler = _GenericHandler(servicer)

    # zwm.ZWM/Tick
    grpc_server.add_generic_rpc_handlers((
        _make_method_handler("/zwm.ZWM/Tick", unary_handler=handler.handle_tick),
        _make_method_handler("/zwm.ZWM/StreamTick", stream_handler=handler.handle_stream_tick),
        _make_method_handler("/zwm.ZWM/GetInfo", unary_handler=handler.handle_get_info),
        _make_method_handler("/zwm.ZWM/HealthCheck", unary_handler=handler.handle_health_check),
    ))


class _GenericHandler:
    """泛型 RPC handler — 解析 JSON 负载并调用 servicer 方法."""

    def __init__(self, servicer: ZWMGrpcServicer) -> None:
        self._servicer = servicer

    def handle_tick(self, request: bytes, context: Any) -> bytes:
        import json
        try:
            req_dict = json.loads(request.decode("utf-8"))
            req = TickRequest.from_dict(req_dict)
            resp = self._servicer.tick(req)
            return json.dumps(resp.to_dict(), ensure_ascii=False).encode("utf-8")
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return b"{}"

    def handle_stream_tick(self, request: bytes, context: Any) -> Any:
        import json
        try:
            req_list = json.loads(request.decode("utf-8"))
            reqs = [TickRequest.from_dict(r) for r in req_list]
            for resp in self._servicer.stream_tick(reqs):
                yield json.dumps(resp.to_dict(), ensure_ascii=False).encode("utf-8")
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return

    def handle_get_info(self, request: bytes, context: Any) -> bytes:
        import json
        resp = self._servicer.get_info()
        return json.dumps(resp, ensure_ascii=False).encode("utf-8")

    def handle_health_check(self, request: bytes, context: Any) -> bytes:
        import json
        resp = {"status": "SERVING", "agent_ready": self._servicer.agent is not None}
        return json.dumps(resp).encode("utf-8")


def _make_method_handler(
    method: str,
    unary_handler: Callable | None = None,
    stream_handler: Callable | None = None,
) -> Any:
    """创建一个 generic RPC method handler."""
    if not _GRPC_AVAILABLE:
        return None

    if stream_handler is not None:
        rpc_handler = grpc.unary_stream_rpc_method_handler(
            stream_handler,
            request_deserializer=lambda b: b,
            response_serializer=lambda b: b,
        )
    elif unary_handler is not None:
        rpc_handler = grpc.unary_unary_rpc_method_handler(
            unary_handler,
            request_deserializer=lambda b: b,
            response_serializer=lambda b: b,
        )
    else:
        return None

    _method = method
    _rpc_handler = rpc_handler

    class _ServiceHandler(grpc.GenericRpcHandler):
        def service(self, handler_call_details):
            if handler_call_details.method == _method:
                return _rpc_handler
            return None

    return _ServiceHandler()
