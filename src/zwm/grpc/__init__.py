"""ZWM gRPC API — 高性能替代 REST + WebSocket 的服务间通信.

基于 protobuf 的强类型 RPC, 比 JSON/REST 快 3-10× (gRPC benchmark).
适用于:
  - 微服务间 OODA 调用
  - 多智能体 A2A 协调 (替代 HTTP 轮询)
  - 实时流式传输 (server-streaming tick)

用法:
    # 启动 gRPC 服务器
    zwm serve-grpc --port 50051

    # 或代码调用
    from zwm.grpc import serve_grpc
    serve_grpc(port=50051)
"""

from zwm.grpc.server import serve_grpc, ZWMGrpcServicer

__all__ = ["serve_grpc", "ZWMGrpcServicer"]
