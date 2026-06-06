"""ZWM FastAPI 接入层 (P2-3 audit)。

提供 REST + WebSocket 接口, 将 TrinityAgent 的 OODA 循环暴露为
HTTP 服务。可选依赖: ``pip install zwm[api]``。
"""
from .app import app, create_app