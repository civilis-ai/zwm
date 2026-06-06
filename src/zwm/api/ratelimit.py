"""H4 — 速率限制器 (Token Bucket + Sliding Window)。

提供 3 种限流策略, 全部线程安全 + 异步友好:

* :class:`TokenBucket`  — 经典令牌桶, 适合 QPS 限流 (WebSocket tick/秒)
* :class:`SlidingWindow` — 滑动窗口, 适合突发流量 (REST 端点 / 分钟)
* :class:`RateLimiterRegistry` — 全局注册表, 按 (scope, key) 维度管理

设计目标
--------
1. **零外部依赖** — 仅用标准库, 避免引入 ``slowapi`` / ``limits`` 等
2. **可观测** — 每次拒绝都会触发 Prometheus 计数器
3. **可配置** — 通过环境变量 ``ZWM_RATE_LIMIT_*`` 调整
4. **优雅降级** — 限流器内部错误不应阻塞业务
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

_LOG = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Token Bucket (WebSocket tick/秒)
# ----------------------------------------------------------------------
@dataclass
class TokenBucket:
    """经典令牌桶, 线程安全.

    Parameters
    ----------
    capacity : int
        桶容量 (突发上限).
    refill_rate : float
        每秒补充令牌数.
    initial : float | None
        初始令牌数, 默认等于 capacity (允许初始突发).
    """

    capacity: float
    refill_rate: float
    initial: Optional[float] = None
    _tokens: float = field(init=False)
    _last: float = field(init=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.refill_rate <= 0:
            raise ValueError("refill_rate must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self._tokens = float(self.initial if self.initial is not None else self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta > 0:
            self._tokens = min(
                self.capacity,
                self._tokens + delta * self.refill_rate,
            )
            self._last = now

    def try_consume(self, tokens: float = 1.0) -> bool:
        """非阻塞获取令牌. 成功返回 True, 失败返回 False."""
        if tokens <= 0:
            return True
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def retry_after(self) -> float:
        """返回距离下一个令牌可用的秒数 (用于响应 Retry-After header)."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                return 0.0
            deficit = 1.0 - self._tokens
            return deficit / self.refill_rate

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            self._refill()
            return {
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "tokens": self._tokens,
            }


# ----------------------------------------------------------------------
# Sliding Window (REST 端点 / 分钟)
# ----------------------------------------------------------------------
@dataclass
class SlidingWindow:
    """滑动窗口限流器, 线程安全.

    Parameters
    ----------
    window_seconds : float
        窗口长度 (秒).
    max_requests : int
        窗口内最大请求数.
    """

    window_seconds: float
    max_requests: int
    _hits: Deque[float] = field(init=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or self.max_requests <= 0:
            raise ValueError("window_seconds and max_requests must be positive")
        self._hits = deque()
        self._lock = threading.Lock()

    def try_record(self, now: Optional[float] = None) -> bool:
        """记录一次请求, 命中窗口返回 True, 超限返回 False."""
        ts = now if now is not None else time.monotonic()
        with self._lock:
            # Pop expired
            cutoff = ts - self.window_seconds
            while self._hits and self._hits[0] < cutoff:
                self._hits.popleft()
            if len(self._hits) >= self.max_requests:
                return False
            self._hits.append(ts)
            return True

    def retry_after(self) -> float:
        with self._lock:
            if len(self._hits) < self.max_requests:
                return 0.0
            oldest = self._hits[0]
            return max(0.0, oldest + self.window_seconds - time.monotonic())


# ----------------------------------------------------------------------
# Registry — 按 (scope, key) 维度管理限流器
# ----------------------------------------------------------------------
class RateLimiterRegistry:
    """集中管理所有限流器, 支持自动清理过期桶.

    Scopes:
      * "ws"  : WebSocket 连接 / session
      * "ip"  : 客户端 IP
      * "session" : REST 会话
      * "token" : Bearer token
    """

    _instance: Optional["RateLimiterRegistry"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._buckets: Dict[Tuple[str, str], TokenBucket] = {}
        self._windows: Dict[Tuple[str, str], SlidingWindow] = {}
        self._last_seen: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = 300.0  # 5 min
        self._last_cleanup = time.monotonic()
        # Default policies (can be overridden via env)
        self._ws_capacity = float(os.environ.get("ZWM_WS_BUCKET_CAPACITY", "10"))
        self._ws_refill = float(os.environ.get("ZWM_WS_BUCKET_REFILL", "5"))
        self._ws_window_s = float(os.environ.get("ZWM_WS_WINDOW_SECONDS", "60"))
        self._ws_window_max = int(os.environ.get("ZWM_WS_WINDOW_MAX", "120"))
        # Per-IP
        self._ip_capacity = float(os.environ.get("ZWM_IP_BUCKET_CAPACITY", "30"))
        self._ip_refill = float(os.environ.get("ZWM_IP_BUCKET_REFILL", "10"))
        self._ip_window_s = float(os.environ.get("ZWM_IP_WINDOW_SECONDS", "60"))
        self._ip_window_max = int(os.environ.get("ZWM_IP_WINDOW_MAX", "600"))
        # REST endpoints — higher limits than WS since they're explicit
        # request-response pairs (not a persistent stream).
        self._rest_capacity = float(os.environ.get("ZWM_REST_BUCKET_CAPACITY", "30"))
        self._rest_refill = float(os.environ.get("ZWM_REST_BUCKET_REFILL", "10"))
        self._rest_window_s = float(os.environ.get("ZWM_REST_WINDOW_SECONDS", "60"))
        self._rest_window_max = int(os.environ.get("ZWM_REST_WINDOW_MAX", "300"))

    @classmethod
    def instance(cls) -> "RateLimiterRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ---- token bucket (rate per second) ---------------------------------
    def _scope_params(self, scope: str) -> tuple[float, float, float, int]:
        """Return (bucket_capacity, bucket_refill, window_seconds, window_max)
        for the given scope."""
        if scope == "ip":
            return (self._ip_capacity, self._ip_refill,
                    self._ip_window_s, self._ip_window_max)
        if scope == "rest":
            return (self._rest_capacity, self._rest_refill,
                    self._rest_window_s, self._rest_window_max)
        # "ws" and any other scope fall back to WebSocket defaults.
        return (self._ws_capacity, self._ws_refill,
                self._ws_window_s, self._ws_window_max)

    def get_bucket(self, scope: str, key: str) -> TokenBucket:
        bucket_key = (scope, key)
        now = time.monotonic()
        with self._lock:
            self._maybe_cleanup_locked(now)
            if bucket_key not in self._buckets:
                cap, rate, _, _ = self._scope_params(scope)
                self._buckets[bucket_key] = TokenBucket(capacity=cap, refill_rate=rate)
            self._last_seen[bucket_key] = now
            return self._buckets[bucket_key]

    # ---- sliding window (rate per minute) -------------------------------
    def get_window(self, scope: str, key: str) -> SlidingWindow:
        win_key = (scope, key)
        now = time.monotonic()
        with self._lock:
            self._maybe_cleanup_locked(now)
            if win_key not in self._windows:
                _, _, ws, wm = self._scope_params(scope)
                self._windows[win_key] = SlidingWindow(
                    window_seconds=ws, max_requests=wm
                )
            self._last_seen[win_key] = now
            return self._windows[win_key]

    # ---- combined check --------------------------------------------------
    def check_and_record(self, scope: str, key: str) -> Tuple[bool, float, str]:
        """Check both token bucket and sliding window.

        Returns (allowed, retry_after_seconds, rejection_reason).
        rejection_reason is "" if allowed.
        """
        try:
            bucket = self.get_bucket(scope, key)
            window = self.get_window(scope, key)
            if not window.try_record():
                return False, window.retry_after(), "window_exceeded"
            if not bucket.try_consume(1.0):
                return False, bucket.retry_after(), "bucket_empty"
            return True, 0.0, ""
        except Exception as exc:  # pragma: no cover - fail open
            _LOG.warning("rate limiter internal error: %s (failing open)", exc)
            return True, 0.0, ""

    # ---- cleanup --------------------------------------------------------
    def _maybe_cleanup_locked(self, now: float) -> None:
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        # Drop entries idle for > 30 min
        threshold = now - 1800.0
        for key in list(self._last_seen.keys()):
            if self._last_seen.get(key, 0) < threshold:
                self._buckets.pop(key, None)
                self._windows.pop(key, None)
                self._last_seen.pop(key, None)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "buckets": len(self._buckets),
                "windows": len(self._windows),
                "tracked_keys": len(self._last_seen),
            }

    def reset(self) -> None:
        """For tests only."""
        with self._lock:
            self._buckets.clear()
            self._windows.clear()
            self._last_seen.clear()
            self._last_cleanup = time.monotonic()


# ----------------------------------------------------------------------
# FastAPI 依赖 — REST 端点限流
# ----------------------------------------------------------------------
def _client_key_from_request(request) -> str:
    """Derive a stable client identity from the incoming request.

    Priority: Bearer token prefix > X-Forwarded-For > client host.
    The token prefix provides per-user isolation; the IP fallback
    provides a coarse floor when auth is disabled.
    """
    # Bearer token (first 8 chars) — stable per-user identity.
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return f"tok:{auth[7:15]}"
    # X-Forwarded-For (first hop) — respects reverse proxies.
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return f"ip:{xff.split(',')[0].strip()}"
    # Direct client host.
    if request.client:
        return f"ip:{request.client.host}"
    return "ip:unknown"


async def require_rate_limit(request: "Request") -> None:
    """FastAPI 依赖 — 对 REST 端点实施双限流 (TokenBucket + SlidingWindow)。

    用法::

        @app.post("/tick", dependencies=[Depends(require_rate_limit)])
        async def do_tick(...): ...

    限流策略通过环境变量配置:
      * ``ZWM_REST_BUCKET_CAPACITY``  — 令牌桶容量 (默认 30)
      * ``ZWM_REST_BUCKET_REFILL``    — 每秒补充令牌数 (默认 10)
      * ``ZWM_REST_WINDOW_SECONDS``   — 滑动窗口秒数 (默认 60)
      * ``ZWM_REST_WINDOW_MAX``       — 窗口内最大请求数 (默认 300)

    超限时返回 429 Too Many Requests, 携带 ``Retry-After`` 头。
    """
    from fastapi import Request as _Request, HTTPException, status as http_status
    from zwm.observability import metrics as _obs_metrics

    rl = RateLimiterRegistry.instance()
    key = _client_key_from_request(request)
    allowed, retry_after_s, reason = rl.check_and_record("rest", key)
    if not allowed:
        try:
            _obs_metrics.inc_rate_limit_rejected(scope="rest", reason=reason)
        except Exception:
            pass
        raise HTTPException(
            status_code=http_status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {reason}",
            headers={"Retry-After": str(max(1, int(retry_after_s + 0.5)))},
        )


__all__ = [
    "TokenBucket",
    "SlidingWindow",
    "RateLimiterRegistry",
    "require_rate_limit",
]
