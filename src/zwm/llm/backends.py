"""多提供商 LLM 后端 — Anthropic / OpenAI / DeepSeek 统一接口.

每个后端实现相同的 :class:`LLMBackend` 协议:
  - ``generate(prompt, system, **kwargs) → str``
  - ``supports_thinking`` — 是否支持 extended thinking
  - ``model_name`` — 当前模型名

:class:`CompositeBackend` 支持 fallback 链: 主模型 → 备用模型 → ...
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

_log = logging.getLogger(__name__)

# ─── 公开类型 ───────────────────────────────────────────
__all__ = [
    "LLMBackend",
    "AnthropicBackend",
    "OpenAIBackend",
    "DeepSeekBackend",
    "CompositeBackend",
    "create_backend",
    "auto_detect_backend",
    "LLMResponse",
]


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """统一的 LLM 响应."""
    text: str
    model: str
    backend_name: str
    usage: dict = field(default_factory=dict)  # {"input_tokens": N, "output_tokens": M}
    latency_ms: float = 0.0
    finish_reason: str = "stop"


# ─── 抽象基类 ────────────────────────────────────────────

class LLMBackend(ABC):
    """LLM 后端的统一抽象.

    子类必须实现 ``_generate_impl``；可选覆盖 ``supports_thinking``。
    """

    name: str = "base"
    supports_thinking: bool = False
    supports_vision: bool = False

    @abstractmethod
    def _generate_impl(
        self,
        prompt: str,
        system: str = "",
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking: bool = False,
        json_mode: bool = False,
    ) -> LLMResponse:
        ...

    def generate(
        self,
        prompt: str,
        system: str = "",
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking: bool = False,
        json_mode: bool = False,
        retries: int = 2,
        backoff_s: float = 1.0,
    ) -> LLMResponse:
        """带重试和退避的生成入口."""
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self._generate_impl(
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking=thinking,
                    json_mode=json_mode,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    wait = backoff_s * (2 ** attempt)
                    _log.warning(
                        "LLM %s attempt %d/%d failed: %s; retrying in %.1fs",
                        self.name, attempt + 1, retries + 1, exc, wait,
                    )
                    time.sleep(wait)
        raise RuntimeError(
            f"LLM {self.name} failed after {retries + 1} attempts: {last_exc}"
        )

    @property
    def model_name(self) -> str:
        return getattr(self, "_model", "unknown")


# ─── Anthropic Claude ────────────────────────────────────

class AnthropicBackend(LLMBackend):
    """Anthropic Claude API 后端 (Messages API).

    环境变量:
      ANTHROPIC_API_KEY — API 密钥
      ANTHROPIC_BASE_URL — 自定义 base URL (可选)

    支持模型:
      - claude-opus-4-8 (最高推理能力, extended thinking)
      - claude-sonnet-4-6 (平衡速度与质量)
      - claude-haiku-4-5-20251001 (最快, 最低成本)
    """

    name = "anthropic"
    supports_thinking = True
    supports_vision = True

    THINKING_MODELS = frozenset({
        "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
        "claude-sonnet-4-6", "claude-sonnet-4-5",
    })

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        if not self._api_key:
            _log.warning("ANTHROPIC_API_KEY not set; AnthropicBackend will fail")
        self._client = None  # lazy init

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic SDK not installed; pip install anthropic"
                ) from exc
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = Anthropic(**kwargs)
        return self._client

    def _generate_impl(
        self,
        prompt: str,
        system: str = "",
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking: bool = False,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        t0 = time.perf_counter()

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        # Anthropic 的 extended thinking 通过 thinking 参数控制
        if thinking and self._model in self.THINKING_MODELS:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": min(max_tokens // 2, 16000)}
            kwargs["temperature"] = 1.0  # thinking 模式要求 temperature=1
        else:
            kwargs["temperature"] = temperature

        resp = client.messages.create(**kwargs)

        # 提取文本内容
        text_parts = []
        input_tokens = 0
        output_tokens = 0
        for blk in resp.content:
            if getattr(blk, "type", "") == "text":
                text_parts.append(blk.text)
        if hasattr(resp, "usage"):
            input_tokens = getattr(resp.usage, "input_tokens", 0) or 0
            output_tokens = getattr(resp.usage, "output_tokens", 0) or 0

        latency = (time.perf_counter() - t0) * 1000
        return LLMResponse(
            text="".join(text_parts),
            model=self._model,
            backend_name=self.name,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            latency_ms=latency,
            finish_reason=getattr(resp, "stop_reason", "stop") or "stop",
        )


# ─── OpenAI GPT ─────────────────────────────────────────

class OpenAIBackend(LLMBackend):
    """OpenAI API 后端 (Chat Completions).

    环境变量:
      OPENAI_API_KEY — API 密钥
      OPENAI_BASE_URL — 自定义 base URL (可用于兼容 API)

    支持模型:
      - gpt-4o, gpt-4o-mini
      - o4-mini, o3-mini (reasoning models)
    """

    name = "openai"
    supports_vision = True

    REASONING_MODELS = frozenset({"o4-mini", "o3-mini", "o1-mini", "o1"})

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not self._api_key:
            _log.warning("OPENAI_API_KEY not set; OpenAIBackend will fail")
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_thinking(self) -> bool:
        return self._model in self.REASONING_MODELS

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai SDK not installed; pip install openai"
                ) from exc
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _generate_impl(
        self,
        prompt: str,
        system: str = "",
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking: bool = False,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        t0 = time.perf_counter()

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        # reasoning models use max_completion_tokens
        if self._model in self.REASONING_MODELS:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)

        text = resp.choices[0].message.content or ""
        latency = (time.perf_counter() - t0) * 1000
        return LLMResponse(
            text=text,
            model=self._model,
            backend_name=self.name,
            usage={
                "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
            latency_ms=latency,
            finish_reason=resp.choices[0].finish_reason or "stop",
        )


# ─── DeepSeek ────────────────────────────────────────────

class DeepSeekBackend(LLMBackend):
    """DeepSeek API 后端 (兼容 OpenAI 协议).

    环境变量:
      DEEPSEEK_API_KEY — API 密钥
      DEEPSEEK_BASE_URL — 自定义 base URL (默认 https://api.deepseek.com/v1)

    支持模型:
      - deepseek-v4-pro (最强推理, 1M context)
      - deepseek-chat (通用对话)
      - deepseek-reasoner (reasoning 专用)
    """

    name = "deepseek"
    supports_thinking = True
    supports_vision = False

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

    REASONING_MODELS = frozenset({"deepseek-v4-pro", "deepseek-reasoner"})

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self._base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", self.DEFAULT_BASE_URL,
        )
        if not self._api_key:
            _log.warning("DEEPSEEK_API_KEY not set; DeepSeekBackend will fail")
        self._client = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_thinking(self) -> bool:
        return self._model in self.REASONING_MODELS

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai SDK not installed; DeepSeek uses OpenAI-compatible API"
                ) from exc
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def _generate_impl(
        self,
        prompt: str,
        system: str = "",
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking: bool = False,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        t0 = time.perf_counter()

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # DeepSeek 的 reasoning models 也支持 temperature
        kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)

        text = resp.choices[0].message.content or ""
        latency = (time.perf_counter() - t0) * 1000
        return LLMResponse(
            text=text,
            model=self._model,
            backend_name=self.name,
            usage={
                "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
            latency_ms=latency,
            finish_reason=resp.choices[0].finish_reason or "stop",
        )


# ─── Composite Backend (fallback 链) ─────────────────────

class CompositeBackend(LLMBackend):
    """Fallback 链后端 — 主模型失败时自动切换到备用模型.

    用法::

        primary = AnthropicBackend(model="claude-sonnet-4-6")
        fallback = OpenAIBackend(model="gpt-4o-mini")
        composite = CompositeBackend([primary, fallback])
        # 主模型可用时使用主模型，失败时自动切换到 fallback
    """

    name = "composite"

    def __init__(self, backends: list[LLMBackend]) -> None:
        if not backends:
            raise ValueError("至少需要一个后端")
        self._backends = backends
        self._healthy: dict[str, bool] = {b.name: True for b in backends}
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return ",".join(b.model_name for b in self._backends)

    @property
    def supports_thinking(self) -> bool:
        return any(b.supports_thinking for b in self._backends)

    @property
    def supports_vision(self) -> bool:
        return any(b.supports_vision for b in self._backends)

    def _generate_impl(
        self,
        prompt: str,
        system: str = "",
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking: bool = False,
        json_mode: bool = False,
    ) -> LLMResponse:
        errors: list[str] = []
        for backend in self._backends:
            with self._lock:
                healthy = self._healthy.get(backend.name, True)
            if not healthy:
                _log.debug("Skipping unhealthy backend %s", backend.name)
                continue
            try:
                resp = backend._generate_impl(
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking=thinking and backend.supports_thinking,
                    json_mode=json_mode,
                )
                return resp
            except Exception as exc:
                msg = f"{backend.name}: {exc}"
                errors.append(msg)
                _log.warning("Backend %s failed, trying next: %s", backend.name, exc)
                # 标记为不健康（短暂）
                with self._lock:
                    self._healthy[backend.name] = False
        raise RuntimeError(
            f"All backends failed: {'; '.join(errors)}"
        )

    def reset_health(self) -> None:
        """重置所有后端的健康状态 (例如定期探测后)."""
        with self._lock:
            for name in self._healthy:
                self._healthy[name] = True


# ─── 工厂函数 ────────────────────────────────────────────

def create_backend(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> LLMBackend:
    """通过名称创建 LLM 后端.

    provider:
      - "anthropic" / "claude"
      - "openai" / "gpt"
      - "deepseek"
      - "auto" — 自动检测环境变量中的 API key
      - "anthropic+openai" — Anthropic 主 + OpenAI fallback
      - "deepseek+openai" — DeepSeek 主 + OpenAI fallback

    示例:
        be = create_backend("anthropic", model="claude-sonnet-4-6")
        be = create_backend("auto")  # 检测环境变量
    """
    provider = provider.lower()

    if provider in ("anthropic", "claude"):
        return AnthropicBackend(
            api_key=api_key,
            model=model or "claude-sonnet-4-6",
            base_url=base_url,
        )
    elif provider in ("openai", "gpt"):
        return OpenAIBackend(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            base_url=base_url,
        )
    elif provider == "deepseek":
        return DeepSeekBackend(
            api_key=api_key,
            model=model or "deepseek-chat",
            base_url=base_url,
        )
    elif provider == "anthropic+openai":
        primary = AnthropicBackend(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            model=model or "claude-sonnet-4-6",
            base_url=base_url,
        )
        fallback = OpenAIBackend(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )
        return CompositeBackend([primary, fallback])
    elif provider == "deepseek+openai":
        primary = DeepSeekBackend(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            model=model or "deepseek-chat",
            base_url=base_url,
        )
        fallback = OpenAIBackend(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )
        return CompositeBackend([primary, fallback])
    elif provider == "auto":
        return auto_detect_backend()
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: anthropic, openai, deepseek, auto, "
            f"anthropic+openai, deepseek+openai"
        )


def auto_detect_backend() -> LLMBackend:
    """根据环境变量自动检测并创建最佳可用后端.

    检测顺序:
      1. DEEPSEEK_API_KEY → DeepSeekBackend
      2. ANTHROPIC_API_KEY → AnthropicBackend
      3. OPENAI_API_KEY → OpenAIBackend
      4. 都没有 → 抛出异常
    """
    backends: list[LLMBackend] = []

    if os.environ.get("DEEPSEEK_API_KEY"):
        backends.append(DeepSeekBackend(
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        ))
    if os.environ.get("ANTHROPIC_API_KEY"):
        backends.append(AnthropicBackend(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        ))
    if os.environ.get("OPENAI_API_KEY"):
        backends.append(OpenAIBackend(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        ))

    if not backends:
        raise RuntimeError(
            "No LLM API key found. Set one of: DEEPSEEK_API_KEY, "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY"
        )

    if len(backends) == 1:
        return backends[0]
    return CompositeBackend(backends)
