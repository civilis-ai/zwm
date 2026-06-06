"""ZWM LLM 推理后端 — 多提供商统一抽象层.

支持:
  - Anthropic Claude (claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5)
  - OpenAI GPT (gpt-4o, gpt-4o-mini, o4-mini)
  - DeepSeek (deepseek-v4-pro, deepseek-chat, deepseek-reasoner)

架构:
  LLMBackend (抽象基类)
    ├─ AnthropicBackend
    ├─ OpenAIBackend
    ├─ DeepSeekBackend
    └─ CompositeBackend (fallback 链)

  LLMRouter — 按任务复杂度自动路由到合适的模型
  ZWMContext — 将 ZWM 内部状态 (卦象/EFE/JEPA) 转为 LLM 提示

用法:
    from zwm.llm import create_backend, LLMRouter, ZWMContext

    backend = create_backend("anthropic", api_key="sk-...", model="claude-sonnet-4-6")
    # 或自动检测
    backend = create_backend("auto")  # 读取环境变量选择

    router = LLMRouter(backend)
    thought = router.generate_thought(hexagram=..., context=...)
"""

from zwm.llm.backends import (
    LLMBackend,
    AnthropicBackend,
    OpenAIBackend,
    DeepSeekBackend,
    CompositeBackend,
    create_backend,
    auto_detect_backend,
)
from zwm.llm.router import LLMRouter, TaskComplexity, ModelTier
from zwm.llm.context import ZWMContext, build_react_prompt, build_planning_prompt

__all__ = [
    "LLMBackend",
    "AnthropicBackend",
    "OpenAIBackend",
    "DeepSeekBackend",
    "CompositeBackend",
    "create_backend",
    "auto_detect_backend",
    "LLMRouter",
    "TaskComplexity",
    "ModelTier",
    "ZWMContext",
    "build_react_prompt",
    "build_planning_prompt",
]
