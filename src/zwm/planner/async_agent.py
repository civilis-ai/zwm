"""P3-8 (audit) — async/await 并发支持。

提供 ``AsyncAgent`` 包装器, 将 TrinityAgent 的同步 OODA 循环暴露为
async 接口, 并支持:
  * 并发批量 OODA 步 (``batch_tick``)
  * 并行参数扫描 (``sweep``)
  * 背景训练循环 (``run_loop`` → asyncio.Task)
  * 与 FastAPI / WebSocket 无缝集成

设计原则:
  - 所有重型计算 (torch/numpy) 在线程池中执行, 不阻塞事件循环
  - 批量操作使用 asyncio.gather 并发
  - 优雅关闭 (graceful shutdown)
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from zwm.planner.agent_data import TickReport


@dataclass
class AsyncTickRequest:
    """异步 OODA 步的请求参数。"""
    sensor_data: dict[str, float] | None = None
    h_current: int | None = None
    year: int = 2026
    month: int = 1
    day: int = 1
    hour: int = 0
    time_phase: float | None = None
    target_palace: int | None = None
    day_gan: str | None = None
    reward: float = 0.0
    # P0-2: multimodal input fields
    language_text: str | None = None
    vision_features: list[float] | None = None


class AsyncAgent:
    """TrinityAgent 的 async 包装器。

    用法:
        # 方式1: 自动创建 agent
        async with AsyncAgent() as agent:
            report = await agent.tick(AsyncTickRequest(sensor_data={...}))

        # 方式2: 复用已有 agent (P1 — WebSocket 复用全局实例)
        async with AsyncAgent(agent=existing_agent, owns_agent=False) as agent:
            report = await agent.tick(...)
    """

    def __init__(
        self,
        db_path: str = "zwm_async.db",
        checkpoint_path: str | None = None,
        mcts_iterations: int = 200,
        max_workers: int | None = None,
        agent: Any = None,
        owns_agent: bool | None = None,
        **kwargs,
    ) -> None:
        self._db_path = db_path
        self._checkpoint_path = checkpoint_path
        self._mcts_iterations = mcts_iterations
        self._kwargs = kwargs
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="zwm-async",
        )
        self._agent: Any = agent
        # owns_agent: True = 自己创建的, 需要 close; False = 外部传入的, 不 close
        self._owns_agent = owns_agent if owns_agent is not None else (agent is None)
        self._loop_task: asyncio.Task | None = None

    async def __aenter__(self) -> "AsyncAgent":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def start(self) -> None:
        """在后台线程构造 agent (torch 初始化可能耗时)。"""
        loop = asyncio.get_running_loop()
        self._agent = await loop.run_in_executor(self._executor, self._build_agent)

    def _build_agent(self):
        from zwm.planner.agent import TrinityAgent
        return TrinityAgent(
            db_path=self._db_path,
            checkpoint_path=self._checkpoint_path,
            mcts_iterations=self._mcts_iterations,
            **self._kwargs,
        )

    async def close(self) -> None:
        """优雅关闭: 停止背景循环 → 关闭 agent → 关闭线程池。"""
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        if self._agent is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._agent.close)
            self._agent = None
        self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # 核心 async API
    # ------------------------------------------------------------------
    async def tick(self, req: AsyncTickRequest) -> TickReport:
        """单步 async OODA — 在线程池中执行, 不阻塞事件循环。"""
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._tick_sync, req,
        )

    def _tick_sync(self, req: AsyncTickRequest) -> TickReport:
        from zwm.core.hexagram import hexagram_from_bits
        h = None
        if req.h_current is not None:
            h = hexagram_from_bits(req.h_current)
        return self._agent.observe_predict_evaluate_act(
            sensor_data=req.sensor_data,
            h_current=h,
            year=req.year,
            month=req.month,
            day=req.day,
            hour=req.hour,
            time_phase=req.time_phase,
            target_palace=req.target_palace,
            day_gan=req.day_gan,
            reward=req.reward,
            language_text=req.language_text,
            vision_features=np.array(req.vision_features, dtype=np.float32) if req.vision_features else None,
        )

    async def batch_tick(
        self, requests: list[AsyncTickRequest],
    ) -> list[TickReport]:
        """并发批量 OODA — 多个请求在线程池中并行执行。

        注意: 所有请求共享同一个 agent 实例, 因此内部状态 (如
        _palace_visits, _step_count) 会按请求到达顺序交错更新。
        如有需要隔离状态, 请使用多个 AsyncAgent 实例。
        """
        tasks = [self.tick(req) for req in requests]
        return await asyncio.gather(*tasks)

    async def sweep(
        self,
        base_req: AsyncTickRequest,
        param: str,
        values: list[Any],
    ) -> list[tuple[Any, TickReport]]:
        """参数扫描 — 对某个参数在多个值上并行运行 OODA。

        例: sweep(base_req, "reward", [-1.0, -0.5, 0.0, 0.5, 1.0])
        """
        requests = [
            AsyncTickRequest(**{**base_req.__dict__, param: v})
            for v in values
        ]
        reports = await self.batch_tick(requests)
        return list(zip(values, reports))

    # ------------------------------------------------------------------
    # 背景循环
    # ------------------------------------------------------------------
    async def run_loop(
        self,
        steps: int = 100,
        interval: float = 0.0,
        on_tick: callable | None = None,
    ) -> list[TickReport]:
        """异步运行 N 步 OODA 循环, 每步可选调用 on_tick 回调。

        on_tick 可以是 async 函数, 接收 TickReport 参数。
        """
        results: list[TickReport] = []
        for i in range(steps):
            report = await self.tick(AsyncTickRequest())
            results.append(report)
            if on_tick is not None:
                if asyncio.iscoroutinefunction(on_tick):
                    await on_tick(report)
                else:
                    on_tick(report)
            if interval > 0:
                await asyncio.sleep(interval)
        return results

    def start_background_loop(
        self, steps: int = 100, interval: float = 0.1,
        on_tick: callable | None = None,
    ) -> asyncio.Task:
        """启动后台 OODA 循环 (返回 Task, 可 await/cancel)。"""
        self._loop_task = asyncio.create_task(
            self.run_loop(steps=steps, interval=interval, on_tick=on_tick)
        )
        return self._loop_task

    # ------------------------------------------------------------------
    # 属性代理
    # ------------------------------------------------------------------
    @property
    def step_count(self) -> int:
        return self._agent._step_count if self._agent is not None else 0

    @property
    def store(self):
        return self._agent.store if self._agent is not None else None