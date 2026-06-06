"""P3a — Elastic Weight Consolidation (EWC) 防灾难遗忘.

EWC (Kirkpatrick et al. 2017, PNAS) 是持续学习中防止灾难遗忘的核心方法。
在 2026 年, EWC 仍然是 online continual learning 的最佳 baseline。

核心思想:
  1. 对每个已学习的任务, 计算 Fisher Information Matrix (FIM) 对角线
  2. 在新任务训练时, 添加 L2 正则化项:  Σ (F_i / 2) * (θ_i - θ*_i)^2
  3. F_i 大的参数对旧任务重要 → 大惩罚 → 不容易被新任务覆盖

ZWM 中的应用:
  - 卦象空间有 64 种状态, 每个新 hexagram 对应一个 "task"
  - 防止学习新卦象时忘记旧卦象的世界模型表示
  - 与 OnlineLearner 协同工作: EWC 负责参数级巩固, Hebbian 负责关联级巩固

用法:
    from zwm.learning.ewc import EWCRegularizer

    ewc = EWCRegularizer(model=jepa_predictor, importance=100.0)
    # 训练前: 记录当前任务的重要性
    ewc.register_task(task_id="hex_1_乾为天")
    # 训练时: 添加 EWC 惩罚到损失
    loss = task_loss + ewc.penalty()
    # 切换任务: 合并新任务的重要性
    ewc.register_task(task_id="hex_2_坤为地")
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_log = logging.getLogger(__name__)

__all__ = ["EWCRegularizer", "EWCState", "compute_fisher_diagonal"]


@dataclass
class EWCState:
    """单个任务的 EWC 状态: 最优参数 + Fisher 对角线."""
    params: dict[str, np.ndarray]  # 参数名 → 最优值
    fisher: dict[str, np.ndarray]  # 参数名 → Fisher 对角线
    task_id: str = ""


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader: Any = None,
    num_samples: int = 100,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """计算模型的 Fisher Information Matrix 对角线.

    F_i = E[(∂L/∂θ_i)^2] 在训练数据分布上的期望。

    对于 ZWM 的在线场景 (无 dataloader), 使用最近的参数梯度
    作为 proxy — 大梯度意味着参数对当前任务重要。

    Args:
        model: PyTorch 模型
        dataloader: 数据加载器 (None 时使用最近的梯度)
        num_samples: 采样数 (仅当有 dataloader 时)
        device: 计算设备

    Returns:
        dict: 参数名 → Fisher 对角线 (numpy array)
    """
    if device is None:
        device = next(model.parameters()).device

    fisher: dict[str, np.ndarray] = {}

    if dataloader is not None:
        # 有数据时: 从数据分布采样估计 Fisher
        model.train()
        fisher_accum: dict[str, torch.Tensor] = {}
        n_samples = 0

        for batch in dataloader:
            if n_samples >= num_samples:
                break
            model.zero_grad()
            # 用负对数似然估计 Fisher
            if isinstance(batch, (tuple, list)):
                loss = -model(*[b.to(device) if isinstance(b, torch.Tensor) else b
                               for b in batch])
            else:
                loss = -model(batch.to(device) if isinstance(batch, torch.Tensor) else batch)
            if isinstance(loss, torch.Tensor):
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        g = param.grad.data.clone()
                        if name not in fisher_accum:
                            fisher_accum[name] = torch.zeros_like(g)
                        fisher_accum[name] += g ** 2
            n_samples += 1

        for name, val in fisher_accum.items():
            fisher[name] = (val / max(n_samples, 1)).cpu().numpy()
    else:
        # 无数据时: 使用最近的参数梯度作为 proxy
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] = (param.grad.data ** 2).cpu().numpy()
            else:
                # 没有梯度 → 使用参数大小的 L2 作为均匀先验
                fisher[name] = (param.data ** 2).cpu().numpy() * 1e-6

    return fisher


class EWCRegularizer:
    """EWC 正则化器 — 持续学习防遗忘.

    用法::

        ewc = EWCRegularizer(model, importance=100.0)

        # 任务 1: 训练
        ewc.register_task("hex_1")
        loss = task1_loss + ewc.penalty()
        loss.backward(); opt.step()

        # 任务 2: 训练 (不会忘记任务 1)
        ewc.register_task("hex_2")
        loss = task2_loss + ewc.penalty()
        loss.backward(); opt.step()

    参考:
      - Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
      - 2026 SOTA: Online EWC + Fisher merging + importance decay
    """

    def __init__(
        self,
        model: nn.Module,
        importance: float = 100.0,
        decay: float = 0.999,
        max_tasks: int = 64,
    ) -> None:
        self._model = model
        self._importance = importance
        self._decay = decay           # 旧任务重要性衰减率
        self._max_tasks = max_tasks

        # 每个任务的 EWC 状态
        self._tasks: OrderedDict[str, EWCState] = OrderedDict()

        # 合并后的 Fisher 对角线 (所有任务的加权和)
        self._merged_fisher: dict[str, np.ndarray] = {}
        self._merged_params: dict[str, np.ndarray] = {}

        # 计算设备
        self._device = next(model.parameters()).device

    @property
    def n_tasks(self) -> int:
        return len(self._tasks)

    @property
    def task_ids(self) -> list[str]:
        return list(self._tasks.keys())

    def register_task(
        self,
        task_id: str,
        dataloader: Any = None,
        num_samples: int = 100,
    ) -> None:
        """注册一个任务并更新合并的 Fisher 信息.

        Args:
            task_id: 任务标识 (如 "hex_1", "hex_2")
            dataloader: 可选 — 用于计算 Fisher 的数据
            num_samples: Fisher 估计的采样数
        """
        # 计算当前参数的 Fisher 对角线
        fisher = compute_fisher_diagonal(
            self._model, dataloader=dataloader,
            num_samples=num_samples, device=self._device,
        )

        # 保存当前最优参数
        params = {
            name: param.data.clone().cpu().numpy()
            for name, param in self._model.named_parameters()
        }

        # 创建 EWC 状态
        state = EWCState(params=params, fisher=fisher, task_id=task_id)

        # 如果已经注册过, 合并 fisher (取最大值 — 保留对两个任务都重要的参数)
        if task_id in self._tasks:
            old_state = self._tasks[task_id]
            for name in fisher:
                if name in old_state.fisher:
                    fisher[name] = np.maximum(fisher[name], old_state.fisher[name])

        self._tasks[task_id] = state
        if len(self._tasks) > self._max_tasks:
            # 删除最旧的任务
            oldest = next(iter(self._tasks))
            del self._tasks[oldest]
            _log.debug("EWC: evicted oldest task %s (max_tasks=%d)", oldest, self._max_tasks)

        # 重新合并所有任务的 Fisher
        self._merge_all_tasks()

        _log.info(
            "EWC: registered task %s (%d total tasks, merged_fisher params=%d)",
            task_id, len(self._tasks), len(self._merged_fisher),
        )

    def _merge_all_tasks(self) -> None:
        """合并所有已注册任务的 Fisher 信息.

        合并策略:
          - 每个任务的 Fisher 按 (decay)^age 加权
          - 合并参数取最近任务的值
        """
        if not self._tasks:
            self._merged_fisher = {}
            self._merged_params = {}
            return

        merged_f: dict[str, np.ndarray] = {}
        merged_p: dict[str, np.ndarray] = {}

        tasks_list = list(self._tasks.values())
        n = len(tasks_list)

        for i, state in enumerate(tasks_list):
            age = n - 1 - i  # 0 for newest, n-1 for oldest
            weight = self._decay ** age
            for name, f_val in state.fisher.items():
                if name not in merged_f:
                    merged_f[name] = np.zeros_like(f_val)
                merged_f[name] += weight * f_val

        # 合并参数 — 使用最新任务的参数作为锚点
        newest = tasks_list[-1]
        merged_p = copy.deepcopy(newest.params)

        self._merged_fisher = merged_f
        self._merged_params = merged_p

    def penalty(
        self,
        scale: float | None = None,
    ) -> torch.Tensor:
        """计算 EWC 惩罚项.

        L_ewc = (importance / 2) * Σ_i F_i * (θ_i - θ*_i)^2

        Returns:
            标量损失张量 (在模型设备上)
        """
        if not self._merged_fisher:
            return torch.tensor(0.0, device=self._device)

        loss = torch.tensor(0.0, device=self._device)
        imp = scale if scale is not None else self._importance

        for name, param in self._model.named_parameters():
            if name not in self._merged_fisher:
                continue
            # 加载到设备
            fisher_i = torch.from_numpy(self._merged_fisher[name]).to(self._device)
            params_i = torch.from_numpy(self._merged_params[name]).to(self._device)
            # EWC 惩罚: F_i * (θ_i - θ*_i)^2
            diff = param - params_i
            loss += (fisher_i * (diff ** 2)).sum()

        return (imp / 2.0) * loss

    def apply_to_loss(
        self,
        task_loss: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        """便捷方法: 将 EWC 惩罚加到任务损失上."""
        return task_loss + self.penalty(scale=scale)

    def importance_vector(
        self,
        normalize: bool = True,
    ) -> dict[str, float]:
        """返回每个参数的重要性摘要 (标量).

        用于监控哪些参数被 EWC '冻结' 了。
        """
        summary: dict[str, float] = {}
        for name, f_val in self._merged_fisher.items():
            total = float(np.sum(f_val))
            if normalize and total > 0:
                total = total / f_val.size
            summary[name] = total
        return summary

    def state_dict(self) -> dict[str, Any]:
        """序列化 EWC 状态 (用于 checkpoint)."""
        tasks_data = {}
        for tid, state in self._tasks.items():
            tasks_data[tid] = {
                "params": {k: v.tolist() for k, v in state.params.items()},
                "fisher": {k: v.tolist() for k, v in state.fisher.items()},
                "task_id": state.task_id,
            }
        return {
            "tasks": tasks_data,
            "importance": self._importance,
            "decay": self._decay,
            "max_tasks": self._max_tasks,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """从 checkpoint 恢复 EWC 状态."""
        self._importance = sd.get("importance", self._importance)
        self._decay = sd.get("decay", self._decay)
        self._max_tasks = sd.get("max_tasks", self._max_tasks)
        self._tasks = OrderedDict()
        for tid, data in sd.get("tasks", {}).items():
            state = EWCState(
                params={k: np.array(v) for k, v in data["params"].items()},
                fisher={k: np.array(v) for k, v in data["fisher"].items()},
                task_id=data.get("task_id", tid),
            )
            self._tasks[tid] = state
        self._merge_all_tasks()
        _log.info("EWC: restored %d tasks from checkpoint", len(self._tasks))
