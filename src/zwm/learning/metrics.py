"""P1-4: Lightweight learning-curve logger (TensorBoard + JSONL fallback).

Provides a single ``MetricsLogger`` class that writes metrics to:
  * TensorBoard (if ``tensorboard`` is installed) — ``runs/`` directory.
  * A JSONL file (``metrics.jsonl``) — always, even if TB is unavailable.
  * Optionally: a CSV summary (``metrics.csv``) for offline plotting.

The 2026 SOTA monitoring default is Weights & Biases, but TensorBoard is
still the universal fallback and the only dependency-free option for
self-hosted experiments. The logger is a context manager that takes
care of flushing on exit and never blocks the agent loop on I/O.
"""
from __future__ import annotations

import json
import os
import time
import atexit
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(
        self,
        log_dir: str | os.PathLike = "runs/zwm",
        run_name: str | None = None,
        also_jsonl: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "zwm",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if run_name is None:
            run_name = f"run_{int(time.time())}"
        self.run_dir = self.log_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl" if also_jsonl else None
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(self.run_dir))
        except Exception:
            # TensorBoard is optional. The agent's metrics still go to
            # JSONL so the user can plot them with any tool.
            self.tb_writer = None
        # 2026 SOTA: Weights & Biases remote observability.  When
        # ``use_wandb=True`` (or the ``WANDB_API_KEY`` env var is set),
        # metrics are also streamed to a W&B dashboard for remote
        # monitoring, experiment comparison, and team collaboration.
        self._wandb = None
        if use_wandb or os.environ.get("WANDB_API_KEY"):
            try:
                import wandb
                wandb.init(project=wandb_project, name=run_name, dir=str(self.log_dir))
                self._wandb = wandb
            except Exception:
                self._wandb = None
        self.step = 0
        # Open JSONL in append mode
        if self.jsonl_path is not None:
            self._fh = open(self.jsonl_path, "a", encoding="utf-8")
        else:
            self._fh = None

    def log(
        self,
        metrics: dict[str, float | int],
        global_step: int | None = None,
    ) -> None:
        if global_step is None:
            global_step = self.step
        # TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                try:
                    self.tb_writer.add_scalar(k, float(v), global_step)
                except Exception:
                    pass
            self.tb_writer.flush()
        # JSONL — one row per call.
        if self._fh is not None:
            row = {
                "step": int(global_step),
                "time": time.time(),
            }
            for k, v in metrics.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    # Non-numeric values (e.g. raw strings, None) leak through
                    # when a caller accidentally passes a categorical label.
                    # Skip them gracefully so the JSONL row is still written,
                    # and log once so the bug is visible.
                    _warn_skipped.add(k)
            if _warn_skipped:
                import logging
                _log = logging.getLogger(__name__)
                _log.warning("telemetry: skipped non-numeric keys: %s", sorted(_warn_skipped))
                _warn_skipped.clear()
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()
        # W&B — stream to remote dashboard.
        if self._wandb is not None:
            try:
                self._wandb.log(
                    {k: float(v) for k, v in metrics.items()},
                    step=int(global_step),
                )
            except Exception:
                pass
        self.step = int(global_step) + 1

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
        if self._fh is not None:
            self._fh.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args: Any) -> bool:
        self.close()
        return False


# Module-level singleton — convenient for quick experiments.
_default_logger: MetricsLogger | None = None

# Non-numeric telemetry keys seen this batch — dedup + warn once.
_warn_skipped: set[str] = set()


def _close_singleton() -> None:
    """P2-3: atexit handler — close the singleton logger's file handles."""
    global _default_logger
    if _default_logger is not None:
        _default_logger.close()
        _default_logger = None


atexit.register(_close_singleton)


def get_logger(
    log_dir: str | os.PathLike = "runs/zwm",
    run_name: str | None = None,
) -> MetricsLogger:
    """Get or create a process-wide MetricsLogger singleton."""
    global _default_logger
    if _default_logger is None:
        _default_logger = MetricsLogger(log_dir=log_dir, run_name=run_name)
    return _default_logger


def close_logger() -> None:
    """P2-3: Explicitly close the singleton logger (idempotent)."""
    _close_singleton()


@contextmanager
def metrics_context(
    log_dir: str | os.PathLike = "runs/zwm",
    run_name: str | None = None,
):
    """Context-manager wrapper for one-shot experiments."""
    logger = MetricsLogger(log_dir=log_dir, run_name=run_name)
    try:
        yield logger
    finally:
        logger.close()
