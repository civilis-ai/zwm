"""P2-X (audit) — FSDP2 distributed training wrappers for JEPA.

Split from ``predictor.py`` to reduce its 1300+ line monolith.  FSDP2 is
the 2026 SOTA PyTorch distributed training API.  It shards model
parameters across GPUs, reducing per-GPU memory by 1/N and enabling
training of larger models (wider latent_dim, deeper MLPs).

Usage:
    import torch.distributed as dist
    dist.init_process_group("nccl")
    model = JEPAPredictor(input_dim=106)  # uses Z_WORLD_DIM default
    model = wrap_fsdp2(model)

When the distributed package is not available or only a single GPU
is detected, returns the model unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import logging

_log = logging.getLogger(__name__)

__all__ = [
    "wrap_fsdp2",
    "wrap_fsdp2_hierarchical",
]


def _fsdp_available() -> bool:
    """Check whether FSDP2 is importable in this environment."""
    try:
        import torch.distributed as dist  # noqa: F401
        from torch.distributed.fsdp import fully_shard  # noqa: F401
    except ImportError:
        return False
    return True


def _should_skip_sharding() -> bool:
    """Skip sharding when not in a multi-GPU environment."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return True
    if torch.cuda.device_count() <= 1:
        return True
    return False


def wrap_fsdp2(model: nn.Module) -> nn.Module:
    """Wrap a JEPAPredictor for multi-GPU training with FSDP2.

    FSDP2 (``torch.distributed.fsdp.fully_shard``) is the 2026 SOTA
    PyTorch distributed training API.  It shards model parameters across
    GPUs, reducing per-GPU memory by 1/N and enabling training of larger
    models (wider latent_dim, deeper MLPs).

    Usage:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        model = JEPAPredictor(input_dim=106)  # uses Z_WORLD_DIM default
        model = wrap_fsdp2(model)

    When the distributed package is not available or only a single GPU
    is detected, returns the model unchanged.
    """
    if not _fsdp_available():
        return model
    if _should_skip_sharding():
        return model

    from torch.distributed.fsdp import fully_shard

    # Apply FSDP2 to the main submodules.
    # fully_shard recursively shards parameters and their gradients.
    for name, child in model.named_children():
        if name in ("context_encoder", "predictor", "target_encoder"):
            try:
                fully_shard(child)
            except Exception as exc:
                _log.warning("FSDP2 fully_shard(%s) failed: %s — returning unwrapped model", name, exc)
                return model

    try:
        fully_shard(model)
    except Exception as exc:
        _log.warning("FSDP2 fully_shard(model) failed: %s — returning unwrapped model", exc)
        return model

    return model


def wrap_fsdp2_hierarchical(model) -> nn.Module:
    """Wrap a HierarchicalJEPAPredictor for multi-GPU FSDP2 training.

    Shards each level (short/mid/long) independently, then shards the
    top-level model.
    """
    if not _fsdp_available():
        return model
    if _should_skip_sharding():
        return model

    from torch.distributed.fsdp import fully_shard

    for level_name in ("short", "mid", "long"):
        level = getattr(model, level_name, None)
        if level is not None:
            for name, child in level.named_children():
                if name in ("context_encoder", "predictor", "target_encoder"):
                    try:
                        fully_shard(child)
                    except Exception as exc:
                        _log.warning("FSDP2 hierarchical fully_shard(%s.%s) failed: %s — returning unwrapped model",
                                     level_name, name, exc)
                        return model
            try:
                fully_shard(level)
            except Exception as exc:
                _log.warning("FSDP2 hierarchical fully_shard(%s) failed: %s — returning unwrapped model",
                             level_name, exc)
                return model

    try:
        fully_shard(model)
    except Exception as exc:
        _log.warning("FSDP2 hierarchical fully_shard(model) failed: %s — returning unwrapped model", exc)
        return model

    return model
