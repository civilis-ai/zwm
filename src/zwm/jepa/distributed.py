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

import os
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


def _build_mp_policy(
    mixed_precision: bool | None = None,
) -> object:
    """Build a MixedPrecisionPolicy for FSDP2.

    When ``mixed_precision`` is True (or left as None and the env var
    ``ZWM_FSDP_MIXED_PRECISION`` is set to ``1``), returns a policy that
    uses bf16 for forward computation and fp32 for gradient reduction.
    Otherwise returns the default (no mixed precision).
    """
    if mixed_precision is None:
        mixed_precision = os.environ.get("ZWM_FSDP_MIXED_PRECISION", "0") == "1"

    if not mixed_precision:
        return None  # type: ignore[return-value]

    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=True,
        )
        _log.info("FSDP2 mixed precision enabled: param_dtype=bf16, reduce_dtype=fp32")
        return mp_policy
    except ImportError:
        _log.warning("MixedPrecisionPolicy not available in this PyTorch version; skipping mixed precision")
        return None


def _build_offload_policy(
    cpu_offload: bool | None = None,
) -> object:
    """Build an OffloadPolicy for FSDP2.

    When ``cpu_offload`` is True (or left as None and the env var
    ``ZWM_FSDP_CPU_OFFLOAD`` is set to ``1``), returns a
    ``CPUOffloadPolicy`` that offloads parameters to CPU when not in use.
    Otherwise returns the default (no offload).
    """
    if cpu_offload is None:
        cpu_offload = os.environ.get("ZWM_FSDP_CPU_OFFLOAD", "0") == "1"

    if not cpu_offload:
        return None  # type: ignore[return-value]

    try:
        from torch.distributed.fsdp import CPUOffloadPolicy
        offload_policy = CPUOffloadPolicy(pin_memory=True)
        _log.info("FSDP2 CPU offload enabled (pin_memory=True)")
        return offload_policy
    except ImportError:
        _log.warning("CPUOffloadPolicy not available in this PyTorch version; skipping CPU offload")
        return None


def _apply_activation_checkpointing(
    model: nn.Module,
    activation_checkpointing: bool | None = None,
) -> None:
    """Apply activation checkpointing to eligible submodules.

    Wraps submodules with ``torch.utils.checkpoint.checkpoint`` so that
    their forward activations are not stored — they are recomputed during
    the backward pass, trading compute for memory.

    Controlled by the ``activation_checkpointing`` parameter or the
    ``ZWM_FSDP_ACTIVATION_CHECKPOINTING`` environment variable.

    Only modules whose names are in ``_CHECKPOINT_ELIGIBLE`` are wrapped,
    to avoid checkpointing tiny modules where the overhead would dominate.
    """
    if activation_checkpointing is None:
        activation_checkpointing = os.environ.get("ZWM_FSDP_ACTIVATION_CHECKPOINTING", "0") == "1"

    if not activation_checkpointing:
        return

    _CHECKPOINT_ELIGIBLE = {
        "context_encoder", "predictor", "target_encoder",
        "encoder", "decoder", "transformer",
    }

    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
        )
    except ImportError:
        _log.warning(
            "torch.distributed.algorithms._checkpoint.checkpoint_wrapper "
            "not available; skipping activation checkpointing"
        )
        return

    count = 0
    for name, child in model.named_children():
        if name in _CHECKPOINT_ELIGIBLE:
            wrapped = checkpoint_wrapper(
                child,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            setattr(model, name, wrapped)
            count += 1
            _log.debug("Activation checkpointing applied to %s", name)

    if count > 0:
        _log.info("FSDP2 activation checkpointing enabled on %d submodule(s)", count)


def wrap_fsdp2(
    model: nn.Module,
    *,
    mixed_precision: bool | None = None,
    cpu_offload: bool | None = None,
    activation_checkpointing: bool | None = None,
) -> nn.Module:
    """Wrap a JEPAPredictor for multi-GPU training with FSDP2.

    FSDP2 (``torch.distributed.fsdp.fully_shard``) is the 2026 SOTA
    PyTorch distributed training API.  It shards model parameters across
    GPUs, reducing per-GPU memory by 1/N and enabling training of larger
    models (wider latent_dim, deeper MLPs).

    Optional features (controlled by parameters or environment variables):

    * **Mixed precision**: bf16 forward + fp32 gradient reduction.
      Enable via ``mixed_precision=True`` or ``ZWM_FSDP_MIXED_PRECISION=1``.
    * **CPU offload**: offload sharded parameters to CPU when not in use.
      Enable via ``cpu_offload=True`` or ``ZWM_FSDP_CPU_OFFLOAD=1``.
    * **Activation checkpointing**: recompute forward activations during
      backward to trade compute for memory.
      Enable via ``activation_checkpointing=True`` or
      ``ZWM_FSDP_ACTIVATION_CHECKPOINTING=1``.

    Usage:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        model = JEPAPredictor(input_dim=106)
        model = wrap_fsdp2(model, mixed_precision=True, cpu_offload=True)

    When the distributed package is not available or only a single GPU
    is detected, returns the model unchanged.
    """
    if not _fsdp_available():
        return model
    if _should_skip_sharding():
        return model

    from torch.distributed.fsdp import fully_shard

    mp_policy = _build_mp_policy(mixed_precision)
    offload_policy = _build_offload_policy(cpu_offload)

    # Apply activation checkpointing *before* FSDP sharding so that the
    # checkpoint wrappers are in place when fully_shard traverses the
    # module tree.
    _apply_activation_checkpointing(model, activation_checkpointing)

    # Build kwargs for fully_shard — only pass non-None policies so we
    # remain compatible with older PyTorch versions that lack these
    # parameters.
    def _shard_kwargs() -> dict:
        kwargs: dict = {}
        if mp_policy is not None:
            kwargs["mp_policy"] = mp_policy
        if offload_policy is not None:
            kwargs["offload_policy"] = offload_policy
        return kwargs

    # Apply FSDP2 to the main submodules.
    # fully_shard recursively shards parameters and their gradients.
    for name, child in model.named_children():
        if name in ("context_encoder", "predictor", "target_encoder"):
            try:
                fully_shard(child, **_shard_kwargs())
            except Exception as exc:
                _log.warning("FSDP2 fully_shard(%s) failed: %s — returning unwrapped model", name, exc)
                return model

    try:
        fully_shard(model, **_shard_kwargs())
    except Exception as exc:
        _log.warning("FSDP2 fully_shard(model) failed: %s — returning unwrapped model", exc)
        return model

    return model


def wrap_fsdp2_hierarchical(
    model,
    *,
    mixed_precision: bool | None = None,
    cpu_offload: bool | None = None,
    activation_checkpointing: bool | None = None,
) -> nn.Module:
    """Wrap a HierarchicalJEPAPredictor for multi-GPU FSDP2 training.

    Shards each level (short/mid/long) independently, then shards the
    top-level model.

    Optional features (same as ``wrap_fsdp2``):
    * **Mixed precision**: bf16 forward + fp32 gradient reduction.
    * **CPU offload**: offload sharded parameters to CPU when not in use.
    * **Activation checkpointing**: recompute forward activations during
      backward to trade compute for memory.
    """
    if not _fsdp_available():
        return model
    if _should_skip_sharding():
        return model

    from torch.distributed.fsdp import fully_shard

    mp_policy = _build_mp_policy(mixed_precision)
    offload_policy = _build_offload_policy(cpu_offload)

    # Apply activation checkpointing *before* FSDP sharding.
    _apply_activation_checkpointing(model, activation_checkpointing)

    def _shard_kwargs() -> dict:
        kwargs: dict = {}
        if mp_policy is not None:
            kwargs["mp_policy"] = mp_policy
        if offload_policy is not None:
            kwargs["offload_policy"] = offload_policy
        return kwargs

    for level_name in ("short", "mid", "long"):
        level = getattr(model, level_name, None)
        if level is not None:
            for name, child in level.named_children():
                if name in ("context_encoder", "predictor", "target_encoder"):
                    try:
                        fully_shard(child, **_shard_kwargs())
                    except Exception as exc:
                        _log.warning("FSDP2 hierarchical fully_shard(%s.%s) failed: %s — returning unwrapped model",
                                     level_name, name, exc)
                        return model
            try:
                fully_shard(level, **_shard_kwargs())
            except Exception as exc:
                _log.warning("FSDP2 hierarchical fully_shard(%s) failed: %s — returning unwrapped model",
                             level_name, exc)
                return model

    try:
        fully_shard(model, **_shard_kwargs())
    except Exception as exc:
        _log.warning("FSDP2 hierarchical fully_shard(model) failed: %s — returning unwrapped model", exc)
        return model

    return model
