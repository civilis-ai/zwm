"""M1 — Multi-GPU / FSDP2 integration smoke tests.

These tests don't require a GPU.  They verify:

1. The FSDP2 wrapper is importable and degrades gracefully when
   ``torch.distributed`` is not initialised (returns the model
   unchanged — the standard FSDP2 no-op pattern).
2. The ``_fsdp_available`` probe correctly reports the runtime
   environment.
3. The hierarchical wrapper is a no-op on a non-distributed
   environment.
4. When run under ``torchrun`` (multi-GPU), the wrappers do not
   raise.  This is verified by a separate ``test_fsdp2_multigpu``
   marker; the CI runs the single-GPU smoke test by default.

For real multi-GPU validation use:

    torchrun --nproc_per_node=2 -m pytest \\
        tests/test_m1_distributed_fsdp.py -m fsdp2_multigpu

which only runs when CUDA is available and the user explicitly
opts in.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# A trivial stand-in for JEPAPredictor — the wrapper only inspects
# ``named_children()`` looking for the three named submodules.
# ----------------------------------------------------------------------
class _DummyJEPA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_encoder = nn.Linear(8, 8)
        self.predictor = nn.Linear(8, 8)
        self.target_encoder = nn.Linear(8, 8)
        self.head = nn.Linear(8, 1)

    def forward(self, x):  # pragma: no cover - never called
        return self.head(self.predictor(self.context_encoder(x)))


class _DummyHierarchical(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for level in ("short", "mid", "long"):
            setattr(self, level, _DummyJEPA())


# ----------------------------------------------------------------------
# Single-GPU smoke tests
# ----------------------------------------------------------------------
class TestFSDP2NoOp:
    """FSDP2 wrappers must no-op gracefully when not in a distributed env."""

    def test_wrap_fsdp2_returns_model_unchanged(self) -> None:
        from zwm.jepa.distributed import wrap_fsdp2, _fsdp_available
        m = _DummyJEPA()
        out = wrap_fsdp2(m)
        assert out is m
        # Probe must be bool.
        assert isinstance(_fsdp_available(), bool)

    def test_wrap_fsdp2_hierarchical_returns_model_unchanged(self) -> None:
        from zwm.jepa.distributed import wrap_fsdp2_hierarchical
        m = _DummyHierarchical()
        out = wrap_fsdp2_hierarchical(m)
        assert out is m

    def test_wrap_fsdp2_does_not_raise_on_unknown_submodules(self) -> None:
        """If a module lacks the expected submodules, wrap_fsdp2 should
        still return a model (best-effort sharding)."""
        from zwm.jepa.distributed import wrap_fsdp2
        class _Bare(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)
        m = _Bare()
        out = wrap_fsdp2(m)
        assert out is m

    def test_wrap_fsdp2_does_not_break_parameters(self) -> None:
        """Wrapping a model should not alter its parameters on a non-distributed env."""
        from zwm.jepa.distributed import wrap_fsdp2
        m = _DummyJEPA()
        params_before = sum(p.numel() for p in m.parameters())
        wrap_fsdp2(m)
        params_after = sum(p.numel() for p in m.parameters())
        assert params_before == params_after

    def test_fsdp_available_probe_consistent(self) -> None:
        from zwm.jepa.distributed import _fsdp_available
        # In a normal CI env (no FSDP2 import) this is False; if
        # torch is recent enough and FSDP2 imports cleanly it could
        # be True.  Either way, the probe must agree with the
        # wrapper's no-op behaviour.
        from zwm.jepa.distributed import wrap_fsdp2
        m = _DummyJEPA()
        wrapped = wrap_fsdp2(m)
        if not _fsdp_available():
            assert wrapped is m


# ----------------------------------------------------------------------
# Multi-GPU integration test (only runs under torchrun with CUDA)
# ----------------------------------------------------------------------
@pytest.mark.fsdp2_multigpu
class TestFSDP2MultiGPU:
    """These tests only run on a real multi-GPU host via torchrun.

    Run with::

        torchrun --nproc_per_node=2 -m pytest \\
            tests/test_m1_distributed_fsdp.py -m fsdp2_multigpu

    Each rank validates that wrap_fsdp2 does not raise and that
    parameters are still iterable post-shard.
    """

    def test_wrap_fsdp2_under_torchrun(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for multi-GPU integration test")
        if torch.cuda.device_count() < 2:
            pytest.skip("2+ GPUs required for multi-GPU integration test")
        import torch.distributed as dist
        if not dist.is_initialized():
            pytest.skip("must be run under torchrun")
        from zwm.jepa.distributed import wrap_fsdp2
        m = _DummyJEPA().cuda()
        wrapped = wrap_fsdp2(m)
        # Even under FSDP2, parameters are iterable (just sharded).
        params = list(wrapped.parameters())
        assert len(params) > 0
        dist.barrier()
