"""Shared test fixtures — determinism, isolation, seed control.

When this module is loaded, every test gets:

  * ``observability.metrics`` reset before/after  (no cross-test accumulation)
  * ``learning.metrics._default_logger`` closed    (no stale file handles)
  * ``ZWM_DATA_DIR``→ temp dir        (no CWD artifact writes)
  * torch / numpy random seeds fixed  (reproducible any order)

Tests that need a persistent *agent* should pass ``db_path=":memory:"``
(most existing tests already do); the conftest guarantees that even
tests that forget still write into the ephemeral temp dir.
"""
from __future__ import annotations

import os
import tempfile
import random as _random

import pytest
import numpy as np
import torch


@pytest.fixture(autouse=True)
def _zwm_data_dir(monkeypatch) -> str:
    """Redirect ZWM data artifacts into a per-process temp directory.

    The same temp dir is used for the entire process so that sequential
    tests that intentionally share a DB file can do so, but every
    ``pytest`` invocation gets a fresh directory.
    """
    data_dir = os.environ.get(
        "ZWM_TEST_DATA",
        os.path.join(tempfile.gettempdir(), f"zwm_test_{_random.randint(0, 999_999)}"),
    )
    os.makedirs(data_dir, exist_ok=True)
    monkeypatch.setenv("ZWM_DATA_DIR", data_dir)
    return data_dir


@pytest.fixture(autouse=True)
def _reset_observability():
    """Reset the process-wide metrics and logging singletons per-test."""
    from zwm.observability import metrics

    metrics.reset()

    import zwm.learning.metrics as lm

    if lm._default_logger is not None:
        try:
            lm._default_logger.close()
        except Exception:
            pass
        lm._default_logger = None

    yield

    metrics.reset()
    if lm._default_logger is not None:
        try:
            lm._default_logger.close()
        except Exception:
            pass
        lm._default_logger = None


@pytest.fixture(autouse=True)
def _fix_seeds():
    """Fix torch and numpy random seeds for deterministic tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
