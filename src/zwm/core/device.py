"""Single source of truth for torch device placement.

All torch modules and tensors in the OODA loop go through
``get_device()`` so the entire model stack can be placed on
CUDA / XPU / MPS / CPU by setting ``ZWM_DEVICE`` (or auto-
detecting the best available accelerator).

Usage::

    from zwm.core.device import get_device, get_dtype

    device = get_device()
    model.to(device)
    x = torch.from_numpy(arr).to(device)
"""
from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    """Return the configured accelerator device.

    Resolution order:
      1. ``ZWM_DEVICE`` env var (e.g. ``cuda:0``, ``cpu``, ``xpu``)
      2. CUDA (if ``torch.cuda.is_available()``)
      3. XPU  (Intel, if available)
      4. MPS  (Apple Silicon)
      5. CPU  (fallback)
    """
    env = os.environ.get("ZWM_DEVICE", "").strip()
    if env:
        return torch.device(env)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype() -> torch.dtype:
    """Return the best AMP dtype for the current accelerator.

    bf16 is preferred on Ampere+ (SM 8.0+) — it has the same range
    as fp32 and avoids the fp16 gradient scaling overhead.
    Falls back to fp32 on older hardware.
    """
    device = get_device()
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "xpu":
        return torch.bfloat16 if torch.xpu.has_fp64_dtype() else torch.float32
    return torch.float32


def autocast_device_type() -> str:
    """Return the device_type string ``torch.autocast`` expects."""
    d = get_device()
    if d.type == "cuda":
        return "cuda"
    if d.type == "xpu":
        return "xpu"
    return "cpu"
