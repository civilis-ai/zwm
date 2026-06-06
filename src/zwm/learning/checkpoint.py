"""P1-5: MCTS / agent state checkpoint + restore.

Persists the cross-tick learning state (visit counts, palace visits,
Hebbian associations, preference weights, V-table, JEPA parameters)
to a portable JSON + torch state-dict file. Restore is symmetrical.

The 2026 MLOps default — without it, every restart loses 100% of the
learning progress, which is unacceptable for any non-trivial world model.

M3 — Checkpoint schema versioning
--------------------------------
Each checkpoint carries a ``zwm_checkpoint_version`` field.  Loading a
checkpoint whose version is *newer* than the running framework raises
:class:`IncompatibleCheckpointError`; older versions are migrated
forward (``v1`` → ``v2`` → ``v3``).  This lets the user load historic
checkpoints after a forward-only schema change without losing progress.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

# M3: bump this whenever the on-disk JSON schema changes in a
# backwards-incompatible way (renamed/removed keys, new mandatory
# fields, etc.).  ``save_checkpoint`` always writes the latest version;
# ``load_checkpoint`` understands every version <= CURRENT_VERSION.
CURRENT_CHECKPOINT_VERSION = 3
# Field name used in the JSON header.
_VERSION_KEY = "zwm_checkpoint_version"
# App version captured at write time, useful for cross-referencing
# changelog entries.
_APP_VERSION_KEY = "zwm_app_version"


class IncompatibleCheckpointError(Exception):
    """Raised when a checkpoint was written by a *newer* version of ZWM
    than the one running."""

    def __init__(self, found_version: int, max_supported: int, path: str):
        self.found_version = found_version
        self.max_supported = max_supported
        self.path = path
        super().__init__(
            f"checkpoint at {path!r} has version {found_version} which is "
            f"newer than this build's max supported {max_supported}; "
            f"please upgrade zwm"
        )


def _to_jsonable(obj):
    """Recursively coerce numpy / dataclass objects into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def save_checkpoint(
    agent,
    path: str | os.PathLike,
) -> str:
    """Save the agent's full state to ``path`` (``.json`` + ``.pt``).

    Returns the path of the JSON file as a string for convenience.
    The torch state-dict (JEPA + square encoder + router) is stored at
    ``<path>.pt`` next to the JSON file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Collect serialisable state
    state: dict[str, Any] = {
        "version": 2,  # legacy field, kept for old loaders
        # M3 — schema versioning
        _VERSION_KEY: CURRENT_CHECKPOINT_VERSION,
        _APP_VERSION_KEY: _get_app_version(),
        "agent_class": type(agent).__name__,
        "step_count": agent._step_count,
        "palace_visits": dict(agent._palace_visits),
        # Hebbian
        "hebbian": {
            "associations": dict(agent.hebbian.associations),
            "pre_count": {str(k): v for k, v in agent.hebbian._pre_count.items()},
            "post_count": {str(k): v for k, v in agent.hebbian._post_count.items()},
            "learning_rate": agent.hebbian.learning_rate,
            "oja_w_max": agent.hebbian.oja_w_max,
            "forget": agent.hebbian.forget,
        },
        # Online learner
        "learner": {
            "preference_weights": dict(agent.learner.preference_weights),
            "visit_counts": {str(k): v for k, v in agent.learner.visit_counts.items()},
            "value_table": {str(k): v for k, v in agent.learner._value_table.items()},
            "total_visits": agent.learner.total_visits,
            "learning_rate": agent.learner.learning_rate,
            "gae_lambda": agent.learner.gae_lambda,
            "gamma": agent.learner.gamma,
        },
        # Curiosity / growth
        "curiosity": {
            "step_count": agent.curiosity.step_count,
            "beta_initial": agent.curiosity.beta_initial,
        },
        "growth": {"total_episodes": agent.growth.total_episodes},
        # ReAct state
        "react": {
            "enabled": agent._react_loop is not None,
            "max_steps": agent._react_loop.max_steps if agent._react_loop else 3,
        },
        # EWC state (anti-catastrophic-forgetting)
        "ewc": agent._ewc.state_dict() if agent._ewc is not None else None,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(state), f, ensure_ascii=False, indent=2)
    # Torch state
    pt_path = Path(str(p) + ".pt")
    try:
        import torch
        torch_state = {
            "jepa_context": agent.jepa.context_encoder.state_dict(),
            "jepa_predictor": agent.jepa.predictor.state_dict(),
            "jepa_target": agent.jepa.target_encoder.state_dict(),
            "router": agent.planner._moe.router.state_dict(),
        }
        if agent._square_learnable is not None:
            torch_state["square_learnable"] = (
                agent._square_learnable.state_dict()
            )
        # LoRA adapters (if quantized)
        if hasattr(agent.jepa, "is_quantized") and agent.jepa.is_quantized:
            lora_state = {}
            for name, mod in agent.jepa.named_modules():
                if hasattr(mod, "lora_a"):  # _LoRALinear
                    lora_state[name + ".lora_a"] = mod.lora_a.state_dict()
                    lora_state[name + ".lora_b"] = mod.lora_b.state_dict()
            if lora_state:
                torch_state["lora"] = lora_state
        # VQ codebook
        if hasattr(agent.jepa, "_vq") and agent.jepa._vq is not None:
            torch_state["vq"] = agent.jepa._vq.state_dict()
        # Value head
        if hasattr(agent.jepa, "_value_head") and agent.jepa._value_head is not None:
            torch_state["value_head"] = agent.jepa._value_head.state_dict()
        # Multimodal encoder fusion weights
        if hasattr(agent, "_mm_encoder") and agent._mm_encoder is not None:
            torch_state["mm_fusion"] = agent._mm_encoder.state_dict()
        torch.save(torch_state, pt_path)
    except Exception as exc:
        _log.warning("Checkpoint torch.save failed: %s — structural state saved but neural state not persisted", exc)
    return str(p)


def _get_app_version() -> str:
    """Read the running ZWM version (best-effort)."""
    try:
        from zwm import __version__
        return str(__version__)
    except Exception:
        return "unknown"


# M3 — migration table.  Each entry says "from version → to version",
# paired with a function that mutates the loaded state dict in place.
# Versions must form a forward-only chain: 1 → 2 → 3 → ...
_MIGRATIONS: list[tuple[int, int, Any]] = [
    # (from_v, to_v, mutate(state) -> None)
    # v1 had no "growth.total_episodes" key — set to 0 in v2.
    (1, 2, lambda s: s.setdefault("growth", {}).setdefault("total_episodes", 0)),
    # v2 had no schema-version header — backfill it from "version".
    (2, 3, lambda s: s.setdefault(_VERSION_KEY, s.get("version", 2))),
]


def _migrate(state: dict) -> int:
    """Apply forward migrations in order. Returns the final version."""
    cur = int(state.get(_VERSION_KEY, state.get("version", 1)))
    for from_v, to_v, fn in _MIGRATIONS:
        while cur == from_v:
            fn(state)
            cur = to_v
            state[_VERSION_KEY] = cur
    return cur


class _DimensionalDriftError(Exception):
    """Raised when a checkpoint's tensor shapes don't match the current
    model's expected dimensions (e.g. loading a ``test``-preset checkpoint
    into a ``large``-preset agent)."""

    def __init__(self, path: str, mismatches: list[str]) -> None:
        self.mismatches = mismatches
        super().__init__(
            f"Dimension drift detected in checkpoint {path!r}:\n  "
            + "\n  ".join(mismatches)
            + "\n\nChange the ZWM_SIZE_PRESET env var to match the checkpoint's "
            + "training configuration."
        )


def _validate_dimensional_compatibility(
    agent,
    torch_state: dict,
    pt_path: Path,
) -> None:
    """Check that the checkpoint's tensor shapes match the current model.

    Catches silent dimension mismatches where a checkpoint trained with
    SIZE_PRESET=test (LATENT_DIM=64) is loaded into a SIZE_PRESET=large
    (LATENT_DIM=256) agent.  Without this check, ``load_state_dict``
    would either raise an opaque error or, worse, silently skip keys.
    """
    import torch
    mismatches: list[str] = []

    # Map state-dict keys to the agent's live modules for shape comparison.
    _checks: list[tuple[str, object, str | None]] = [
        ("jepa_context", agent.jepa.context_encoder, "context_encoder"),
        ("jepa_predictor", agent.jepa.predictor, "predictor"),
        ("jepa_target", agent.jepa.target_encoder, "target_encoder"),
    ]
    if hasattr(agent, "_square_learnable") and agent._square_learnable is not None:
        _checks.append(("square_learnable", agent._square_learnable, "square_encoder"))
    if hasattr(agent, "_mm_encoder") and agent._mm_encoder is not None:
        _checks.append(("mm_fusion", agent._mm_encoder, "multimodal_encoder"))

    for key, live_module, _label in _checks:
        if key not in torch_state:
            continue
        ckpt_sd = torch_state[key]
        live_sd = live_module.state_dict()
        for pname, ckpt_param in ckpt_sd.items():
            live_param = live_sd.get(pname)
            if live_param is None:
                mismatches.append(
                    f"  {key}.{pname}: exists in checkpoint but not in live model"
                )
            elif ckpt_param.shape != live_param.shape:
                mismatches.append(
                    f"  {key}.{pname}: checkpoint {list(ckpt_param.shape)} "
                    f"≠ live {list(live_param.shape)}"
                )

    # VQ codebook dimension check.
    if "vq" in torch_state and hasattr(agent.jepa, "_vq") and agent.jepa._vq is not None:
        vq_sd = torch_state["vq"]
        live_vq_sd = agent.jepa._vq.state_dict()
        for pname, ckpt_param in vq_sd.items():
            live_param = live_vq_sd.get(pname)
            if live_param is not None and ckpt_param.shape != live_param.shape:
                mismatches.append(
                    f"  vq.{pname}: checkpoint {list(ckpt_param.shape)} "
                    f"≠ live {list(live_param.shape)}"
                )

    if mismatches:
        raise _DimensionalDriftError(path=str(pt_path), mismatches=mismatches)


def load_checkpoint(agent, path: str | os.PathLike) -> None:
    """Restore the agent's state from ``path``.

    P1-dim-drift: validates tensor shape compatibility between the
    checkpoint and the live model before loading.  Prevents silent
    dimension corruption when SIZE_PRESET changes between save and load.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        state = json.load(f)
    # M3 — schema version check + migration.
    raw_v = state.get(_VERSION_KEY, state.get("version", 1))
    try:
        found_v = int(raw_v)
    except (TypeError, ValueError):
        found_v = 1
    if found_v > CURRENT_CHECKPOINT_VERSION:
        raise IncompatibleCheckpointError(
            found_version=found_v,
            max_supported=CURRENT_CHECKPOINT_VERSION,
            path=str(p),
        )
    final_v = _migrate(state)
    if final_v != CURRENT_CHECKPOINT_VERSION:
        _log.warning(
            "checkpoint at %s: partial migration v%d → v%d; some fields may be missing",
            p, found_v, final_v,
        )
    agent._step_count = int(state.get("step_count", 0))
    agent._palace_visits = {int(k): int(v) for k, v in state.get("palace_visits", {}).items()}
    h_state = state.get("hebbian", {})
    agent.hebbian.associations = dict(h_state.get("associations", {}))
    agent.hebbian._pre_count = {int(k): int(v) for k, v in h_state.get("pre_count", {}).items()}
    agent.hebbian._post_count = {int(k): int(v) for k, v in h_state.get("post_count", {}).items()}
    agent.hebbian.learning_rate = float(h_state.get("learning_rate", agent.hebbian.learning_rate))
    agent.hebbian.oja_w_max = float(h_state.get("oja_w_max", agent.hebbian.oja_w_max))
    agent.hebbian.forget = float(h_state.get("forget", agent.hebbian.forget))
    l_state = state.get("learner", {})
    agent.learner.preference_weights = dict(l_state.get("preference_weights", {}))
    agent.learner.visit_counts = {int(k): int(v) for k, v in l_state.get("visit_counts", {}).items()}
    agent.learner._value_table = {int(k): float(v) for k, v in l_state.get("value_table", {}).items()}
    agent.learner.total_visits = int(l_state.get("total_visits", 0))
    agent.learner.learning_rate = float(l_state.get("learning_rate", agent.learner.learning_rate))
    agent.learner.gae_lambda = float(l_state.get("gae_lambda", agent.learner.gae_lambda))
    agent.learner.gamma = float(l_state.get("gamma", agent.learner.gamma))
    c_state = state.get("curiosity", {})
    agent.curiosity.step_count = int(c_state.get("step_count", 0))
    agent.curiosity.beta_initial = float(c_state.get("beta_initial", agent.curiosity.beta_initial))
    g_state = state.get("growth", {})
    agent.growth.total_episodes = int(g_state.get("total_episodes", 0))
    # Torch state-dict
    pt_path = Path(str(p) + ".pt")
    if pt_path.exists():
        try:
            import torch
            torch_state = torch.load(pt_path, map_location="cpu")
            # P1-dim-drift: validate tensor shapes before loading to
            # prevent silent corruption when SIZE_PRESET mismatches.
            _validate_dimensional_compatibility(agent, torch_state, pt_path)
            if "jepa_context" in torch_state:
                agent.jepa.context_encoder.load_state_dict(torch_state["jepa_context"])
            if "jepa_predictor" in torch_state:
                agent.jepa.predictor.load_state_dict(torch_state["jepa_predictor"])
            if "jepa_target" in torch_state:
                agent.jepa.target_encoder.load_state_dict(torch_state["jepa_target"])
            if "router" in torch_state:
                agent.planner._moe.router.load_state_dict(torch_state["router"])
            if "square_learnable" in torch_state and agent._square_learnable is not None:
                agent._square_learnable.load_state_dict(torch_state["square_learnable"])
            # LoRA adapters
            if "lora" in torch_state:
                for name, mod in agent.jepa.named_modules():
                    if hasattr(mod, "lora_a") and (name + ".lora_a") in torch_state["lora"]:
                        mod.lora_a.load_state_dict(torch_state["lora"][name + ".lora_a"])
                        mod.lora_b.load_state_dict(torch_state["lora"][name + ".lora_b"])
            # VQ codebook
            if "vq" in torch_state and hasattr(agent.jepa, "_vq") and agent.jepa._vq is not None:
                agent.jepa._vq.load_state_dict(torch_state["vq"])
            # Value head
            if "value_head" in torch_state and hasattr(agent.jepa, "_value_head") and agent.jepa._value_head is not None:
                agent.jepa._value_head.load_state_dict(torch_state["value_head"])
            # Multimodal fusion weights
            if "mm_fusion" in torch_state and hasattr(agent, "_mm_encoder") and agent._mm_encoder is not None:
                agent._mm_encoder.load_state_dict(torch_state["mm_fusion"])
        except Exception as exc:
            _log.warning("Checkpoint torch.load failed: %s — neural state not restored, structural state is intact", exc)
        # EWC state (JSON, not torch — restore even if torch load failed)
        ewc_state = state.get("ewc")
        if ewc_state is not None and agent._ewc is not None:
            try:
                agent._ewc.load_state_dict(ewc_state)
            except Exception as exc:
                _log.warning("EWC state restoration failed: %s", exc)
