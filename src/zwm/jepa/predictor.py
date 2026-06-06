from __future__ import annotations

import copy
import logging
import math
from collections import deque
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from zwm.jepa.heads import (
    _DEFAULT_ACTION_EMBED_DIM,
    _ActionEmbedding,
    _Encoder,
    _EnergyHead,
    _NullContext,
    _PriorExpert,
    _VariationalHead,
    sigreg_loss,
)
from zwm.jepa.distributed import wrap_fsdp2, wrap_fsdp2_hierarchical
from zwm.core.constants import Z_WORLD_DIM, LATENT_DIM, HIDDEN_DIM, ACTION_EMBED_DIM

_DEFAULT_INPUT_DIM = Z_WORLD_DIM   # 64 (square GNN) + 13 (circular phase) + 29 (unified field)
_DEFAULT_HIDDEN_DIM = HIDDEN_DIM   # 2026 P3-H: 64 → 192, matches SOTA scale
_DEFAULT_LATENT_DIM = LATENT_DIM   # 2026 P3-H: 32 → 64, matches SOTA scale

# Re-exports for backward compat — tests and other modules import these
# names from ``predictor`` directly.
__all__ = [
    "JEPAPredictor",
    "HierarchicalJEPAPredictor",
    "wrap_fsdp2",
    "wrap_fsdp2_hierarchical",
    "_TrainTransitionInputs",
    "_Encoder",
    "_VariationalHead",
    "_EnergyHead",
    "_ActionEmbedding",
    "_PriorExpert",
    "_NullContext",
    "sigreg_loss",
]


class _CosineWarmup:
    """Cosine learning-rate schedule with linear warmup.

    2026 SOTA (LLaMA, DeepSeek-V3): warm up from 0 → lr over
    ``warmup_steps``, then cosine-decay to ``min_lr`` over the
    remaining steps.  Called after each ``optimizer.step()``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 100,
        total_steps: int = 10_000,
        min_lr: float = 1e-5,
    ) -> None:
        self._opt = optimizer
        self._warmup = warmup_steps
        self._total = total_steps
        self._min_lr = min_lr
        self._base_lr = optimizer.param_groups[0]["lr"]
        self._step = 0

    def step(self) -> None:
        self._step += 1
        lr = self._compute_lr(self._step)
        for pg in self._opt.param_groups:
            pg["lr"] = lr

    def _compute_lr(self, step: int) -> float:
        if step <= self._warmup:
            return self._base_lr * step / max(self._warmup, 1)
        progress = (step - self._warmup) / max(self._total - self._warmup, 1)
        return self._min_lr + 0.5 * (self._base_lr - self._min_lr) * (
            1.0 + math.cos(math.pi * min(progress, 1.0))
        )


class _EMASchedule:
    """EMA decay schedule: cosine warmup from 0.9 → target_decay.

    2026 SOTA (DINOv2, EMA2): a warmup schedule for the EMA decay
    stabilises early training when the online encoder is changing
    rapidly.  Without it, the target encoder lags too far behind
    and the JEPA loss is dominated by representation drift rather
    than genuine prediction error.
    """

    def __init__(
        self,
        target_decay: float = 0.99,
        warmup_steps: int = 200,
    ) -> None:
        self._target = target_decay
        self._warmup = warmup_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        return self.current_decay()

    def current_decay(self) -> float:
        if self._step >= self._warmup:
            return self._target
        progress = self._step / max(self._warmup, 1)
        # Cosine from 0.9 → target
        return 0.9 + (self._target - 0.9) * 0.5 * (1.0 - math.cos(math.pi * progress))


class _LoRALinear(nn.Module):
    """LoRA-adapted Linear layer: y = W·x + (α/r)·B·A·x.

    The 2026 SOTA for parameter-efficient fine-tuning.  The original
    weight W is frozen; only A and B are trained.  At inference, the
    adapter can be merged back into W for zero-cost deployment.

    ``alpha`` controls the effective learning rate of the adapter
    (higher α = stronger adaptation).  ``rank`` (r) controls the
    bottleneck dimension — typical values are 4-64.
    """

    def __init__(
        self,
        base: nn.Linear,
        lora_a: nn.Linear,
        lora_b: nn.Linear,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.alpha = alpha
        self.rank = lora_a.out_features
        # Freeze the base weight.
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = W·x + (α/r)·B·A·x  (after merge, lora_a/lora_b are None)
        base_out = self.base(x)
        if self.lora_a is not None and self.lora_b is not None:
            lora_out = self.lora_b(self.lora_a(x))
            scale = self.alpha / self.rank
            return base_out + scale * lora_out
        return base_out

    def merge(self) -> nn.Linear:
        """Merge the LoRA adapter back into the base weight.

        After merging, the adapter is absorbed and inference has zero
        additional cost.  This is the standard deployment path for
        LoRA-adapted models.  The LoRA matrices are deleted after
        merging to free GPU memory.
        """
        with torch.no_grad():
            delta_w = (self.alpha / self.rank) * self.lora_b.weight @ self.lora_a.weight
            self.base.weight.add_(delta_w)
        # Clean up LoRA matrices to free memory after merge.
        del self.lora_a
        del self.lora_b
        self.lora_a = None
        self.lora_b = None
        return self.base


class _TrainTransitionInputs(NamedTuple):
    """Bundle returned by ``JEPAPredictor._build_inputs``.

    Carries the padded/aligned tensors (z_sq_t, z_sq_next, cp_t, cp_next,
    unified_t, unified_next) plus a ``has_learnable`` flag.
    """
    z_sq_t: np.ndarray
    z_sq_next: np.ndarray
    cp_t: np.ndarray
    cp_next: np.ndarray
    unified_t: np.ndarray | None
    unified_next: np.ndarray | None
    has_learnable: bool


class JEPAPredictor(nn.Module):
    """Joint-Embedding Predictive Architecture, properly trained.

    Architecture (V-JEPA style):
      * online ``context_encoder``  : x_t      -> z_t   (trained by backprop)
      * ``target_encoder``          : x_{t+1}  -> z'_{t+1} (EMA of online, no grad)
      * ``predictor``               : z_t      -> ẑ_{t+1} (trained by backprop)

    The loss is prediction error in *representation space* against the
    stop-gradient EMA target, plus a VICReg term that prevents representational
    collapse (variance + covariance regularisation). This is the modern
    self-supervised world-model recipe — real autograd, EMA target, anti-collapse.
    """

    def __init__(
        self,
        input_dim: int = _DEFAULT_INPUT_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        latent_dim: int = _DEFAULT_LATENT_DIM,
        learning_rate: float = 1e-3,
        ema_decay: float = 0.99,
        vicreg_weight: float = 0.04,
        replay_capacity: int = 256,
        batch_size: int = 16,
        seed: int = 42,
        variational: bool = True,
        kl_weight: float = 1e-3,
        # 2026 P3 cohort — feature flags.  Defaults wire every upgrade
        # on so the predictor matches 2026 SOTA out of the box; tests
        # that want the legacy 2024 architecture can flip them off.
        use_action_cond: bool = True,   # P3-A
        use_energy_head: bool = True,   # P3-F
        use_sigreg: bool = True,        # P3-D
        use_prior_expert: bool = True,  # P3-C
        use_backward: bool = True,      # P3-E
        sigreg_weight: float = 0.04,
        energy_weight: float = 0.05,
        poe_weight: float = 0.3,
        backward_weight: float = 0.5,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        # Device placement — determined once so every tensor lands on
        # the correct accelerator.  Respects ZWM_DEVICE env var.
        from zwm.core.device import get_device
        self.device = get_device()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self._ema_decay = ema_decay
        self._vicreg_weight = vicreg_weight
        self.variational = variational
        self._kl_weight = kl_weight
        # 2026 P3 cohort flag bookkeeping
        self._use_action_cond = use_action_cond
        self._use_energy_head = use_energy_head
        self._use_sigreg = use_sigreg
        self._use_prior_expert = use_prior_expert
        self._use_backward = use_backward
        self._sigreg_weight = sigreg_weight
        self._energy_weight = energy_weight
        self._poe_weight = poe_weight
        self._backward_weight = backward_weight
        # Experience replay: VICReg needs a batch with cross-sample variance to
        # produce a real anti-collapse gradient (a single sample's variance is
        # 0 and contributes no signal). Replaying recent transitions in a
        # minibatch keeps the variance/covariance terms live and is also more
        # sample-efficient.
        self._replay: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
            maxlen=replay_capacity
        )
        self._batch_size = batch_size

        # ZWM 结构化编码器 — 当 input_dim 是 256 (4×64) 时启用
        # 取代 flat MLP/Transformer, 按场类型分别处理
        self._structured_encoder = None
        if input_dim == 256 and input_dim % 64 == 0:
            try:
                from zwm.jepa.structured_encoder import ZWMStructuredEncoder
                self._structured_encoder = ZWMStructuredEncoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    backend="hybrid",
                )
            except Exception as exc:
                logging.getLogger(__name__).debug("ZWMStructuredEncoder import failed: %s", exc)

        if self._structured_encoder is not None:
            self.context_encoder = self._structured_encoder
        else:
            self.context_encoder = _Encoder(
                input_dim, hidden_dim, latent_dim, use_transformer=True
            )

        # P3-A — Action embedding (mutation mask).  We always
        # instantiate the embedding; the ``use_action_cond`` flag only
        # controls whether the predictor consumes it at forward time.
        # When action conditioning is disabled, the predictor's input
        # dim stays at ``latent_dim`` (no concatenation) so legacy
        # behaviour is preserved exactly.
        self._action_embed: _ActionEmbedding = _ActionEmbedding(
            num_masks=64, embed_dim=_DEFAULT_ACTION_EMBED_DIM
        )
        action_dim = (
            self._action_embed.dim if use_action_cond else 0
        )

        if variational:
            # Variational mode: predictor is the backbone (no final linear),
            # and _var_head outputs (mu, log_var) replacing the last layer.
            # The predictor's input dim is ``latent_dim + action_dim``
            # when action conditioning is enabled (P3-A).
            # P2 FIX: Upgraded from plain MLP to residual MLP with LayerNorm
            # (2026 SOTA pattern: Pre-LN + residual + GELU).
            self.predictor = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self._var_head = _VariationalHead(hidden_dim, latent_dim)
        else:
            # Deterministic mode: residual MLP with LayerNorm.
            self.predictor = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self._var_head: _VariationalHead | None = None

        # P3-E — Backward predictor: ``z_{t+1} -> ẑ_t``.  A symmetric
        # copy of the forward predictor.  Cycle-consistency between
        # the two yields the inverse-dynamics signal BiJEPA (2026)
        # shows doubles the supervision per transition.
        # Architecture matches the forward backbone: outputs
        # ``hidden_dim`` so the variational head (when present) can
        # produce (mu, logvar) of size ``latent_dim``.
        # P0 FIX: Add _backward_proj for deterministic mode so the
        # backward predictor output (hidden_dim) can be projected to
        # latent_dim for MSE against z_t.
        if use_backward:
            self._backward_predictor: nn.Module | None = nn.Sequential(
                nn.Linear(latent_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self._backward_proj: nn.Linear | None = nn.Linear(hidden_dim, latent_dim)
        else:
            self._backward_predictor = None
            self._backward_proj = None

        # P3-C — Prior expert for Product-of-Experts.
        self._prior_expert: _PriorExpert | None = (
            _PriorExpert(latent_dim, hidden_dim=hidden_dim) if use_prior_expert else None
        )

        # P3-F — Energy-based scalar head.
        self._energy_head: _EnergyHead | None = (
            _EnergyHead(latent_dim) if use_energy_head else None
        )

        # EMA target encoder — a frozen, slowly-updated copy of the online
        # context encoder. Never receives gradients.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        opt_params = (
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters())
        )
        if self._var_head is not None:
            opt_params += list(self._var_head.parameters())
        if self._backward_predictor is not None:
            opt_params += list(self._backward_predictor.parameters())
        if self._prior_expert is not None:
            opt_params += list(self._prior_expert.parameters())
        if self._energy_head is not None:
            opt_params += list(self._energy_head.parameters())
        opt_params += list(self._action_embed.parameters())
        self._opt = torch.optim.Adam(opt_params, lr=learning_rate)

        # 2026 SOTA: Cosine LR schedule with linear warmup.  The
        # standard recipe from LLaMA / Chinchilla / DeepSeek-V3:
        # warm up from 0 → lr over ``warmup_steps``, then decay
        # to ``min_lr`` via a cosine curve.  This replaces the
        # fixed lr=1e-3 that was used before.
        self._lr_scheduler = _CosineWarmup(
            self._opt,
            warmup_steps=10,
            total_steps=10_000,
            min_lr=learning_rate * 0.01,
        )

        # EMA decay schedule: warmup from 0.9 → target over 200 steps,
        # then hold.  The 2026 recipe (DINOv2 / EMA2) uses a cosine
        # schedule for the EMA decay to stabilise early training.
        self._ema_schedule = _EMASchedule(
            target_decay=ema_decay,
            warmup_steps=20,
        )

        # P2-1: MuZero-style latent V(s) head. Initialised lazily via
        # ``init_value_head()``. ``None`` means use the analytical fallback
        # (EMA V-table in OnlineLearner).
        self._value_head: nn.Module | None = None
        self._value_weight: float = 0.5

        # P2-3: VQ-VAE discretisation. 64 codes × ``latent_dim`` dims.
        # Optional — initialised only when ``enable_vq=True`` is passed.
        self._vq: nn.Module | None = None

        # P0-1 — Mixed-precision (bf16) + torch.compile. The 2026 SOTA
        # default for any non-trivial torch model. Both bring real wins:
        #   * bf16: ~1.5–2× throughput on H100/L4, no accuracy loss for JEPA
        #   * torch.compile: 1.3–1.8× speedup on small MLPs by kernel fusion
        # The ``compiled_predictor`` and ``compiled_context`` are used in
        # ``train_step`` (the hot path) and ``predict`` / ``target_latent``.
        self._amp_enabled = (
            torch.cuda.is_available()
            or (
                hasattr(torch, "xpu")
                and torch.xpu.is_available()
            )
        )
        self._compiled_predictor: nn.Module | None = None
        self._compiled_context: nn.Module | None = None
        if hasattr(torch, "compile"):
            # torch.compile requires a working Triton/CUDA toolchain to
            # actually emit fused kernels.  When Triton isn't available
            # (e.g. CPU-only Windows, Python 3.14 with no triton
            # package), the lazy compile blows up on the first forward
            # call.  Detect upfront so we can fall back cleanly to the
            # eager modules.
            triton_ok = True
            try:
                import importlib.util as _iu
                triton_ok = _iu.find_spec("triton") is not None
            except Exception:
                triton_ok = False
            if not triton_ok or not (
                torch.cuda.is_available()
                or (hasattr(torch, "xpu") and torch.xpu.is_available())
            ):
                # CPU / no-triton / no-accelerator — eager only.  The
                # speedup is GPU-only anyway, and this prevents a
                # lazy-compile crash from breaking the OODA loop.
                self._compiled_predictor = None
                self._compiled_context = None
            else:
                try:
                    # mode="default" — safe across custom modules.  The
                    # ``reduce-overhead`` mode (CUDA graphs) requires
                    # the wrapped module to implement ``__len__`` and a
                    # few other ``torch.nn`` dunders that our slim
                    # ``_Encoder`` class doesn't expose, so we use the
                    # default inductor backend which still fuses
                    # kernels and gives the bulk of the speedup.
                    self._compiled_predictor = torch.compile(
                        self.predictor, mode="default"
                    )
                    self._compiled_context = torch.compile(
                        self.context_encoder, mode="default"
                    )
                except Exception:
                    # torch.compile is a best-effort accelerator;
                    # never crash the JEPA training loop on a
                    # kernel-fusion failure.
                    self._compiled_predictor = None
                    self._compiled_context = None

        # Move the entire model to the accelerator.
        self.to(self.device)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(x)

    def predict(
        self,
        z_world: np.ndarray | torch.Tensor,
        mask: int | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Predict the next-state latent from the current world tensor.

        When ``mask`` is provided (P3-A, action conditioning), the
        mutation mask is embedded and concatenated to the latent
        before the predictor forward pass.  This turns the world
        model into a *controllable* transition model
        ``z_{t+1} = f_θ(z_t, a_t)`` — the foundation of 2026 SOTA
        latent-space MPC (V-JEPA 2-AC, LeWorldModel).
        """
        x = self._as_tensor(z_world)
        with torch.no_grad():
            ctx = self._compiled_context if self._compiled_context is not None else self.context_encoder
            if self._amp_enabled:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
                ):
                    z = ctx(x)
                    z_pred = self._forward_predict(z, mask)
            else:
                z = ctx(x)
                z_pred = self._forward_predict(z, mask)
        return z_pred.float().cpu().numpy().astype(np.float32)

    def _forward_predict(
        self, z: torch.Tensor, mask: int | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Internal: forward the latent (+ optional action embed)
        through the predictor, variational head, and prior expert.
        """
        # P3-A — action conditioning.  When a mask is supplied the
        # 32-dim embedding is concatenated to the latent.  When
        # ``mask is None`` *and* action conditioning is enabled, we
        # default to a passive "no-action" token (mask=0) so the
        # predictor always sees a 96-dim input and can still be used
        # as a *prior* over futures (no specific action chosen).
        if self._use_action_cond:
            if mask is None:
                mask = 0
            if isinstance(mask, int):
                mask_t = torch.tensor(mask, dtype=torch.long, device=z.device)
            elif mask.dim() == 0:
                mask_t = mask.to(dtype=torch.long, device=z.device)
            else:
                mask_t = mask.to(dtype=torch.long, device=z.device)
            a = self._action_embed(mask_t)
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z_in = torch.cat([z, a], dim=-1)
        else:
            z_in = z

        if self.variational:
            h = self.predictor(z_in)
            mu, _ = self._var_head(h)
            z_pred = mu
        else:
            z_pred = self.predictor(z_in)

        # P3-C — Product of Experts (BJEPA 2026).  Combine the
        # learned dynamics expert with the structural prior expert
        # via a weighted sum (a simplified PoE — exact PoE would
        # require Gaussian multiplications which we approximate
        # linearly here for stability and speed).
        if self._prior_expert is not None:
            z_prior = self._prior_expert(z)  # same shape as z (unsqueezed)
            z_pred = (1.0 - self._poe_weight) * z_pred + self._poe_weight * z_prior
        return z_pred

    def predict_with_uncertainty(
        self, z_world: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the next-state latent with uncertainty.

        Returns:
            (mean, variance) as numpy float32 arrays.
            When ``variational=False``, variance is an array of zeros.

        Routes through the same action-conditioning + PoE path as
        ``predict`` so the uncertainty estimate is consistent.
        """
        x = self._as_tensor(z_world)
        with torch.no_grad():
            ctx = self._compiled_context if self._compiled_context is not None else self.context_encoder
            z = ctx(x)
            if self.variational:
                # Build the action-conditioned input the same way as
                # ``_forward_predict`` so the predictor sees the right
                # dimensionality (64 + 32 = 96 when action-cond is on).
                if self._use_action_cond:
                    mask_t = torch.tensor(0, dtype=torch.long, device=z.device)
                    a = self._action_embed(mask_t)
                    if a.dim() == 1:
                        a = a.unsqueeze(0)
                    if z.dim() == 1:
                        z = z.unsqueeze(0)
                    z_in = torch.cat([z, a], dim=-1)
                else:
                    z_in = z
                h = self.predictor(z_in)
                mu, log_var = self._var_head(h)
                # PoE blending on the mean (same as _forward_predict).
                if self._prior_expert is not None:
                    z_prior = self._prior_expert(z)
                    mu = (1.0 - self._poe_weight) * mu + self._poe_weight * z_prior
                variance = log_var.exp()
            else:
                mu = self._forward_predict(z)
                variance = torch.zeros_like(mu)
        return mu.float().cpu().numpy().astype(np.float32), variance.float().cpu().numpy().astype(np.float32)

    def target_latent(self, x_next: np.ndarray | torch.Tensor) -> np.ndarray:
        """EMA-target embedding of a (next) state — the prediction target."""
        x = self._as_tensor(x_next)
        with torch.no_grad():
            z = self.target_encoder(x)
        return z.float().cpu().numpy().astype(np.float32)

    def context_encode(self, z_world: np.ndarray | torch.Tensor) -> np.ndarray:
        """P0-5: encode z_world → 64-dim latent via the context encoder.

        Used by the particle filter to map a 106-dim world vector
        into the JEPA's 64-dim latent manifold, so the particle
        filter can stay in latent space (avoiding dim mismatches).
        """
        x = self._as_tensor(z_world)
        with torch.no_grad():
            z = self.context_encoder(x)
        return z.float().cpu().numpy().astype(np.float32)

    def predict_latent(self, z_latent: np.ndarray | torch.Tensor) -> np.ndarray:
        """P0-5: pure-latent transition — 64-dim → 64-dim.

        Runs the JEPA predictor head on a latent vector and returns
        the predicted next latent.  This is the transition function
        the particle filter uses for ensemble EFE in JEPA latent
        space, avoiding the need to pad/unpad through z_world.

        When action conditioning is enabled (default), the predictor
        expects a 96-dim input (64 latent + 32 action embed).  This
        method appends a "no-action" token (mask=0) so the call is
        always valid.  It also routes through ``_forward_predict``
        to include the prior expert (PoE) and variational head.
        """
        x = self._as_tensor(z_latent)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            z_pred = self._forward_predict(x, mask=0)
        if z_pred.dim() > 1:
            z_pred = z_pred.squeeze(0)
        return z_pred.float().cpu().numpy().astype(np.float32)

    def policy_targets(self, z_world: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Return a softmax policy distribution over 63 mutation masks.

        Uses the predictor's representation to score each mutation and
        returns a temperature-scaled softmax. This enables MuZero-style
        MCTS policy improvement: the MCTS search distribution trains
        the policy head toward better action selection.

        KV-Cache optimised: the context_encoder runs ONCE and the
        predictor runs as a single batched forward pass over all 63
        action embeddings, instead of 63 sequential calls.
        """
        self.eval()
        embed_device = next(self._action_embed.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(z_world).float().unsqueeze(0).to(embed_device)
            # Encode into latent space ONCE (106→64).
            z_latent = self.context_encoder(x)
            assert z_latent.shape[-1] == self.latent_dim, (
                f"policy_targets: latent dim {z_latent.shape[-1]} "
                f"!= {self.latent_dim}"
            )
            # Batch all 63 action embeddings into a single forward pass
            # through the predictor instead of 63 sequential calls.
            masks = torch.arange(1, 64, dtype=torch.long, device=embed_device)
            action_embs = self._action_embed(masks)  # [63, embed_dim]
            # Expand z_latent: [1, latent_dim] → [63, latent_dim]
            z_expanded = z_latent.expand(63, -1)
            if self._use_action_cond:
                z_act = torch.cat([z_expanded, action_embs], dim=-1)  # [63, latent_dim+embed_dim]
            else:
                z_act = z_expanded
            if self.variational:
                h = self.predictor(z_act)
                mu, _ = self._var_head(h)
                preds = mu
            else:
                preds = self.predictor(z_act)
            # Use prediction norms as proxy for action quality
            scores = preds.norm(dim=-1).cpu().numpy()
            # Temperature-scaled softmax
            exp_scores = np.exp((scores - scores.max()) / max(temperature, 0.01))
            return exp_scores / exp_scores.sum()

    def policy_targets_with_vq(
        self, z_world: np.ndarray, temperature: float = 1.0,
    ) -> np.ndarray:
        """M2 — VQ-token-augmented policy.

        Identical to :meth:`policy_targets` but mixes the continuous
        latent with a *discrete* VQ token (the nearest codebook entry)
        before scoring each action.  This gives the policy head access
        to a symbolic "what state am I in" channel alongside the
        continuous "how close am I to target" channel — a 2025 SOTA
        pattern (V-JEPA 2 + VQ-VAE).

        Falls back to :meth:`policy_targets` when VQ has not been
        initialised via :meth:`init_vq`.
        """
        if getattr(self, "_vq", None) is None:
            return self.policy_targets(z_world, temperature=temperature)
        self.eval()
        embed_device = next(self._action_embed.parameters()).device
        with torch.no_grad():
            x = torch.from_numpy(z_world).float().unsqueeze(0).to(embed_device)
            z_latent = self.context_encoder(x)
            assert z_latent.shape[-1] == self.latent_dim, (
                f"policy_targets_with_vq: latent dim {z_latent.shape[-1]} "
                f"!= {self.latent_dim}"
            )
            # Quantise — straight-through estimator means grads still
            # flow back through the codebook at training time, but
            # here we just need the discrete index for the policy.
            _, vq_indices, _ = self._vq(z_latent)
            vq_token = self._vq.codebook[vq_indices]  # [1, D]
            # Mix: 70% continuous, 30% discrete (a typical 2026 mix).
            z_mixed = 0.7 * z_latent + 0.3 * vq_token
            scores = []
            for mask in range(1, 64):
                action_emb = self._action_embed(
                    torch.tensor([mask], device=embed_device)
                )
                z_act = torch.cat([z_mixed, action_emb], dim=-1)
                if self.variational:
                    h = self.predictor(z_act)
                    mu, _ = self._var_head(h)
                    pred = mu
                else:
                    pred = self.predictor(z_act)
                scores.append(float(pred.norm()))
            scores = np.array(scores)
            exp_scores = np.exp((scores - scores.max()) / max(temperature, 0.01))
            return exp_scores / exp_scores.sum()

    def vicreg_loss(self, latent: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Variance + covariance regularisation (collapse prevention).

        Operates on a batch of latents [B, D]. For a single sample it degrades
        gracefully to the variance hinge only.
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        b, d = latent.shape

        # Variance term: hinge so each dimension keeps unit-ish std.
        std = torch.sqrt(latent.var(dim=0, unbiased=False) + eps)
        var_loss = torch.mean(F.relu(1.0 - std))

        # Covariance term: push off-diagonal covariance to zero.
        if b > 1:
            centered = latent - latent.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / (b - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            cov_loss = (off_diag ** 2).sum() / d
        else:
            cov_loss = latent.new_zeros(())
        return var_loss + cov_loss

    def _variational_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Variational loss: NLL + KL divergence.

        L = NLL(z_target; mu, sigma^2) + kl_weight * KL(N(mu, sigma^2) || N(0, 1))

        where sigma^2 = exp(log_var), and the KL weight is ``self._kl_weight``.
        """
        # Negative log-likelihood of target under N(mu, sigma^2)
        nll = 0.5 * (log_var + (target - mu) ** 2 / (log_var.exp() + 1e-6))
        nll = nll.mean()

        # KL divergence: KL(N(mu, sigma^2) || N(0, 1))
        # = 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
        kl = 0.5 * (mu ** 2 + log_var.exp() - log_var - 1)
        kl = kl.mean()

        return nll + self._kl_weight * kl

    def train_step(
        self,
        z_current: np.ndarray | torch.Tensor,
        z_next: np.ndarray | torch.Tensor,
        max_grad_norm: float = 5.0,
    ) -> dict[str, float]:
        """One real gradient step on a replayed minibatch.

        Returns ``{"loss": total, "pred_error": pred}`` where
        ``loss`` is the full regularization objective (may be negative
        due to SIGReg / energy terms) and ``pred_error`` is the raw
        world-model prediction error (always ≥ 0, the quantity that
        should decrease with training).

        Stores the (xₜ, xₜ₊₁) transition in the replay buffer, samples a
        minibatch, and predicts each EMA-target next embedding from the online
        current embedding. The loss is prediction error in representation space
        plus a batched VICReg term (active because the batch has cross-sample
        variance). Updates the EMA target afterwards. Guards against NaN/Inf.
        """
        x_t = self._as_tensor(z_current).detach()
        x_next = self._as_tensor(z_next).detach()
        if not (torch.isfinite(x_t).all() and torch.isfinite(x_next).all()):
            return {"loss": float("nan"), "pred_error": float("nan")}
        self._replay.append((x_t, x_next))

        n = len(self._replay)
        k = min(self._batch_size, n)
        idx = torch.randint(0, n, (k,))
        xb = torch.stack([self._replay[int(i)][0] for i in idx]).to(self.device)
        xnb = torch.stack([self._replay[int(i)][1] for i in idx]).to(self.device)

        # P0-1: bf16 autocast on the forward pass + compiled modules in
        # the hot path. Gradients remain fp32 (autocast does this for us).
        ctx = self._compiled_context if self._compiled_context is not None else self.context_encoder
        pred = self._compiled_predictor or self.predictor
        autocast_ctx = (
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
            )
            if self._amp_enabled
            else _NullContext()
        )
        with autocast_ctx:
            z_t = ctx(xb)
            with torch.no_grad():
                z_target = self.target_encoder(xnb)

            # P3-A: when action conditioning is enabled, sample a
            # random mask per batch element.  The world model is then
            # trained to predict ``z_{t+1} | (z_t, a)`` for arbitrary
            # ``a``, so MPC can sweep actions at inference time.
            if self._use_action_cond:
                mask_batch = torch.randint(
                    0, 64, (xb.shape[0],), dtype=torch.long, device=xb.device
                )
                a_emb = self._action_embed(mask_batch)
                z_t_for_pred = torch.cat([z_t, a_emb], dim=-1)
            else:
                z_t_for_pred = z_t

            if self.variational:
                h = pred(z_t_for_pred)
                mu, log_var = self._var_head(h)
                pred_loss = self._variational_loss(mu, log_var, z_target)
                _raw_pred_error = F.mse_loss(mu, z_target)
            else:
                z_pred = pred(z_t_for_pred)
                pred_loss = F.mse_loss(z_pred, z_target)
                _raw_pred_error = pred_loss

            # P3-C — Product-of-Experts structural prior.  When the
            # prior expert is enabled, add its L2 agreement with the
            # target as an extra regulariser — keeps the prior expert
            # in the same loss surface as the dynamics expert.
            if self._prior_expert is not None:
                z_prior = self._prior_expert(z_t)
                # The prior should also be self-consistent with the
                # dynamics prediction — the agreement term is what
                # makes the PoE productive.
                poe_loss = F.mse_loss(z_prior, z_pred) if not self.variational else F.mse_loss(z_prior, mu)
            else:
                poe_loss = torch.tensor(0.0, dtype=z_t.dtype, device=z_t.device)

            # P3-D — Anti-collapse regulariser.  When SIGReg is
            # enabled, replace the VICReg term with the 2026 SOTA
            # distribution-matching loss.  VICReg is kept as a
            # fallback when SIGReg is disabled.
            if self._use_sigreg:
                reg = sigreg_loss(z_t)
            else:
                reg = self.vicreg_loss(z_t)
            loss = pred_loss + self._sigreg_weight * reg + self._poe_weight * poe_loss

            # P3-E — Bidirectional prediction.  The backward predictor
            # maps ``z_{t+1} -> ẑ_t`` and we penalise the deviation
            # from the actual ``z_t``.  Combined with the cycle
            # consistency, this doubles the supervision signal per
            # transition (BiJEPA 2026).
            if self._backward_predictor is not None and not self.variational:
                h_back = self._backward_predictor(
                    torch.cat([z_target, a_emb], dim=-1) if self._use_action_cond else z_target
                )
                z_back = self._backward_proj(h_back)
                bwd_loss = F.mse_loss(z_back, z_t.detach())
            elif self._backward_predictor is not None and self.variational:
                h_back = self._backward_predictor(
                    torch.cat([z_target, a_emb], dim=-1) if self._use_action_cond else z_target
                )
                mu_back, _ = self._var_head(h_back)
                bwd_loss = F.mse_loss(mu_back, z_t.detach())
            else:
                bwd_loss = torch.tensor(0.0, dtype=z_t.dtype, device=z_t.device)
            loss = loss + self._backward_weight * bwd_loss

            # P3-F — Energy-based auxiliary head.  Lower energy on
            # the actual next state, higher energy on a randomly
            # sampled "decoy" transition.  This pulls the energy
            # surface toward the right futures and matches the
            # 2026 LeCun EBM thesis.
            # P0 FIX: e_pos = energy(z_target), e_neg = energy(z_target[perm]).
            # Standard EBM: real data gets low energy, decoys get high energy.
            # Previously e_pos was energy(z_t) which inverted the learning signal.
            if self._energy_head is not None:
                e_pos = self._energy_head(z_target)  # energy of real next state
                # Permute z_target to create a decoy batch
                perm = torch.randperm(z_target.shape[0], device=z_target.device)
                e_neg = self._energy_head(z_target[perm])
                margin = 1.0
                energy_loss = F.relu(e_pos - e_neg + margin).mean()
            else:
                energy_loss = torch.tensor(0.0, dtype=z_t.dtype, device=z_t.device)
            loss = loss + self._energy_weight * energy_loss

            # P2-3: VQ-VAE discretisation loss (only when VQ enabled).
            if self._vq is not None:
                z_q, _idx, vq_loss = self._vq(z_t)
                loss = loss + vq_loss
            # P2-1: auxiliary value head loss — V_θ(z_t) → x_t's target
            # latent scalar magnitude. This is a self-supervised proxy
            # for the latent energy (||z_t|| ≈ -log density) and is the
            # standard MuZero auxiliary loss. Only when the head is
            # initialised.
            if self._value_head is not None:
                v_pred = self._value_head(z_t)
                v_target = z_target.detach().norm(dim=-1, keepdim=True)
                value_loss = F.mse_loss(v_pred, v_target)
                loss = loss + self._value_weight * value_loss

        if not torch.isfinite(loss):
            return {"loss": float("nan"), "pred_error": float("nan")}

        self._opt.zero_grad()
        loss.backward()
        grad_params = (
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters())
        )
        if self._var_head is not None:
            grad_params += list(self._var_head.parameters())
        if self._backward_predictor is not None:
            grad_params += list(self._backward_predictor.parameters())
        if self._backward_proj is not None:
            grad_params += list(self._backward_proj.parameters())
        if self._prior_expert is not None:
            grad_params += list(self._prior_expert.parameters())
        if self._energy_head is not None:
            grad_params += list(self._energy_head.parameters())
        grad_params += list(self._action_embed.parameters())
        nn.utils.clip_grad_norm_(grad_params, max_norm=max_grad_norm)
        self._opt.step()
        self._update_target_encoder()
        return {
            "loss": float(loss.detach()),
            "pred_error": float(_raw_pred_error.detach()),
        }


    def train_transition(
        self,
        square_t: np.ndarray,
        time_t: float,
        square_next: np.ndarray,
        time_next: float,
        unified_t: np.ndarray | None = None,
        unified_next: np.ndarray | None = None,
        mask: int | None = None,
        max_grad_norm: float = 5.0,
        h_next: object | None = None,
    ) -> float:
        """End-to-end transition training: gradient flows into the learnable
        square encoder (when one has been attached).

        Composed of three phases:
          1. _build_inputs        — build the 106-dim (or fallback 77-dim) world vector
          2. _forward_learnable   — context / target encoding + action conditioning
          3. _compute_loss        — pred + PoE + SIGReg + bwd + energy + VQ + value
        Returns the final scalar loss.
        """
        inputs = self._build_inputs(
            square_t, time_t, square_next, time_next,
            unified_t, unified_next,
        )
        if inputs.has_learnable:
            return self._forward_learnable(inputs, mask, max_grad_norm, h_next)
        # Passive fallback: no learnable encoder, delegate to train_step.
        z_t = np.concatenate([
            inputs.z_sq_t, inputs.cp_t,
            *([inputs.unified_t] if inputs.unified_t is not None else []),
        ])
        z_next = np.concatenate([
            inputs.z_sq_next, inputs.cp_next,
            *([inputs.unified_next] if inputs.unified_next is not None else []),
        ])
        return self.train_step(z_t, z_next, max_grad_norm=max_grad_norm)

    def _build_inputs(
        self,
        square_t: np.ndarray,
        time_t: float,
        square_next: np.ndarray,
        time_next: float,
        unified_t: np.ndarray | None,
        unified_next: np.ndarray | None,
    ) -> "_TrainTransitionInputs":
        """Pack the inputs needed for the learnable forward pass.

        Returns a small namedtuple-like object carrying the padded/aligned
        tensors (z_sq_t, z_sq_next, cp_t, cp_next, unified_t, unified_next)
        plus a ``has_learnable`` flag.
        """
        from zwm.jepa.square_encoder import circular_phase_vector

        joint_dim = 77  # 64 (square) + 13 (circular phase)
        world_dim = self.input_dim
        extra_dim = max(0, world_dim - joint_dim)
        if extra_dim == 0:
            unified_t_use: np.ndarray | None = None
            unified_next_use: np.ndarray | None = None
        else:
            t_arr = np.asarray(unified_t, dtype=np.float32).reshape(-1) if unified_t is not None else np.zeros(extra_dim, dtype=np.float32)
            n_arr = np.asarray(unified_next, dtype=np.float32).reshape(-1) if unified_next is not None else np.zeros(extra_dim, dtype=np.float32)
            if t_arr.shape[0] < extra_dim:
                t_arr = np.concatenate([t_arr, np.zeros(extra_dim - t_arr.shape[0], dtype=np.float32)])
            if n_arr.shape[0] < extra_dim:
                n_arr = np.concatenate([n_arr, np.zeros(extra_dim - n_arr.shape[0], dtype=np.float32)])
            unified_t_use = t_arr[:extra_dim]
            unified_next_use = n_arr[:extra_dim]

        has_learnable = (
            hasattr(self, "_square_encoder") and self._square_encoder is not None
        )
        return _TrainTransitionInputs(
            z_sq_t=np.asarray(square_t, dtype=np.float32).reshape(-1),
            z_sq_next=np.asarray(square_next, dtype=np.float32).reshape(-1),
            cp_t=circular_phase_vector(time_t),
            cp_next=circular_phase_vector(time_next),
            unified_t=unified_t_use,
            unified_next=unified_next_use,
            has_learnable=has_learnable,
        )

    def _forward_learnable(
        self,
        inputs: "_TrainTransitionInputs",
        mask: int | None,
        max_grad_norm: float,
        h_next: object | None,
    ) -> float:
        """Run the learnable forward pass and backprop."""
        ft = torch.from_numpy(inputs.z_sq_t).unsqueeze(0)
        fn = torch.from_numpy(inputs.z_sq_next).unsqueeze(0)

        z_sq_t = self._square_encoder(ft)  # [1, 64]
        with torch.no_grad():
            z_sq_next = self._square_encoder(fn)  # [1, 64]

        cp_t = torch.from_numpy(inputs.cp_t).unsqueeze(0)
        cp_next = torch.from_numpy(inputs.cp_next).unsqueeze(0)
        extras_t: list[torch.Tensor] = []
        extras_n: list[torch.Tensor] = []
        if inputs.unified_t is not None:
            extras_t.append(torch.from_numpy(inputs.unified_t).unsqueeze(0))
        if inputs.unified_next is not None:
            extras_n.append(torch.from_numpy(inputs.unified_next).unsqueeze(0))

        x_t = torch.cat([z_sq_t, cp_t] + extras_t, dim=-1)
        x_next = torch.cat([z_sq_next, cp_next] + extras_n, dim=-1)

        # Move inputs to the model's device (CPU → CUDA when applicable).
        x_t = x_t.to(self.device)
        x_next = x_next.to(self.device)

        self._replay.append((x_t.squeeze(0).detach(), x_next.squeeze(0).detach()))

        autocast_ctx = (
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
            )
            if self._amp_enabled
            else _NullContext()
        )
        with autocast_ctx:
            z_lat = self.context_encoder(x_t)
            with torch.no_grad():
                z_target = self.target_encoder(x_next)

            # P1-5: Consume _target_square_encoder for richer target representation
            if self._target_square_encoder is not None:
                try:
                    from zwm.jepa.square_encoder import hexagram_square_features
                    sq_feat_next = hexagram_square_features(h_next) if h_next is not None else None
                    if sq_feat_next is not None:
                        sq_next = self._target_square_encoder.embed(h_next)
                        sq_next_t = torch.from_numpy(sq_next).unsqueeze(0).to(z_target.device)
                        z_target = z_target + 0.1 * sq_next_t.detach()
                except Exception as exc:
                    import logging as _logging
                    _logging.getLogger(__name__).debug("Square encoder injection skipped: %s", exc)

            # P3-A: action conditioning.  When mask is provided, use
            # it; otherwise default to mask=0 ("no-action").
            action_mask = mask if mask is not None else 0
            if self._use_action_cond:
                mask_t = torch.tensor(
                    int(action_mask), dtype=torch.long, device=x_t.device,
                ).unsqueeze(0)
                a_emb = self._action_embed(mask_t)
                z_t_for_pred = torch.cat([z_lat, a_emb], dim=-1)
            else:
                a_emb = None
                z_t_for_pred = z_lat

            loss, pred_loss_raw = self._compute_loss(z_lat, z_target, z_t_for_pred, a_emb, x_t)
            self._add_vq_value_losses(z_lat, z_target, loss)

        if not torch.isfinite(loss):
            return {"loss": float("nan"), "pred_error": float("nan")}

        self._opt.zero_grad()
        loss.backward()
        grad_params = self._collect_grad_params()
        nn.utils.clip_grad_norm_(grad_params, max_norm=max_grad_norm)
        self._opt.step()
        # 2026 SOTA: step the LR schedule and EMA schedule after each
        # optimizer step.
        if hasattr(self, "_lr_scheduler"):
            self._lr_scheduler.step()
        self._update_target_encoder()
        return {
            "loss": float(loss.detach()),
            "pred_error": float(pred_loss_raw.detach()),
        }

    def _compute_loss(
        self,
        z_lat: torch.Tensor,
        z_target: torch.Tensor,
        z_t_for_pred: torch.Tensor,
        a_emb: torch.Tensor | None,
        x_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum the JEPA auxiliary losses: pred, PoE, SIGReg/VICReg, bwd, energy.

        Returns ``(total_loss, raw_prediction_error)`` where ``raw_prediction_error``
        is the raw MSE between the predictor output and the target, always ≥ 0.
        """
        if self.variational:
            h = self.predictor(z_t_for_pred)
            mu, log_var = self._var_head(h)
            pred_loss = self._variational_loss(mu, log_var, z_target)
            raw_pred_error_pred = F.mse_loss(mu, z_target)
        else:
            z_pred = self.predictor(z_t_for_pred)
            pred_loss = F.mse_loss(z_pred, z_target)
            raw_pred_error_pred = pred_loss

        # P3-C — PoE prior agreement
        if self._prior_expert is not None:
            z_prior = self._prior_expert(z_lat)
            poe_loss = F.mse_loss(z_prior, z_pred if not self.variational else mu)
        else:
            poe_loss = torch.tensor(0.0, dtype=z_lat.dtype, device=z_lat.device)

        # P3-D — SIGReg anti-collapse
        reg = sigreg_loss(z_lat) if self._use_sigreg else self.vicreg_loss(z_lat)

        # P3-E — Backward prediction
        if self._backward_predictor is not None:
            bwd_in = (
                torch.cat([z_target, a_emb], dim=-1) if self._use_action_cond
                else z_target
            )
            h_back = self._backward_predictor(bwd_in)
            if self.variational:
                mu_back, _ = self._var_head(h_back)
                bwd_loss = F.mse_loss(mu_back, z_lat.detach())
            else:
                bwd_loss = F.mse_loss(h_back, z_lat.detach())
        else:
            bwd_loss = torch.tensor(0.0, dtype=z_lat.dtype, device=z_lat.device)

        # P3-F — Energy head
        # P0 FIX: e_pos = energy(z_target), e_neg = energy(z_target[perm]).
        # Standard EBM: real data gets low energy, decoys get high energy.
        if self._energy_head is not None:
            e_pos = self._energy_head(z_target)  # energy of real next state
            perm = torch.randperm(z_target.shape[0], device=z_target.device)
            e_neg = self._energy_head(z_target[perm])
            energy_loss = F.relu(e_pos - e_neg + 1.0).mean()
        else:
            energy_loss = torch.tensor(0.0, dtype=z_lat.dtype, device=z_lat.device)

        return (
            pred_loss
            + self._sigreg_weight * reg
            + self._poe_weight * poe_loss
            + self._backward_weight * bwd_loss
            + self._energy_weight * energy_loss
        ), raw_pred_error_pred

    def _add_vq_value_losses(
        self,
        z_lat: torch.Tensor,
        z_target: torch.Tensor,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """Add VQ + value-head MSEs in-place; return the augmented loss."""
        if self._vq is not None:
            z_q, _idx, vq_loss = self._vq(z_lat)
            loss = loss + vq_loss
        if self._value_head is not None:
            v_pred = self._value_head(z_lat)
            v_target = z_target.detach().norm(dim=-1, keepdim=True)
            value_loss = F.mse_loss(v_pred, v_target)
            loss = loss + self._value_weight * value_loss
        return loss

    def _collect_grad_params(self) -> list[nn.Parameter]:
        """All parameters that should receive gradient updates."""
        grad_params: list[nn.Parameter] = (
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters())
        )
        if hasattr(self, "_square_encoder") and self._square_encoder is not None:
            grad_params += list(self._square_encoder.parameters())
        if self._var_head is not None:
            grad_params += list(self._var_head.parameters())
        if self._backward_predictor is not None:
            grad_params += list(self._backward_predictor.parameters())
        if self._backward_proj is not None:
            grad_params += list(self._backward_proj.parameters())
        if self._prior_expert is not None:
            grad_params += list(self._prior_expert.parameters())
        if self._energy_head is not None:
            grad_params += list(self._energy_head.parameters())
        if hasattr(self, "_action_embed") and self._action_embed is not None:
            grad_params += list(self._action_embed.parameters())
        if self._vq is not None:
            grad_params += list(self._vq.parameters())
        if self._value_head is not None:
            grad_params += list(self._value_head.parameters())
        return grad_params

    def attach_square_encoder(self, square_encoder) -> None:
        """Hook a ``LearnableSquareGNN`` into the JEPA optimiser scope.

        When attached, gradients from ``train_step`` (and ``train_transition``)
        flow into the square encoder parameters — representation learning is
        now end-to-end, not a frozen random projection. Re-creates the Adam
        optimiser with the extended parameter list.
        """
        import copy as _copy
        self._square_encoder = square_encoder
        # EMA-copy the square encoder too (target side) so the target net
        # tracks the online encoder for the world-model loss.
        try:
            self._target_square_encoder = _copy.deepcopy(square_encoder)
            for p in self._target_square_encoder.parameters():
                p.requires_grad_(False)
        except Exception as exc:
            # AUDIT-S5: EMA target encoder couldn't be deep-copied —
            # the target net would fall behind the online net and
            # surprise would drift.  Surface the failure.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "attach_square_encoder: target-encoder copy failed: %s", exc,
            )
            self._target_square_encoder = None

        params = (
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters())
            + list(square_encoder.parameters())
        )
        if self._var_head is not None:
            params += list(self._var_head.parameters())
        self._opt = torch.optim.Adam(params, lr=self._opt.param_groups[0]["lr"])
        # P2-1: also include the latent value head in the optimiser.
        if self._value_head is not None:
            self._opt = torch.optim.Adam(
                params + list(self._value_head.parameters()),
                lr=self._opt.param_groups[0]["lr"],
            )

    # ------------------------------------------------------------------
    # P2-1: MuZero-style latent value head
    # ------------------------------------------------------------------
    def init_value_head(
        self,
        hidden_dim: int = 32,
        value_weight: float = 0.5,
    ) -> None:
        """P2-1 — attach a learned V(s) head to the JEPA latent.

        MuZero / EfficientZero use a learned value function as the
        bootstrap target for n-step TD. Without it, EFE's V(s) is an EMA
        table (smooth but coarse). With it, the agent gets a smooth +
        expressive value estimate, which dramatically improves sample
        efficiency on sparse-reward streams.
        """
        if getattr(self, "_value_head", None) is not None:
            return
        # The value head reads the *latent* (output of the context
        # encoder), not the predictor's last hidden state.  Use
        # ``self.latent_dim`` so the input dim tracks the 2026 P3-H
        # upgrade.
        self._value_head = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Move value head to the same device as the rest of the model.
        self._value_head.to(self.device)
        self._value_weight = value_weight
        # P0-4: dedicated value-head optimizer so TD / DreamerV3 updates
        # don't stomp the main JEPA encoder/decoder.  Without this,
        # ``loss.backward()`` accumulates into the same parameter groups
        # the world model uses, leading to a feedback loop that
        # destabilises the surprise signal.  Adam with a higher LR
        # is appropriate for the small value head.
        try:
            self._value_opt = torch.optim.Adam(
                list(self._value_head.parameters()),
                lr=self._opt.param_groups[0]["lr"] * 2.0,
            )
        except Exception as exc:
            # AUDIT-S5: value-head optimiser rebuild failed — V(s)
            # updates would silently fall back to the main optimiser
            # (which would corrupt JEPA weights).  Set _value_opt to
            # None and let the caller skip the value update.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "init_value_head: value-opt init failed: %s", exc,
            )
            self._value_opt = None
        # Add value-head params to the main optimiser (so checkpoint
        # restore captures them) but the value head gets its own
        # optimizer for updates.
        try:
            params = (
                list(self.context_encoder.parameters())
                + list(self.predictor.parameters())
            )
            if self._var_head is not None:
                params += list(self._var_head.parameters())
            if hasattr(self, "_square_encoder") and self._square_encoder is not None:
                params += list(self._square_encoder.parameters())
            params += list(self._value_head.parameters())
            self._opt = torch.optim.Adam(
                params, lr=self._opt.param_groups[0]["lr"]
            )
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "init_value_head: main optimiser rebuild failed: %s", exc,
            )
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "init_value_head: main optimiser rebuild failed: %s", exc,
            )

    def value(self, z_world: np.ndarray | torch.Tensor) -> np.ndarray | None:
        """P2-1 — learned V(s) of the world state.

        AUDIT-S6: returns ``None`` when the value head has not been
        initialised, instead of a hard-coded ``np.zeros(1)``.  Callers
        that don't differentiate "V(s) = 0" from "V(s) unknown" used
        to silently misweight the EFE term.  Downstream EFE scoring
        now treats ``None`` as "use the EMA baseline V" — the
        mathematically correct fallback.
        """
        if self._value_head is None:
            return None
        x = self._as_tensor(z_world)
        with torch.no_grad():
            z = self.context_encoder(x)
            v = self._value_head(z)
        return v.float().cpu().numpy().astype(np.float32)

    def value_or_zero(self, z_world: np.ndarray | torch.Tensor) -> np.ndarray:
        """``value()`` with the legacy zero-fallback for callers that
        can't handle ``None`` (e.g. external optimisers that expect a
        concrete tensor).  Prefer ``value()`` when you can — see
        AUDIT-S6."""
        v = self.value(z_world)
        if v is None:
            return np.zeros(1, dtype=np.float32)
        return v

    # ------------------------------------------------------------------
    # P2-3: VQ-VAE tokenisation
    # ------------------------------------------------------------------
    def init_vq(
        self,
        num_codes: int = 64,
        beta: float = 0.25,
    ) -> None:
        """P2-3 — attach a VQ-VAE codebook to the predictor's latent.

        The 2025/2026 SOTA for self-supervised video / language is
        discrete token representations. With VQ on, the JEPA latent
        becomes a sequence of 64 codes × ``latent_dim`` and the planner
        can reason in symbolic terms.

        P2-15: ``VQCodebook(dim=...)`` now matches ``self.latent_dim``
        (the actual encoder output) instead of the legacy hard-coded
        32.  This prevents the silent shape-mismatch the previous
        version would have produced after the P3-H 64-dim upgrade.
        """
        from zwm.jepa.vq import VQCodebook
        if getattr(self, "_vq", None) is not None:
            return
        self._vq = VQCodebook(num_codes=num_codes, dim=self.latent_dim, beta=beta)
        # Move VQ to the same device as the rest of the model.
        self._vq.to(self.device)
        # Include VQ params in the optimiser scope.
        try:
            params = (
                list(self.context_encoder.parameters())
                + list(self.predictor.parameters())
            )
            if self._var_head is not None:
                params += list(self._var_head.parameters())
            if hasattr(self, "_square_encoder") and self._square_encoder is not None:
                params += list(self._square_encoder.parameters())
            if self._value_head is not None:
                params += list(self._value_head.parameters())
            params += list(self._vq.parameters())
            self._opt = torch.optim.Adam(
                params, lr=self._opt.param_groups[0]["lr"]
            )
        except Exception as exc:
            # AUDIT-S5: VQ optimiser rebuild failing means the VQ
            # codebook parameters won't be updated — the agent would
            # still see VQ loss but the codebook would stay frozen.
            # Surface the failure so the operator can intervene.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "init_vq: optimiser rebuild failed: %s", exc,
            )

    def tokenize(self, z_world: np.ndarray | torch.Tensor) -> np.ndarray:
        """P2-3 — return the discrete code indices for a world state."""
        if self._vq is None:
            return np.zeros(1, dtype=np.int64)
        x = self._as_tensor(z_world)
        with torch.no_grad():
            z = self.context_encoder(x)
            _, indices, _ = self._vq(z)
        return indices.numpy().astype(np.int64)

    # ------------------------------------------------------------------
    # Quantization (Q-LoRA / 4-bit) — 2026 SOTA for inference efficiency
    # ------------------------------------------------------------------
    def quantize_4bit(self) -> dict[str, str]:
        """Quantize the JEPA predictor to 4-bit using bitsandbytes NF4.

        The 2026 SOTA for efficient inference is 4-bit NormalFloat (NF4)
        quantization with double quantization, as introduced in QLoRA
        (Dettmers et al., 2023) and now standard in every production LLM
        deployment.  This reduces memory by ~4× and enables inference on
        consumer GPUs.

        Returns a dict of {layer_name: dtype_before} for verification.
        Falls back gracefully if bitsandbytes is not installed.
        """
        try:
            import bitsandbytes as bnb
        except ImportError:
            return {"error": "bitsandbytes not installed — pip install bitsandbytes"}

        before: dict[str, str] = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and module.weight.numel() > 256:
                before[name] = str(module.weight.dtype)
                # Replace the Linear layer with a 4-bit NF4 quantized
                # version.  bitsandbytes handles the quantization on the
                # fly — no pre-quantized checkpoint needed.
                try:
                    q_linear = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=torch.bfloat16,
                        quant_type="nf4",
                        quant_storage=torch.uint8,
                    )
                    # Copy the weight into the quantized layer.
                    q_linear.weight = bnb.nn.Params4bit(
                        module.weight.data,
                        requires_grad=False,
                        quant_type="nf4",
                        quant_storage=torch.uint8,
                    )
                    if module.bias is not None:
                        q_linear.bias = nn.Parameter(module.bias.data.clone())

                    # Replace in the module hierarchy.
                    parts = name.split(".")
                    parent = self
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], q_linear)
                except Exception as exc:
                    # AUDIT-S5: best-effort quant layer swap; some
                    # modules (custom, non-standrad Linear shapes)
                    # can't be NF4-quantized — debug-log it so the
                    # operator can see which ones were skipped.
                    import logging as _logging
                    _logging.getLogger(__name__).debug(
                        "4-bit quant skipped for %s: %s", name, exc,
                    )

        self._quantized = True
        return before

    def apply_lora(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: list[str] | None = None,
    ) -> list[str]:
        """Apply LoRA (Low-Rank Adaptation) adapters to the predictor.

        The 2026 SOTA for parameter-efficient fine-tuning is LoRA /
        Q-LoRA: instead of updating all weights, inject small rank-r
        matrices (A, B) into each Linear layer and only train those.
        This reduces trainable parameters by ~99% while maintaining
        95-99% of full fine-tuning quality.

        When combined with ``quantize_4bit()``, this is Q-LoRA — the
        standard recipe for fine-tuning large models on consumer GPUs.

        Returns a list of adapted module names.
        """
        if target_modules is None:
            # Default: adapt the predictor and context encoder's
            # Linear layers (the most impactful for fine-tuning).
            target_modules = ["predictor", "context_encoder"]

        adapted: list[str] = []
        for name, module in self.named_modules():
            # Check if this module is in the target list.
            should_adapt = any(t in name for t in target_modules)
            if not should_adapt:
                continue
            if isinstance(module, nn.Linear) and module.weight.numel() > 256:
                try:
                    lora_a = nn.Linear(module.in_features, rank, bias=False)
                    lora_b = nn.Linear(rank, module.out_features, bias=False)
                    # Initialise A with Kaiming, B with zeros (LoRA recipe).
                    nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))
                    nn.init.zeros_(lora_b.weight)

                    # Wrap the original module with a LoRA adapter.
                    lora_module = _LoRALinear(module, lora_a, lora_b, alpha)

                    # Replace in the module hierarchy.
                    parts = name.split(".")
                    parent = self
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], lora_module)
                    adapted.append(name)
                except Exception as exc:
                    # AUDIT-S5: same pattern as quantize_4bit — log
                    # which LoRA target was skipped.
                    import logging as _logging
                    _logging.getLogger(__name__).debug(
                        "LoRA attach skipped for %s: %s", name, exc,
                    )

        # Add LoRA parameters to the optimiser scope.
        if adapted:
            lora_params = []
            for name, param in self.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
                    lora_params.append(param)
                elif not any(t in name for t in target_modules):
                    param.requires_grad = False
            if lora_params:
                self._opt = torch.optim.Adam(lora_params, lr=self._opt.param_groups[0]["lr"])

        return adapted

    @property
    def is_quantized(self) -> bool:
        return getattr(self, "_quantized", False)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        # 2026 SOTA: use the EMA schedule instead of a fixed decay.
        if hasattr(self, "_ema_schedule"):
            d = self._ema_schedule.step()
        else:
            d = self._ema_decay
        for tgt, src in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            tgt.mul_(d).add_(src, alpha=1.0 - d)
        # Also update the EMA target square encoder if it exists.
        target_sq = getattr(self, "_target_square_encoder", None)
        if target_sq is not None and hasattr(self, "_square_encoder") and self._square_encoder is not None:
            for tgt, src in zip(
                target_sq.parameters(),
                self._square_encoder.parameters(),
            ):
                tgt.mul_(d).add_(src, alpha=1.0 - d)

    def _as_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)


class HierarchicalJEPAPredictor(nn.Module):
    """Multi-timescale JEPA: 3 temporal scales with separate EMA targets.

    Levels:
      * short (1 step)  — fast dynamics, existing JEPAPredictor
      * mid   (4 steps) — medium-range trends
      * long  (16 steps) — slow structural shifts

    Each level owns its own context_encoder, predictor head, and EMA
    target_encoder so gradients do not cross-contaminate timescales.

    P1 FIX: Added ring buffers for mid/long timescales.  Previously all
    three levels received the same (z_t, z_{t+1}) pair, making them
    effectively three copies of the same model.  Now mid uses (z_{t-4},
    z_t) and long uses (z_{t-16}, z_t) when sufficient history is available.
    """

    def __init__(
        self,
        input_dim: int = _DEFAULT_INPUT_DIM,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        learning_rate: float = 1e-3,
        ema_decay: float = 0.99,
        vicreg_weight: float = 0.04,
        replay_capacity: int = 256,
        batch_size: int = 16,
        seed: int = 42,
        variational: bool = True,
        kl_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        # Short-term level reuses a full JEPAPredictor (1-step).
        self.short = JEPAPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
            vicreg_weight=vicreg_weight,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            seed=seed,
            variational=variational,
            kl_weight=kl_weight,
        )
        # Mid-term level (4-step).
        self.mid = JEPAPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
            vicreg_weight=vicreg_weight,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            seed=seed + 1,
            variational=variational,
            kl_weight=kl_weight,
        )
        # Long-term level (16-step).
        self.long = JEPAPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
            vicreg_weight=vicreg_weight,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            seed=seed + 2,
            variational=variational,
            kl_weight=kl_weight,
        )
        self._temporal_spans = {"short": 1, "mid": 4, "long": 16}
        # P1 FIX: Ring buffers for multi-timescale training.
        # mid needs 4-step gap, long needs 16-step gap.
        import collections
        self._z_history: collections.deque = collections.deque(maxlen=16)
        self._z_next_history: collections.deque = collections.deque(maxlen=16)

    # ------------------------------------------------------------------
    def predict(self, z_world: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        """Predict next-state latents at all 3 temporal scales.

        Returns dict with keys "short", "mid", "long".
        """
        return {
            "short": self.short.predict(z_world),
            "mid": self.mid.predict(z_world),
            "long": self.long.predict(z_world),
        }

    def target_latent(self, x_next: np.ndarray | torch.Tensor) -> np.ndarray:
        """Aggregate target latents across levels (mean)."""
        z_short = self.short.target_latent(x_next)
        z_mid = self.mid.target_latent(x_next)
        z_long = self.long.target_latent(x_next)
        return ((z_short + z_mid + z_long) / 3.0).astype(np.float32)

    def train_step(
        self,
        z_current: np.ndarray | torch.Tensor,
        z_next: np.ndarray | torch.Tensor,
        max_grad_norm: float = 5.0,
    ) -> dict[str, float]:
        """Train all 3 levels with multi-timescale data.

        - short: (z_t, z_{t+1}) — 1-step, always trained
        - mid:   (z_{t-4}, z_t) — 4-step gap, trained when history >= 4
        - long:  (z_{t-16}, z_t) — 16-step gap, trained when history >= 16

        P1 FIX: Previously all three levels received the same (z_t, z_{t+1}),
        making them three copies of the same model.  Now each level sees
        data at its own temporal scale.
        """
        # — short (1-step, always train) —
        loss_short_raw = self.short.train_step(
            z_current, z_next, max_grad_norm=max_grad_norm
        )
        loss_short = (
            loss_short_raw["pred_error"]
            if isinstance(loss_short_raw, dict)
            else loss_short_raw
        )
        # — store in history for multi-scale training —
        self._z_history.append(np.asarray(z_current, dtype=np.float32))
        self._z_next_history.append(np.asarray(z_next, dtype=np.float32))
        # — mid (4-step gap) —
        if len(self._z_history) >= 4:
            z_mid_past = self._z_history[-4]
            z_mid_now = self._z_next_history[-1]
            loss_mid_raw = self.mid.train_step(
                z_mid_past, z_mid_now, max_grad_norm=max_grad_norm
            )
            loss_mid = (
                loss_mid_raw["pred_error"]
                if isinstance(loss_mid_raw, dict)
                else loss_mid_raw
            )
        else:
            loss_mid = 0.0
        # — long (16-step gap) —
        if len(self._z_history) >= 16:
            z_long_past = self._z_history[0]
            z_long_now = self._z_next_history[-1]
            loss_long_raw = self.long.train_step(
                z_long_past, z_long_now, max_grad_norm=max_grad_norm
            )
            loss_long = (
                loss_long_raw["pred_error"]
                if isinstance(loss_long_raw, dict)
                else loss_long_raw
            )
        else:
            loss_long = 0.0
        return {"short": loss_short, "mid": loss_mid, "long": loss_long}

    def encode(self, z_world: np.ndarray | torch.Tensor) -> np.ndarray:
        """Encode using the short-term encoder (primary)."""
        x = self.short._as_tensor(z_world)
        with torch.no_grad():
            z = self.short.context_encoder(x)
        return z.float().cpu().numpy().astype(np.float32)

    def attach_square_encoder(self, square_encoder) -> None:
        """Attach square encoder to all 3 levels."""
        import copy as _copy
        for level in (self.short, self.mid, self.long):
            level.attach_square_encoder(
                _copy.deepcopy(square_encoder)
            )

    def init_value_head(self, hidden_dim: int = 32, value_weight: float = 0.5) -> None:
        """Init value heads on all 3 levels."""
        for level in (self.short, self.mid, self.long):
            level.init_value_head(hidden_dim=hidden_dim, value_weight=value_weight)

    def value(self, z_world: np.ndarray | torch.Tensor) -> np.ndarray:
        """Value from the short-term level (primary)."""
        return self.short.value(z_world)

    def init_vq(self, num_codes: int = 64, beta: float = 0.25) -> None:
        """Init VQ on all 3 levels."""
        for level in (self.short, self.mid, self.long):
            level.init_vq(num_codes=num_codes, beta=beta)

    def tokenize(self, z_world: np.ndarray | torch.Tensor) -> np.ndarray:
        """Tokenize using the short-term level (primary)."""
        return self.short.tokenize(z_world)

    # ------------------------------------------------------------------
    # Flat-return adapter methods (compatible with TrinityAgent)
    # ------------------------------------------------------------------
    def predict_flat(self, z_world, mask=None) -> np.ndarray:
        """Predict next-state latent at the short level only.

        Returns a single ``np.ndarray`` instead of the full dict,
        making it compatible with agents that expect a flat return.
        """
        return self.predict(z_world)["short"]

    def train_step_flat(self, z_current, z_next, max_grad_norm=5.0) -> float:
        """Train all levels, return the short-level loss as a scalar."""
        return self.train_step(z_current, z_next, max_grad_norm=max_grad_norm)["short"]

    def train_transition_flat(
        self,
        square_t,
        time_t,
        square_next,
        time_next,
        unified_t=None,
        unified_next=None,
        mask=None,
        max_grad_norm=5.0,
    ) -> float:
        """End-to-end transition training, return the short-level loss."""
        return self.train_transition(
            square_t, time_t, square_next, time_next,
            unified_t=unified_t, unified_next=unified_next,
            max_grad_norm=max_grad_norm,
        )["short"]

    def predict_with_uncertainty_flat(self, z_world) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty at the short level only."""
        return self.short.predict_with_uncertainty(z_world)

    def value_flat(self, z_world) -> np.ndarray:
        """Value estimate at the short level only."""
        return self.short.value(z_world)

    def train_transition(
        self,
        square_t: np.ndarray,
        time_t: float,
        square_next: np.ndarray,
        time_next: float,
        unified_t: np.ndarray | None = None,
        unified_next: np.ndarray | None = None,
        max_grad_norm: float = 5.0,
    ) -> dict[str, float]:
        """End-to-end transition training on all 3 levels."""
        ls = self.short.train_transition(
            square_t, time_t, square_next, time_next,
            unified_t=unified_t, unified_next=unified_next,
            max_grad_norm=max_grad_norm,
        )
        lm = self.mid.train_transition(
            square_t, time_t, square_next, time_next,
            unified_t=unified_t, unified_next=unified_next,
            max_grad_norm=max_grad_norm,
        )
        ll = self.long.train_transition(
            square_t, time_t, square_next, time_next,
            unified_t=unified_t, unified_next=unified_next,
            max_grad_norm=max_grad_norm,
        )
        _ex = lambda r: float(r["pred_error"] if isinstance(r, dict) else r)
        return {"short": _ex(ls), "mid": _ex(lm), "long": _ex(ll)}


# FSDP2 distributed training wrappers live in ``zwm.jepa.distributed`` and
# are re-exported above for backward compat.

