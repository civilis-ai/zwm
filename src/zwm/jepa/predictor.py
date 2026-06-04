from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_INPUT_DIM = 77  # 64 (square GNN) + 13 (circular phase)


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        hidden_dim: int = 64,
        latent_dim: int = 32,
        learning_rate: float = 1e-3,
        ema_decay: float = 0.99,
        vicreg_weight: float = 0.04,
        replay_capacity: int = 256,
        batch_size: int = 16,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self._ema_decay = ema_decay
        self._vicreg_weight = vicreg_weight
        # Experience replay: VICReg needs a batch with cross-sample variance to
        # produce a real anti-collapse gradient (a single sample's variance is
        # 0 and contributes no signal). Replaying recent transitions in a
        # minibatch keeps the variance/covariance terms live and is also more
        # sample-efficient.
        self._replay: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
            maxlen=replay_capacity
        )
        self._batch_size = batch_size

        self.context_encoder = _Encoder(input_dim, hidden_dim, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # EMA target encoder — a frozen, slowly-updated copy of the online
        # context encoder. Never receives gradients.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self._opt = torch.optim.Adam(
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters()),
            lr=learning_rate,
        )

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(x)

    def predict(self, z_world: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict the next-state latent from the current world tensor."""
        x = self._as_tensor(z_world)
        with torch.no_grad():
            z = self.context_encoder(x)
            z_pred = self.predictor(z)
        return z_pred.numpy().astype(np.float32)

    def target_latent(self, x_next: np.ndarray | torch.Tensor) -> np.ndarray:
        """EMA-target embedding of a (next) state — the prediction target."""
        x = self._as_tensor(x_next)
        with torch.no_grad():
            z = self.target_encoder(x)
        return z.numpy().astype(np.float32)

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

    def train_step(
        self,
        z_current: np.ndarray | torch.Tensor,
        z_next: np.ndarray | torch.Tensor,
        max_grad_norm: float = 5.0,
    ) -> float:
        """One real gradient step on a replayed minibatch. Returns the loss.

        Stores the (xₜ, xₜ₊₁) transition in the replay buffer, samples a
        minibatch, and predicts each EMA-target next embedding from the online
        current embedding. The loss is prediction error in representation space
        plus a batched VICReg term (active because the batch has cross-sample
        variance). Updates the EMA target afterwards. Guards against NaN/Inf.
        """
        x_t = self._as_tensor(z_current).detach()
        x_next = self._as_tensor(z_next).detach()
        if not (torch.isfinite(x_t).all() and torch.isfinite(x_next).all()):
            return float("nan")
        self._replay.append((x_t, x_next))

        n = len(self._replay)
        k = min(self._batch_size, n)
        idx = torch.randint(0, n, (k,))
        xb = torch.stack([self._replay[int(i)][0] for i in idx])
        xnb = torch.stack([self._replay[int(i)][1] for i in idx])

        z_t = self.context_encoder(xb)
        z_pred = self.predictor(z_t)

        with torch.no_grad():
            z_target = self.target_encoder(xnb)

        pred_loss = F.mse_loss(z_pred, z_target)
        vic = self.vicreg_loss(z_t)
        loss = pred_loss + self._vicreg_weight * vic

        if not torch.isfinite(loss):
            return float("nan")

        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.context_encoder.parameters())
            + list(self.predictor.parameters()),
            max_norm=max_grad_norm,
        )
        self._opt.step()
        self._update_target_encoder()
        return float(loss.detach())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_target_encoder(self) -> None:
        d = self._ema_decay
        for tgt, src in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            tgt.mul_(d).add_(src, alpha=1.0 - d)

    @staticmethod
    def _as_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(np.asarray(x, dtype=np.float32))
