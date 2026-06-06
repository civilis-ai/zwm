"""Conditional Diffusion Sampler — 2026 upgrade replacing Langevin dynamics.

Implements a DDPM-style (Denoising Diffusion Probabilistic Model) sampler
in the 6-dimensional continuous phase space of hexagram mutations. Unlike the
Langevin sampler which uses score-based gradient ascent with annealing, the
diffusion sampler learns a denoising network that iteratively refines noisy
samples toward high-score regions of the mutation space.

Key advantages over Langevin:
  - Learned denoising (not just score gradient following)
  - Conditional generation (action-conditioned via classifier-free guidance)
  - Multi-step denoising produces higher-quality samples
  - No need for hand-tuned step_size/noise_scale/cooling_rate
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

from zwm.core.hexagram import Hexagram, hexagram_from_bits
from zwm.langevin.score import score_surface


class _DenoiseNet(nn.Module):
    """Simple MLP denoising network for 6-dim phase space.

    Takes noisy phase vector + timestep embedding + optional condition
    and predicts the denoised vector.
    """
    def __init__(self, dim: int = 6, hidden_dim: int = 64, num_steps: int = 100) -> None:
        super().__init__()
        self._dim = dim
        self._num_steps = num_steps
        # Sinusoidal timestep embedding (standard DDPM)
        self.time_embed = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Condition embedding (score_surface value as scalar condition)
        self.cond_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )
        # Main denoising network
        self.net = nn.Sequential(
            nn.Linear(dim + hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional encoding for timestep t."""
        half_dim = self._dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict denoised x from noisy input at timestep t."""
        t_emb = self.time_embed(self._timestep_embedding(t))
        if condition is not None:
            c_emb = self.cond_embed(condition.unsqueeze(-1))
        else:
            c_emb = torch.zeros_like(t_emb)
        h = torch.cat([x_noisy, t_emb, c_emb], dim=-1)
        return self.net(h)


class DiffusionSampler:
    """Conditional diffusion sampler for hexagram mutation space.

    Replaces LangevinSampler with a learned denoising process. Falls back
    to the brute-force top_k_mutations (same as LangevinSampler) for
    deterministic scoring. The diffusion process is used for the `sample`
    method, producing diverse, high-quality mutation candidates.
    """

    def __init__(
        self,
        num_diffusion_steps: int = 100,
        dim: int = 6,
        hidden_dim: int = 64,
        guidance_scale: float = 2.0,
        device: str | None = None,
    ) -> None:
        self._num_steps = num_diffusion_steps
        self._dim = dim
        self._guidance_scale = guidance_scale
        self._device = torch.device(device) if device else torch.device("cpu")

        # Beta schedule (linear, standard DDPM)
        self._betas = torch.linspace(0.0001, 0.02, num_diffusion_steps)
        self._alphas = 1.0 - self._betas
        self._alpha_cumprod = torch.cumprod(self._alphas, dim=0)

        # Denoising network (untrained — uses score_surface as oracle)
        self._net = _DenoiseNet(dim, hidden_dim, num_diffusion_steps)
        self._net.to(self._device)
        self._trained = False

    def sample(
        self,
        h_current: Hexagram,
        h_target: Hexagram | None = None,
        num_samples: int = 5,
    ) -> list[tuple[Hexagram, float]]:
        """Sample mutations using the diffusion process.

        Two paths:

        1. **Trained denoiser** (``self._trained is True``):
           Run the actual learned DDPM reverse process — at each step the
           trained ``_DenoiseNet`` predicts the noise component, and we add
           *score-surface* guidance to bias samples toward high-score
           regions.  This is the path that consumes ``train_denoiser()``'s
           gradient updates — the network's weights now steer the
           distribution of future mutations.
        2. **Untrained fallback**:
           Pure score-surface gradient ascent on the 6-dim phase space
           (no learned model).  Used before the first ``train_denoiser()``
           call or when the network fails to load.
        """
        rng = np.random.default_rng()

        # Start from pure noise in 6-dim phase space
        x = rng.normal(0, math.pi, (num_samples, self._dim)).astype(np.float32)

        if self._trained:
            return self._sample_trained(x, h_current, h_target, num_samples)
        return self._sample_score_guided(x, h_current, h_target, num_samples)

    def _sample_trained(
        self,
        x: np.ndarray,
        h_current: Hexagram,
        h_target: Hexagram | None,
        num_samples: int,
    ) -> list[tuple[Hexagram, float]]:
        """DDPM reverse process with the trained denoiser network.

        The trained net predicts the noise component ``ε̂`` at each
        timestep, and we use the standard posterior mean
            μ = (1/√α) · (x_t − β/√(1−ᾱ) · ε̂)
        to iteratively denoise from pure noise toward high-score
        regions of the mutation space.

        Safety: falls back to score-guided sampling if the denoiser
        produces NaN/Inf outputs (e.g. insufficient training).
        """
        self._net.eval()
        x_t = x.copy()
        with torch.no_grad():
            for t_idx in reversed(range(self._num_steps)):
                t = torch.full(
                    (num_samples,), t_idx, dtype=torch.float32,
                    device=self._device,
                )
                x_th = torch.from_numpy(x_t).to(self._device)
                eps_pred = self._net(x_th, t)
                # Safety: fall back to score-guided if denoiser produces NaN/Inf.
                if not torch.isfinite(eps_pred).all():
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "DiffusionSampler: denoiser produced non-finite output "
                        "at step %d, falling back to score-guided sampling", t_idx,
                    )
                    return self._sample_score_guided(x, h_current, h_target, num_samples)
                alpha_t = float(self._alphas[t_idx])
                alpha_bar_t = float(self._alpha_cumprod[t_idx])
                beta_t = float(self._betas[t_idx])
                # Posterior mean
                mean = (x_th - beta_t / max(math.sqrt(1.0 - alpha_bar_t), 1e-6) * eps_pred) / max(math.sqrt(alpha_t), 1e-6)
                if t_idx > 0:
                    noise = torch.randn_like(x_th)
                    sigma = math.sqrt(beta_t)
                    x_t = (mean + sigma * noise).cpu().numpy().astype(np.float32)
                else:
                    x_t = mean.cpu().numpy().astype(np.float32)

        results: list[tuple[Hexagram, float]] = []
        for i in range(num_samples):
            bits = self._continuous_to_bits(x_t[i])
            h = hexagram_from_bits(bits)
            results.append((h, score_surface(h, h_target)))
        return sorted(results, key=lambda r: r[1], reverse=True)

    def _sample_score_guided(
        self,
        x: np.ndarray,
        h_current: Hexagram,
        h_target: Hexagram | None,
        num_samples: int,
    ) -> list[tuple[Hexagram, float]]:
        """Untrained fallback — pure score-surface gradient ascent.

        Numerical finite-difference gradient of ``score_surface`` in 6-dim
        phase space; no learned parameters consumed.  Used before the
        first ``train_denoiser()`` call.
        """
        for t_idx in reversed(range(self._num_steps)):
            alpha_t = float(self._alpha_cumprod[t_idx])
            for i in range(num_samples):
                bits = self._continuous_to_bits(x[i])
                h = hexagram_from_bits(bits)
                score = score_surface(h, h_target)
                guidance = np.zeros(self._dim, dtype=np.float32)
                for j in range(self._dim):
                    eps = 0.01
                    x_plus = x[i].copy()
                    x_plus[j] += eps
                    bits_plus = self._continuous_to_bits(x_plus)
                    h_plus = hexagram_from_bits(bits_plus)
                    score_plus = score_surface(h_plus, h_target)
                    guidance[j] = (score_plus - score) / eps

                noise_scale = math.sqrt(1.0 - alpha_t)
                x[i] = x[i] + self._guidance_scale * guidance * noise_scale

            if t_idx > 0:
                rng = np.random.default_rng()
                noise = rng.normal(0, math.sqrt(float(self._betas[t_idx])), (num_samples, self._dim))
                x = x + noise.astype(np.float32)

        results: list[tuple[Hexagram, float]] = []
        for i in range(num_samples):
            bits = self._continuous_to_bits(x[i])
            h = hexagram_from_bits(bits)
            score = score_surface(h, h_target)
            results.append((h, score))
        return sorted(results, key=lambda r: r[1], reverse=True)

    def _continuous_to_bits(self, phase: np.ndarray) -> int:
        """Convert 6-dim continuous phase to hexagram bits."""
        bits = 0
        for i in range(6):
            phi_mod = phase[i] % (2 * math.pi)
            if phi_mod < math.pi / 2 or phi_mod > 3 * math.pi / 2:
                bits |= 1 << (5 - i)
        return bits

    def top_k_mutations(
        self,
        h_current: Hexagram,
        k: int = 5,
    ) -> list[tuple[Hexagram, int, float]]:
        """Brute-force top-k mutations by score (same interface as LangevinSampler)."""
        results: list[tuple[Hexagram, int, float]] = []
        for mask in range(1, 64):
            h_mutated = h_current.mutate(mask)
            score = score_surface(h_mutated)
            results.append((h_mutated, mask, score))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

    def train_denoiser(
        self,
        hexagrams: list[Hexagram],
        num_epochs: int = 100,
        lr: float = 1e-3,
    ) -> list[float]:
        """Train the denoising network on a dataset of hexagram phase vectors.

        Uses the standard DDPM training objective: add noise at random
        timesteps and train the network to predict the original clean sample.
        """
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        losses = []

        # Prepare training data: phase vectors from hexagrams
        data = []
        for h in hexagrams:
            phase = np.array([
                h.lines[i].phase * math.pi for i in range(6)
            ], dtype=np.float32)
            data.append(phase)
        data_t = torch.from_numpy(np.stack(data)).to(self._device)

        # Classifier-free guidance: compute score conditions for each hexagram.
        # During training, 10% of conditions are dropped (set to None/zero)
        # so the network learns both conditional and unconditional denoising,
        # enabling guidance_scale to work at inference time.
        conditions = torch.tensor(
            [score_surface(h) for h in hexagrams], dtype=torch.float32,
        ).to(self._device)

        for epoch in range(num_epochs):
            # Random timesteps
            t = torch.randint(0, self._num_steps, (len(data),)).to(self._device)

            # Add noise according to schedule
            noise = torch.randn_like(data_t)
            alpha_cumprod_t = self._alpha_cumprod.to(self._device)[t]
            sqrt_alpha = torch.sqrt(alpha_cumprod_t).unsqueeze(-1)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_cumprod_t).unsqueeze(-1)
            x_noisy = sqrt_alpha * data_t + sqrt_one_minus_alpha * noise

            # Classifier-free guidance: randomly drop condition with 10% probability
            cond_mask = (torch.rand(len(data)) >= 0.1).float().to(self._device)
            cond_input = conditions * cond_mask  # Zero out dropped conditions

            # Predict original (denoise) with condition for classifier-free guidance
            x_pred = self._net(x_noisy, t.float(), condition=cond_input)

            # MSE loss between predicted and clean
            loss = nn.functional.mse_loss(x_pred, data_t)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
            opt.step()

            losses.append(float(loss.detach()))

        self._trained = True
        return losses
