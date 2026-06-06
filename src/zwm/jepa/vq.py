"""P2-3: VQ-VAE discretisation for the JEPA latent.

The 2025/2026 self-supervised recipe for video and language is
**tokenised** representations — VQ-VAE, VQ-KD, RVQ. A discrete code
makes the world model composable and lets the planner reason in
terms of "code transitions" rather than continuous vectors.

This module provides a small VQ layer that turns a continuous latent
[batch, D] into a finite codebook of size K and returns a
straight-through gradient estimator (the standard VQ-VAE trick).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQCodebook(nn.Module):
    """Vector-quantised codebook of K codes, each of dimension D.

    Forward:
        z_e: [B, D] continuous embeddings
    Returns:
        z_q: [B, D] quantised embeddings (with straight-through grad)
        indices: [B] integer code indices
        vq_loss: scalar commitment + codebook loss (added to outer loss)
    """

    def __init__(self, num_codes: int = 64, dim: int = 32, beta: float = 0.25) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        # K-means initialisation: random unit vectors.
        codebook = torch.randn(num_codes, dim)
        codebook = F.normalize(codebook, dim=-1)
        self.codebook = nn.Parameter(codebook)

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B, D]
        # Distance to each code: ||z_e - e_k||^2
        flat = z_e.reshape(-1, self.dim)
        # Compute distances
        d = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )
        indices = d.argmin(dim=1)
        z_q_flat = self.codebook[indices]
        # Straight-through estimator.
        z_q = flat + (z_q_flat - flat).detach()
        z_q = z_q.reshape(z_e.shape)
        # VQ loss: codebook + commitment.
        codebook_loss = F.mse_loss(z_q_flat, flat.detach())
        commitment_loss = F.mse_loss(flat, z_q_flat.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss
        return z_q, indices, vq_loss
