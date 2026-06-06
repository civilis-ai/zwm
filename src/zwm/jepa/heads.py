"""P2-X (audit) — auxiliary heads and regularisation primitives for JEPA.

Split from ``predictor.py`` to reduce its 1300+ line monolith.  This module
holds the *small reusable building blocks* that the main ``JEPAPredictor``
composes — encoders, value / variational / energy heads, action embeddings,
prior expert, and the SIGReg regularisation loss.  Each piece is small,
independently testable, and conceptually separate from the training loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from zwm.core.constants import ACTION_EMBED_DIM as _DEFAULT_ACTION_EMBED_DIM


# ======================================================================
# 2026 P3 cohort — frontier upgrades
# ======================================================================
class _EnergyHead(nn.Module):
    """P3-F — Energy-Based Model scalar head.

    Replaces the variational head with a single scalar energy
    ``E(z) ∈ R``.  Lower energy ↔ higher plausibility.  The 2026
    LeCun position (``JEPA + EBM``) emphasises that EBMs *do not
    require a softmax over alternatives* and can therefore represent
    *multimodal* futures — e.g. several equally-valid next hexagrams.
    Used as an auxiliary loss against the prediction L2 error.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _ActionEmbedding(nn.Module):
    """P3-A — Action (mutation-mask) embedding.

    Encodes the integer mutation mask ``m ∈ {0..63}`` as a dense
    32-dim vector.  Concatenated with the latent ``z_t`` before the
    predictor forward pass, turning the world model into a
    *controllable* transition model ``z_{t+1} = f_θ(z_t, a_t)``.
    This is the 2026 SOTA (V-JEPA 2-AC, LeWorldModel) — without
    action conditioning, latent-space MPC is impossible.
    """

    def __init__(
        self,
        num_masks: int = 64,
        embed_dim: int = _DEFAULT_ACTION_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_masks, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.05)
        self._dim = embed_dim

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.embed(mask)


class _PriorExpert(nn.Module):
    """P3-C — Structural prior expert for Bayesian-JEPA (PoE).

    A second small MLP that maps a latent to a "plausible"
    latent, but trained on a *prior* signal (e.g. I-Ching symmetry
    rules — trigram inversions, element cycles, palace homologies).
    Its output is combined with the dynamics expert via a Product
    of Experts (PoE), allowing domain-knowledge injection without
    retraining the dynamics model.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _VariationalHead(nn.Module):
    """Variational prediction head: outputs mean and log-variance.

    Replaces the predictor's last linear layer when ``variational=True``.
    The latent distribution is modelled as N(mu, sigma^2) where mu and
    log(sigma^2) are produced by two parallel linear projections.
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fc_mu(x), self.fc_logvar(x)


class _NullContext:
    """Tiny context-manager that does nothing — used as the no-op AMP fallback."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Encoder(nn.Module):
    """Context / target encoder used inside the JEPA predictor.

    P2 FIX: Added optional Transformer backbone (use_transformer=True).
    When enabled, the input is chunked into tokens, processed through a
    2-layer Transformer encoder with RoPE-style positional encoding, then
    pooled and projected to latent_dim.  This replaces the 2022-era pure
    MLP with a 2026 SOTA architecture (Transformer + Pre-LN + GELU).
    When use_transformer=False, the original MLP path is preserved for
    backward compatibility.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        use_transformer: bool = True,
        num_heads: int = 4,
        num_layers: int = 2,
        num_chunks: int = 8,
    ) -> None:
        super().__init__()
        self._use_transformer = use_transformer
        if use_transformer:
            self._num_chunks = num_chunks
            # Round up chunk_size so num_chunks * chunk_size >= input_dim
            self._chunk_size = (input_dim + num_chunks - 1) // num_chunks
            self._actual_input_dim = self._chunk_size * num_chunks
            # Project each chunk to hidden_dim
            self.chunk_proj = nn.Linear(self._chunk_size, hidden_dim)
            # Learnable positional encoding
            self.pos_embed = nn.Parameter(
                torch.randn(1, num_chunks, hidden_dim) * 0.02
            )
            # Pre-LN Transformer encoder (2026 SOTA: norm_first=True)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers,
                enable_nested_tensor=False,  # Pre-LN incompatible with nested tensors
            )
            # Pool and project to latent_dim
            self.pool_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, latent_dim),
            )
            # For __len__ compatibility
            self.net = self.transformer
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_transformer:
            return self.net(x)
        # Transformer path: chunk input, process as sequence, pool
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Move input to the same device as model parameters
        target_device = self.chunk_proj.weight.device
        if x.device != target_device:
            x = x.to(target_device)
        b = x.shape[0]
        # Pad if needed
        if x.shape[-1] < self._actual_input_dim:
            x = F.pad(x, (0, self._actual_input_dim - x.shape[-1]))
        # Chunk: [b, input_dim] → [b, num_chunks, chunk_size]
        x = x.view(b, self._num_chunks, self._chunk_size)
        # Project chunks: [b, num_chunks, hidden_dim]
        x = self.chunk_proj(x)
        # Add positional encoding
        x = x + self.pos_embed
        # Transformer: [b, num_chunks, hidden_dim]
        x = self.transformer(x)
        # Mean pool + project: [b, latent_dim]
        x = x.mean(dim=1)
        x = self.pool_proj(x)
        return x

    # torch.compile's dynamo wraps the module and, in the default
    # inductor backend, calls ``len(self._orig_mod)`` to introspect the
    # layer count. ``nn.Sequential`` provides this for free, but our
    # custom ``_Encoder`` does not, so we add a forwarder.
    def __len__(self) -> int:
        if hasattr(self, "net"):
            return len(self.net)
        return 4  # chunk_proj + transformer + pool_proj layers


def sigreg_loss(
    z: torch.Tensor,
    num_slices: int = 256,
    sigma: float = 1.0,
) -> torch.Tensor:
    """P3-D — Slice-Induced Gaussian Regularisation (SIGReg, LeWM 2026).

    Replaces VICReg's variance+covariance terms with a single
    distribution-matching loss:  take random 1-D projections of
    ``z``, sort them, and compare the empirical CDF to the standard
    Gaussian CDF.  The 2026 LeWorldModel paper shows this collapses
    the loss surface into a single convex minimum and reduces the
    number of loss terms from 3 (VICReg: var+cov+inv) to 1.
    Falls back to a variance hinge when the batch is a single
    sample (matches VICReg's single-sample behaviour).
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    b, d = z.shape
    if b < 2:
        # Degraded: 1 sample → no cross-sample variance, fall back
        # to a unit-norm hinge so the gradient keeps the latent
        # spread above 1.0 std.
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(1.0 - std).mean()
    device = z.device
    dtype = z.dtype
    slices = torch.randn(num_slices, d, device=device, dtype=dtype)
    proj = z @ slices.t()  # [b, num_slices]
    proj_sorted, _ = torch.sort(proj, dim=0)
    # CDF of the standard normal at the sorted projected values
    normal = torch.distributions.Normal(0.0, sigma)
    cdf_vals = normal.cdf(proj_sorted / sigma)
    # Empirical CDF: i / b for the i-th sorted element
    emp = torch.arange(1, b + 1, device=device, dtype=dtype) / b
    # Mean squared distance between the two CDFs.
    return ((cdf_vals - emp.unsqueeze(1)) ** 2).mean()


__all__ = [
    "_DEFAULT_ACTION_EMBED_DIM",
    "_ActionEmbedding",
    "_Encoder",
    "_EnergyHead",
    "_NullContext",
    "_PriorExpert",
    "_VariationalHead",
    "sigreg_loss",
]
