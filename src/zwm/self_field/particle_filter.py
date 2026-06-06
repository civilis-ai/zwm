"""P2-2: Particle-filtering active inference (PILCO 风格).

The 2025/2026 SOTA active-inference approximation is **deeply
probabilistic** — instead of evaluating a single deterministic EFE,
the agent maintains a *particle ensemble* of belief states and
estimates the EFE as the average across the ensemble.

Why this matters:
  * Analytical EFE (current) averages over a single belief — fragile
    under model mismatch.
  * Particle EFE = expected free energy under the posterior belief,
    approximated by Monte-Carlo sampling.

The implementation is deliberately simple:
  * ``ParticleBelief`` — N weighted particles in latent space.
  * ``ParticleFilter`` — predict/update with importance weights.
  * ``particle_efe`` — EFE averaged over the ensemble.

The agent calls ``particle_efe`` from its planner when ``n_particles > 0``;
otherwise the analytical EFE is used (legacy path).
"""
from __future__ import annotations

import numpy as np


class ParticleBelief:
    """Weighted particle ensemble in latent (z_world) space."""

    def __init__(self, particles: np.ndarray, weights: np.ndarray) -> None:
        # particles: [N, D] latent vectors
        # weights:  [N]    non-negative, sums to 1
        self.particles = np.asarray(particles, dtype=np.float32)
        w = np.asarray(weights, dtype=np.float32)
        s = w.sum()
        self.weights = w / s if s > 0 else np.full(len(w), 1.0 / len(w), dtype=np.float32)
        self.n = self.particles.shape[0]
        self.dim = self.particles.shape[1] if self.n > 0 else 0

    def resample(self) -> "ParticleBelief":
        """Systematic resample (low-variance)."""
        n = self.n
        if n == 0:
            return self
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # avoid fp drift
        positions = (np.arange(n, dtype=np.float32) + np.random.random()) / n
        idx = np.searchsorted(cumsum, positions)
        new_particles = self.particles[idx]
        return ParticleBelief(new_particles, np.full(n, 1.0 / n, dtype=np.float32))

    def ess(self) -> float:
        """Effective sample size — 1 / Σ wᵢ²."""
        return float(1.0 / max(np.sum(self.weights ** 2), 1e-12))

    def mean(self) -> np.ndarray:
        return np.average(self.particles, axis=0, weights=self.weights)


class ParticleFilter:
    """Simple particle filter for EFE-based planning.

    Each tick:
      * ``predict(transition_fn, noise_std)`` advances all particles through
        a deterministic transition (the JEPA world model) + Gaussian noise.
      * ``update(observation, observation_fn, obs_std)`` re-weights by the
        observation likelihood and resamples when ESS < N/2.
    """

    def __init__(
        self,
        n_particles: int = 16,
        dim: int = 32,
        noise_std: float = 0.05,
        obs_std: float = 0.1,
    ) -> None:
        self.n_particles = n_particles
        self.dim = dim
        self.noise_std = noise_std
        self.obs_std = obs_std
        self.belief = ParticleBelief(
            np.random.randn(n_particles, dim).astype(np.float32) * 0.1,
            np.full(n_particles, 1.0 / n_particles, dtype=np.float32),
        )

    def predict(self, transition_fn) -> None:
        """Apply ``transition_fn(z) -> z_next`` to each particle + noise."""
        new_particles = np.stack(
            [transition_fn(self.belief.particles[i]) for i in range(self.n_particles)],
            axis=0,
        )
        new_particles += np.random.randn(*new_particles.shape).astype(np.float32) * self.noise_std
        self.belief = ParticleBelief(new_particles, self.belief.weights)

    def update(
        self,
        observation: np.ndarray,
        observation_fn,
    ) -> None:
        """Re-weight by the observation likelihood. Resample if ESS low."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape[0] != self.dim:
            # Skip if observation dim doesn't match.
            return
        new_weights = np.zeros(self.n_particles, dtype=np.float32)
        for i in range(self.n_particles):
            pred_obs = observation_fn(self.belief.particles[i])
            diff = pred_obs - obs
            # Gaussian likelihood
            new_weights[i] = np.exp(
                -0.5 * float((diff ** 2).sum()) / (self.obs_std ** 2)
            )
        if new_weights.sum() < 1e-10:
            # Re-initialise if all weights are zero.
            self.belief = ParticleBelief(
                np.random.randn(self.n_particles, self.dim).astype(np.float32) * 0.1,
                np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float32),
            )
            return
        self.belief = ParticleBelief(self.belief.particles, new_weights)
        if self.belief.ess() < self.n_particles / 2.0:
            self.belief = self.belief.resample()


def particle_efe(
    belief: ParticleBelief,
    pragmatic_fn,
    epistemic_fn,
    intrinsic_fn=None,
) -> float:
    """EFE averaged over the particle ensemble.

    ``pragmatic_fn(z) -> float`` and ``epistemic_fn(z) -> float`` are
    called per particle; the result is the importance-weighted mean.
    The 2026 SOTA active-inference recipe averages EFE across the
    posterior belief rather than evaluating it at a single point.
    """
    if belief.n == 0:
        return 0.0
    pragmatic_vals = np.array(
        [pragmatic_fn(belief.particles[i]) for i in range(belief.n)],
        dtype=np.float32,
    )
    epistemic_vals = np.array(
        [epistemic_fn(belief.particles[i]) for i in range(belief.n)],
        dtype=np.float32,
    )
    if intrinsic_fn is not None:
        intrinsic_vals = np.array(
            [intrinsic_fn(belief.particles[i]) for i in range(belief.n)],
            dtype=np.float32,
        )
    else:
        intrinsic_vals = np.zeros(belief.n, dtype=np.float32)
    weights = belief.weights
    p = float(np.sum(weights * pragmatic_vals))
    e = float(np.sum(weights * epistemic_vals))
    i = float(np.sum(weights * intrinsic_vals))
    # P0 FIX: negate so particle_efe is a cost (lower = better),
    # consistent with the analytical EFE.  pragmatic_fn returns
    # value (negated EFE), so -(p + e + i) = EFE cost - epistemic
    # bonus — same semantics as expected_free_energy.
    return -(p + e + i)
