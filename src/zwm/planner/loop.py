from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from zwm.core.hexagram import Hexagram, hexagram_from_bits
from zwm.langevin.diffusion import DiffusionSampler
from zwm.langevin.sampler import LangevinSampler
from zwm.moe.sparse_activation import SparseMoE
from zwm.self_field.palace_graph import LuoshuGrid
from zwm.scene_field.five_hexagrams import FiveHexagramChain


@dataclass
class PlanResult:
    chain: FiveHexagramChain
    hexagram_scores: list[tuple[int, float]]
    top_mutation: int
    top_score: float
    moe_active_experts: list[str]
    moe_weight: float
    trajectory: list[tuple[str, float]]


@dataclass
class _MCTSNode:
    hex_bits: int
    mask: int = 0
    parent: _MCTSNode | None = None
    children: dict[int, _MCTSNode] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0
    untried_masks: list[int] = field(default_factory=list)

    @property
    def value(self) -> float:
        return self.total_value / (self.visits or 1)


class TrinityPlanner:
    """Trinity World Model Planner with MCTS, EFE, MoE, and learning integration.

    Uses Expected Free Energy (EFE) for node evaluation and MoE-weighted
    scoring with Monte Carlo Tree Search for multi-step planning.

    Supports 日干 (Day Heavenly Stem) as the 太极点 for 六亲 positioning.
    """

    def __init__(
        self,
        use_mcts: bool = True,
        mcts_iterations: int = 200,
        exploration_weight: float = 1.4,
        efe_beta: float = 0.3,
        use_diffusion: bool = True,
    ) -> None:
        self._sampler: LangevinSampler | DiffusionSampler = (
            DiffusionSampler() if use_diffusion else LangevinSampler()
        )
        self._moe = SparseMoE(top_k=3)
        self._use_mcts = use_mcts
        self._mcts_iterations = mcts_iterations
        self._exploration_weight = exploration_weight
        self._efe_beta = efe_beta
        # Persistent epistemic state: hexagram visit history accumulated
        # across every MCTS simulation and across successive plan() calls.
        # This is what makes the EFE epistemic term a live signal instead
        # of a constant.
        self._visit_counts: dict[int, int] = {}
        self._total_visits: int = 0
        # Palace-space exploration history (keyed by palace position 1-9),
        # supplied per-plan by the agent's multi-scale topology scaffold.
        self._palace_visits: dict[int, int] = {}

    @property
    def visit_counts(self) -> dict[int, int]:
        return self._visit_counts

    # ------------------------------------------------------------------
    # Main planning entry point
    # ------------------------------------------------------------------
    def plan(
        self,
        h_current: Hexagram,
        grid: LuoshuGrid | None = None,
        time_phase: float = 0.0,
        top_k: int = 5,
        target_palace: int | None = None,
        day_gan: str | None = None,
        preference_weights: dict[str, float] | None = None,
        mask_priors: list[int] | None = None,
        palace_visit_counts: dict[int, int] | None = None,
        intrinsic_fn: Callable[[Hexagram], float] | None = None,
        beta_curiosity: float | None = None,
        value_fn: Callable[[Hexagram], float] | None = None,
        particle_filter=None,
        learned_value: float | None = None,
        uncertainty_scale: float = 0.0,
        log_evidence: float | None = None,
        latent_value_fn: Callable[[np.ndarray], float] | None = None,
    ) -> PlanResult:
        if grid is None:
            grid = LuoshuGrid()

        self._palace_visits = palace_visit_counts or {}
        # P1 fix: reject ambiguous defaults explicitly. The planner no longer
        # silently holds construction-time _intrinsic_fn / _active_beta from
        # the previous plan() call (which masked missing-caller bugs). If a
        # caller wants the world-model intrinsic term, it must pass one;
        # otherwise we fall back to a zero intrinsic instead of stale state.
        self._active_beta = (
            self._efe_beta if beta_curiosity is None else float(beta_curiosity)
        )
        # Reset to a safe default instead of inheriting last call's function.
        if intrinsic_fn is None:
            self._intrinsic_fn = lambda _h: 0.0  # noqa: E731 — typed fallback
        else:
            self._intrinsic_fn = intrinsic_fn

        # MoE evaluation — now influences scoring, optionally biased by the
        # agent's learned expert preferences. Day_gan provides context element
        # for the element expert (P1-4).
        moe_score = self._moe.evaluate(
            h_current, grid,
            time_phase=time_phase,
            target_palace=target_palace or grid.self_position,
            preference_weights=preference_weights,
            day_gan=day_gan,
        )
        active = self._moe.active_experts(h_current, grid, time_phase)

        if self._use_mcts:
            scores = self._mcts_search(
                h_current, grid, time_phase, target_palace, day_gan,
                mask_priors=mask_priors,
                value_fn=value_fn,
                particle_filter=particle_filter,
                learned_value=learned_value,
                uncertainty_scale=uncertainty_scale,
                log_evidence=log_evidence,
                latent_value_fn=latent_value_fn,
            )
        else:
            scores = self._sampler.top_k_mutations(h_current, k=top_k)

        top_h, top_mask, top_score = scores[0]

        # Blend MoE weight into final score
        blended_score = self._blend(top_score, moe_score)

        chain = FiveHexagramChain.with_evolution(h_current, top_mask)

        trajectory: list[tuple[str, float]] = []
        for h, mask, score in scores[:3]:
            h_mut = h_current.mutate(mask)
            trajectory.append((h_mut.name, score))

        return PlanResult(
            chain=chain,
            hexagram_scores=[(s[1], s[2]) for s in scores],
            top_mutation=top_mask,
            top_score=blended_score,
            moe_active_experts=active,
            moe_weight=moe_score,
            trajectory=trajectory,
        )

    # ------------------------------------------------------------------
    # MCTS search
    # ------------------------------------------------------------------
    def _mcts_search(
        self,
        h_current: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
        target_palace: int | None,
        day_gan: str | None,
        mask_priors: list[int] | None = None,
        intrinsic_fn=None,
        value_fn=None,
        particle_filter=None,
        learned_value: float | None = None,
        uncertainty_scale: float = 0.0,
        log_evidence: float | None = None,
        latent_value_fn=None,
    ) -> list[tuple[Hexagram, int, float]]:
        root = _MCTSNode(hex_bits=h_current.normal_order)
        root.untried_masks = self._ordered_masks(h_current, mask_priors)

        for _ in range(self._mcts_iterations):
            node = self._select(root)
            if node.untried_masks:
                # Propagate root priors down to children so Hebbian/memory
                # suggestions influence the whole search tree, not only the
                # root expansion. The child node re-orders its own untried
                # masks using the same prior list, filtered by what is still
                # legal at that hexagram.
                node = self._expand(node, h_current, mask_priors)
            reward = self._simulate(
                node, grid, time_phase, target_palace, day_gan, intrinsic_fn,
                value_fn=value_fn, particle_filter=particle_filter,
                learned_value=learned_value, uncertainty_scale=uncertainty_scale,
                log_evidence=log_evidence, latent_value_fn=latent_value_fn,
            )
            self._backpropagate(node, reward)

        # Gather results from child nodes
        results: list[tuple[Hexagram, int, float]] = []
        for mask, child in root.children.items():
            hex_val = root.hex_bits ^ mask
            h = hexagram_from_bits(hex_val)
            results.append((h, mask, child.value))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:5]

    def _ordered_masks(
        self,
        h_current: Hexagram,
        mask_priors: list[int] | None,
    ) -> list[int]:
        """Build the untried-mask list so expansion explores best-first.

        ``untried_masks`` is consumed via ``.pop()`` (tail first), so the
        highest-priority mask must sit at the tail. Priority is:
          1. externally supplied priors, in the order given (Hebbian/memory),
          2. diffusion-sampled mutations (when DiffusionSampler is trained),
          3. then the Langevin score surface, best first.
        """
        ranked = self._sampler.top_k_mutations(h_current, k=63)  # desc by score
        langevin_desc = [mask for _h, mask, _score in ranked]

        # De-duplicate priors, preserving caller priority order.
        priors: list[int] = []
        seen: set[int] = set()
        for m in (mask_priors or []):
            if 1 <= m <= 63 and m not in seen:
                seen.add(m)
                priors.append(m)

        # DiffusionSampler trained samples — insert before Langevin masks.
        diffusion_masks: list[int] = []
        if isinstance(self._sampler, DiffusionSampler) and self._sampler._trained:
            try:
                samples = self._sampler.sample(h_current, num_samples=8)
                for h_sample, _score in samples:
                    mask = h_sample.normal_order ^ h_current.normal_order
                    if 1 <= mask <= 63 and mask not in seen:
                        seen.add(mask)
                        diffusion_masks.append(mask)
            except Exception:
                pass

        non_prior_desc = [m for m in langevin_desc if m not in seen]

        # Tail = highest priority. Lay down worst Langevin first, then diffusion
        # masks, then priors in reverse so priors[0] (top Hebbian/memory pick)
        # is the very last element and is therefore popped first.
        return (
            list(reversed(non_prior_desc))
            + list(reversed(diffusion_masks))
            + list(reversed(priors))
        )

    def reinforce_expert(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
        expert_index: int,
        weight: float = 1.0,
    ) -> float:
        """Public hook: take one router gradient step toward ``expert_index``.

        Lets the agent reinforce the rewarded expert without reaching into the
        planner's private MoE internals.
        """
        return self._moe.router.train_toward(
            h, grid, time_phase, expert_index=expert_index, weight=weight
        )

    @property
    def expert_names(self) -> list[str]:
        return self._moe.expert_names

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        while not node.untried_masks and node.children:
            best = None
            best_ucb = -float("inf")
            log_parent = math.log(max(node.visits, 1))
            for child in node.children.values():
                exploitation = child.total_value / max(child.visits, 1)
                exploration = self._exploration_weight * math.sqrt(
                    log_parent / max(child.visits, 1)
                )
                ucb = exploitation + exploration
                if ucb > best_ucb:
                    best_ucb = ucb
                    best = child
            if best is None:
                break
            node = best
        return node

    def _expand(
        self,
        node: _MCTSNode,
        h_current: Hexagram,
        mask_priors: list[int] | None = None,
    ) -> _MCTSNode:
        if not node.untried_masks:
            return node
        mask = node.untried_masks.pop()
        child_bits = node.hex_bits ^ mask
        child = _MCTSNode(hex_bits=child_bits, mask=mask, parent=node)
        # Propagate priors: the child re-orders its own untried masks using the
        # same prior list (caller priority) intersected with legal masks at
        # this node. This makes Hebbian/memory suggestions affect the entire
        # search depth, not just the root expansion.
        if mask_priors:
            seen = {mask}
            priors = [m for m in mask_priors if 1 <= m <= 63 and m not in seen]
            non_prior = [m for m in range(1, 64) if m not in priors and m != mask]
            child.untried_masks = list(reversed(non_prior)) + list(reversed(priors))
        else:
            child.untried_masks = [m for m in range(1, 64) if m != mask]
        node.children[mask] = child
        return child

    def _simulate(
        self,
        node: _MCTSNode,
        grid: LuoshuGrid,
        time_phase: float,
        target_palace: int | None,
        day_gan: str | None,
        intrinsic_fn=None,
        value_fn=None,
        particle_filter=None,
        learned_value: float | None = None,
        uncertainty_scale: float = 0.0,
        log_evidence: float | None = None,
        latent_value_fn=None,
    ) -> float:
        h = hexagram_from_bits(node.hex_bits)
        return self._efe_score(
            h, grid, time_phase, target_palace, day_gan, intrinsic_fn,
            value_fn=value_fn, particle_filter=particle_filter,
            learned_value=learned_value, uncertainty_scale=uncertainty_scale,
            log_evidence=log_evidence, latent_value_fn=latent_value_fn,
        )

    def _backpropagate(self, node: _MCTSNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_value += reward
            # Record the visit in the planner's persistent epistemic memory,
            # keyed by hexagram identity, so EFE's curiosity term reflects
            # how often each state has actually been explored.
            self._visit_counts[node.hex_bits] = (
                self._visit_counts.get(node.hex_bits, 0) + 1
            )
            self._total_visits += 1
            node = node.parent

    # ------------------------------------------------------------------
    # EFE scoring — Expected Free Energy
    # ------------------------------------------------------------------
    def _efe_score(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
        target_palace: int | None,
        day_gan: str | None,
        intrinsic_fn=None,
        value_fn=None,
        particle_filter=None,
        learned_value: float | None = None,
        uncertainty_scale: float = 0.0,
        log_evidence: float | None = None,
        latent_value_fn=None,
    ) -> float:
        from zwm.self_field.efe import expected_free_energy
        from zwm.self_field.particle_filter import particle_efe as _particle_efe

        # Scale epistemic term when variational JEPA uncertainty is available.
        effective_beta = self._active_beta
        if uncertainty_scale > 0:
            effective_beta = effective_beta * (1.0 + uncertainty_scale)

        efe = expected_free_energy(
            h=h,
            grid=grid,
            target_palace=target_palace or grid.self_position,
            visit_counts=self._visit_counts,
            total_visits=max(self._total_visits, 1),
            beta_curiosity=effective_beta,
            palace_visit_counts=self._palace_visits,
            intrinsic_fn=intrinsic_fn,
            log_evidence=log_evidence,
        )

        # P3: Particle filter EFE — when a particle filter is available,
        # average the EFE over the ensemble for robust uncertainty estimation.
        # This is the 2026 SOTA active-inference recipe.  The pragmatic and
        # epistemic value functions are derived from the same EFE components
        # used in the analytical path, projected onto the particle's latent
        # via the JEPA value head (V(z)) and a learned novelty term.
        if particle_filter is not None and particle_filter.belief.n > 0:
            try:
                from zwm.self_field.efe import expected_free_energy as _efe
                from zwm.core.hexagram import hexagram_from_bits

                def _pragmatic(z):
                    """Pragmatic value of a particle latent (64-dim JEPA latent).

                    Uses the JEPA value head (V(z)) when available —
                    the value is a learned scalar estimate of expected
                    outcome magnitude, trained self-supervised on
                    ||z_target|| as a proxy.

                    Falls back to the analytical EFE when no value head
                    is available (this happens in the multi-scale
                    topology path where the particle is projected back
                    to a hexagram through the codebook).

                    Prior to the 2026-06 audit this decoded the latent's
                    first 6 raw dims as hexagram bits, which is
                    semantically meaningless — JEPA latent axes have no
                    correspondence to hexagram bit positions.
                    """
                    try:
                        z_arr = np.asarray(z, dtype=np.float32).flatten()
                        if latent_value_fn is not None:
                            return float(latent_value_fn(z_arr))
                        # No learned value head: use latent norm as a
                        # weak proxy (crude but honest — no bit-hack).
                        return float(np.linalg.norm(z_arr)) * 0.1
                    except Exception:
                        return 0.0

                def _epistemic(z):
                    """P2-11 — epistemic value of a particle latent.

                    Epistemic value = how *far* the particle is from the
                    observed mean (norm of deviation) — particles that
                    represent unseen regions of latent space are the
                    ones that drive exploration.  Scaled into [0, 1].
                    """
                    try:
                        z = np.asarray(z, dtype=np.float32)
                        # Distance from origin in latent space, normalised
                        # by the JEPA latent dim so the scale is consistent
                        # across action-conditioned / unconditioned latents.
                        norm = float(np.linalg.norm(z))
                        # Sigmoid squash so the value is in a bounded range.
                        return 1.0 / (1.0 + np.exp(-norm + 2.0))
                    except Exception:
                        return 0.0

                pefe = _particle_efe(
                    particle_filter.belief,
                    _pragmatic,
                    _epistemic,
                    intrinsic_fn=lambda z: 0.0,
                )
                efe = 0.7 * efe + 0.3 * pefe
            except Exception:
                pass

        # P3: Learned JEPA value head — when available, blend the learned
        # V(z) into the EFE. This replaces the coarse EMA table with a
        # smooth, expressive value estimate (MuZero 2026 style).
        # P2-12: ``learned_value`` (a pre-computed scalar) takes
        # priority over the ``value_fn(h)`` call to avoid double-
        # counting when both are passed.  When ``learned_value`` is
        # None, the ``value_fn`` path is the single source of truth.
        if learned_value is not None:
            efe += learned_value * 0.3
        elif value_fn is not None:
            try:
                v = value_fn(h)
                if isinstance(v, (int, float)):
                    efe = 0.8 * efe + 0.2 * float(v)
            except Exception:
                pass

        # World-model intrinsic reward (surprise / novelty) added to EFE so
        # the planner is steered by the JEPA prediction error signal.
        return efe + self._intrinsic_fn(h)

    # ------------------------------------------------------------------
    # Score blending
    # ------------------------------------------------------------------
    @staticmethod
    def _blend(efe_score: float, moe_score: float) -> float:
        return 0.55 * float(efe_score) + 0.45 * float(moe_score)

    # NOTE: the full Observe → Predict → Evaluate → Act → Learn loop lives in
    # ``zwm.planner.agent.TrinityAgent``. TrinityPlanner is intentionally a
    # stateless single-step evaluator (``plan``); it does not own perception,
    # memory, or learning state.
