from __future__ import annotations

import math
from dataclasses import dataclass, field

from zwm.core.hexagram import Hexagram, hexagram_from_bits
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
    ) -> None:
        self._sampler = LangevinSampler()
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
    ) -> PlanResult:
        if grid is None:
            grid = LuoshuGrid()

        self._palace_visits = palace_visit_counts or {}

        # MoE evaluation — now influences scoring, optionally biased by the
        # agent's learned expert preferences.
        moe_score = self._moe.evaluate(
            h_current, grid,
            time_phase=time_phase,
            target_palace=target_palace or grid.self_position,
            preference_weights=preference_weights,
        )
        active = self._moe.active_experts(h_current, grid, time_phase)

        if self._use_mcts:
            scores = self._mcts_search(
                h_current, grid, time_phase, target_palace, day_gan,
                mask_priors=mask_priors,
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
    ) -> list[tuple[Hexagram, int, float]]:
        root = _MCTSNode(hex_bits=h_current.normal_order)
        root.untried_masks = self._ordered_masks(h_current, mask_priors)

        for _ in range(self._mcts_iterations):
            node = self._select(root)
            if node.untried_masks:
                node = self._expand(node)
            reward = self._simulate(node, grid, time_phase, target_palace, day_gan)
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
          2. then the Langevin score surface, best first.
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

        non_prior_desc = [m for m in langevin_desc if m not in seen]

        # Tail = highest priority. Lay down worst Langevin first, then priors in
        # reverse so priors[0] (top Hebbian/memory pick) is the very last element
        # and is therefore popped first.
        return list(reversed(non_prior_desc)) + list(reversed(priors))

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

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        if not node.untried_masks:
            return node
        mask = node.untried_masks.pop()
        child_bits = node.hex_bits ^ mask
        child = _MCTSNode(hex_bits=child_bits, mask=mask, parent=node)
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
    ) -> float:
        h = hexagram_from_bits(node.hex_bits)
        return self._efe_score(h, grid, time_phase, target_palace, day_gan)

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
    ) -> float:
        from zwm.self_field.efe import expected_free_energy

        return expected_free_energy(
            h=h,
            grid=grid,
            target_palace=target_palace or grid.self_position,
            visit_counts=self._visit_counts,
            total_visits=max(self._total_visits, 1),
            beta_curiosity=self._efe_beta,
            palace_visit_counts=self._palace_visits,
        )

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
