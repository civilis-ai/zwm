from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from zwm.core.hexagram import Hexagram


@dataclass
class OnlineLearner:
    """Online preference learner with PPO/GRPO-style GAE(λ) advantage estimation.

    Two update paths:
      * ``update_from_outcome(..., trajectory_end=False)`` — 1-step online update
        (low latency, used every tick). Stores the transition in a trajectory
        buffer for later GAE.
      * ``update_from_outcome(..., trajectory_end=True)`` OR
        ``end_trajectory()`` — full GAE(λ) sweep across the buffered trajectory.
        Normalises advantages, applies them to preference weights.

    The 2025/2026 PPO recipe (DeepSeek-V3 GRPO, OpenAI PPO) reduces variance
    vs 1-step TD by ~5–10× and stabilises sparse-reward learning. The value
    table is an EMA per-state V(s) used as the bootstrap in GAE.
    """

    learning_rate: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    baseline_alpha: float = 0.2
    baseline_window: int = 128
    grpo_group_size: int = 8  # 2026 P0-5: real GRPO group size
    # Trajectory buffer for n-step GAE(λ).
    _trajectory: list[tuple[Hexagram, float]] = field(default_factory=list)
    _value_table: dict[int, float] = field(default_factory=dict)
    _reward_window: deque[float] = field(
        default_factory=lambda: deque(maxlen=128)
    )
    # P0-5: per-tick GRPO record buffer — stores (h_state, expert_list, reward)
    # so the GRPO update can compare real completions (not j % N approximations).
    _grpo_buffer: deque[tuple[int, tuple[str, ...], float]] = field(
        default_factory=lambda: deque(maxlen=128)
    )
    # P0-1: DPO preference pairs buffer — stores (chosen, rejected, reward_diff)
    # collected from human feedback or automated outcome comparison.
    _preference_pairs: deque[tuple[tuple[str, ...], tuple[str, ...], float]] = field(
        default_factory=lambda: deque(maxlen=256)
    )
    # Backwards-compat fields
    preference_weights: dict[str, float] = field(default_factory=lambda: {
        "time": 0.17, "space": 0.17, "social": 0.17,
        "element": 0.17, "risk": 0.16, "narrative": 0.16,
    })
    visit_counts: dict[int, int] = field(default_factory=dict)
    total_visits: int = 0

    def update_from_outcome(
        self,
        h: Hexagram,
        reward: float,
        moe_weights: dict[str, float] | None = None,
        trajectory_end: bool = False,
        advantage: float | None = None,
    ) -> None:
        """Append a (state, reward) to the trajectory buffer and either:
          * apply a 1-step online update (default) — fast per-tick path, or
          * apply a full GAE(λ) sweep (when ``trajectory_end=True``).
        """
        self.visit_counts[h.normal_order] = (
            self.visit_counts.get(h.normal_order, 0) + 1
        )
        self.total_visits += 1
        self._reward_window.append(float(reward))
        self._trajectory.append((h, float(reward)))

        # P0-5: per-tick GRPO record.  We need the *actual* expert list
        # active at this tick (not a j % N rotation), so callers should
        # pass ``moe_active_experts`` via ``record_grpo_step()`` before
        # invoking this method.  Fall back to the keys of moe_weights if
        # the caller didn't bother to call record_grpo_step explicitly.
        if moe_weights is not None and not self._grpo_buffer:
            active = tuple(moe_weights.keys())
            self._grpo_buffer.append((h.normal_order, active, float(reward)))

        if trajectory_end:
            self._gae_update(moe_weights)
        else:
            self._one_step_update(h, float(reward), moe_weights, advantage)

    def record_grpo_step(
        self,
        h: Hexagram,
        active_experts: list[str] | tuple[str, ...],
        reward: float,
    ) -> None:
        """P0-5: store a per-tick GRPO completion.

        Should be called once per tick from the agent's learn phase,
        passing the **real** expert list that the MCTS-MoE selected
        (not a synthetic rotation).  GRPO update uses this buffer to
        evaluate "completions of the same prompt" against each other.
        """
        self._grpo_buffer.append(
            (h.normal_order, tuple(active_experts), float(reward))
        )

    def get_grpo_group(self, group_size: int | None = None) -> list[tuple[tuple[str, ...], float]]:
        """P0-5: return the most recent GRPO group of (expert_list, reward) tuples.

        Filters to records that share the same state class (same h.normal_order)
        so they truly represent "completions of the same prompt" — the
        DeepSeek-V3 / R1 GRPO definition.  Falls back to a mixed-state group
        when the state-class filter yields < 2 entries.
        """
        gs = group_size or self.grpo_group_size
        records = list(self._grpo_buffer)[-gs:]
        if not records:
            return []
        # Group by state class.
        by_state: dict[int, list[tuple[tuple[str, ...], float]]] = {}
        for state, experts, reward in records:
            by_state.setdefault(state, []).append((experts, reward))
        # Pick the largest group (most "completions" of the same prompt).
        if by_state:
            largest = max(by_state.values(), key=len)
            if len(largest) >= 2:
                return largest
        # Fallback: heterogeneous group (still better than a synthetic rotation).
        return [(experts, reward) for _, experts, reward in records]

    def _one_step_update(
        self,
        h: Hexagram,
        reward: float,
        moe_weights: dict[str, float] | None,
        advantage: float | None = None,
    ) -> None:
        if not self._reward_window:
            baseline = 0.0
        else:
            baseline = sum(self._reward_window) / len(self._reward_window)
        v_old = self._value_table.get(h.normal_order, baseline)
        v_new = (1 - self.baseline_alpha) * v_old + self.baseline_alpha * reward
        self._value_table[h.normal_order] = v_new
        delta = reward - baseline
        # Use advantage (reward - baseline) for more stable updates
        update_signal = advantage if advantage is not None else delta
        self._apply_preference_update(moe_weights, update_signal)

    def _gae_update(
        self, moe_weights: dict[str, float] | None
    ) -> None:
        """Full PPO/GRPO-style GAE(λ) sweep over the buffered trajectory.

        Computes ``A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}`` where
        ``δ_t = r_t + γ · V(s_{t+1}) − V(s_t)``. Uses the EMA value table
        for the bootstrap. This is the standard 2025+ PPO advantage estimator.
        """
        if not self._trajectory:
            return
        traj = self._trajectory
        T = len(traj)
        v_states = [
            self._value_table.get(h.normal_order, 0.0) for h, _ in traj
        ]
        deltas: list[float] = []
        for t in range(T):
            r_t = traj[t][1]
            v_t = v_states[t]
            v_next = v_states[t + 1] if t + 1 < T else 0.0
            deltas.append(r_t + self.gamma * v_next - v_t)
        # Backward accumulate A_t.
        advantages = [0.0] * T
        a_next = 0.0
        for t in reversed(range(T)):
            a = deltas[t] + self.gamma * self.gae_lambda * a_next
            advantages[t] = a
            a_next = a
        # Normalise (PPO trick).
        if T > 1:
            mean_a = sum(advantages) / T
            var_a = sum((a - mean_a) ** 2 for a in advantages) / T
            std_a = max(var_a ** 0.5, 1e-6)
            advantages = [(a - mean_a) / std_a for a in advantages]
        # Apply weight to preference deltas, summed over the trajectory.
        for t, (h_t, _r_t) in enumerate(traj):
            v_old = self._value_table.get(h_t.normal_order, 0.0)
            v_new = (1 - self.baseline_alpha) * v_old + self.baseline_alpha * _r_t
            self._value_table[h_t.normal_order] = v_new
            a_t = advantages[t]
            self._apply_preference_update(moe_weights, a_t)
        # Clear buffer for the next trajectory.
        self._trajectory = []

    def _apply_preference_update(
        self, moe_weights: dict[str, float] | None, advantage: float
    ) -> None:
        if not moe_weights:
            return
        for expert, weight in moe_weights.items():
            if expert in self.preference_weights:
                delta = self.learning_rate * advantage * weight
                self.preference_weights[expert] += delta
        total_w = sum(self.preference_weights.values())
        if total_w > 0:
            for k in self.preference_weights:
                self.preference_weights[k] /= total_w

    def end_trajectory(
        self, moe_weights: dict[str, float] | None = None
    ) -> None:
        """Explicitly trigger a GAE(λ) update at episode boundaries."""
        self._gae_update(moe_weights)

    def get_visit_count(self, h: Hexagram) -> int:
        return self.visit_counts.get(h.normal_order, 0)

    def novelty_bonus(self, h: Hexagram) -> float:
        visits = self.get_visit_count(h)
        return 1.0 / (1.0 + visits)

    def value_estimate(self, h: Hexagram) -> float:
        """Return V(s) — the EMA per-state value estimate, or 0.0 if unseen."""
        return self._value_table.get(h.normal_order, 0.0)

    # ── P0-1: DPO alignment pipeline ──────────────────────────────

    def record_preference_pair(
        self,
        chosen_experts: list[str] | tuple[str, ...],
        rejected_experts: list[str] | tuple[str, ...],
        reward_diff: float = 1.0,
    ) -> None:
        """Record a real preference pair for DPO alignment.

        ``reward_diff``: positive = chosen is better, higher = stronger.
        """
        self._preference_pairs.append(
            (tuple(chosen_experts), tuple(rejected_experts), float(reward_diff))
        )

    @property
    def preference_pair_count(self) -> int:
        """Number of unprocessed preference pairs in the buffer."""
        return len(self._preference_pairs)

    def dpo_step(self, beta: float = 0.1, min_pairs: int = 4) -> dict[str, float] | None:
        """Run one DPO alignment step using collected preference pairs.

        Consumes up to ``min_pairs`` entries and applies the DPO gradient.
        Returns the updated preference weights dict, or None if not enough pairs.
        """
        if len(self._preference_pairs) < min_pairs:
            return None
        for _ in range(min_pairs):
            chosen, rejected, diff = self._preference_pairs.popleft()
            dpo_update(
                self.preference_weights,
                chosen_experts=list(chosen),
                rejected_experts=list(rejected),
                beta=beta * abs(diff),
                learning_rate=self.learning_rate,
            )
        return self.preference_weights

    def dpo_router_step(
        self,
        router,
        chosen_experts: list[str],
        rejected_experts: list[str],
        lr: float = 0.01,
    ) -> None:
        """Apply DPO gradient signal to the MoE router's gating weights.

        Connects the DPO preference alignment pipeline to the MoE
        router, ensuring that expert selection aligns with learned
        preferences.  Adjusts the router's gate biases to favor
        chosen experts over rejected ones.

        Should be called after ``dpo_update()`` or ``dpo_step()`` when
        the MoE router is available.
        """
        apply_dpo_to_router(router, chosen_experts, rejected_experts, lr)

    @property
    def reward_baseline(self) -> float:
        """Return the global reward baseline (EMA over recent rewards)."""
        if not self._reward_window:
            return 0.0
        return sum(self._reward_window) / len(self._reward_window)


@dataclass
class CuriosityScheduler:
    beta_initial: float = 0.5
    beta_final: float = 0.05
    decay_rate: float = 0.001
    step_count: int = 0

    @property
    def beta(self) -> float:
        return self.beta_final + (self.beta_initial - self.beta_final) * (
            1.0 / (1.0 + self.decay_rate * self.step_count)
        )

    def step(self) -> float:
        self.step_count += 1
        return self.beta


@dataclass
class GrowthManager:
    total_episodes: int = 0

    @property
    def phase(self) -> str:
        if self.total_episodes < 100:
            return "explore"
        elif self.total_episodes < 500:
            return "exploit"
        return "expert"

    @property
    def curiosity_weight(self) -> float:
        weights = {"explore": 0.5, "exploit": 0.2, "expert": 0.05}
        return weights[self.phase]

    def _phase_code(self) -> int:
        """Numeric encoding for telemetry (0=explore, 1=exploit, 2=expert)."""
        codes = {"explore": 0, "exploit": 1, "expert": 2}
        return codes.get(self.phase, -1)

    def advance(self) -> None:
        self.total_episodes += 1


# ------------------------------------------------------------------
# P3: DPO (Direct Preference Optimization) style preference alignment
# ------------------------------------------------------------------
def dpo_update(
    preference_weights: dict[str, float],
    chosen_experts: list[str],
    rejected_experts: list[str],
    beta: float = 0.1,
    learning_rate: float = 0.01,
) -> dict[str, float]:
    """DPO-style preference alignment for expert weights.

    Given a pair of (chosen, rejected) expert lists from a human or
    automated preference signal, apply the DPO loss gradient to the
    preference weights.  This is the 2024/2026 SOTA replacement for
    RLHF PPO — it's simpler, more stable, and doesn't require a
    separate reward model.

    ``chosen_experts``: experts that led to a preferred outcome.
    ``rejected_experts``: experts that led to a dispreferred outcome.
    ``beta``: temperature for the DPO loss.
    ``learning_rate``: step size for the weight update.

    Returns the updated preference_weights dict (mutated in-place).
    """
    if not chosen_experts or not rejected_experts:
        return preference_weights

    # Compute the log-ratio of weights for the chosen vs rejected sets.
    # DPO gradient: -β * ∇ log(σ(chosen_logit - rejected_logit))
    chosen_logit = sum(
        preference_weights.get(e, 0.0) for e in chosen_experts
    )
    rejected_logit = sum(
        preference_weights.get(e, 0.0) for e in rejected_experts
    )
    diff = chosen_logit - rejected_logit

    # DPO gradient: for each chosen expert, increase weight;
    # for each rejected expert, decrease weight.
    import math
    grad_scale = -beta * (1.0 / (1.0 + math.exp(diff)))

    for e in chosen_experts:
        if e in preference_weights:
            preference_weights[e] -= learning_rate * grad_scale
    for e in rejected_experts:
        if e in preference_weights:
            preference_weights[e] += learning_rate * grad_scale

    # Re-normalize to sum to 1.
    total_w = sum(preference_weights.values())
    if total_w > 0:
        for k in preference_weights:
            preference_weights[k] /= total_w

    return preference_weights


# ------------------------------------------------------------------
# DPO → MoE router gradient propagation
# ------------------------------------------------------------------
def apply_dpo_to_router(
    router,
    chosen_experts: list[str],
    rejected_experts: list[str],
    lr: float = 0.01,
) -> None:
    """Adjust the MoE router's gate biases to favor chosen experts over rejected ones.

    After computing the DPO loss on preference_weights, this function
    propagates the gradient signal to the MoE router's gating network.
    For each chosen expert, its gate bias is increased by ``lr``; for
    each rejected expert, its gate bias is decreased by ``lr``.  This
    is a simple but effective approach that directly shifts the router's
    prior toward preferred experts.

    Supports both ``MoERouter`` (6-expert legacy) and
    ``FineGrainedSparseMoE`` (64-expert + shared) router types.
    """
    import torch

    # Determine the gate module and expert name list based on router type.
    if hasattr(router, "gate") and hasattr(router.gate, "bias"):
        # MoERouter: gate is nn.Linear with bias
        gate = router.gate
        expert_names = [
            "time", "space", "social",
            "element", "risk", "narrative",
        ]
    elif hasattr(router, "_gate") and hasattr(router._gate, "bias"):
        # FineGrainedSparseMoE: _gate is nn.Linear with bias
        gate = router._gate
        expert_names = list(router.expert_names)
    else:
        return  # no adjustable gate bias found

    name_to_idx = {name: i for i, name in enumerate(expert_names)}

    with torch.no_grad():
        for name in chosen_experts:
            if name in name_to_idx:
                gate.bias[name_to_idx[name]] += lr
        for name in rejected_experts:
            if name in name_to_idx:
                gate.bias[name_to_idx[name]] -= lr


# ------------------------------------------------------------------
# GRPO (Group Relative Policy Optimization) — DeepSeek-V3 风格
# ------------------------------------------------------------------
def grpo_update(
    preference_weights: dict[str, float],
    group_rewards: list[tuple[list[str], float]],
    beta: float = 0.2,
    learning_rate: float = 0.01,
    kl_weight: float = 0.01,
) -> dict[str, float]:
    """GRPO — Group Relative Policy Optimization (DeepSeek-V3, 2025/2026).

    The key insight of GRPO vs PPO/DPO: instead of requiring a separate
    value function (PPO) or paired preferences (DPO), GRPO evaluates a
    *group* of completions (expert selections) against each other and
    pushes the policy toward the best-in-group.  This eliminates the
    need for a reward model and is the training recipe behind
    DeepSeek-V3 / R1.

    ``group_rewards`` is a list of (expert_list, reward) tuples from
    the same prompt/state.  The advantage of each completion is
    computed *relative to the group mean* — no value function needed.

    Algorithm:
      1. Compute group advantage: A_i = (r_i - mean(r)) / std(r)
      2. For each expert in completion i:
         - If A_i > 0 (better than group average): increase weight
         - If A_i < 0 (worse than group average): decrease weight
      3. Clip the update magnitude (GRPO clipping, analogous to PPO)
         to prevent destructively large updates.
      4. Apply a KL penalty against a reference policy (the pre-update
         weights) to prevent the policy from drifting too far.  Since we
         lack an explicit reference policy, we use L2 regularization on
         the weight change as a proxy: ``kl_weight * ||π - π_ref||₂``.

    ``kl_weight``: coefficient for the reference-policy KL penalty (L2
      regularization on weight drift).  Higher values keep the policy
      closer to the pre-update weights.

    Returns the updated preference_weights dict (mutated in-place).
    """
    if len(group_rewards) < 2:
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "GRPO skipped: need ≥2 group rewards, got %d", len(group_rewards),
        )
        return preference_weights

    # Step 1: Compute group-normalised advantages.
    # Snapshot pre-update weights for KL penalty (reference policy proxy).
    ref_weights = dict(preference_weights)
    rewards = [r for _, r in group_rewards]
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = max(var_r ** 0.5, 1e-6)
    advantages = [(r - mean_r) / std_r for _, r in group_rewards]

    # Step 2: Apply GRPO-weighted updates.
    for (experts, _r), adv in zip(group_rewards, advantages):
        # GRPO clipping: limit the advantage magnitude to prevent
        # destructively large updates (analogous to PPO's clip).
        clipped_adv = max(-2.0, min(2.0, adv))
        for e in experts:
            if e in preference_weights:
                # Gradient direction: +adv pushes the expert up,
                # -adv pushes it down.  Scaled by β (temperature).
                update = learning_rate * beta * clipped_adv
                preference_weights[e] += update

    # Step 3: Re-normalize to sum to 1.
    total_w = sum(preference_weights.values())
    if total_w > 0:
        for k in preference_weights:
            preference_weights[k] /= total_w

    # Step 4: KL penalty against reference policy (pre-update weights).
    # L2 regularization on weight drift: kl_weight * ||π - π_ref||₂
    # This prevents the policy from drifting too far from the reference,
    # analogous to the KL penalty in DeepSeek-V3's GRPO.
    if kl_weight > 0.0 and ref_weights:
        l2_drift = sum(
            (preference_weights.get(k, 0.0) - ref_weights.get(k, 0.0)) ** 2
            for k in preference_weights
        ) ** 0.5
        penalty = kl_weight * l2_drift
        if penalty > 0.0:
            for k in preference_weights:
                preference_weights[k] -= penalty * (
                    preference_weights[k] - ref_weights.get(k, 0.0)
                ) / max(l2_drift, 1e-8)
            # Re-normalize after penalty.
            total_w = sum(preference_weights.values())
            if total_w > 0:
                for k in preference_weights:
                    preference_weights[k] /= total_w

    return preference_weights
