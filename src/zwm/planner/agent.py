"""TrinityAgent — the persistent OODA orchestrator.

This is the object the WIRING_PLAN calls for: a single owner of cross-tick
learning state that closes the Observe → Predict → Evaluate → Act → Learn loop
and feeds every previously-orphaned subsystem real data.

  Observe   sensor_data -> RuleBasedEncoder -> hexagram; calendar -> time_phase
  Predict   UnifiedField -> SquareCircularJoint -> z_world -> JEPAPredictor
  Evaluate  TrinityPlanner.plan (MCTS + live EFE + preference-biased MoE),
            warm-started by Hebbian + episodic memory priors
  Act       apply the top mutation -> next hexagram
  Learn     EpisodicStore + VSA memory, OnlineLearner preference feedback,
            MoE router gradient step, Hebbian association update,
            JEPA latent-prediction training (real backprop)

The planner stays a stateless evaluator; all mutable state lives here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from zwm.core.hexagram import Hexagram, hexagram_from_bits
from zwm.encoder.base import RuleBasedEncoder
from zwm.hexaembed.vsa import VSACodebook, VSAEpisode, VSAMemoryBuffer
from zwm.jepa.predictor import JEPAPredictor
from zwm.jepa.square_encoder import FixedWeightSquareGNN, SquareCircularJoint
from zwm.learning.hebbian import HebbianAssociator
from zwm.learning.online import CuriosityScheduler, GrowthManager, OnlineLearner
from zwm.planner.loop import PlanResult, TrinityPlanner
from zwm.scene_field.calendar import MultiScaleCalendar
from zwm.scene_field.unified_field import UnifiedField
from zwm.self_field.palace_graph import LuoshuGrid
from zwm.storage.episodic_db import EpisodicStore, SemanticStore
from zwm.topology.recursive import expand_topology

# Reward at/above this is treated as a "good" outcome worth consolidating
# and reinforcing through the learners.
_GOOD_OUTCOME = 0.5


@dataclass(frozen=True, slots=True)
class TickReport:
    """Everything one OODA tick produced — returned for inspection/logging."""

    plan: PlanResult
    h_current: Hexagram
    h_next: Hexagram
    reward: float
    jepa_loss: float | None
    router_loss: float | None
    episode_id: int
    surprise: float = 0.0

    # Convenience pass-throughs so callers can treat a TickReport like a result.
    @property
    def top_mutation(self) -> int:
        return self.plan.top_mutation

    @property
    def top_score(self) -> float:
        return self.plan.top_score


class TrinityAgent:
    """Owns persistent learning state and runs the closed OODA loop."""

    def __init__(
        self,
        db_path: str = "zwm_episodes.db",
        semantic_path: str | None = None,
        mcts_iterations: int = 200,
        grid: LuoshuGrid | None = None,
    ) -> None:
        # Stateless evaluator.
        self.planner = TrinityPlanner(mcts_iterations=mcts_iterations)

        # Perception + time.
        self.encoder = RuleBasedEncoder()
        self.calendar = MultiScaleCalendar()

        # Persistent learners.
        self.learner = OnlineLearner()
        self.curiosity = CuriosityScheduler()
        self.growth = GrowthManager()
        self.hebbian = HebbianAssociator()

        # In-memory VSA structures (cheap, no external handle).
        self.vsa = VSACodebook()
        self.vsa_buffer = VSAMemoryBuffer()

        # World model — construct the torch components BEFORE opening any OS
        # handle, so a torch failure cannot leak an open SQLite connection.
        self.joint = SquareCircularJoint(FixedWeightSquareGNN())
        self.jepa = JEPAPredictor(input_dim=77)

        self.grid = grid if grid is not None else LuoshuGrid()

        # Multi-scale palace scaffold: the recursive 九宫 topology enumerates the
        # palace space the agent can localise into. It drives the EFE
        # palace-exploration term — the agent steers toward least-visited
        # generative palaces (epistemic drive over space, not just over states).
        self.topology = expand_topology(max_depth=2)
        self._palace_candidates = [
            n.palace_position
            for n in self.topology.nodes_at_depth(1)
            if n.palace_position != self.grid.self_position
        ]
        self._palace_visits: dict[int, int] = {}

        # Cross-tick caches for transition-based JEPA training.
        self._prev_z = None
        self._prev_hex: int | None = None

        # Memory — opened LAST. If anything above threw, no handle was leaked.
        self.store = EpisodicStore(db_path=db_path)
        try:
            self.semantic = (
                SemanticStore(file_path=semantic_path) if semantic_path else None
            )
        except Exception:
            self.store.close()
            raise

    # Context-manager support so the SQLite handle is always released, even on
    # an exception mid-loop.
    def __enter__(self) -> "TrinityAgent":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # OBSERVE → ... → LEARN  (full loop from raw sensors)
    # ------------------------------------------------------------------
    def observe_predict_evaluate_act(
        self,
        sensor_data: dict | None = None,
        h_current: Hexagram | None = None,
        grid: LuoshuGrid | None = None,
        year: int = 2026,
        time_phase: float | None = None,
        target_palace: int | None = None,
        day_gan: str | None = None,
        reward: float = 0.0,
    ) -> TickReport:
        # OBSERVE — perception turns sensors into a hexagram.
        if h_current is None:
            if sensor_data is None:
                raise ValueError("Provide either sensor_data or h_current")
            h_current = self.encoder.encode(sensor_data)

        # OBSERVE — calendar supplies the time phase if not given explicitly.
        if time_phase is None:
            time_phase = self.calendar.time_layers(year)["年"]

        return self.tick(
            h_current=h_current,
            grid=grid,
            time_phase=time_phase,
            target_palace=target_palace,
            day_gan=day_gan,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # One tick given a hexagram state (Predict → Evaluate → Act → Learn)
    # ------------------------------------------------------------------
    def tick(
        self,
        h_current: Hexagram,
        grid: LuoshuGrid | None = None,
        time_phase: float = 0.0,
        target_palace: int | None = None,
        day_gan: str | None = None,
        reward: float = 0.0,
    ) -> TickReport:
        grid = grid if grid is not None else self.grid
        reward = self._validate_reward(reward)

        # OBSERVE (space) — when no target palace is given, the topology scaffold
        # steers toward the least-visited generative palace. Counts are read for
        # the decision but incremented AFTER planning, so this tick's chosen
        # palace still carries its unknown-bonus during EFE evaluation.
        if target_palace is None:
            target_palace = self._next_palace_to_explore()

        # PREDICT — encode the unified world state and predict the next latent.
        world = UnifiedField.snapshot(h_current, grid, time_phase, day_gan=day_gan)
        z_world = self.joint.encode(h_current, time_phase)
        z_pred = self.jepa.predict(z_world)

        # Priors — Hebbian suggestions + episodic memory, warm-start the search.
        mask_priors = self._combined_priors(h_current)

        # EVALUATE — plan with live EFE (state + palace curiosity),
        # preference-biased MoE, and a Hebbian/memory-primed MCTS.
        result = self.planner.plan(
            h_current,
            grid=grid,
            time_phase=time_phase,
            target_palace=target_palace,
            day_gan=day_gan,
            preference_weights=dict(self.learner.preference_weights),
            mask_priors=mask_priors,
            palace_visit_counts=dict(self._palace_visits),
        )

        # ACT — the chosen mutation defines the next state.
        h_next = result.chain.evolved

        # PREDICT (consume) — world-model surprise: squared error between the
        # predicted next latent and the EMA-target embedding of the state the
        # planner actually chose. A real signal, not a discarded value: high
        # surprise means the world model has not yet learned this transition.
        z_actual = self.joint.encode(h_next, time_phase)
        z_target = self.jepa.target_latent(z_actual)
        surprise = float(((z_pred - z_target) ** 2).mean())

        # Now record the palace visit (after EFE has seen the pre-visit counts).
        self._palace_visits[target_palace] = (
            self._palace_visits.get(target_palace, 0) + 1
        )

        # LEARN — write back into every persistent subsystem.
        jepa_loss = self._train_jepa(z_world, z_actual)
        router_loss = self._reinforce_router(
            h_current, grid, time_phase, result, reward
        )
        self._update_preferences(h_current, result, reward)
        self.hebbian.update_from_episode(
            [h_current.normal_order, h_next.normal_order], reward
        )
        episode_id = self._store_episode(h_current, h_next, world, reward)

        self.curiosity.step()
        self.growth.advance()
        self._prev_z = z_world
        self._prev_hex = h_current.normal_order

        return TickReport(
            plan=result,
            h_current=h_current,
            h_next=h_next,
            reward=reward,
            jepa_loss=jepa_loss,
            router_loss=router_loss,
            episode_id=episode_id,
            surprise=surprise,
        )

    @staticmethod
    def _validate_reward(reward: float) -> float:
        """Validate the reward at the loop boundary; clamp to [-1, 1].

        A NaN/Inf reward would silently corrupt every learner (preference
        renormalisation, router CE weight, Hebbian deltas), so reject it.
        """
        try:
            r = float(reward)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reward must be a real number, got {reward!r}") from exc
        if not math.isfinite(r):
            raise ValueError(f"reward must be finite, got {reward!r}")
        return max(-1.0, min(1.0, r))

    def _next_palace_to_explore(self) -> int:
        """Least-visited palace from the recursive topology scaffold.

        Realises the epistemic drive over *space*: the agent prefers palaces it
        has localised into least often, breaking ties by palace order.
        """
        if not self._palace_candidates:
            return self.grid.self_position
        return min(
            self._palace_candidates,
            key=lambda p: (self._palace_visits.get(p, 0), p),
        )

    # ------------------------------------------------------------------
    # Memory-derived priors
    # ------------------------------------------------------------------
    def memory_priors(self, h_current: Hexagram, k: int = 5) -> dict[int, float]:
        """Mask priors learned from past episodes with similar starting states.

        Returns {mutation_mask: cumulative_reward}. Uses VSA similarity over the
        stored episode fingerprints, then reconstructs the mask that was taken.
        """
        query = self.vsa.encode_hexagram(h_current.normal_order)
        similar = self.store.query_similar_vector(query, limit=k)
        priors: dict[int, float] = {}
        for ep in similar:
            main = ep.get("main_hex_bits")
            evolved = ep.get("evolved_hex_bits")
            if main is None or evolved is None:
                continue
            mask = (main ^ evolved) & 0b111111
            if 1 <= mask <= 63:
                priors[mask] = priors.get(mask, 0.0) + float(ep.get("reward", 0.0))
        return priors

    def _combined_priors(self, h_current: Hexagram) -> list[int]:
        cur = h_current.normal_order
        masks: list[int] = []

        # Hebbian: high-association successor hexagrams -> the mask reaching them.
        for h2, _strength in self.hebbian.suggest_next(cur, top_k=5):
            mask = (cur ^ h2) & 0b111111
            if 1 <= mask <= 63:
                masks.append(mask)

        # Episodic memory: masks that paid off from similar states.
        mem = self.memory_priors(h_current)
        masks.extend(
            m for m, _w in sorted(mem.items(), key=lambda x: x[1], reverse=True)
        )

        # De-duplicate, preserving priority order.
        seen: set[int] = set()
        ordered: list[int] = []
        for m in masks:
            if m not in seen:
                seen.add(m)
                ordered.append(m)
        return ordered

    # ------------------------------------------------------------------
    # LEARN helpers
    # ------------------------------------------------------------------
    def _train_jepa(self, z_world, z_next) -> float | None:
        """Real JEPA training step on the observed transition (z_t -> z_{t+1})."""
        loss = self.jepa.train_step(z_world, z_next)
        return None if loss != loss else loss  # filter NaN

    def _reinforce_router(
        self,
        h_current: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
        result: PlanResult,
        reward: float,
    ) -> float | None:
        """Gradient step that reinforces the highest-routed active expert.

        Only reinforce on good outcomes; the strength scales with reward.
        Routed through the planner's public ``reinforce_expert`` hook rather
        than reaching into its private MoE.
        """
        if reward < _GOOD_OUTCOME or not result.moe_active_experts:
            return None
        names = self.planner.expert_names
        top_expert = result.moe_active_experts[0]
        if top_expert not in names:
            return None
        idx = names.index(top_expert)
        return self.planner.reinforce_expert(
            h_current, grid, time_phase, expert_index=idx, weight=reward
        )

    def _update_preferences(
        self, h_current: Hexagram, result: PlanResult, reward: float
    ) -> None:
        moe_weights = {name: 1.0 for name in result.moe_active_experts}
        self.learner.update_from_outcome(h_current, reward, moe_weights=moe_weights)

    def _store_episode(
        self,
        h_current: Hexagram,
        h_next: Hexagram,
        world: UnifiedField,
        reward: float,
    ) -> int:
        vsa_vec = self.vsa.encode_hexagram(h_current.normal_order)
        outcome = "吉" if reward >= _GOOD_OUTCOME else "凶"
        chain = world.five_chain
        episode_id = self.store.store(
            main_bits=h_current.normal_order,
            inter_bits=chain.inter.normal_order,
            evolved_bits=h_next.normal_order,
            reversed_bits=chain.reversed_.normal_order,
            complement_bits=chain.complement.normal_order,
            outcome=outcome,
            reward=reward,
            encoded_vector=vsa_vec,
            context={"time_phase": world.time_phase},
        )
        # VSA episodic buffer + semantic frequency table.
        self.vsa_buffer.add(
            VSAEpisode(
                hexagram_vector=vsa_vec,
                context_vector=self.vsa.encode_hexagram(h_next.normal_order),
                outcome_vector=self.vsa.encode_trigram(h_current.lower_trigram.index),
                reward=reward,
            )
        )
        if reward >= _GOOD_OUTCOME:
            self.vsa_buffer.consolidate()
        if self.semantic is not None:
            self.semantic.increment_frequency(h_current.normal_order)
            self.semantic.update_association(
                h_current.normal_order, h_next.normal_order, delta=reward * 0.01
            )
        return episode_id

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.store.close()
        if self.semantic is not None:
            self.semantic.close()
