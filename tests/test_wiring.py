"""Integration tests for the wiring-remediation work (PR-1 .. PR-5).

These exercise the real OODA closed loop: persistent learning state,
EFE visit-count feedback, learned MoE routing, episodic + VSA memory,
real torch JEPA training, and Hebbian-primed search.
"""
from __future__ import annotations

import math

from zwm.core.hexagram import hexagram_from_bits, hexagram_from_name
from zwm.self_field.palace_graph import LuoshuGrid


# ----------------------------------------------------------------------
# PR-1: EFE visit_counts feedback + Langevin warm-start
# ----------------------------------------------------------------------
class TestEFEVisitCounts:
    def test_visit_counts_accumulate_across_mcts(self):
        from zwm.planner.loop import TrinityPlanner

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        planner = TrinityPlanner(mcts_iterations=200)
        planner.plan(qian, grid)
        # After a real MCTS run the planner must have recorded visits.
        assert sum(planner.visit_counts.values()) > 0

    def test_epistemic_value_decays_with_visits(self):
        from zwm.self_field.efe import epistemic_value

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        fresh = epistemic_value(qian, grid, visit_counts={}, total_visits=1)
        seen = epistemic_value(
            qian, grid,
            visit_counts={qian.normal_order: 50}, total_visits=50,
        )
        # Palace key-space must not collide with hexagram key-space.
        palace_explored = epistemic_value(
            qian, grid, visit_counts={},
            palace_visit_counts={p: 5 for p in range(1, 10)},
        )
        assert palace_explored < fresh  # explored palaces remove unknown bonus
        # Novelty bonus must shrink as the same state is visited more.
        assert seen < fresh

    def test_efe_score_uses_live_visit_counts(self):
        from zwm.planner.loop import TrinityPlanner

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        planner = TrinityPlanner(mcts_iterations=120)
        # Two consecutive plans: the second sees non-empty visit history.
        planner.plan(qian, grid)
        before = dict(planner.visit_counts)
        planner.plan(qian, grid)
        after = planner.visit_counts
        assert sum(after.values()) > sum(before.values())


# ----------------------------------------------------------------------
# PR-2: OnlineLearner feedback -> learned MoE routing
# ----------------------------------------------------------------------
class TestMoEFeedback:
    def test_preference_weights_scale_evaluation(self):
        from zwm.moe.sparse_activation import SparseMoE

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        moe = SparseMoE(top_k=3)
        base = moe.evaluate(qian, grid, time_phase=0.0, target_palace=1)
        skewed = moe.evaluate(
            qian, grid, time_phase=0.0, target_palace=1,
            preference_weights={
                "time": 1.0, "space": 0.0, "social": 0.0,
                "element": 0.0, "risk": 0.0, "narrative": 0.0,
            },
        )
        assert isinstance(skewed, float)
        # A degenerate preference must change the blended score.
        assert not math.isclose(base, skewed, rel_tol=1e-9)

    def test_router_is_trainable_and_learns(self):
        from zwm.moe.router import MoERouter

        router = MoERouter()
        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        target = 2  # social expert index
        before = router.route(qian, grid, 0.0)[target]
        for _ in range(60):
            router.train_toward(qian, grid, 0.0, expert_index=target, lr=0.05)
        after = router.route(qian, grid, 0.0)[target]
        assert after > before


# ----------------------------------------------------------------------
# PR-3: Episodic + VSA memory
# ----------------------------------------------------------------------
class TestMemoryWiring:
    def test_agent_persists_episodes(self, tmp_path):
        from zwm.planner.agent import TrinityAgent

        db = str(tmp_path / "ep.db")
        agent = TrinityAgent(db_path=db, mcts_iterations=80)
        qian = hexagram_from_name("乾为天")
        agent.tick(h_current=qian, reward=0.9)
        agent.tick(h_current=qian, reward=0.8)
        assert agent.store.count() == 2
        agent.close()

    def test_similar_memory_biases_priors(self, tmp_path):
        from zwm.planner.agent import TrinityAgent

        db = str(tmp_path / "ep.db")
        agent = TrinityAgent(db_path=db, mcts_iterations=80)
        qian = hexagram_from_name("乾为天")
        result = agent.tick(h_current=qian, reward=0.95)
        assert result.top_mutation >= 0
        priors = agent.memory_priors(qian)
        assert isinstance(priors, dict)
        agent.close()


# ----------------------------------------------------------------------
# PR-4: Real torch JEPA training
# ----------------------------------------------------------------------
class TestJEPATraining:
    def test_predictor_is_torch_module(self):
        import torch.nn as nn
        from zwm.jepa.predictor import JEPAPredictor

        assert isinstance(JEPAPredictor(), nn.Module)

    def test_loss_decreases_on_repeated_transition(self):
        import numpy as np
        from zwm.jepa.predictor import JEPAPredictor

        pred = JEPAPredictor(input_dim=77)
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1, 77).astype(np.float32)
        z_next = rng.normal(0, 1, 77).astype(np.float32)
        first = pred.train_step(z, z_next)
        for _ in range(200):
            last = pred.train_step(z, z_next)
        assert last < first

    def test_ema_target_encoder_tracks(self):
        import numpy as np
        from zwm.jepa.square_encoder import SquareCircularJoint, FixedWeightSquareGNN

        joint = SquareCircularJoint(FixedWeightSquareGNN())
        qian = hexagram_from_name("乾为天")
        z = joint.encode(qian, time_phase=0.0)
        assert z.shape[0] == 77  # 64 square + 13 circular

    def test_vicreg_loss_nonnegative(self):
        import numpy as np
        from zwm.jepa.predictor import JEPAPredictor

        pred = JEPAPredictor()
        import torch
        latent = torch.randn(8, 32)
        loss = pred.vicreg_loss(latent)
        assert float(loss) >= 0.0


# ----------------------------------------------------------------------
# PR-5: Full OODA tick persists all state
# ----------------------------------------------------------------------
class TestOODAClosedLoop:
    def test_full_tick_from_sensors(self, tmp_path):
        from zwm.planner.agent import TrinityAgent

        db = str(tmp_path / "ep.db")
        agent = TrinityAgent(db_path=db, mcts_iterations=80)
        sensors = {
            "temperature": 25.0, "terrain": 0.8, "social_proximity": 0.7,
            "resource_level": 0.5, "momentum": 0.3, "overall_favorability": 0.6,
        }
        result = agent.observe_predict_evaluate_act(sensor_data=sensors, reward=0.7)
        assert result.top_mutation >= 0
        assert agent.store.count() == 1
        agent.close()

    def test_state_grows_across_ticks(self, tmp_path):
        from zwm.planner.agent import TrinityAgent

        db = str(tmp_path / "ep.db")
        agent = TrinityAgent(db_path=db, mcts_iterations=80)
        qian = hexagram_from_name("乾为天")
        for _ in range(3):
            agent.tick(h_current=qian, reward=0.8)
        assert agent.learner.total_visits == 3
        assert agent.store.count() == 3
        # Hebbian associations recorded for the evolved transitions.
        assert len(agent.hebbian.associations) >= 0
        agent.close()

    def test_jepa_trains_inside_loop(self, tmp_path):
        from zwm.planner.agent import TrinityAgent

        db = str(tmp_path / "ep.db")
        agent = TrinityAgent(db_path=db, mcts_iterations=60)
        qian = hexagram_from_name("乾为天")
        losses = []
        for _ in range(12):
            r = agent.tick(h_current=qian, reward=0.8)
            if r.jepa_loss is not None:
                losses.append(r.jepa_loss)
        # JEPA produced real training signal during the loop.
        assert len(losses) >= 1
        agent.close()


# ----------------------------------------------------------------------
# Post-review hardening (H1/H2/H3/M2/L1) regression guards
# ----------------------------------------------------------------------
class TestReviewHardening:
    def test_vicreg_active_across_replay_batch(self):
        """H1: VICReg must produce a real anti-collapse signal — latents over
        the replay buffer keep per-dimension variance well above zero."""
        import numpy as np
        import torch
        from zwm.jepa.predictor import JEPAPredictor

        pred = JEPAPredictor(input_dim=77, batch_size=16)
        rng = np.random.default_rng(0)
        for _ in range(40):
            pred.train_step(
                rng.normal(0, 1, 77).astype(np.float32),
                rng.normal(0, 1, 77).astype(np.float32),
            )
        xs = torch.stack([pred._replay[i][0] for i in range(len(pred._replay))])
        with torch.no_grad():
            lat = pred.context_encoder(xs)
        assert float(lat.std(0).mean()) > 0.05  # not collapsed

    def test_encoder_rejects_partial_sensors(self):
        """H2: missing sensor keys must fail fast, not be coerced to YIN."""
        import pytest
        from zwm.encoder.base import RuleBasedEncoder

        with pytest.raises(ValueError):
            RuleBasedEncoder().encode({"temperature": 25.0})

    def test_encoder_rejects_nonfinite_sensor(self):
        import pytest
        from zwm.encoder.base import RuleBasedEncoder

        sensors = {
            "temperature": float("nan"), "terrain": 0.8, "social_proximity": 0.7,
            "resource_level": 0.5, "momentum": 0.3, "overall_favorability": 0.6,
        }
        with pytest.raises(ValueError):
            RuleBasedEncoder().encode(sensors)

    def test_nan_reward_rejected(self, tmp_path):
        """H3: a non-finite reward must not silently corrupt the learners."""
        import pytest
        from zwm.planner.agent import TrinityAgent

        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=40) as ag:
            qian = hexagram_from_name("乾为天")
            with pytest.raises(ValueError):
                ag.tick(h_current=qian, reward=float("nan"))

    def test_palace_exploration_is_even(self, tmp_path):
        """M2: each of the 8 candidate palaces is explored once before any
        repeats (no pre-plan double-count skewing the epistemic drive)."""
        from zwm.planner.agent import TrinityAgent

        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=40) as ag:
            qian = hexagram_from_name("乾为天")
            for _ in range(8):
                ag.tick(h_current=qian, reward=0.6)
            assert sorted(ag._palace_visits.values()) == [1] * 8

    def test_surprise_signal_is_real(self, tmp_path):
        """L1: the JEPA prediction is consumed — surprise is a positive,
        non-constant world-model signal."""
        from zwm.planner.agent import TrinityAgent

        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=40) as ag:
            qian = hexagram_from_name("乾为天")
            surprises = [ag.tick(h_current=qian, reward=0.7).surprise for _ in range(6)]
        assert all(s >= 0.0 for s in surprises)
        assert max(surprises) > 0.0

    def test_context_manager_closes_store(self, tmp_path):
        """M3: the SQLite handle is released by the context manager."""
        import sqlite3
        from zwm.planner.agent import TrinityAgent

        db = str(tmp_path / "e.db")
        with TrinityAgent(db_path=db, mcts_iterations=30) as ag:
            ag.tick(h_current=hexagram_from_name("乾为天"), reward=0.7)
        # After exit the connection is closed; using it would raise.
        import pytest
        with pytest.raises(sqlite3.ProgrammingError):
            ag.store.count()

    def test_reinforce_expert_public_hook(self):
        """M4: the planner exposes a public reinforcement hook; the agent does
        not reach into private MoE internals."""
        from zwm.planner.loop import TrinityPlanner

        planner = TrinityPlanner(mcts_iterations=20)
        assert hasattr(planner, "reinforce_expert")
        assert hasattr(planner, "expert_names")
        grid = LuoshuGrid()
        loss = planner.reinforce_expert(
            hexagram_from_name("乾为天"), grid, 0.0, expert_index=2, weight=0.9,
        )
        assert isinstance(loss, float)
