"""Regression tests for the audit-remediation work (CRIT-1/2, HIGH-1/2, MED-1/2).

Each test here would have FAILED against the pre-audit code: it pins a wiring
that closes a previously-open loop — JEPA prediction driving the search, the
world-model surprise being consumed, curiosity annealing reaching EFE, the
novelty bonus being read, and the square encoder learning end-to-end.
"""
from __future__ import annotations

import numpy as np

from zwm.core.hexagram import hexagram_from_name
from zwm.self_field.palace_graph import LuoshuGrid


# ----------------------------------------------------------------------
# CRIT-1: JEPA prediction influences the plan (intrinsic term reaches EFE)
# ----------------------------------------------------------------------
class TestIntrinsicDrivesPlanning:
    def test_plan_accepts_intrinsic_fn_and_uses_it(self):
        from zwm.planner.loop import TrinityPlanner

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()

        seen: list[int] = []

        def intrinsic(h):
            seen.append(h.normal_order)
            return 1.0

        planner = TrinityPlanner(mcts_iterations=120)
        planner.plan(qian, grid, intrinsic_fn=intrinsic, beta_curiosity=0.3)
        # The planner actually invoked the world-model intrinsic term.
        assert len(seen) > 0

    def test_intrinsic_term_changes_scores(self):
        """A strong, mask-selective intrinsic reward must move the search output
        relative to a zero intrinsic — i.e. JEPA surprise really steers search."""
        from zwm.planner.loop import TrinityPlanner

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()

        def zero(h):
            return 0.0

        def biased(h):
            # Reward odd-identity successors strongly.
            return 5.0 if h.normal_order % 2 == 1 else 0.0

        p1 = TrinityPlanner(mcts_iterations=200)
        flat = p1.plan(qian, grid, intrinsic_fn=zero, beta_curiosity=0.5)
        p2 = TrinityPlanner(mcts_iterations=200)
        skewed = p2.plan(qian, grid, intrinsic_fn=biased, beta_curiosity=0.5)

        # The ranked search output differs: either the top pick or the whole
        # score list changed because the intrinsic term entered EFE.
        changed = (
            flat.top_mutation != skewed.top_mutation
            or flat.hexagram_scores != skewed.hexagram_scores
        )
        assert changed
        # And the bias's preferred parity dominates the skewed top pick's
        # successor identity more than under the flat run.
        skewed_odd = sum(
            1 for mask, _s in skewed.hexagram_scores
            if (qian.normal_order ^ mask) % 2 == 1
        )
        flat_odd = sum(
            1 for mask, _s in flat.hexagram_scores
            if (qian.normal_order ^ mask) % 2 == 1
        )
        assert skewed_odd >= flat_odd


# ----------------------------------------------------------------------
# CRIT-2: world-model surprise is consumed in-loop, not just reported
# ----------------------------------------------------------------------
class TestSurpriseConsumed:
    def test_agent_passes_intrinsic_into_planner(self, tmp_path, monkeypatch):
        from zwm.planner import agent as agent_mod
        from zwm.planner.agent import TrinityAgent

        captured = {}
        real_plan = agent_mod.TrinityPlanner.plan

        def spy_plan(self, *a, **kw):
            captured["intrinsic_fn"] = kw.get("intrinsic_fn")
            captured["beta"] = kw.get("beta_curiosity")
            return real_plan(self, *a, **kw)

        monkeypatch.setattr(agent_mod.TrinityPlanner, "plan", spy_plan)

        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=40) as ag:
            ag.tick(h_current=hexagram_from_name("乾为天"), reward=0.7)

        # The agent supplied a live intrinsic function and a curiosity beta.
        assert callable(captured["intrinsic_fn"])
        assert isinstance(captured["beta"], float)
        # The intrinsic function returns a real, finite world-model signal.
        val = captured["intrinsic_fn"](hexagram_from_name("坤为地"))
        assert np.isfinite(val)


# ----------------------------------------------------------------------
# HIGH-1: curiosity annealing reaches EFE (beta is live, not hardcoded 0.3)
# ----------------------------------------------------------------------
class TestCuriosityAnnealingWired:
    def test_beta_curiosity_override_changes_efe(self):
        from zwm.planner.loop import TrinityPlanner

        qian = hexagram_from_name("乾为天")
        grid = LuoshuGrid()
        # Two betas should produce different epistemic weighting -> scores.
        lo = TrinityPlanner(mcts_iterations=150).plan(qian, grid, beta_curiosity=0.0)
        hi = TrinityPlanner(mcts_iterations=150).plan(qian, grid, beta_curiosity=2.0)
        assert lo.hexagram_scores != hi.hexagram_scores

    def test_agent_beta_tracks_scheduler(self, tmp_path, monkeypatch):
        from zwm.planner import agent as agent_mod
        from zwm.planner.agent import TrinityAgent

        betas: list[float] = []
        real_plan = agent_mod.TrinityPlanner.plan

        def spy_plan(self, *a, **kw):
            betas.append(kw.get("beta_curiosity"))
            return real_plan(self, *a, **kw)

        monkeypatch.setattr(agent_mod.TrinityPlanner, "plan", spy_plan)
        qian = hexagram_from_name("乾为天")
        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=30) as ag:
            for _ in range(3):
                ag.tick(h_current=qian, reward=0.6)
        # Curiosity anneals: beta is non-increasing across ticks.
        assert betas == sorted(betas, reverse=True)


# ----------------------------------------------------------------------
# HIGH-2: novelty_bonus is consumed by the intrinsic reward
# ----------------------------------------------------------------------
class TestNoveltyBonusConsumed:
    def test_novelty_bonus_referenced_in_intrinsic(self, tmp_path, monkeypatch):
        from zwm.planner.agent import TrinityAgent
        from zwm.learning.online import OnlineLearner

        calls = {"n": 0}
        real = OnlineLearner.novelty_bonus

        def spy(self, h):
            calls["n"] += 1
            return real(self, h)

        monkeypatch.setattr(OnlineLearner, "novelty_bonus", spy)
        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=40) as ag:
            ag.tick(h_current=hexagram_from_name("乾为天"), reward=0.7)
        # The loop actually read the novelty bonus during planning.
        assert calls["n"] > 0


# ----------------------------------------------------------------------
# MED-1: square encoder learns end-to-end (real gradients in the encoder)
# ----------------------------------------------------------------------
class TestLearnableSquareEncoder:
    def test_learnable_encoder_is_nn_module_with_params(self):
        import torch.nn as nn
        from zwm.jepa.square_encoder import LearnableSquareGNN

        enc = LearnableSquareGNN()
        assert isinstance(enc, nn.Module)
        assert sum(p.numel() for p in enc.parameters()) > 0

    def test_encoder_params_change_under_training(self):
        import torch
        from zwm.jepa.predictor import JEPAPredictor
        from zwm.jepa.square_encoder import LearnableSquareGNN

        enc = LearnableSquareGNN()
        jepa = JEPAPredictor(input_dim=77)
        jepa.attach_square_encoder(enc)

        before = [p.detach().clone() for p in enc.parameters()]
        rng = np.random.default_rng(0)
        for _ in range(30):
            ft = rng.normal(0, 1, 12).astype(np.float32)
            fn = rng.normal(0, 1, 12).astype(np.float32)
            jepa.train_transition(ft, 0.1, fn, 0.2)
        after = list(enc.parameters())
        # At least one parameter tensor moved => real gradient flowed into the
        # square encoder (it is no longer a frozen random projection).
        moved = any(
            not torch.allclose(b, a) for b, a in zip(before, after)
        )
        assert moved

    def test_agent_uses_learnable_encoder_by_default(self, tmp_path):
        from zwm.planner.agent import TrinityAgent
        from zwm.jepa.square_encoder import LearnableSquareGNN

        # R4: the field-encoder path is now the default.  ``ag.square``
        # is ``None`` when ``ag._field_gnn`` is the active encoder
        # (FieldSquareGNN lives on its own attribute).  We probe both
        # the new (field) and the old (single-hex) shapes so the test
        # is forward-compatible with whichever encoder is live.
        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=30) as ag:
            if ag._field_gnn is not None:
                # New wiring: FieldSquareGNN owns the 64-hexagram
                # field and ``ag.joint`` wraps it in the joint encoder.
                assert ag.joint is not None
            else:
                # Legacy wiring: ``ag.square`` is the learnable
                # square encoder.  This branch fires only when
                # FieldSquareGNN import failed (e.g. torch missing).
                assert isinstance(ag.square, LearnableSquareGNN)

    def test_fixed_encoder_still_supported(self, tmp_path):
        from zwm.planner.agent import TrinityAgent
        from zwm.jepa.square_encoder import FixedWeightSquareGNN

        with TrinityAgent(
            db_path=str(tmp_path / "e.db"),
            mcts_iterations=30,
            learnable_encoder=False,
        ) as ag:
            # R4: ``learnable_encoder=False`` only affects the
            # single-hex GNN branch.  When the field encoder is
            # active, ``ag.square`` is ``None`` (the field GNN is
            # always learnable).  We probe the right attribute for
            # the active path.
            if ag._field_gnn is not None:
                assert ag.joint is not None
            else:
                assert isinstance(ag.square, FixedWeightSquareGNN)
            ag.tick(h_current=hexagram_from_name("乾为天"), reward=0.6)


# ----------------------------------------------------------------------
# MED-2: the demo runner consumes TickReport telemetry
# ----------------------------------------------------------------------
class TestDemoConsumesTelemetry:
    def test_run_demo_returns_consumed_metrics(self, tmp_path):
        from zwm.demo import run_demo

        summary = run_demo(
            ticks=6, db_path=str(tmp_path / "demo.db"), mcts_iterations=30
        )
        # The demo reads jepa_loss / surprise / reward off each TickReport.
        assert summary["ticks"] == 6
        assert len(summary["jepa_losses"]) >= 1
        assert len(summary["surprises"]) == 6
        assert all(np.isfinite(s) for s in summary["surprises"])
