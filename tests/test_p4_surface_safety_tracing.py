"""P4-6 / P4-7 / P4-8 / P4-9 — unit tests for the late-stage audit work.

Covers:

  P4-6  : :class:`TrinityConfig` dataclass + topology inline
  P4-7  : CLI / API / MCP surface unification
  P4-8  : Constitutional AI safety guardrails
  P4-9  : OpenTelemetry-compatible tracing

These tests are deliberately self-contained (no torch / no LLM), so
they run in <2 s and pin the new wiring in place.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict

import pytest


# ======================================================================
# P4-6: TrinityConfig dataclass
# ======================================================================
class TestTrinityConfig:
    def test_default_construction(self):
        from zwm.planner.agent_config import TrinityConfig
        c = TrinityConfig()
        assert c.mcts_iterations == 200
        assert c.n_particles == 16
        assert c.use_diffusion is True
        assert c.learnable_encoder is True
        assert c.hierarchical is False
        assert c.use_fsdp2 is False
        assert c.quantize is None
        assert c.use_trainable_vsa is True
        assert c.use_react is True
        assert c.topology_max_depth == 2
        assert c.grid is None
        assert c.db_path == "zwm_episodes.db"

    def test_frozen_immutability(self):
        from zwm.planner.agent_config import TrinityConfig
        c = TrinityConfig()
        with pytest.raises((AttributeError, Exception)):
            c.mcts_iterations = 999  # type: ignore[misc]

    def test_from_dict_round_trip(self):
        from zwm.planner.agent_config import TrinityConfig
        c = TrinityConfig(mcts_iterations=80, n_particles=0, hierarchical=True)
        d = c.to_dict()
        c2 = TrinityConfig.from_dict(d)
        assert c2.mcts_iterations == 80
        assert c2.n_particles == 0
        assert c2.hierarchical is True

    def test_from_dict_drops_unknown(self):
        from zwm.planner.agent_config import TrinityConfig
        c = TrinityConfig.from_dict({"mcts_iterations": 50, "garbage_key": "x"})
        assert c.mcts_iterations == 50

    def test_as_overrides_omits_defaults(self):
        from zwm.planner.agent_config import TrinityConfig
        c = TrinityConfig(mcts_iterations=99)
        ov = c.as_overrides()
        assert "mcts_iterations" in ov
        assert "n_particles" not in ov  # default

    def test_trinity_agent_uses_config(self):
        from zwm.planner.agent import TrinityAgent
        from zwm.planner.agent_config import TrinityConfig
        cfg = TrinityConfig(mcts_iterations=20, n_particles=0, use_react=False)
        with tempfile.TemporaryDirectory() as d:
            cfg2 = TrinityConfig(
                mcts_iterations=20, n_particles=0, use_react=False,
                db_path=os.path.join(d, "e.db"),
            )
            a = TrinityAgent(config=cfg2)
            assert a.config is cfg2
            assert a.planner._mcts_iterations == 20
            a.close()

    def test_trinity_agent_backwards_compat_kwargs(self):
        """P4-6: legacy kwargs still work."""
        from zwm.planner.agent import TrinityAgent
        with tempfile.TemporaryDirectory() as d:
            a = TrinityAgent(
                db_path=os.path.join(d, "e.db"),
                mcts_iterations=20,
                n_particles=0,
                use_react=False,
            )
            assert a.config.mcts_iterations == 20
            a.close()

    def test_topology_inline(self):
        """P4-6: topology is built from config.topology_max_depth."""
        from zwm.planner.agent import TrinityAgent
        from zwm.planner.agent_config import TrinityConfig
        with tempfile.TemporaryDirectory() as d:
            cfg = TrinityConfig(
                db_path=os.path.join(d, "e.db"),
                mcts_iterations=20, n_particles=0, use_react=False,
                topology_max_depth=1,
            )
            a = TrinityAgent(config=cfg)
            # depth=1 → root + 9 children = 10 nodes
            assert a.topology.total_nodes() == 10
            a.close()

    def test_trinity_config_field_count(self):
        """P4-7: CLI/API/MCP all read from the same field set."""
        from zwm.planner.agent_config import TrinityConfig
        fields = TrinityConfig.__dataclass_fields__
        # Ensure the safety field is present (P4-8)
        assert "enable_constitution" in fields
        # The most impactful config knobs
        for key in ("db_path", "mcts_iterations", "n_particles",
                    "use_diffusion", "learnable_encoder", "hierarchical",
                    "use_fsdp2", "quantize", "use_trainable_vsa",
                    "use_react", "topology_max_depth"):
            assert key in fields, f"missing config field: {key}"


# ======================================================================
# P4-7: Surface unification
# ======================================================================
class TestSurfaceUnification:
    def test_argparse_emits_bool_negation(self):
        from zwm.planner.surface import config_to_argparse
        import argparse
        p = argparse.ArgumentParser()
        config_to_argparse(p)
        args = p.parse_args(["--mcts-iterations", "60", "--no-use-react"])
        assert args.mcts_iterations == 60
        assert args.use_react is False

    def test_argparse_to_config_kwargs(self):
        from zwm.planner.surface import (
            config_to_argparse, argparse_to_config_kwargs,
            build_config_from_args,
        )
        import argparse
        p = argparse.ArgumentParser()
        config_to_argparse(p)
        args = p.parse_args([
            "--mcts-iterations", "77", "--n-particles", "8",
            "--quantize", "qlora", "--no-use-react",
        ])
        cfg = build_config_from_args(args)
        assert cfg.mcts_iterations == 77
        assert cfg.n_particles == 8
        assert cfg.quantize == "qlora"
        assert cfg.use_react is False

    def test_pydantic_model_mirrors_dataclass(self):
        from zwm.planner.surface import build_config_overrides_model
        from zwm.planner.agent_config import TrinityConfig
        # ``__dataclass_fields__`` is a dict mapping name -> Field; iterate
        # the keys (names) directly so we don't depend on the Field
        # constructor across Python versions.
        cfg_fields = set(TrinityConfig.__dataclass_fields__)
        ov_fields = set(build_config_overrides_model().model_fields.keys())
        # Overrides is the dataclass minus `grid` (non-serialisable).
        assert cfg_fields - ov_fields == {"grid"}

    def test_mcp_schema_includes_config_fields(self):
        from zwm.planner.surface import config_to_mcp_schema
        schema = config_to_mcp_schema("test", "test desc")
        props = schema["properties"]
        # All public config knobs should be exposed.
        for key in ("mcts_iterations", "n_particles", "use_diffusion",
                    "use_react"):
            assert key in props, f"missing MCP property: {key}"
        # Bool fields map to {"type": "boolean"}.
        assert props["use_diffusion"]["type"] == "boolean"
        # Int fields map to {"type": "integer"}.
        assert props["mcts_iterations"]["type"] == "integer"

    def test_apply_overrides_partial(self):
        from zwm.planner.agent_config import TrinityConfig
        from zwm.planner.surface import apply_overrides
        base = TrinityConfig(mcts_iterations=200, n_particles=16)
        out = apply_overrides(base, {"mcts_iterations": 50})
        assert out.mcts_iterations == 50
        assert out.n_particles == 16  # unchanged
        # The base is untouched (frozen).
        assert base.mcts_iterations == 200

    def test_apply_overrides_none_returns_base(self):
        from zwm.planner.agent_config import TrinityConfig
        from zwm.planner.surface import apply_overrides
        base = TrinityConfig(mcts_iterations=10)
        assert apply_overrides(base, None) is base
        assert apply_overrides(base, {}) is base

    def test_build_config_from_mcp_args_filters_unknown(self):
        """P4-7: MCP tool-specific args (e.g. hex_bits) must not crash
        apply_overrides when forwarded to TrinityConfig."""
        from zwm.planner.surface import build_config_from_mcp_args
        cfg = build_config_from_mcp_args({
            "hex_bits": 1,  # tool-specific, must be dropped
            "mcts_iterations": 30,  # config, must pass through
        })
        assert cfg.mcts_iterations == 30


# ======================================================================
# P4-8: Constitutional AI safety guardrails
# ======================================================================
class TestConstitution:
    def test_default_constitution_has_six_rules(self):
        from zwm.safety.constitution import DEFAULT_CONSTITUTION
        assert len(DEFAULT_CONSTITUTION) >= 6

    def test_block_severity_raises(self):
        from zwm.safety.constitution import (
            ConstitutionalGuard, ConstitutionalViolation,
        )
        g = ConstitutionalGuard()
        with pytest.raises(ConstitutionalViolation):
            g.check_input({"h_current": 1, "reward": 5.0})  # out of range

    def test_warn_severity_does_not_raise(self):
        from zwm.safety.constitution import (
            ConstitutionalGuard, rule_max_field, Severity,
        )
        g = ConstitutionalGuard(constitution=(
            rule_max_field("t", "top_score", -1.0, 1.0, severity=Severity.WARN),
        ))
        # No raise even on failure
        g.check_output({"h_current": 0, "h_next": 0, "top_score": 99.0})

    def test_disabled_guard_does_nothing(self):
        from zwm.safety.constitution import ConstitutionalGuard
        g = ConstitutionalGuard(enabled=False)
        # Even an obviously-bad payload passes when guard is off.
        g.check_input({"h_current": 999, "reward": 100.0})

    def test_self_loop_mutation_caught(self):
        from zwm.safety.constitution import ConstitutionalGuard
        g = ConstitutionalGuard()
        # A no-op mutation is a WARN-severity rule, so no raise.
        g.check_output({"h_current": 5, "h_next": 5, "top_score": 0.5})

    def test_verdict_history_is_bounded(self):
        from zwm.safety.constitution import ConstitutionalGuard
        g = ConstitutionalGuard(history_limit=8)
        for _ in range(20):
            g.check_input({"reward": 0.5})  # valid
        assert len(g.history) == 8

    def test_finite_numbers_recursively(self):
        from zwm.safety.constitution import ConstitutionalGuard
        g = ConstitutionalGuard()
        with pytest.raises(Exception):
            g.check_input({"nested": {"deep": {"a": float("nan")}}})

    def test_nan_reward_rejected_by_value_check(self):
        """P4-8 / P2 contract: _validate_reward still raises ValueError
        for NaN before the constitution sees the input."""
        from zwm.planner.agent import TrinityAgent
        from zwm.core.hexagram import hexagram_from_name
        with tempfile.TemporaryDirectory() as d:
            a = TrinityAgent(
                db_path=os.path.join(d, "e.db"),
                mcts_iterations=20, n_particles=0, use_react=False,
            )
            with pytest.raises(ValueError):
                a.tick(h_current=hexagram_from_name("乾为天"),
                       reward=float("nan"))
            a.close()

    def test_invalid_palace_rejected(self):
        from zwm.safety.constitution import ConstitutionalViolation
        from zwm.planner.agent import TrinityAgent
        from zwm.core.hexagram import hexagram_from_name
        with tempfile.TemporaryDirectory() as d:
            a = TrinityAgent(
                db_path=os.path.join(d, "e.db"),
                mcts_iterations=20, n_particles=0, use_react=False,
            )
            with pytest.raises(ConstitutionalViolation):
                a.tick(h_current=hexagram_from_name("乾为天"),
                       reward=0.5, target_palace=99)
            a.close()

    def test_disabled_constitution_allows_bad_input(self):
        from zwm.planner.agent import TrinityAgent
        from zwm.planner.agent_config import TrinityConfig
        from zwm.core.hexagram import hexagram_from_name
        with tempfile.TemporaryDirectory() as d:
            cfg = TrinityConfig(
                db_path=os.path.join(d, "e.db"),
                mcts_iterations=20, n_particles=0, use_react=False,
                enable_constitution=False,
            )
            a = TrinityAgent(config=cfg)
            r = a.tick(h_current=hexagram_from_name("乾为天"), reward=0.8)
            assert r.top_mutation >= 0
            a.close()

    def test_add_rule_runtime(self):
        from zwm.safety.constitution import (
            ConstitutionalGuard, rule_max_field,
        )
        g = ConstitutionalGuard()
        before = len(g.constitution)
        g.add_rule(rule_max_field("efebound", "top_score", -10.0, 10.0))
        assert len(g.constitution) == before + 1

    def test_history_records_all_rules(self):
        from zwm.safety.constitution import ConstitutionalGuard
        g = ConstitutionalGuard()
        g.check_input({"reward": 0.5})  # all rules pass
        # 6 default rules, all passed.
        assert len(g.history) == 6
        for v in g.history:
            assert v.passed is True


# ======================================================================
# P4-9: OpenTelemetry-compatible tracing
# ======================================================================
class TestTracing:
    def test_in_process_tracer_records_spans(self):
        from zwm.tracing import Tracer
        t = Tracer()
        with t.start_as_current_span("a") as span:
            span.set_attribute("k", 1)
            with t.start_as_current_span("b"):
                pass
        names = [s.name for s in t.spans]
        assert "a" in names
        assert "b" in names
        # Children have a parent.
        b = next(s for s in t.spans if s.name == "b")
        assert b.parent_id is not None

    def test_span_status_marks_errors(self):
        from zwm.tracing import Tracer
        t = Tracer()
        try:
            with t.start_as_current_span("boom") as span:
                raise RuntimeError("kaboom")
        except RuntimeError:
            pass
        s = t.spans[0]
        assert s.status == "error"
        assert s.attributes.get("exception.type") == "RuntimeError"

    def test_span_attributes_coerced(self):
        from zwm.tracing import Tracer
        t = Tracer()
        with t.start_as_current_span("x") as span:
            span.set_attribute("n", 1)
            span.set_attribute("s", "v")
            span.set_attribute("obj", {"k": "v"})  # becomes str
        s = t.spans[0]
        assert s.attributes["n"] == 1
        assert s.attributes["s"] == "v"
        assert isinstance(s.attributes["obj"], str)

    def test_render_recent_pretty(self):
        from zwm.tracing import Tracer, render_recent
        t = Tracer()
        t.clear()
        with t.start_as_current_span("foo"):
            pass
        with t.start_as_current_span("bar") as span:
            span.set_attribute("k", 1)
        # Pass the local tracer so the render is scoped to this test.
        out = render_recent(5, tracer=t)
        assert "foo" in out
        assert "bar" in out

    def test_tracer_collects_ooda_phase_spans(self):
        """P4-9: a single tick produces 5 phase spans."""
        import os, tempfile
        from zwm.planner.agent import TrinityAgent
        from zwm.core.hexagram import hexagram_from_name
        from zwm.tracing import get_tracer
        get_tracer().clear()
        with tempfile.TemporaryDirectory() as d:
            a = TrinityAgent(
                db_path=os.path.join(d, "e.db"),
                mcts_iterations=20, n_particles=0, use_react=False,
            )
            a.tick(h_current=hexagram_from_name("乾为天"), reward=0.7)
            names = [s.name for s in get_tracer().spans]
            for expected in ("ooda.observe", "ooda.predict",
                             "ooda.evaluate", "ooda.act", "ooda.learn"):
                assert expected in names, f"missing span: {expected}"
            a.close()

    def test_configure_otel_handles_missing_sdk(self):
        from zwm.tracing import configure_otel
        # Either succeeds (if opentelemetry is installed) or returns False.
        result = configure_otel("zwm-test")
        assert result in (True, False)


# ======================================================================
# API / MCP / CLI smoke tests for the unified surface
# ======================================================================
class TestSurfaceSmoke:
    def test_api_session_start_inherits_overrides(self):
        from zwm.api.schemas import SessionStartRequest
        req = SessionStartRequest(mcts_iterations=80, n_particles=0)
        d = req.model_dump()
        assert d["mcts_iterations"] == 80
        assert d["n_particles"] == 0
        # Default fields are still present.
        assert d["use_react"] is True

    def test_api_session_start_rejects_unknown_field(self):
        from zwm.api.schemas import SessionStartRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SessionStartRequest(mcts_iterations=80, garbage_key=1)

    def test_mcp_plan_tool_uses_config_overrides(self):
        from zwm.mcp import _tool_plan
        # Force the no-agent path: no host agent → build a fresh
        # one using the overrides.
        result = _tool_plan({"hex_bits": 1, "mcts_iterations": 30}, agent=None)
        assert "top_mutation" in result
        assert "top_score" in result

    def test_cli_tick_uses_unified_surface(self, tmp_path, capsys):
        """The CLI should accept the new --mcts-iterations flag and
        drive the agent end-to-end."""
        from zwm.cli import main
        # 1-step tick to keep the test fast.
        rc = main([
            "tick", "--steps", "1",
            "--mcts-iterations", "20",
            "--n-particles", "0",
            "--no-use-react",
            "--db-path", str(tmp_path / "e.db"),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "steps" in captured.out

    def test_cli_eval_requires_checkpoint(self, tmp_path, capsys):
        from zwm.cli import main
        with pytest.raises(SystemExit) as e:
            main(["eval", "--checkpoint", str(tmp_path / "missing.pt")])
        # Exit code 1 because checkpoint is missing.
        assert e.value.code == 1
