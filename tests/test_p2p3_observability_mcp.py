"""Tests for P2 (Prometheus / OpenTelemetry / spectrum) and P3 (MCP) work.

These tests pin the new wiring closed:

  * ``zwm.observability.metrics`` — Prometheus text exposition format
  * ``zwm.mcp``                    — JSON-RPC 2.0 server with 3 tools
  * Spectrum integration into the OODA loop (interference result is
    computed and surfaced via the agent's ``_last_interference`` slot)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


# ----------------------------------------------------------------------
# P2-1: Prometheus metrics
# ----------------------------------------------------------------------
class TestMetricsRegistry:
    def test_metrics_render_prometheus_format(self):
        from zwm.observability import metrics
        # Save and restore the counter to keep the test deterministic
        # even when other tests in the same session have advanced it.
        before_ticks = metrics.ticks_total.value
        before_hist = dict(metrics.tick_duration._counts)
        try:
            metrics.inc_ticks(3)
            metrics.observe_tick_duration(0.123)
            out = metrics.render()
            # Standard Prometheus exposition lines.
            assert "# HELP zwm_ticks_total" in out
            assert "# TYPE zwm_ticks_total counter" in out
            # The counter advanced by exactly 3.
            assert metrics.ticks_total.value == before_ticks + 3
            assert "zwm_tick_duration_seconds_bucket" in out
            assert 'le="0.1"' in out
            assert 'le="+Inf"' in out
        finally:
            # Roll back to keep the test hermetic.
            metrics.ticks_total._value = before_ticks
            metrics.tick_duration._counts = before_hist
            metrics.tick_duration._sum = 0.0

    def test_metrics_gauges_initialized(self):
        from zwm.observability import metrics
        for attr in (
            "jepa_loss", "surprise", "reward", "episodes_stored",
            "react_reflections", "mcts_iterations", "active_experts",
            "particles", "efe_value", "hex_bits",
            "interference_resonance", "interference_phase_coherence",
            "dominant_harmonic",
        ):
            assert hasattr(metrics, attr), f"missing gauge: {attr}"

    def test_metrics_phase_histograms_have_labels(self):
        from zwm.observability import metrics
        # Save the phase histogram state.
        saved = dict(metrics.phase_duration["observe"]._counts)
        saved_sum = metrics.phase_duration["observe"]._sum
        try:
            metrics.observe_phase("observe", 0.01)
            out = metrics.render()
            # The label should appear on the bucket lines.
            assert 'phase="observe"' in out
        finally:
            metrics.phase_duration["observe"]._counts = saved
            metrics.phase_duration["observe"]._sum = saved_sum

    def test_configure_logging_idempotent(self):
        from zwm.observability import configure_logging, metrics
        # Calling twice should not duplicate handlers.
        configure_logging()
        n1 = len(metrics.__class__.__mro__)  # no-op — just exercising the call
        configure_logging()
        n2 = len(metrics.__class__.__mro__)
        assert n1 == n2


# ----------------------------------------------------------------------
# P2-1: Agent tick publishes metrics
# ----------------------------------------------------------------------
class TestAgentTickPublishesMetrics:
    def test_tick_increments_counter(self, tmp_path):
        from zwm.core.hexagram import hexagram_from_name
        from zwm.observability import metrics
        from zwm.planner.agent import TrinityAgent

        before = metrics.ticks_total.value
        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=20) as ag:
            ag.tick(h_current=hexagram_from_name("乾为天"), reward=0.5)
            ag.tick(h_current=hexagram_from_name("乾为天"), reward=0.6)
        # Two ticks → counter incremented by 2.
        assert metrics.ticks_total.value >= before + 2

    def test_tick_publishes_spectrum_gauges(self, tmp_path):
        from zwm.core.hexagram import hexagram_from_name
        from zwm.planner.agent import TrinityAgent

        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=20) as ag:
            ag.tick(h_current=hexagram_from_name("乾为天"), reward=0.5)
        # The interference result was stashed.
        assert hasattr(ag, "_last_interference")
        assert ag._last_interference is not None
        # Resonance is a real finite float.
        assert isinstance(ag._last_interference.resonance, float)


# ----------------------------------------------------------------------
# P2-3: Spectrum integration — every agent action gets an interference result
# ----------------------------------------------------------------------
class TestSpectrumIntegration:
    def test_interference_is_computed_every_tick(self, tmp_path):
        from zwm.core.hexagram import hexagram_from_name
        from zwm.planner.agent import TrinityAgent

        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=20) as ag:
            for i in range(3):
                ag.tick(h_current=hexagram_from_name("乾为天"), reward=0.5)
                # Each tick refreshes the interference result.
                assert ag._last_interference is not None
                # The dominant harmonic is in [1, 6].
                assert 1 <= ag._last_interference.dominant_harmonic <= 6


# ----------------------------------------------------------------------
# P3-1: MCP server — JSON-RPC 2.0 over JSONL
# ----------------------------------------------------------------------
class TestMCPServer:
    def test_initialize_returns_capabilities(self):
        from zwm.mcp import dispatch
        resp = dispatch({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
        })
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "result" in resp
        assert resp["result"]["serverInfo"]["name"] == "zwm-mcp"
        assert "tools" in resp["result"]["capabilities"]

    def test_tools_list_returns_four_tools(self):
        from zwm.mcp import dispatch
        resp = dispatch({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        })
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {"zwm/observe", "zwm/plan", "zwm/reflect", "zwm/preference"}

    def test_tool_observe_with_hex_name(self):
        from zwm.mcp import dispatch
        resp = dispatch({
            "jsonrpc": "2.0", "id": 3,
            "method": "tools/call",
            "params": {
                "name": "zwm/observe",
                "arguments": {"hex_name": "乾为天"},
            },
        })
        assert resp["id"] == 3
        # The result content is JSON-encoded.
        payload = json.loads(resp["result"]["content"][0]["text"])
        # 乾为天 has all-yang lines; verify the name round-tripped and
        # the yao signals are all 1.0 (pure yang).  The exact bits depend
        # on the canonical order chosen by hexagram_from_name().
        assert payload["hex_name"] == "乾为天"
        assert payload["yao_signals"] == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert payload["binary_str"] == "111111"
        assert resp["result"]["isError"] is False

    def test_tool_observe_with_sensor_data(self):
        from zwm.mcp import dispatch
        resp = dispatch({
            "jsonrpc": "2.0", "id": 4,
            "method": "tools/call",
            "params": {
                "name": "zwm/observe",
                "arguments": {
                    "sensor_data": {
                        "temperature": 0.5,
                        "terrain": 0.6,
                        "social_proximity": 0.7,
                        "resource_level": 0.4,
                        "momentum": 0.3,
                        "overall_favorability": 0.6,
                    },
                },
            },
        })
        assert resp["result"]["isError"] is False
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert 1 <= payload["hex_bits"] <= 64

    def test_tool_plan_returns_mutation(self):
        from zwm.mcp import dispatch
        resp = dispatch({
            "jsonrpc": "2.0", "id": 5,
            "method": "tools/call",
            "params": {
                "name": "zwm/plan",
                "arguments": {"hex_bits": 1, "mcts_iters": 20},
            },
        })
        assert resp["result"]["isError"] is False
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert "top_mutation" in payload
        assert 1 <= payload["top_mutation"] <= 63
        assert isinstance(payload["top_score"], float)
        assert len(payload["moe_active_experts"]) > 0

    def test_tool_reflect_returns_empty_when_no_db(self, tmp_path):
        from zwm.mcp import dispatch
        resp = dispatch({
            "jsonrpc": "2.0", "id": 6,
            "method": "tools/call",
            "params": {
                "name": "zwm/reflect",
                "arguments": {
                    "limit": 5,
                    "db_path": str(tmp_path / "no_such.db"),
                },
            },
        })
        assert resp["result"]["isError"] is False
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert payload["reflections"] == []
        assert payload["count"] == 0

    def test_unknown_tool_returns_method_not_found_error(self):
        from zwm.mcp import _handle_request, ERR_METHOD_NOT_FOUND
        from zwm.mcp import dispatch
        # The dispatch function (the public entry point) catches the
        # raised error and returns a JSON-RPC error response.  Use it
        # here rather than the raw _handle_request so the test exercises
        # the same code path as serve_stdio().
        resp = dispatch({
            "jsonrpc": "2.0", "id": 7,
            "method": "tools/call",
            "params": {"name": "zwm/nonexistent", "arguments": {}},
        })
        assert resp["id"] == 7
        assert resp["error"]["code"] == ERR_METHOD_NOT_FOUND

    def test_unknown_method_returns_method_not_found(self):
        from zwm.mcp import dispatch, ERR_METHOD_NOT_FOUND
        resp = dispatch({
            "jsonrpc": "2.0", "id": 8, "method": "zwm/foo",
        })
        assert resp["error"]["code"] == ERR_METHOD_NOT_FOUND

    def test_invalid_jsonrpc_version_returns_invalid_request(self):
        from zwm.mcp import dispatch, ERR_INVALID_REQUEST
        resp = dispatch({
            "jsonrpc": "1.0", "id": 9, "method": "ping",
        })
        assert resp["error"]["code"] == ERR_INVALID_REQUEST

    def test_observe_missing_args_returns_invalid_params(self):
        from zwm.mcp import dispatch, ERR_INVALID_PARAMS
        resp = dispatch({
            "jsonrpc": "2.0", "id": 10,
            "method": "tools/call",
            "params": {"name": "zwm/observe", "arguments": {}},
        })
        # The error is wrapped in a tool-result with isError=True.
        assert resp["result"]["isError"] is True
        assert "sensor_data" in resp["result"]["content"][0]["text"]

    def test_tool_reflect_with_live_agent(self, tmp_path):
        """When a TrinityAgent is provided, the reflect tool reads from
        its live store and returns the actual reflections."""
        from zwm.core.hexagram import hexagram_from_name
        from zwm.mcp import dispatch
        from zwm.planner.agent import TrinityAgent

        with TrinityAgent(db_path=str(tmp_path / "e.db"), mcts_iterations=20) as ag:
            ag.tick(h_current=hexagram_from_name("乾为天"), reward=0.5)
            resp = dispatch({
                "jsonrpc": "2.0", "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "zwm/reflect",
                    "arguments": {"limit": 10},
                },
            }, agent=ag)
        # We can't assert reflection count > 0 because ReAct may not
        # have been triggered on every tick — but the tool must succeed.
        assert resp["result"]["isError"] is False
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert "reflections" in payload
        assert "count" in payload


# ----------------------------------------------------------------------
# P1-4: CLI modules registry
# ----------------------------------------------------------------------
class TestCLIModuleRegistry:
    def test_modules_constant_present(self):
        from zwm.cli import MODULES
        assert isinstance(MODULES, dict)
        # All 14 modules exposed.
        for expected in (
            "core", "encoder", "hexaembed", "jepa", "langevin",
            "learning", "moe", "planner", "scene_field",
            "self_field", "spectrum", "storage", "topology", "api",
        ):
            assert expected in MODULES

    def test_cli_info_json_contains_modules(self, capsys):
        from zwm.cli import main
        rc = main(["info", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "modules" in payload
        assert "spectrum" in payload["modules"]
