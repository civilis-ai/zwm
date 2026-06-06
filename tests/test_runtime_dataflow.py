from __future__ import annotations


def _sensors() -> dict[str, float]:
    return {
        "temperature": 0.62,
        "terrain": 0.41,
        "social_proximity": 0.73,
        "resource_level": 0.58,
        "momentum": 0.35,
        "overall_favorability": 0.67,
    }


def test_self_state_preserves_trinity_vertical_targets():
    from zwm.self_field.self_state import SelfState

    ss = SelfState(day_gan="庚")
    seen = []
    for _ in range(10):
        target = ss.next_to_explore()
        seen.append(target)
        ss.record_visit(target)

    assert set(seen) == {1, 2, 3, 4, 6, 7, 8, 9, 10, 11}
    assert ss.to_luoshu_palace(10) == 5
    assert ss.to_luoshu_palace(11) == 5


def test_self_state_has_planar_target_for_efe_planner():
    from zwm.self_field.self_state import SelfState

    ss = SelfState(day_gan="庚")
    seen = []
    for _ in range(8):
        target = ss.next_spatial_to_explore()
        assert target in {1, 2, 3, 4, 6, 7, 8, 9}
        seen.append(target)
        ss.record_visit(target)

    assert set(seen) == {1, 2, 3, 4, 6, 7, 8, 9}


def test_agent_ooda_consumes_provided_sensor_data(tmp_path):
    from zwm.planner.agent import TrinityAgent

    sensors = _sensors()
    with TrinityAgent(
        db_path=str(tmp_path / "e.db"),
        mcts_iterations=5,
        n_particles=0,
        use_react=False,
    ) as agent:
        report = agent.observe_predict_evaluate_act(
            sensor_data=sensors,
            reward=0.6,
            target_palace=1,
        )

        assert agent._last_sensor_data == sensors
        assert agent._last_hex_field.shape == (64, 6)
        assert agent._last_target_palace == 1
        assert agent._last_prediction.z_world.shape[0] == agent.jepa.input_dim
        assert report.plan is agent._last_plan


def test_runtime_tick_exposes_complete_ooda_state(tmp_path):
    from zwm.runtime import ZWMEngine

    sensors = _sensors()
    engine = ZWMEngine(
        day_gan="庚",
        db_path=str(tmp_path / "runtime.db"),
        mcts_iterations=5,
        n_particles=0,
        use_react=False,
        enable_tracing=False,
        enable_otlp=False,
    ).activate()
    try:
        state = engine.tick(
            sensor_data=sensors,
            reward=0.6,
            target_palace=1,
            language_text="向北探索",
        )

        assert state.sensor_data == sensors
        assert state.hex_field is not None
        assert state.hex_field.shape == (64, 6)
        assert state.plan is not None
        assert state.report is not None
        assert state.z_world is not None
        assert state.z_world.shape[0] == engine.agent.jepa.input_dim
        assert state.z_pred is not None
        assert state.target_palace == 1
        assert state.exploration_target == 1
        assert engine.agent._last_sensor_data == sensors
        assert engine.self_state.palace_visits[1] == 1
    finally:
        engine.close()


def test_otlp_exporter_is_opt_in_by_default(monkeypatch):
    from zwm.tracing import configure_otlp_from_env

    monkeypatch.delenv("ZWM_OTLP_ENABLED", raising=False)
    assert configure_otlp_from_env() is False
