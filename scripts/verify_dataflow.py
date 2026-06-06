"""
Production-level end-to-end data flow verification for ZWM.

Covers all P0/P1/P2 fixes:
  P0: Rate limiting, dead output connections
  P1: Dimension drift detection, architecture dependency fixes
  P2: AgentBuilder, CLI train command, full OODA flow
"""
import os, sys, json, time, math, traceback, tempfile, pathlib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


class Verifier:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, fn):
        try:
            fn()
            self.passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, str(e), traceback.format_exc()))
            print(f"  ✗ {name}: {e}")

    def summary(self):
        print(f"\n{'='*60}")
        print(f"Results: {self.passed} passed, {self.failed} failed")
        if self.errors:
            for name, err, tb in self.errors:
                print(f"\n--- {name} ---")
                print(tb)
        return self.failed == 0


v = Verifier()

# ============================================================
# SECTION 1: Core Types & Data Format Consistency
# ============================================================
print("=== Section 1: Core Types & Data Format ===")

from zwm.core.yao import YANG, YIN, YaoLine
from zwm.core.trigram import Trigram, trigram_from_index
from zwm.core.hexagram import (
    Hexagram, hexagram_from_bits, hexagram_from_name, all_hexagrams,
)

v.check("YANG/YIN are singleton", lambda: YaoLine(True) is YANG and YaoLine(False) is YIN)
v.check("Trigram.QIAN.index == 7", lambda: Trigram.QIAN.index == 7)
v.check("Interlock identity: 乾->乾", lambda: hexagram_from_name("乾为天").interlock().name == "乾为天")
v.check("Reverse: 否->泰", lambda: hexagram_from_name("天地否").reverse().name == "地天泰")

# P1-arch: to_phase_vector method on Hexagram
pv = hexagram_from_bits(0).to_phase_vector()
v.check("Hexagram.to_phase_vector() returns 6 complex", lambda: (
    len(pv) == 6 and all(isinstance(c, complex) for c in pv)
))

# Unicode
v.check("All 64 hexagrams map to CJK Unified block", lambda: (
    all(0x4DC0 <= ord(h.unicode) <= 0x4DFF for h in all_hexagrams())
))

# ============================================================
# SECTION 2: Rate Limiting (P0 fix)
# ============================================================
print("=== Section 2: Rate Limiting ===")

from zwm.api.ratelimit import (
    TokenBucket, SlidingWindow, RateLimiterRegistry, require_rate_limit,
)

b = TokenBucket(capacity=10, refill_rate=5)
v.check("TokenBucket: consume within capacity", lambda: b.try_consume(3))
b2 = TokenBucket(capacity=1, refill_rate=0.01)
# drain it
b2.try_consume(1)
v.check("TokenBucket: empty bucket rejects", lambda: not b2.try_consume(1))

w = SlidingWindow(window_seconds=0.1, max_requests=3)
w.try_record(); w.try_record(); w.try_record()
v.check("SlidingWindow: 4th request in window rejected", lambda: not w.try_record())

rl = RateLimiterRegistry.instance()
allowed, retry, reason = rl.check_and_record("rest", "verify-test")
v.check("RateLimiterRegistry: check_and_record returns allowed", lambda: allowed and reason == "")
rl.reset()

# ============================================================
# SECTION 3: Dead Outputs Now Wired (P0 fix)
# ============================================================
print("=== Section 3: Dead Outputs Wired ===")

from zwm.scene_field.calendar import MultiScaleCalendar
from zwm.self_field.palace_graph import LuoshuGrid
from zwm.scene_field.unified_field import UnifiedField

# 3a: cosmic_phases flows through calendar_context
cal = MultiScaleCalendar()
cosmic = cal.cosmic_phases(2026)
ctx = cal.calendar_context(2026, cosmic_phases=cosmic)
v.check("cosmic_phases -> calendar_context passthrough", lambda: (
    ctx["元"] == cosmic["元"] and ctx["会"] == cosmic["会"]
    and ctx["运"] == cosmic["运"] and ctx["世"] == cosmic["世"]
))

# 3b: UnifiedField.snapshot() works without self_field import
h = hexagram_from_name("乾为天")
g = LuoshuGrid()
uf = UnifiedField.snapshot(h, g, 0.0)
v.check("UnifiedField.snapshot() produces 9-palace field", lambda: len(uf.luoshu_field) == 9)
v.check("UnifiedField.to_tensor() produces 29-dim vector", lambda: len(uf.to_tensor()) == 29)

# 3c: spectrum uses to_phase_vector() not lazy Hexagram import
from zwm.spectrum.complex_phase import HexagramPhaseVector
h42 = hexagram_from_bits(42)
pv42 = HexagramPhaseVector.from_hexagram(h42)
v.check("from_hexagram via to_phase_vector() works", lambda: len(pv42.phases) == 6)

# 3d: score_surface uses cosine_similarity
from zwm.langevin.score import score_surface
s_same = score_surface(hexagram_from_bits(0), hexagram_from_bits(0))
s_diff = score_surface(hexagram_from_bits(0), hexagram_from_bits(63))
v.check("score_surface: same hex > opposite hex", lambda: s_same > s_diff)

# 3e: FineGrainedSparseMoE has train_toward
from zwm.moe.sparse_activation import FineGrainedSparseMoE
fg = FineGrainedSparseMoE()
v.check("FineGrainedSparseMoE.train_toward exists", lambda: hasattr(fg, "train_toward"))

# 3f: EpisodicStore.update_context
from zwm.storage.episodic_db import EpisodicStore
store = EpisodicStore(db_path=":memory:", use_index=False)
ep_id = store.store(main_bits=0, evolved_bits=1, reward=0.5)
store.update_context(ep_id, {"test_key": "verified"})
stored = store.query_recent(1)[0]
v.check("update_context persists to DB", lambda: stored.get("context", {}).get("test_key") == "verified")
store.close()

# ============================================================
# SECTION 4: Dimension Drift Detection (P1 fix)
# ============================================================
print("=== Section 4: Dimension Drift Detection ===")

from zwm.learning.checkpoint import _validate_dimensional_compatibility, _DimensionalDriftError
from zwm.jepa.predictor import JEPAPredictor

v.check("_DimensionalDriftError is Exception subclass", lambda: issubclass(_DimensionalDriftError, Exception))

class MockAgent:
    pass

ma = MockAgent()
ma.jepa = JEPAPredictor(input_dim=106, latent_dim=64)
ma.jepa.init_value_head()
ma.jepa.init_vq(num_codes=64, beta=0.25)
ma._square_learnable = None
ma._mm_encoder = None

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    pt_path = f.name

torch_state = {
    "jepa_context": ma.jepa.context_encoder.state_dict(),
    "jepa_predictor": ma.jepa.predictor.state_dict(),
    "jepa_target": ma.jepa.target_encoder.state_dict(),
    "vq": ma.jepa._vq.state_dict(),
    "value_head": ma.jepa._value_head.state_dict(),
}
torch.save(torch_state, pt_path)

# Should pass for matching dimensions
_validate_dimensional_compatibility(ma, torch_state, pathlib.Path(pt_path))
v.check("Drift detection passes for matching dims", lambda: True)

# Inject wrong-dim value head
bad_state = dict(torch_state)
bad_state["value_head"] = {
    "0.weight": torch.randn(32, 32),  # expect 64
    "0.bias": torch.randn(32),
    "2.weight": torch.randn(1, 32),
    "2.bias": torch.randn(1),
}
drift_caught = False
try:
    _validate_dimensional_compatibility(ma, bad_state, pathlib.Path(pt_path))
except _DimensionalDriftError:
    drift_caught = True
v.check("Drift detection catches dimension mismatch", lambda: drift_caught)

os.unlink(pt_path)

# ============================================================
# SECTION 5: AgentBuilder (P2 fix)
# ============================================================
print("=== Section 5: AgentBuilder ===")

from zwm.planner.agent_builder import AgentBuilder
from zwm.planner.agent_config import TrinityConfig
from zwm.planner.agent import TrinityAgent

cfg = TrinityConfig(db_path=":memory:", mcts_iterations=10, n_particles=0, use_react=False)
agent = TrinityAgent.__new__(TrinityAgent)
object.__setattr__(agent, "config", cfg)
# Set essential attributes that __init__ normally sets
agent._step_count = 0
agent._square_learnable = None

builder = AgentBuilder(config=cfg).with_agent(agent)
builder.build_planning()
v.check("AgentBuilder.build_planning() sets planner", lambda: hasattr(agent, "planner"))
builder.build_perception()
v.check("AgentBuilder.build_perception() sets encoder/calendar", lambda: (
    hasattr(agent, "encoder") and hasattr(agent, "calendar")
))
builder.build_learning_systems()
v.check("AgentBuilder.build_learning_systems() sets learner/hebbian", lambda: (
    hasattr(agent, "learner") and hasattr(agent, "hebbian")
))
builder.build_constitution()
v.check("AgentBuilder.build_constitution() sets constitution", lambda: (
    hasattr(agent, "constitution")
))

# ============================================================
# SECTION 6: CLI Train Command (P2 fix)
# ============================================================
print("=== Section 6: CLI Train Command ===")

from zwm.cli import build_parser
parser = build_parser()
subs = list(parser._subparsers._group_actions[0].choices.keys())
v.check("train subcommand registered", lambda: "train" in subs)
v.check(f"CLI has {len(subs)} subcommands (>=14)", lambda: len(subs) >= 14)

# ============================================================
# SECTION 7: Full OODA Data Flow (End-to-End)
# ============================================================
print("=== Section 7: Full OODA Data Flow ===")

os.environ["ZWM_SIZE_PRESET"] = "test"

agent_config = TrinityConfig(
    db_path=":memory:",
    mcts_iterations=20,
    n_particles=0,
    use_react=False,
    use_diffusion=False,
    use_fsdp2=False,
    hierarchical=False,
    learnable_encoder=False,
)

h = hexagram_from_name("乾为天")
with TrinityAgent(config=agent_config) as agent:
    for tick_idx in range(5):
        reward = 0.5 + 0.4 * math.sin(tick_idx / 3.0)
        report = agent.tick(
            h_current=h, reward=reward,
            year=2026, month=6, day=6, hour=12 + tick_idx,
        )
        v.check(f"Tick {tick_idx}: valid h_current", lambda r=report: r.h_current is not None)
        v.check(f"Tick {tick_idx}: valid h_next", lambda r=report: r.h_next is not None)
        v.check(f"Tick {tick_idx}: top_mutation 1-63", lambda r=report: 1 <= r.top_mutation <= 63)
        v.check(f"Tick {tick_idx}: episode_id > 0", lambda r=report: r.episode_id > 0)
        v.check(f"Tick {tick_idx}: plan result exists", lambda r=report: r.plan is not None)
        v.check(f"Tick {tick_idx}: moe_active_experts non-empty",
                lambda r=report: len(r.plan.moe_active_experts) > 0)
        v.check(f"Tick {tick_idx}: surprise is finite",
                lambda r=report: math.isfinite(r.surprise) and r.surprise >= 0)
        v.check(f"Tick {tick_idx}: top_score is finite",
                lambda r=report: math.isfinite(r.top_score))
        h = report.h_next

    # Post-loop state verification
    v.check("Planner visit_counts populated", lambda: len(agent.planner.visit_counts) > 0)
    v.check("5 episodes stored", lambda: agent.store.count() == 5)
    v.check("Learner total_visits == 5", lambda: agent.learner.total_visits == 5)
    v.check("_cosmic_phases set on agent", lambda: (
        hasattr(agent, "_cosmic_phases") and agent._cosmic_phases is not None
    ))
    v.check("_step_count incremented", lambda: agent._step_count == 5)

    # Verify last-hexagram progression (shouldn't stay on same hexagram)
    last_episodes = agent.store.query_recent(5)
    hex_changes = sum(
        1 for ep in last_episodes
        if ep.get("main_hex_bits") != ep.get("evolved_hex_bits")
    )
    v.check("Most ticks produced hexagram changes", lambda: hex_changes >= 3)

    report_last = report

v.check("Final h_next != h_current (evolution happened)", lambda: (
    report_last.h_current.normal_order != report_last.h_next.normal_order
))

os.environ.pop("ZWM_SIZE_PRESET", None)

# ============================================================
# SECTION 8: Train Command Smoke Test (quick)
# ============================================================
print("=== Section 8: Train Command Smoke Test ===")
import argparse

train_args = argparse.Namespace(
    steps=3,
    checkpoint_every=10,
    seed="乾为天",
    year=2026,
    period=5.0,
    json=False,
    db_path=":memory:",
    mcts_iterations=10,
    n_particles=0,
    use_react=False,
    use_diffusion=False,
    use_fsdp2=False,
    hierarchical=False,
    learnable_encoder=False,
    checkpoint_path=None,
    semantic_path=None,
    grid=None,
    use_trainable_vsa=False,
    quantize=None,
    topology_max_depth=1,
    enable_constitution=False,
)

from zwm.cli import cmd_train
rc = cmd_train(train_args)
v.check("zwm train 3-step smoke test returns 0", lambda: rc == 0)

# ============================================================
print()
ok = v.summary()
sys.exit(0 if ok else 1)
