"""P2-1 (audit) — OODA 5 阶段方法 (Observe/Predict/Evaluate/Act/Learn)。

从 ``agent.py`` 抽出,使 ``TrinityAgent.tick()`` 调度逻辑清晰。

每个 _phase_* 函数:
  * 只接收显式参数 + agent 引用
  * 返回该阶段的产物 (不直接修改持久态,持久化由 Phase 5 负责)
  * 内部 try/except 保证不破坏主循环
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from zwm.planner.agent_data import GOOD_OUTCOME, TickPrediction
from zwm.planner.codon import codon_amino_acid, hexagram_to_codon
from zwm.planner.mutations import classify_mutation
from zwm.planner.agent_priors import (
    _calendar_context,
    _combined_priors,
    _next_palace_to_explore,
    _world_vector,
)
from zwm.planner.agent_train import (
    _counterfactual_train,
    _dreamer_replay,
    _gae_flush,
    _joint_train_step,
    _log_telemetry,
    _periodic_denoiser_training,
    _train_multimodal,
)
from zwm.scene_field.calendar import GanzhiTime

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zwm.core.hexagram import Hexagram
    from zwm.jepa.predictor import HierarchicalJEPAPredictor
    from zwm.planner.agent import TrinityAgent
    from zwm.planner.loop import PlanResult
    from zwm.scene_field.unified_field import UnifiedField
    from zwm.self_field.palace_graph import LuoshuGrid


# ================= Phase 1 — OBSERVE =================
def _observe(
    agent: "TrinityAgent",
    grid: "LuoshuGrid",
    year: int,
    month: int,
    day: int,
    hour: int,
    target_palace: int | None,
) -> int:
    """Read all time/space signals; pick the next target palace.

    P0: agent.self_state 作为"我"的单一来源。SelfState owns the
    trinity target (八方 + 上/下); this phase projects it to the
    planar Luoshu palace consumed by EFE/MCTS.
    """
    if target_palace is None:
        exploration_target = agent.self_state.next_spatial_to_explore()
        target_palace = agent.self_state.to_luoshu_palace(exploration_target)
    else:
        exploration_target = target_palace
        target_palace = agent.self_state.to_luoshu_palace(target_palace)
    agent._last_exploration_target = exploration_target

    # 构建完整的 TimeContext
    from zwm.scene_field.time_context import TimeContext
    agent.ganzhi = GanzhiTime.from_date(year, month, day, hour)
    cosmic = agent.calendar.cosmic_phases(year)
    agent._cosmic_phases = cosmic
    agent._time_context = TimeContext.compute(
        year, month, day, hour,
        calendar=agent.calendar, ganzhi=agent.ganzhi,
    )
    return target_palace


# ================= Phase 2 — PREDICT =================
def _predict(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    grid: "LuoshuGrid",
    time_phase: float,
    day_gan: str | None,
    year: int, month: int, day: int, hour: int,
) -> TickPrediction:
    """Encode the trinity world vector and run JEPA forward.

    支持两种编码路径:
      - 场编码 (新): 传感器 → FieldEncoder → (64,6) 场 → FieldGNN → z_sq
      - 单卦编码 (旧): 传感器 → RuleBasedEncoder → 1卦 → SquareGNN → z_sq
    """
    cosmic_phases = getattr(agent, "_cosmic_phases", None)
    calendar_context = _calendar_context(
        agent, year, month, day, hour, cosmic_phases=cosmic_phases,
    )

    # ─── 多场编码路径 (新) ───
    # 方图场 + 圆图时间场 + 干支场 + 元会运世场 → 融合 → z_world
    if (agent.field_encoder is not None and agent._field_gnn is not None
            and hasattr(agent, "_last_sensor_data")):
        sensor_data = agent._last_sensor_data
        try:
            # 1) 传感器 → 方图场 (64, 6)
            square_field = agent.field_encoder.encode(sensor_data)
            agent._last_hex_field = square_field

            # 2) TimeContext → 时间多场 (圆图+干支+元会运世+节气)
            from zwm.scene_field.time_field import TimeFieldEncoder, MultiFieldJoint
            tc = getattr(agent, "_time_context", None)
            time_fields = None
            if tc is not None:
                tfe = TimeFieldEncoder()
                time_fields = tfe.encode_all(tc)

            # 3) 多场融合 → z_world (256 dim)
            joint = MultiFieldJoint(
                square_field=square_field,
                time_fields=time_fields,
                square_gnn=agent._field_gnn,
            )
            z_world_multi = joint.encode()

            # 4) 如果 JEPA input_dim 不匹配, pad 到匹配
            target_dim = agent.jepa.input_dim
            if len(z_world_multi) < target_dim:
                z_world = np.pad(z_world_multi, (0, target_dim - len(z_world_multi)),
                                 'constant').astype(np.float32)
            else:
                z_world = z_world_multi[:target_dim].astype(np.float32)

            # 构建 UnifiedField — _act 需要 .evolve() 方法
            from zwm.scene_field.unified_field import UnifiedField
            uf = UnifiedField()
            _, world = uf.encode(
                h_current, grid, calendar_context=calendar_context,
                day_gan=day_gan,
            )

            z_pred = agent.jepa.predict(z_world)
            if isinstance(z_pred, dict):
                z_pred = z_pred["short"]

            _log.debug("Multi-field encode: z_world=%d dim, square+%stime",
                      len(z_world_multi), "" if time_fields else "no ")

            return TickPrediction(
                z_world=z_world, z_pred=z_pred,
                z_var=_compute_uncertainty(agent, z_world),
                world=world, calendar_context=calendar_context,
            )
        except Exception as exc:
            _log.debug("Multi-field path failed: %s; falling back", exc)

    # ─── 单卦编码路径 (旧 — 回退) ───
    z_world, world = _world_vector(
        agent, h_current, time_phase, grid,
        day_gan=day_gan, calendar_context=calendar_context,
    )
    # Pad z_world to match JEPA input_dim when falling back from the
    # multi-field path (256-dim) to the single-hexagram path (106-dim).
    target_dim = agent.jepa.input_dim
    if len(z_world) < target_dim:
        z_world = np.pad(z_world, (0, target_dim - len(z_world)),
                         'constant').astype(np.float32)
    else:
        z_world = z_world[:target_dim].astype(np.float32)
    z_pred = agent.jepa.predict(z_world)
    if isinstance(z_pred, dict):
        z_pred = z_pred["short"]

    z_var = _compute_uncertainty(agent, z_world)

    # Particle filter: advance belief through the learned model.
    # The particle filter operates in JEPA latent space (64-dim) and
    # its transition is the JEPA's pure-latent predictor (which maps
    # a 64-dim latent to a 64-dim predicted latent — the same network
    # the main world-model call uses internally).  This avoids
    # padding shenanigans: the particles stay in the JEPA's
    # representation space, not the 106-dim world vector space.
    if agent._particle_filter is not None:
        try:
            # Cache the 64-dim latent of the *current* state so the
            # transition can return a sensible default if the JEPA
            # latent predictor is unavailable.
            try:
                z_latent_now = agent.jepa.context_encode(z_world)
                z_latent_now = np.asarray(z_latent_now, dtype=np.float32).flatten()[:64]
                if len(z_latent_now) < 64:
                    z_latent_now = np.concatenate([
                        z_latent_now,
                        np.zeros(64 - len(z_latent_now), dtype=np.float32),
                    ])
            except Exception:
                z_latent_now = None
            agent._last_latent = z_latent_now

            def _transition(z_latent: np.ndarray) -> np.ndarray:
                """Map a 64-dim latent particle through the JEPA predictor.

                Operates entirely in 64-dim JEPA latent space:
                  1. Use ``jepa.predict_latent`` if the JEPA exposes it
                     (preferred — runs the predictor head in latent space).
                  2. Fallback: pad to 106-dim and call ``jepa.predict``,
                     then take the 64-dim output.
                """
                z_latent = np.asarray(z_latent, dtype=np.float32).flatten()
                if len(z_latent) < 64:
                    z_latent = np.concatenate([
                        z_latent,
                        np.zeros(64 - len(z_latent), dtype=np.float32),
                    ])
                # Preferred path: pure-latent predictor.
                if hasattr(agent.jepa, "predict_latent"):
                    try:
                        out = agent.jepa.predict_latent(z_latent[:64])
                        out = np.asarray(out, dtype=np.float32).flatten()[:64]
                        if len(out) == 64:
                            return out
                    except Exception as exc:
                        _log.debug("OODA phase error: %s", exc)
                # Fallback: identity transition if no latent predictor.
                return z_latent[:64]
            agent._particle_filter.predict(_transition)
        except Exception as exc:
            _log.warning("ParticleFilter.predict failed: %s", exc)

    return TickPrediction(
        z_world=z_world,
        z_pred=z_pred,
        z_var=z_var,
        world=world,
        calendar_context=calendar_context,
    )


def _compute_uncertainty(agent: "TrinityAgent", z_world: np.ndarray) -> np.ndarray | None:
    """Run the JEPA uncertainty head; return mean variance or None."""
    try:
        if hasattr(agent.jepa, "predict_with_uncertainty_flat"):
            _, z_var = agent.jepa.predict_with_uncertainty_flat(z_world)
        elif hasattr(agent.jepa, "predict_with_uncertainty"):
            _, z_var = agent.jepa.predict_with_uncertainty(z_world)
        else:
            return None
        if isinstance(z_var, dict):
            z_var = z_var["short"]
        return np.asarray(z_var, dtype=np.float32)
    except Exception as exc:
        _log.debug("Uncertainty head unavailable: %s", exc)
        return None


# ================= Phase 3 — EVALUATE =================
def _evaluate(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    grid: "LuoshuGrid",
    time_phase: float,
    target_palace: int,
    day_gan: str | None,
    prediction: TickPrediction,
    vision_features: np.ndarray | None,
    language_features: np.ndarray | None,
) -> "PlanResult":
    """Run TrinityPlanner.plan() with all signals primed.

    2026 SOTA: when the ReAct loop is enabled, run a reasoning-
    acting-observing cycle *before* the planner to enrich the priors
    with tool-derived insights (memory, harmony, risk, topology, time).
    """
    mask_priors = _combined_priors(agent, h_current)
    learned_value = _learned_value(agent, prediction.z_world)

    # ReAct / Tool-Use: run the reasoning loop to gather additional
    # signals before planning.  The ReAct result's tool_scores are
    # blended into the mask_priors to bias the MCTS search toward
    # the tool-recommended mutations.
    react_bonus = np.zeros(64, dtype=np.float32)
    if agent._react_loop is not None:
        try:
            from zwm.planner.react import ReActLoop
            react_result = agent._react_loop.run(
                h_current, grid, target_palace, time_phase,
            )
            # Use the ReAct confidence and recommendation to bias
            # the priors.  High confidence → stronger bias.
            confidence = react_result.confidence
            recommendation = react_result.recommendation
            if recommendation == "caution_high_risk":
                # Penalise high-risk mutations.
                risk = react_result.tool_scores.get("risk_assessor", 0.0)
                react_bonus -= risk * confidence * 0.1
            elif recommendation == "proceed_high_harmony":
                # Boost the current direction.
                react_bonus += confidence * 0.05
            elif recommendation == "follow_precedent":
                # Boost memory-supported mutations.
                react_bonus += confidence * 0.03
            # Store the ReAct result for telemetry.
            agent._last_react_result = react_result
        except Exception as exc:
            _log.debug("ReAct loop failed: %s", exc)

    # Blend ReAct bonus into mask_priors (now a dict[int, float]).
    if np.any(react_bonus != 0) and mask_priors:
        for m in list(mask_priors.keys()):
            if 1 <= m <= 63:
                mask_priors[m] = mask_priors[m] + float(react_bonus[m])
        # Remove near-zero or negative priors.
        mask_priors = {m: w for m, w in mask_priors.items() if w > 0.001}

    # P0-dead-output: consume _last_topology_path as spatial exploration
    # prior.  The topology walk from the previous ACT phase describes
    # which sub-palace the agent landed in; we boost mutations that lead
    # toward *unvisited* palaces at the next depth level.  This closes
    # the loop on the RecursiveTopology work done in _act().
    topo_path = getattr(agent, "_last_topology_path", None)
    if topo_path is not None and hasattr(agent, "topology") and agent.topology is not None:
        try:
            # The last node in the path is where we are; find its children.
            current_node = topo_path[-1]
            unvisited_children = [
                c for c in current_node.children
                if agent._palace_visits.get(c.luoshu_number, 0) == 0
            ]
            if unvisited_children:
                # Boost priors that lead toward unexplored sub-palaces.
                for child in unvisited_children:
                    # Map child palace number → mutation mask bias.
                    # The child's luoshu_number (1-9) maps loosely to
                    # mutation groups: bits 0-2 for position 1-3,
                    # bits 3-5 for position 4-6, wrap for 7-9.
                    child_pos = (child.luoshu_number - 1) % 9
                    for m in range(1, 64):
                        if (m >> (child_pos // 3 * 3)) & 0b111 == (child_pos % 3 + 1):
                            mask_priors[m] = mask_priors.get(m, 0.0) + 0.02
        except Exception as exc:
            _log.debug("Topology-guided exploration bonus failed: %s", exc)

    # P0-dead-output: consume _last_multimodal_emb as perception prior.
    # The multimodal embedding from the previous tick (vision+text+sensor
    # fusion) is used to bias the current tick's exploration toward
    # hexagram states that are semantically similar to the prior
    # sensory context.  This closes the loop on the MultimodalEncoder
    # work done in _act().
    mm_emb = getattr(agent, "_last_multimodal_emb", None)
    if mm_emb is not None and len(mm_emb) > 0 and mask_priors:
        try:
            # Correlate the multimodal embedding with each prior mask's
            # hexagram via VSA encoding — higher VSA similarity means
            # the mask leads to a state that resonates with the prior
            # sensory context.
            for m in list(mask_priors.keys()):
                h_mut = h_current.mutate(m)
                vsa = agent.vsa.encode_hexagram(h_mut.normal_order)
                # Cosine similarity between VSA vector and multimodal embedding
                # (when dimensions match).
                if len(vsa) == len(mm_emb):
                    sim = float(np.dot(vsa, mm_emb) / (
                        np.linalg.norm(vsa) * np.linalg.norm(mm_emb) + 1e-8
                    ))
                    mask_priors[m] += 0.05 * sim
        except Exception as exc:
            _log.debug("Multimodal prior consumption failed: %s", exc)

    # Latent-value function for particle-filter EFE: maps a 64-dim
    # JEPA latent directly to a learned scalar via the value head,
    # replacing the bit-hack that falsely decoded latent dims as hexagram bits.
    latent_value_fn = None
    if agent.jepa._value_head is not None:
        import torch as _t
        def _lvfn(z: np.ndarray) -> float:
            try:
                z_t = _t.from_numpy(
                    np.asarray(z, dtype=np.float32).flatten()[:agent.jepa.latent_dim]
                ).unsqueeze(0)
                v = agent.jepa._value_head(z_t)
                v_arr = np.asarray(v.detach())
                return float(v_arr.item()) if v_arr.ndim == 0 else float(v_arr.flat[0])
            except Exception:
                return 0.0
        latent_value_fn = _lvfn

    return agent.planner.plan(
        h_current,
        grid=grid,
        time_phase=time_phase,
        target_palace=target_palace,
        day_gan=day_gan,
        preference_weights=dict(agent.learner.preference_weights),
        mask_priors=mask_priors,
        palace_visit_counts=dict(agent._palace_visits),
        intrinsic_fn=agent.learner.novelty_bonus,
        beta_curiosity=agent.curiosity.beta,
        value_fn=_value_fn_builder(agent, prediction.calendar_context, grid, time_phase, day_gan),
        particle_filter=agent._particle_filter,
        learned_value=learned_value,
        uncertainty_scale=(
            float(np.mean(prediction.z_var)) if prediction.z_var is not None else 0.0
        ),
        log_evidence=learned_value,
        latent_value_fn=latent_value_fn,
    )


def _learned_value(agent: "TrinityAgent", z_world: np.ndarray) -> float | None:
    """Read the learned V(z) head as a float scalar for EFE scaling."""
    try:
        if hasattr(agent.jepa, "value_flat"):
            v = agent.jepa.value_flat(z_world)
        else:
            v = agent.jepa.value(z_world)
        if isinstance(v, dict):
            v = v["short"]
        v_arr = np.asarray(v)
        return float(v_arr.item()) if v_arr.ndim == 0 else float(v_arr.flat[0])
    except Exception as exc:
        _log.debug("Value head unavailable: %s", exc)
        return None


def _value_fn_builder(
    agent: "TrinityAgent",
    calendar_context: dict,
    grid: "LuoshuGrid",
    time_phase: float,
    day_gan: str | None,
):
    """Closure that returns V(h) by re-encoding the world vector for a candidate h."""
    if agent.jepa._value_head is None:
        return None
    def value_fn(h):
        z = _world_vector(
            agent, h, time_phase, grid, day_gan=day_gan,
            calendar_context=calendar_context,
        )[0]
        return float(agent.jepa.value(z)[0])
    return value_fn


# ================= Phase 4 — ACT =================
def _act(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    grid: "LuoshuGrid",
    time_phase: float,
    day_gan: str | None,
    prediction: TickPrediction,
    result: "PlanResult",
) -> tuple["Hexagram", "UnifiedField", float, str, str, str]:
    """Apply the chosen mutation; measure world-model surprise."""
    h_next = result.chain.evolved
    world_next = prediction.world.evolve(
        result.top_mutation,
        new_time_phase=time_phase,
        day_gan=day_gan,
    )

    # P2-3: Compute the 复频谱 (complex spectrum) interference result for
    # the chosen mutation.  This wires the spectrum module into the
    # agent's loop: every action is now characterised by its harmonic
    # signature (resonance, phase coherence, dominant harmonic), and
    # the result is recorded in the tick telemetry so the /metrics
    # endpoint and the inspector can surface it.
    try:
        from zwm.spectrum.complex_phase import HexagramPhaseVector
        from zwm.spectrum.frequency import FrequencySpectrum
        from zwm.spectrum.interference import compute_interference
        pv = HexagramPhaseVector.from_hexagram(h_next)
        spec = FrequencySpectrum(pv)
        interference = compute_interference(spec)
        # Stash on the agent for downstream /metrics consumers.
        agent._last_interference = interference
    except Exception as exc:
        _log.debug("Spectrum interference failed: %s", exc)

    # Action-conditioned re-prediction (V-JEPA 2-AC style).
    z_pred_actioned = agent.jepa.predict(prediction.z_world, mask=result.top_mutation)
    if isinstance(z_pred_actioned, dict):
        z_pred_actioned = z_pred_actioned["short"]

    z_actual = np.concatenate([
        agent.joint.encode(h_next, time_phase),
        np.asarray(world_next.to_tensor(), dtype=np.float32),
    ]).astype(np.float32)
    # Pad z_actual to match JEPA input_dim when the single-hexagram
    # path produces fewer dimensions than the multi-field path expects.
    jepa_input_dim = agent.jepa.input_dim
    if len(z_actual) < jepa_input_dim:
        z_actual = np.pad(z_actual, (0, jepa_input_dim - len(z_actual)),
                          'constant').astype(np.float32)
    else:
        z_actual = z_actual[:jepa_input_dim].astype(np.float32)
    z_target = agent.jepa.target_latent(z_actual)
    surprise = float(((z_pred_actioned - z_target) ** 2).mean())

    _multi_scale_surprise(agent, surprise, prediction.z_world)

    # F4: RecursiveTopology — wire the multi-scale palace tree into
    # the ACT phase.  When the agent has a topology, descend the
    # tree from the root, walking the chosen mutation into a sub-
    # palace path.  The selected sub-palace is recorded as a
    # structured signal so downstream consumers (A2A consensus,
    # log inspector) can see how the action maps onto the 81 / 729
    # palace resolution.  ``expand_topology()`` is the public entry
    # point — the agent field is built via ``_init_topology``.
    try:
        topo = getattr(agent, "topology", None)
        if topo is not None:
            root = topo.root
            # R6: use ``bit_count(mutation_mask)`` to pick a child.
            # The previous ``mutation_mask % 9`` was biased toward
            # low child indices (since masks are typically small),
            # so 89% of ticks landed on child 0.  ``bit_count`` is
            # a 0..6 hash that distributes much more evenly.
            mutation_mask = int(result.top_mutation)
            n_root_children = max(1, len(root.children))
            n_sub_children = max(1, len(root.children[0].children)) if root.children else 1
            level1_idx = bin(mutation_mask).count("1") % n_root_children
            level1 = root.children[level1_idx]
            level2_idx = bin(mutation_mask).count("1") % n_sub_children if level1.children else 0
            level2 = level1.children[level2_idx] if level1.children else level1
            agent._last_topology_path = (root, level1, level2)
            agent._last_topology_bagua = level2.bagua
            agent._last_topology_direction = level2.direction
    except Exception as exc:
        _log.debug("RecursiveTopology walk failed: %s", exc)

    # F6: MultimodalEncoder — if the agent has been supplied with
    # vision_features or language_text during observe(), the encoder
    # (lazily created in ``_init_perception``) is now called to
    # produce a cross-modal embedding for the chosen action.  This
    # replaces the "skeleton without meat" P0-2 self-claim — the
    # encoder is now actually consumed by the OODA loop.
    try:
        if getattr(agent, "_multimodal", None) is not None:
            vision = getattr(agent, "_last_vision_features", None)
            lang_text = getattr(agent, "_last_language_text", None)
            if vision is not None or lang_text is not None:
                agent._last_multimodal_emb = agent._multimodal.encode_multimodal(
                    visual_features=vision,
                    text_embedding=agent._multimodal.encode_text(lang_text) if lang_text else None,
                    sensor_data={"h_next": h_next.normal_order},
                )
    except Exception as exc:
        _log.debug("MultimodalEncoder encode failed: %s", exc)

    try:
        codon = hexagram_to_codon(h_current.normal_order)
        codon_aa = codon_amino_acid(codon)
    except Exception as exc:
        _log.debug("Codon lookup failed: %s", exc)
        codon = "???"
        codon_aa = "Unknown"
    mutation_class = classify_mutation(result.top_mutation)

    return h_next, world_next, surprise, codon, codon_aa, mutation_class


def _multi_scale_surprise(agent: "TrinityAgent", surprise: float, z_world: np.ndarray) -> None:
    """Drive curiosity beta with long-horizon surprise from the hierarchical model."""
    from zwm.jepa.predictor import HierarchicalJEPAPredictor
    if not isinstance(agent.jepa, HierarchicalJEPAPredictor):
        return
    try:
        z_pred_dict = agent.jepa.predict(z_world)
        multi_surprise = {
            scale: float(((z_s - agent.jepa.target_latent(z_world)) ** 2).mean())
            for scale, z_s in z_pred_dict.items()
        }
        if "long" in multi_surprise:
            agent.curiosity.beta_initial *= (1.0 + 0.1 * min(multi_surprise["long"], 1.0))
    except Exception as exc:
        _log.debug("Multi-scale surprise failed: %s", exc)


# ================= Phase 5 — LEARN =================
def _learn_world_update(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    h_next: "Hexagram",
    grid: "LuoshuGrid",
    time_phase: float,
    result: "PlanResult",
    reward: float,
    world_next: "UnifiedField",
    world_curr: "UnifiedField | None",
    year: int, month: int, day: int, hour: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """AUDIT-A3 (split): compute the world-model gradient signal.

    Returns (z_world, z_actual, joint_losses).  The first half of the
    monolithic ``_learn()`` — encodes both states into the joint
    106-dim world vector, runs the joint train step, and updates
    preferences / Hebbian / GRPO buffer.

    Splitting ``_learn()`` from 175 lines to ~6 focused sub-functions
    (each < 40 lines) makes the OODA phase easier to test in
    isolation and gives us clear error-recovery boundaries.
    """
    z_actual = np.concatenate([
        agent.joint.encode(h_next, time_phase),
        np.asarray(world_next.to_tensor(), dtype=np.float32),
    ]).astype(np.float32)
    if world_curr is None:
        # Lazy compute — only if the caller didn't pass it through.
        calendar_ctx = _calendar_context(agent, year, month, day, hour)
        z_world, world_curr = _world_vector(
            agent, h_current, time_phase, grid, calendar_context=calendar_ctx,
        )
        z_world = np.concatenate([
            agent.joint.encode(h_current, time_phase),
            np.asarray(world_curr.to_tensor(), dtype=np.float32),
        ]).astype(np.float32)
    else:
        z_world = np.concatenate([
            agent.joint.encode(h_current, time_phase),
            np.asarray(world_curr.to_tensor(), dtype=np.float32),
        ]).astype(np.float32)

    joint_losses = _joint_train_step(
        agent, z_world, z_actual, h_current, h_next,
        grid, time_phase, result, reward,
    )
    # 反事实训练: 综卦(对面视角) + 错卦(完全反转) → 增强 JEPA 世界理解
    rev_loss, cmp_loss = _counterfactual_train(
        agent, result.chain, z_world, time_phase,
    )
    if rev_loss or cmp_loss:
        joint_losses["counterfactual_reversed"] = rev_loss
        joint_losses["counterfactual_complement"] = cmp_loss

    from zwm.planner.agent_train import _update_preferences
    _update_preferences(agent, h_current, result, reward)
    agent.hebbian.update_from_episode(
        [h_current.normal_order, h_next.normal_order], reward
    )
    try:
        moe_active = list(getattr(result, "moe_active_experts", []) or [])
        if not moe_active and hasattr(result, "moe_experts"):
            moe_active = list(getattr(result, "moe_experts", []) or [])
        agent.learner.record_grpo_step(h_current, moe_active, reward)
    except Exception as exc:
        _log.debug("GRPO record failed: %s", exc)
    return z_world, z_actual, joint_losses


def _learn_persist_episode(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    h_next: "Hexagram",
    world_next: "UnifiedField",
    reward: float,
    z_world: np.ndarray,
    *,
    surprise: float = 0.0,
    mutation_class: str = "",
    codon: str = "",
    codon_aa: str = "",
    jepa_loss: float | None = None,
    router_loss: float | None = None,
    moe_active_experts: list[str] | None = None,
    top_mutation: int = 0,
    top_score: float = 0.0,
    trajectory: list[tuple[str, float]] | None = None,
) -> int:
    """AUDIT-A3 (split): persist the episode + ReAct reflections to disk.

    Returns the ``episode_id`` so the caller can link later writes
    (e.g. self_reflect) to the row.  ReAct reflection logging
    (Reflexion / Self-Refine) is part of the same write path so
    we don't have an *episode* without its *reasoning trail*.
    """
    episode_id = _store_episode(
        agent, h_current, h_next, world_next, reward, z_world=z_world,
        surprise=surprise,
        mutation_class=mutation_class,
        codon=codon,
        codon_aa=codon_aa,
        jepa_loss=jepa_loss,
        router_loss=router_loss,
        moe_active_experts=moe_active_experts,
        top_mutation=top_mutation,
        top_score=top_score,
        trajectory=trajectory,
    )
    try:
        react_result = getattr(agent, "_last_react_result", None)
        if react_result is not None and hasattr(react_result, "steps"):
            for i, step in enumerate(react_result.steps):
                try:
                    agent.store.store_react_reflection(
                        episode_id=episode_id,
                        step_index=i,
                        thought=str(getattr(step, "thought", "")),
                        tool_name=getattr(step, "tool_name", None),
                        tool_input=getattr(step, "tool_input", None),
                        observation=getattr(step, "observation", None),
                        tool_score=float(getattr(step, "score", 0.0)),
                        confidence=float(getattr(react_result, "confidence", 0.0)),
                        recommendation=str(getattr(react_result, "recommendation", "")),
                    )
                except Exception as exc:
                    _log.debug("React reflection write failed: %s", exc)
            try:
                if hasattr(agent._react_loop, "self_reflect"):
                    critique = agent._react_loop.self_reflect(react_result)
                    agent.store.store_react_reflection(
                        episode_id=episode_id,
                        step_index=len(react_result.steps),
                        thought="[self_reflect] " + critique,
                        tool_name="__self_reflect__",
                        tool_input=None,
                        observation=critique,
                        tool_score=float(react_result.confidence),
                        confidence=float(react_result.confidence),
                        recommendation=str(react_result.recommendation),
                    )
            except Exception as exc:
                _log.debug("self_reflect persist failed: %s", exc)
    except Exception as exc:
        _log.debug("ReAct reflection log skipped: %s", exc)
    return episode_id


def _learn_particle_filter_update(
    agent: "TrinityAgent",
    z_actual: np.ndarray,
) -> None:
    """AUDIT-A3 (split) + A8 (closure extracted): update the particle
    filter belief with the observed z_actual.

    The 64-dim observation_fn is no longer a closure inside
    ``_learn()`` — it's a small local function that survives only for
    the duration of the update.  The transition function used to
    live in ``_predict()`` and the obs function in ``_learn()``;
    with this refactor both are clearly visible at the module
    top-level, making particle-filter changes one-edit instead of
    two-coordinated-edits.
    """
    if agent._particle_filter is None:
        return
    try:
        z_actual_latent = agent.jepa.context_encode(z_actual)
        z_actual_latent = np.asarray(z_actual_latent, dtype=np.float32).flatten()[:64]
        if len(z_actual_latent) < 64:
            z_actual_latent = np.concatenate([
                z_actual_latent,
                np.zeros(64 - len(z_actual_latent), dtype=np.float32),
            ])

        def _obs_fn(z_latent: np.ndarray) -> np.ndarray:
            """Observation model: predict what observation a latent state
            would produce.  Uses the JEPA predictor as a forward model
            so the particle filter can compare predicted vs actual
            observations in latent space."""
            z_latent = np.asarray(z_latent, dtype=np.float32).flatten()[:64]
            if len(z_latent) < 64:
                z_latent = np.concatenate([
                    z_latent,
                    np.zeros(64 - len(z_latent), dtype=np.float32),
                ])
            # Use the JEPA predictor as observation model: given a
            # latent state, predict the next latent (what we'd expect
            # to observe).  This is more informative than identity.
            if hasattr(agent.jepa, "predict_latent"):
                try:
                    pred = agent.jepa.predict_latent(z_latent)
                    pred = np.asarray(pred, dtype=np.float32).flatten()[:64]
                    if len(pred) == 64:
                        return pred
                except Exception:
                    pass
            return z_latent

        agent._particle_filter.update(z_actual_latent, _obs_fn)
    except Exception as exc:
        _log.warning("ParticleFilter.update failed: %s", exc)


def _learn_consolidate(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    h_next: "Hexagram",
    reward: float,
    vision_features: np.ndarray | None,
    language_features: np.ndarray | None,
) -> None:
    """AUDIT-A3 (split): consolidation — VSA / multimodal / denoiser.

    These three updates only fire on "good" outcomes (reward ≥
    GOOD_OUTCOME) or at a fixed cadence.  Splitting them out makes
    the "what counts as learning" policy visible at a glance.
    """
    from zwm.hexaembed.vsa import TrainableVSACodebook
    if isinstance(agent.vsa, TrainableVSACodebook) and reward >= GOOD_OUTCOME:
        target_vec = agent.vsa.encode_hexagram(h_next.normal_order).astype(np.float32)
        agent.vsa.train_step(h_current.normal_order, target_vec)

    if agent._multimodal is not None and reward >= GOOD_OUTCOME:
        _train_multimodal(agent, vision_features, language_features, h_next, reward)

    _periodic_denoiser_training(agent)


def _learn_meta_update(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    joint_losses: dict,
    surprise: float,
    result: "PlanResult",
    reward: float,
) -> None:
    """AUDIT-A3 (split): meta-update — curiosity / growth / telemetry /
    GAE flush / DreamerV3 replay.  These are the *control-plane* ops
    that drive the agent's pace, not its world model.
    """
    agent.curiosity.step()
    agent.growth.advance()
    agent.curiosity.beta_initial = agent.growth.curiosity_weight

    _log_telemetry(agent, h_current, joint_losses, surprise, result, reward)
    agent._step_count += 1

    if agent._step_count % 8 == 0:
        _gae_flush(agent, result)

    if agent._step_count % 16 == 0:
        # Need a world vector for the dreamer — recompute via the
        # world-vector helper.  We pass the *current* hexagram
        # because the dreamer imagines *from* the present, not from
        # the just-completed transition.
        try:
            from zwm.self_field.palace_graph import LuoshuGrid
            grid = agent.grid or LuoshuGrid()
            z_world, _ = _world_vector(agent, h_current, 0.0, grid)
            dreamer_losses = _dreamer_replay(agent, z_world, horizon=5)
            joint_losses.update(dreamer_losses)
        except Exception as exc:
            _log.debug("DreamerV3 replay failed: %s", exc)


def _learn(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    h_next: "Hexagram",
    grid: "LuoshuGrid",
    time_phase: float,
    result: "PlanResult",
    reward: float,
    world_next: "UnifiedField",
    surprise: float,
    vision_features: np.ndarray | None,
    language_features: np.ndarray | None,
    codon: str,
    codon_aa: str,
    mutation_class: str,
    year: int, month: int, day: int, hour: int,
    target_palace: int,
):
    """Write back into every persistent subsystem and emit telemetry.

    AUDIT-A3: this function used to be a 175-line monolith doing
    world-update, persistence, particle filter, consolidation, and
    meta-update all at once.  It is now a thin orchestrator that
    delegates to focused sub-functions.  Each sub-function has a
    single responsibility and an independent try/except boundary
    so a failure in one layer (e.g. particle filter) does not mask
    learning in another (e.g. preference update).

    Returns a TickReport.
    """
    from zwm.planner.agent_data import TickReport

    # 1) Record the visit (cheap, no I/O). ``_palace_visits`` is the
    # planar planner memory and excludes the self-center. SelfState keeps
    # the original trinity target, including 10=上/天 and 11=下/地.
    if target_palace != grid.self_position:
        agent._palace_visits[target_palace] = (
            agent._palace_visits.get(target_palace, 0) + 1
        )
    exploration_target = getattr(agent, "_last_exploration_target", target_palace)
    agent.self_state.record_visit(exploration_target)

    # 2) World-model gradient + preference / Hebbian / GRPO buffer.
    calendar_ctx = _calendar_context(agent, year, month, day, hour)
    _, world_curr = _world_vector(
        agent, h_current, time_phase, grid, calendar_context=calendar_ctx,
    )
    z_world, z_actual, joint_losses = _learn_world_update(
        agent, h_current, h_next, grid, time_phase, result, reward,
        world_next, world_curr, year, month, day, hour,
    )

    # 3) Persist episode + ReAct reflections.
    episode_id = _learn_persist_episode(
        agent, h_current, h_next, world_next, reward, z_world,
        surprise=surprise,
        mutation_class=mutation_class,
        codon=codon,
        codon_aa=codon_aa,
        jepa_loss=joint_losses.get("jepa"),
        router_loss=joint_losses.get("router"),
        moe_active_experts=result.moe_active_experts,
        top_mutation=result.top_mutation,
        top_score=result.top_score,
        trajectory=result.trajectory,
    )

    # 4) Particle-filter belief update (the ensemble EFE).
    _learn_particle_filter_update(agent, z_actual)

    # 5) Consolidation: VSA / multimodal / denoiser.
    _learn_consolidate(
        agent, h_current, h_next, reward, vision_features, language_features,
    )

    # 6) Meta-update: curiosity / growth / telemetry / GAE / DreamerV3.
    _learn_meta_update(agent, h_current, joint_losses, surprise, result, reward)

    return TickReport(
        plan=result,
        h_current=h_current,
        h_next=h_next,
        reward=reward,
        jepa_loss=joint_losses.get("jepa"),
        router_loss=joint_losses.get("router"),
        episode_id=episode_id,
        surprise=surprise,
        mutation_class=mutation_class,
        codon=codon,
        codon_aa=codon_aa,
    )


def _store_episode(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    h_next: "Hexagram",
    world: "UnifiedField",
    reward: float,
    z_world: np.ndarray | None = None,
    *,
    surprise: float = 0.0,
    mutation_class: str = "",
    codon: str = "",
    codon_aa: str = "",
    jepa_loss: float | None = None,
    router_loss: float | None = None,
    moe_active_experts: list[str] | None = None,
    top_mutation: int = 0,
    top_score: float = 0.0,
    trajectory: list[tuple[str, float]] | None = None,
) -> int:
    """存储情节到 EpisodicStore + VectorIndex + VSA buffer + SemanticStore。"""
    from zwm.hexaembed.vsa import VSAEpisode
    vsa_vec = agent.vsa.encode_hexagram(h_current.normal_order)
    outcome = "吉" if reward >= GOOD_OUTCOME else "凶"
    chain = world.five_chain

    # P0-dead-output: build a rich episode context that includes the
    # multimodal embedding and topology path (previously computed in
    # _act() but never persisted).  This makes the episodic memory
    # queryable along sensory and spatial dimensions, not just the
    # hexagram identity.
    ctx: dict = {"time_phase": world.time_phase}
    ctx["surprise"] = surprise
    ctx["mutation_class"] = mutation_class
    ctx["codon"] = codon
    ctx["codon_aa"] = codon_aa
    if jepa_loss is not None:
        ctx["jepa_loss"] = jepa_loss
    if router_loss is not None:
        ctx["router_loss"] = router_loss
    if moe_active_experts:
        ctx["moe_active_experts"] = moe_active_experts
    ctx["top_mutation"] = top_mutation
    ctx["top_score"] = top_score
    if trajectory:
        ctx["trajectory"] = [{"name": n, "score": s} for n, s in trajectory]
    mm_emb = getattr(agent, "_last_multimodal_emb", None)
    if mm_emb is not None and len(mm_emb) > 0:
        ctx["mm_emb_dim"] = len(mm_emb)
        ctx["mm_emb_l2"] = float(np.linalg.norm(mm_emb))
    topo_bagua = getattr(agent, "_last_topology_bagua", None)
    if topo_bagua is not None:
        ctx["topology_bagua"] = topo_bagua
    topo_dir = getattr(agent, "_last_topology_direction", None)
    if topo_dir is not None:
        ctx["topology_direction"] = topo_dir

    episode_id = agent.store.store(
        main_bits=h_current.normal_order,
        inter_bits=chain.inter.normal_order,
        evolved_bits=h_next.normal_order,
        reversed_bits=chain.reversed_.normal_order,
        complement_bits=chain.complement.normal_order,
        outcome=outcome,
        reward=reward,
        encoded_vector=vsa_vec,
        context=ctx,
    )
    try:
        agent.store.add_to_index(episode_id, vsa_vec.astype(np.float32))
    except Exception as exc:
        _log.warning("VectorIndex add failed: %s", exc)
    agent.vsa_buffer.add(
        VSAEpisode(
            hexagram_vector=vsa_vec,
            context_vector=agent.vsa.encode_hexagram(h_next.normal_order),
            outcome_vector=agent.vsa.encode_trigram(h_current.lower_trigram.index),
            reward=reward,
            timestamp=time.time(),
        )
    )
    if reward >= GOOD_OUTCOME:
        agent.vsa_buffer.consolidate()
    if agent.semantic is not None:
        agent.semantic.increment_frequency(h_current.normal_order)
        agent.semantic.update_association(
            h_current.normal_order, h_next.normal_order, delta=reward * 0.01
        )
    try:
        if hasattr(agent.jepa, "_vq") and agent.jepa._vq is not None:
            tokens = agent.jepa.tokenize(z_world)
            agent.store.update_context(episode_id, {"vq_tokens": tokens.tolist()})
    except Exception as exc:
        _log.debug("VQ tokenisation failed: %s", exc)
    return episode_id
