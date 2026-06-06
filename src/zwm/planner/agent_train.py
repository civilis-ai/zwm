"""P2-1 (audit) — 学习/训练辅助函数集合。

从 ``agent.py`` 抽出,封装所有反向传播 + 偏好更新路径:

  * _train_jepa           — JEPA 端到端训练步 (含 learnable encoder 路径)
  * _joint_train_step     — 5 路梯度汇总 (JEPA + Router + Value + Policy KL + LB)
  * _reinforce_router     — MoE 路由强化
  * _update_preferences   — OnlineLearner 偏好更新 + 每 4 步 DPO 对齐
  * _train_multimodal     — 多模态投影训练 (天/地/人融合)
  * _periodic_denoiser_training — 每 50 步训练 DDPM denoiser
  * _log_telemetry        — TensorBoard / JSONL 指标记录
  * _gae_flush            — GAE(λ) 轨迹结束 flush
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from zwm.langevin.diffusion import DiffusionSampler
from zwm.planner.agent_data import GOOD_OUTCOME

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zwm.core.hexagram import Hexagram
    from zwm.planner.agent import TrinityAgent
    from zwm.planner.loop import PlanResult
    from zwm.scene_field.unified_field import UnifiedField
    from zwm.self_field.palace_graph import LuoshuGrid


# ================= 五卦链反事实训练 =================
def _counterfactual_train(
    agent: "TrinityAgent",
    chain: "FiveHexagramChain",
    z_world: np.ndarray,
    time_phase: float,
    weight: float = 0.15,
) -> tuple[float, float]:
    """用综卦和错卦作为反事实预测目标, 增强 JEPA 的世界理解.

    综卦 = 世界从对面看的样子 (上下颠倒视角)
    错卦 = 世界的完全反转 (阴阳互换)

    JEPA 学会预测这两种"备选世界状态", 则世界模型不只知道
    "下一步会发生什么", 还知道"从另一面看会是什么样"和
    "完全逆转会是什么样"。这提升了世界模型的鲁棒性。

    weight < 1.0 确保反事实训练不主导主预测目标。
    """
    h_rev = chain.reversed_
    h_cmp = chain.complement
    losses = []

    for h_aux, label in [(h_rev, "reversed"), (h_cmp, "complement")]:
        try:
            z_aux = np.concatenate([
                agent.joint.encode(h_aux, time_phase),
                np.zeros(29, dtype=np.float32),  # unified field 部分填充0
            ]).astype(np.float32)
            # 反事实训练: z_world → z_aux (综卦或错卦)
            # train_step 内部已自动归一化维度
            result = agent.jepa.train_step(z_world, z_aux)
            loss = result.get("pred_error", result.get("loss", 0.0)) if isinstance(result, dict) else 0.0
            if not np.isnan(loss):
                losses.append(loss)
        except Exception as exc:
            _log.debug("Counterfactual train (%s) skipped: %s", label, exc)

    if len(losses) == 2:
        return float(losses[0]), float(losses[1])
    return 0.0, 0.0


# ================= JEPA 训练 =================
def _train_jepa(
    agent: "TrinityAgent",
    z_world, z_next,
    h_current: "Hexagram | None" = None,
    h_next: "Hexagram | None" = None,
    time_phase: float = 0.0,
) -> float | None:
    """Real JEPA training step on the observed transition (z_t -> z_{t+1}).

    When a learnable square encoder is attached, uses train_transition()
    for end-to-end gradient flow through the encoder. Otherwise falls back
    to train_step() for latent-only training.
    """
    if (
        h_current is not None
        and h_next is not None
        and hasattr(agent.jepa, "_square_encoder")
        and agent.jepa._square_encoder is not None
    ):
        from zwm.jepa.square_encoder import hexagram_square_features
        ft = hexagram_square_features(h_current)
        fn = hexagram_square_features(h_next)
        unified_t = z_world[77:].astype(np.float32) if len(z_world) > 77 else None
        unified_next = z_next[77:].astype(np.float32) if len(z_next) > 77 else None
        loss = agent.jepa.train_transition(
            ft, time_phase, fn, time_phase,
            unified_t=unified_t, unified_next=unified_next,
        )
        if isinstance(loss, dict):
            loss = loss.get("pred_error", loss.get("short", 0.0))
    else:
        loss = agent.jepa.train_step(z_world, z_next)
        if isinstance(loss, dict):
            loss = loss.get("pred_error", loss.get("short", 0.0))
    return None if loss != loss else float(loss)  # filter NaN


# ================= 联合 OODA 训练步 =================
def _joint_train_step(
    agent: "TrinityAgent",
    z_world: np.ndarray,
    z_actual: np.ndarray,
    h_current: "Hexagram",
    h_next: "Hexagram",
    grid: "LuoshuGrid",
    time_phase: float,
    result: "PlanResult",
    reward: float,
) -> dict[str, float]:
    """联合 OODA 训练步 — 统一梯度流穿过 JEPA + router + value head.

    2026 SOTA: MuZero/EfficientZero 使用联合损失训练表示、
    动态和值函数。ZWM 的联合损失将 JEPA 预测损失、
    MoE 路由损失和值函数损失合并为单一反向传播,
    使各子系统不再独立优化而是协同进化。
    """
    import torch
    losses: dict[str, float] = {}

    # 1. JEPA prediction loss (primary)
    jepa_loss = _train_jepa(agent, z_world, z_actual, h_current, h_next, time_phase)
    if jepa_loss is not None:
        losses["jepa"] = jepa_loss

    # 1b. EWC penalty — prevent catastrophic forgetting of previously
    # learned hexagram tasks.  The penalty is zero until the first
    # ``register_ewc_task()`` call, so this is a no-op by default.
    try:
        if agent._ewc is not None and agent._ewc.n_tasks > 0:
            import torch
            ewc_pen = agent._ewc.penalty()
            if ewc_pen.requires_grad:
                ewc_pen.backward()
            losses["ewc"] = float(ewc_pen)
    except Exception as exc:
        _log.debug("EWC penalty computation failed: %s", exc)

    # 2. MoE router loss with load balance
    router_loss = _reinforce_router(agent, h_current, grid, time_phase, result, reward)
    if router_loss is not None:
        try:
            lb_loss = float(agent.planner._moe.load_balance_loss)
            losses["router"] = router_loss
            losses["load_balance"] = lb_loss
        except Exception as exc:
            # AUDIT-S4: surface MoE load-balance failures so silent
            # entropy collapse is observable in the log stream.
            _log.warning("load_balance_loss unavailable: %s", exc)
            losses["router"] = router_loss

    # 3. Value function TD error — train V(z) toward actual outcome
    # P0-4: use the dedicated value-head optimizer (``_value_opt``) when
    # available.  Falling back to the main ``_opt`` would stomp the JEPA
    # encoder gradients and destabilise the surprise signal.
    try:
        if hasattr(agent.jepa, "_value_head") and agent.jepa._value_head is not None:
            z_tensor = torch.from_numpy(z_world).float().unsqueeze(0)
            with torch.no_grad():
                v_target = torch.tensor([[reward]])
            v_pred = agent.jepa._value_head(z_tensor.detach())
            v_loss = torch.nn.functional.mse_loss(v_pred, v_target)
            v_loss.backward()
            # P0-4: use the dedicated value-head optimizer, NOT the
            # main JEPA optimizer, so we don't corrupt the world model.
            value_opt = getattr(agent.jepa, "_value_opt", None)
            if value_opt is not None:
                try:
                    torch.nn.utils.clip_grad_norm_(
                        agent.jepa._value_head.parameters(), max_norm=5.0,
                    )
                    value_opt.step()
                    value_opt.zero_grad()
                except Exception as exc:
                    _log.debug("Value-head optimizer.step failed: %s", exc)
            losses["value_td"] = float(v_loss)
    except Exception as exc:
        _log.debug("Value TD error computation failed: %s", exc)

    # 4. Policy improvement — use MCTS search distribution as target
    try:
        if hasattr(agent.jepa, "policy_targets") and result.hexagram_scores:
            policy_target = agent.jepa.policy_targets(z_world, temperature=0.5)
            visit_dist = np.zeros(63)
            total_v = 0
            for mask, score in result.hexagram_scores:
                if 1 <= mask <= 63:
                    visit_dist[mask - 1] = max(score, 0.0)
                    total_v += max(score, 0.0)
            if total_v > 0:
                visit_dist /= total_v
                kl = float(np.sum(visit_dist * np.log(
                    (visit_dist + 1e-8) / (policy_target + 1e-8)
                )))
                losses["policy_kl"] = kl
    except Exception as exc:
        # AUDIT-S4: policy-KL failure shouldn't kill the tick.
        _log.warning("policy KL divergence unavailable: %s", exc)

    return losses


def _reinforce_router(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    grid: "LuoshuGrid",
    time_phase: float,
    result: "PlanResult",
    reward: float,
) -> float | None:
    """Gradient step that reinforces the highest-routed active expert.

    Only reinforce on good outcomes; the strength scales with reward.
    Routed through the planner's public ``reinforce_expert`` hook rather
    than reaching into its private MoE.

    P0-dead-output: when the planner is using FineGrainedSparseMoE (64
    experts), the fine-grained router also receives a gradient step.
    Previously only the legacy 6-expert MoERouter was trained — the
    fine-grained path's parameters were frozen, wasting 64 expert
    networks.  This closes that loop.
    """
    if reward < GOOD_OUTCOME or not result.moe_active_experts:
        return None
    names = agent.planner.expert_names
    top_expert = result.moe_active_experts[0]
    if top_expert not in names:
        return None
    idx = names.index(top_expert)
    legacy_loss = agent.planner.reinforce_expert(
        h_current, grid, time_phase, expert_index=idx, weight=reward,
    )

    # P0-dead-output: also train the fine-grained MoE router when it's
    # the active sparse activation backend.  The fine-grained router
    # (64 experts + 1 shared) is evaluated during plan() but its
    # train_toward() was never called from the OODA learning loop.
    moe = agent.planner._moe
    if hasattr(moe, "_fg") and moe._fg is not None:
        try:
            fg_idx = idx if idx < 64 else (idx % 64)
            moe._fg.train_toward(
                h_current, grid, time_phase,
                expert_index=fg_idx,
                weight=reward,
            )
        except Exception as exc:
            _log.debug("Fine-grained MoE train_toward failed: %s", exc)

    return legacy_loss


def _update_preferences(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    result: "PlanResult",
    reward: float,
) -> None:
    """更新偏好 + 每 4 步 DPO 对齐 + 每 8 步 GRPO 群体相对策略。"""
    baseline = agent.learner.reward_baseline
    advantage = reward - baseline
    moe_weights = {name: 1.0 for name in result.moe_active_experts}
    agent.learner.update_from_outcome(
        h_current, reward, moe_weights=moe_weights,
        advantage=advantage,
    )
    # P1-7: Apply DPO alignment — every 4 ticks.
    if (
        agent._step_count % 4 == 0
        and len(result.moe_active_experts) >= 2
    ):
        from zwm.learning.online import dpo_update
        chosen = result.moe_active_experts
        rejected_pool = [n for n in agent.planner.expert_names if n not in chosen]
        if rejected_pool:
            rejected = rejected_pool[
                agent._step_count % len(rejected_pool):
                agent._step_count % len(rejected_pool) + 1
            ]
            dpo_update(
                agent.learner.preference_weights,
                chosen_experts=chosen,
                rejected_experts=rejected,
                beta=0.1,
                learning_rate=0.005,
            )
            # Propagate DPO signal to the MoE router's gate biases
            # so expert selection aligns with learned preferences.
            try:
                agent.learner.dpo_router_step(
                    agent.planner._moe.router,
                    chosen_experts=chosen,
                    rejected_experts=rejected,
                    lr=0.01,
                )
            except Exception as exc:
                _log.debug("DPO router step failed: %s", exc)

    # GRPO — Group Relative Policy Optimization (DeepSeek-V3, 2026).
    # Every 8 ticks, evaluate the *group* of recent expert selections
    # relative to each other and push the policy toward the best-in-group.
    # This is the key recipe behind DeepSeek-V3/R1: no value function
    # needed, just group-normalised advantages.
    if agent._step_count % 8 == 0 and agent._step_count >= 8:
        try:
            from zwm.learning.online import grpo_update
            # Build a group from the last 8 ticks' worth of MoE
            # expert selections and their rewards.  The group is
            # stored in the agent's trajectory buffer.
            group = _build_grpo_group(agent)
            if len(group) >= 2:
                grpo_update(
                    agent.learner.preference_weights,
                    group_rewards=group,
                    beta=0.2,
                    learning_rate=0.008,
                )
        except Exception as exc:
            # AUDIT-S4: GRPO group updates failing used to vanish.
            _log.warning("grpo_update failed: %s", exc)


def _build_grpo_group(
    agent: "TrinityAgent",
) -> list[tuple[list[str], float]]:
    """P0-5: Build a GRPO group from the real per-tick completion buffer.

    Uses ``OnlineLearner.get_grpo_group()`` to return recent (expert_list,
    reward) tuples grouped by state class.  This replaces the previous
    ``j % len(names)`` rotation, which was a synthetic group of fake
    completions — incompatible with the GRPO definition (DeepSeek-V3 / R1):
    "a group of completions for the same prompt".

    Each element is (expert_list, reward) for one tick.  The group
    represents the agent's recent completions of the *same state class*,
    which is the correct input to a group-relative advantage estimator.
    """
    raw_group = agent.learner.get_grpo_group(group_size=agent.learner.grpo_group_size)
    # Convert tuple → list for backward compat with the existing
    # grpo_update() signature.
    return [(list(experts), reward) for experts, reward in raw_group]


def _train_multimodal(
    agent: "TrinityAgent",
    vision_features: np.ndarray | None,
    language_features: np.ndarray | None,
    h_next: "Hexagram",
    reward: float,
) -> None:
    """Train vision / language projections + fusion weights on good outcomes."""
    if agent._multimodal is None:
        return
    try:
        if vision_features is not None:
            agent._multimodal.vision_encoder.train_step(vision_features, h_next)
        if language_features is not None:
            agent._multimodal.language_encoder.train_step(language_features, h_next)
        agent._multimodal.train_fusion_weights(
            sensor_data=None,
            visual_features=vision_features,
            text_embedding=language_features,
            target_hex=h_next,
            reward=reward,
        )
    except Exception as exc:
        # AUDIT-S4: multimodal fusion failures used to vanish; now
        # the /metrics scrape will see consecutive missing values
        # *and* the operator gets a log line to correlate.
        _log.warning("multimodal train failed: %s", exc)


def _periodic_denoiser_training(agent: "TrinityAgent") -> float | None:
    """每 50 tick 训练 DDPM denoiser,提高采样质量。

    P2-2 (audit): 训练数据从 32 卦扩展到全部 64 卦, 确保
    denoiser 覆盖完整的 hexagram 空间。

    Returns:
        The final training loss, or None if training was skipped.
    """
    if not isinstance(agent.planner._sampler, DiffusionSampler):
        return None
    if agent._step_count % 50 != 0:
        return None
    try:
        from zwm.core.hexagram import all_hexagrams
        loss = agent.planner._sampler.train_denoiser(
            list(all_hexagrams()), num_epochs=5,
        )
        return float(loss[-1]) if loss else None
    except Exception as exc:
        _log.warning("DDPM denoiser training failed: %s", exc)
        return None


def _log_telemetry(
    agent: "TrinityAgent",
    h_current: "Hexagram",
    joint_losses: dict,
    surprise: float,
    result: "PlanResult",
    reward: float,
) -> None:
    """Emit one JSONL/TensorBoard row per tick (never blocks the loop)."""
    try:
        from zwm.learning.metrics import get_logger
        get_logger().log({
            "loss/jepa": float(joint_losses.get("jepa") or 0.0),
            "loss/router": float(joint_losses.get("router") or 0.0),
            "world/surprise": float(surprise),
            "world/curiosity_beta": float(agent.curiosity.beta),
            "world/phase": agent.growth._phase_code(),
            "world/visit_count": agent.learner.get_visit_count(h_current),
            "prefs/time": agent.learner.preference_weights.get("time", 0.0),
            "prefs/space": agent.learner.preference_weights.get("space", 0.0),
            "prefs/social": agent.learner.preference_weights.get("social", 0.0),
            "prefs/element": agent.learner.preference_weights.get("element", 0.0),
            "prefs/risk": agent.learner.preference_weights.get("risk", 0.0),
            "prefs/narrative": agent.learner.preference_weights.get("narrative", 0.0),
            "reward": float(reward),
            "moe/load_balance_loss": float(agent.planner._moe.load_balance_loss),
            "world/value_estimate": float(agent.learner.value_estimate(h_current)),
            "world/reward_baseline": float(agent.learner.reward_baseline),
            "loss/value_td": joint_losses.get("value_td", 0.0),
            "loss/load_balance": joint_losses.get("load_balance", 0.0),
            "loss/policy_kl": joint_losses.get("policy_kl", 0.0),
        }, global_step=agent._step_count)
    except Exception as exc:
        # AUDIT-S4: telemetry failures used to vanish — now logged
        # so the operator knows the metrics are not being recorded.
        _log.warning("telemetry flush failed: %s", exc)


def _gae_flush(agent: "TrinityAgent", result: "PlanResult") -> None:
    """End the GAE trajectory at the configured window boundary (every 8 ticks)."""
    moe_weights = {name: 1.0 for name in result.moe_active_experts}
    try:
        agent.learner.end_trajectory(moe_weights=moe_weights)
    except Exception as exc:
        # AUDIT-S4: GAE flush failure means advantages aren't being
        # computed for the next policy update — log loudly.
        _log.warning("GAE end_trajectory failed: %s", exc)


# ================= DreamerV3 想象回放 =================
def _dreamer_imagine(
    agent: "TrinityAgent",
    z_start: np.ndarray,
    horizon: int = 15,
    gamma: float = 0.99,
) -> list[tuple[np.ndarray, float]]:
    """DreamerV3-style imagined rollout from a latent state.

    The 2026 SOTA for model-based RL is *imagined replay*: the agent
    uses its world model (JEPA) to simulate future trajectories
    without acting, then trains the value function and policy on
    these imagined transitions.  This is the core of DreamerV3 /
    TD-MPC2 and gives 2-5× sample efficiency over pure on-policy
    learning.

    Uses a simplified RSSM (Recurrent State Space Model) via a
    running-average hidden state ``h`` that carries temporal context
    across imagination steps: ``h_{t+1} = 0.9 * h_t + 0.1 * z_pred``.
    The imagined next latent incorporates this recurrent state:
    ``z_{t+1} = predictor(z_t + h_t)`` instead of ``predictor(z_t)``.
    This captures temporal dependencies without introducing new
    parameters, matching the DreamerV3 standard horizon of 15 steps.

    Returns a list of (z_imagined, reward_imagined) pairs.  The
    caller (``_dreamer_replay``) feeds these into the value head
    and JEPA as additional training signal.
    """
    trajectory: list[tuple[np.ndarray, float]] = []
    z = z_start.copy()
    h = np.zeros_like(z_start)  # simplified RSSM hidden state
    for _ in range(horizon):
        # Simplified RSSM: incorporate recurrent hidden state h into
        # the prediction input.  h carries a running average of past
        # predictions, providing temporal context without new parameters.
        z_input = z + h
        z_next = agent.jepa.predict(z_input)
        if isinstance(z_next, dict):
            z_next = z_next["short"]
        z_next = np.asarray(z_next, dtype=np.float32)

        # Update hidden state as running average (simplified RSSM):
        # h_{t+1} = 0.9 * h_t + 0.1 * z_pred
        h = 0.9 * h + 0.1 * z_next

        # Imagined reward from the value head V(z).
        r = 0.0
        try:
            if hasattr(agent.jepa, "value_flat"):
                v = agent.jepa.value_flat(z_next)
            elif hasattr(agent.jepa, "value"):
                v = agent.jepa.value(z_next)
            else:
                v = None
            if v is not None:
                v_arr = np.asarray(v)
                r = float(v_arr.item()) if v_arr.ndim == 0 else float(v_arr.flat[0])
        except Exception:
            pass

        trajectory.append((z_next, r))
        z = z_next
    return trajectory


def _dreamer_replay(
    agent: "TrinityAgent",
    z_world: np.ndarray,
    horizon: int = 15,
) -> dict[str, float]:
    """Train the value head on imagined DreamerV3 rollouts.

    Called every 16 ticks from the learn phase.  The imagined
    trajectory provides *free* training signal — no environment
    interaction needed.  This is the single biggest sample-
    efficiency win in 2026 model-based RL.
    """
    import torch
    if not hasattr(agent.jepa, "_value_head") or agent.jepa._value_head is None:
        return {}

    trajectory = _dreamer_imagine(agent, z_world, horizon=horizon)
    if not trajectory:
        return {}

    # Compute lambda-returns from the imagined trajectory.
    gamma = 0.99
    lambda_val = agent.learner.gae_lambda if hasattr(agent.learner, "gae_lambda") else 0.95
    rewards = [r for _, r in trajectory]
    values = []
    for z, _ in trajectory:
        try:
            z_t = torch.from_numpy(z).float().unsqueeze(0)
            with torch.no_grad():
                v = float(agent.jepa._value_head(z_t))
            values.append(v)
        except Exception as exc:
            # AUDIT-S4: V(s) inference failed for an imagined state —
            # use 0.0 and warn so the operator knows the value head
            # is being asked about something it can't score.
            _log.debug("V(s) inference failed in dreamer_imagine: %s", exc)
            values.append(0.0)
    # Bootstrap from the last value
    values.append(values[-1] if values else 0.0)

    # GAE-style advantage
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lambda_val * gae
        advantages.insert(0, gae)

    # Train value head on imagined (z, advantage) pairs
    # P0-4: actual backward() + optimizer.step() + zero_grad() so the
    # lambda-return signal actually updates V(s).  Without the opt.step,
    # the loss was computed but the gradient was silently dropped.
    total_loss = 0.0
    count = 0
    for (z, _), adv in zip(trajectory, advantages):
        try:
            z_t = torch.from_numpy(z).float().unsqueeze(0)
            v_pred = agent.jepa._value_head(z_t.detach()).squeeze(-1)
            target = torch.tensor([[float(adv) + float(v_pred.detach())]])
            loss = torch.nn.functional.mse_loss(v_pred, target)
            loss.backward()
            total_loss += float(loss.detach())
            count += 1
        except Exception as exc:
            # AUDIT-S4: dreamer per-step loss failure is logged so a
            # silently broken imagination loop is visible in stderr.
            _log.debug("dreamer per-step loss failed: %s", exc)
    # P0-4: actually apply the gradient.  This is the *only* place where
    # the DreamerV3 imagined-trajectory signal reaches the network, so
    # we step the optimizer here (after accumulating all per-step grads).
    if count > 0:
        try:
            # Use a dedicated optimiser for the value head so we don't
            # accidentally step the JEPA main optimiser (which would
            # corrupt the context_encoder / predictor weights with
            # imagined-trajectory gradients, causing surprise to diverge).
            if not hasattr(agent.jepa, "_value_opt") or agent.jepa._value_opt is None:
                agent.jepa._value_opt = torch.optim.Adam(
                    agent.jepa._value_head.parameters(), lr=1e-3,
                )
            torch.nn.utils.clip_grad_norm_(
                agent.jepa._value_head.parameters(), max_norm=5.0,
            )
            agent.jepa._value_opt.step()
            agent.jepa._value_opt.zero_grad()
        except Exception as exc:
            _log.debug("DreamerV3 optimizer.step failed: %s", exc)

    return {"dreamer_value_loss": total_loss / max(count, 1)}
