from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from zwm.core.hexagram import Hexagram
from zwm.moe.experts import (
    FineGrainedExpertNetwork,
    element_expert,
    narrative_expert,
    risk_expert,
    social_expert,
    space_expert,
    time_expert,
)
from zwm.moe.router import MoERouter
from zwm.self_field.palace_graph import LuoshuGrid


class FineGrainedSparseMoE(nn.Module):
    """64 fine-grained experts + 1 shared expert with top-8 routing.

    Each expert is a small network: Linear(29→8→1) with GELU activation.
    The shared expert processes every input (not routed).
    Includes auxiliary load balancing loss to encourage even expert utilization.
    """

    _NUM_EXPERTS = 64
    _TOP_K = 8
    _FEATURE_DIM = 29

    def __init__(self) -> None:
        super().__init__()
        # 64 fine-grained experts
        self._experts = nn.ModuleList([
            FineGrainedExpertNetwork(self._FEATURE_DIM)
            for _ in range(self._NUM_EXPERTS)
        ])
        # 1 shared expert (processes every input, not routed)
        self._shared_expert = FineGrainedExpertNetwork(self._FEATURE_DIM)
        # Router gate
        self._gate = nn.Linear(self._FEATURE_DIM, self._NUM_EXPERTS)
        nn.init.normal_(self._gate.weight, std=0.1)
        nn.init.zeros_(self._gate.bias)
        self._opt = torch.optim.Adam(self.parameters(), lr=0.01)
        # Feature extraction (reuse MoERouter)
        self._feat_router = MoERouter()
        # Load balancing loss from last forward pass
        self._last_load_balance_loss: float = 0.0
        # Tracked for downstream consumers (shared-expert temperature).
        self._last_trinity_bias: float = 0.0
        self._expert_names = [f"fg_{i}" for i in range(self._NUM_EXPERTS)] + ["shared"]
        # Strategic-2: Track actual routing frequency for load balance
        self._routing_counts = torch.zeros(self._NUM_EXPERTS)
        self._total_routes = 0

    @property
    def expert_names(self) -> list[str]:
        return list(self._expert_names)

    @property
    def load_balance_loss(self) -> float:
        return self._last_load_balance_loss

    def evaluate(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
        target_palace: int | None = None,
        preference_weights: dict[str, float] | None = None,
        day_gan: str | None = None,
    ) -> float:
        # Build feature vector including spatial and elemental context
        from zwm.core.constants import TIAN_GAN_ELEMENTS
        base_features = self._feat_router._extract_features(h, grid, time_phase)
        features = torch.from_numpy(base_features)

        # P1-7: Encode target_palace and day_gan context
        palace_feat = [0.0] * 9
        if target_palace is not None and 1 <= target_palace <= 9:
            palace_feat[target_palace - 1] = 1.0

        element_feat = [0.0] * 5
        if day_gan is not None:
            elem = TIAN_GAN_ELEMENTS.get(day_gan)
            if elem:
                elem_idx = {"金": 0, "木": 1, "水": 2, "火": 3, "土": 4}.get(elem, -1)
                if elem_idx >= 0:
                    element_feat[elem_idx] = 1.0

        features = torch.cat([
            features,
            torch.tensor(palace_feat + element_feat, dtype=torch.float32),
        ])

        with torch.no_grad():
            gate_logits = self._gate(features)
            gate_probs = torch.softmax(gate_logits, dim=-1)

            # Apply preference weights if provided.  Two cases are
            # supported, in order:
            #   1. **Trinity high-level names** ("time", "space", ...):
            #      these don't map 1-to-1 to fine-grained experts, so we
            #      apply them as a *global* temperature-like bias on the
            #      shared expert.  When the OnlineLearner's preferences
            #      for the high-level categories change, the output of
            #      the shared expert moves with them — preserving the
            #      feedback signal the legacy 6-expert path provided.
            #   2. **Fine-grained names** ("fg_0", "fg_1", ...):  direct
            #      per-expert multiplicative bias on gate probabilities.
            if preference_weights is not None:
                trinity_keys = {
                    "time", "space", "social",
                    "element", "risk", "narrative",
                }
                trinity_bias = sum(
                    v for k, v in preference_weights.items() if k in trinity_keys
                )
                # Distribute trinity_bias over the per-expert gate probs.
                pref = torch.tensor([
                    preference_weights.get(name, 0.0)
                    for name in self._expert_names[: self._NUM_EXPERTS]
                ], dtype=torch.float32)
                # Blend: trinity bias affects all experts uniformly
                # (scaled to a per-expert contribution); fine-grained
                # prefs hit their expert directly.
                effective_bias = pref + trinity_bias / self._NUM_EXPERTS
                gate_probs = gate_probs * (1.0 + effective_bias)
                total = gate_probs.sum()
                if total > 1e-10:
                    gate_probs = gate_probs / total

                # Tracked for downstream use (the shared expert gets
                # the same temperature shift).
                self._last_trinity_bias = float(trinity_bias)
            else:
                self._last_trinity_bias = 0.0

            # Top-k routing
            top_k_values, top_k_indices = torch.topk(gate_probs, self._TOP_K)
            top_k_weights = top_k_values / top_k_values.sum()

            # Strategic-2: Track actual routing frequency
            with torch.no_grad():
                for idx in top_k_indices:
                    self._routing_counts[idx] += 1
                self._total_routes += len(top_k_indices)

            # Weighted sum of selected expert outputs
            output = torch.tensor(0.0)
            for i, idx in enumerate(top_k_indices):
                expert_out = self._experts[idx](features.unsqueeze(0))
                output = output + top_k_weights[i] * expert_out.squeeze()

            # Shared expert (always active) — its output is scaled by
            # (1 + trinity_bias) so the high-level preference feedback
            # is consumed by the model and shows up in the final score.
            shared_out = self._shared_expert(features.unsqueeze(0)).squeeze()
            output = output + (1.0 + self._last_trinity_bias) * shared_out

            # Auxiliary load balancing loss: L_aux = E * Σ(f_i * p_i)
            # Strategic-2: Use actual routing frequency instead of uniform f_i
            if self._total_routes > 0:
                f_i = self._routing_counts / self._total_routes
            else:
                f_i = torch.ones(self._NUM_EXPERTS) / self._NUM_EXPERTS
            self._last_load_balance_loss = float(
                self._NUM_EXPERTS * (f_i * gate_probs).sum()
            )

        return float(output)

    def active_experts(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
    ) -> list[str]:
        features_np = self._feat_router._extract_features(h, grid, time_phase)
        features = torch.from_numpy(features_np)
        # Pad with zeros for palace/element features
        features = torch.cat([
            features,
            torch.zeros(14, dtype=features.dtype),
        ])

        with torch.no_grad():
            gate_logits = self._gate(features)
            gate_probs = torch.softmax(gate_logits, dim=-1)
            _, top_k_indices = torch.topk(gate_probs, self._TOP_K)

        return [self._expert_names[idx] for idx in top_k_indices] + ["shared"]

    def train_toward(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
        expert_index: int,
        lr: float | None = None,
        weight: float = 1.0,
        load_balance_weight: float = 0.01,
    ) -> float:
        """Push routing mass toward expert_index + auxiliary load balancing."""
        if lr is not None:
            for group in self._opt.param_groups:
                group["lr"] = lr

        features_np = self._feat_router._extract_features(h, grid, time_phase)
        features = torch.from_numpy(features_np)
        # Pad with zeros for palace/element features (not provided during training)
        features = torch.cat([
            features,
            torch.zeros(14, dtype=features.dtype),
        ])

        gate_logits = self._gate(features)
        target = torch.tensor(int(expert_index), dtype=torch.long)
        ce_loss = nn.functional.cross_entropy(
            gate_logits.unsqueeze(0), target.unsqueeze(0)
        )

        # Auxiliary load balancing loss (DeepSeek-V3 style)
        if load_balance_weight > 0.0:
            with torch.no_grad():
                jitter = torch.randn(
                    self._NUM_EXPERTS, self._FEATURE_DIM, dtype=features.dtype
                ) * 0.05
                batch_feats = features.unsqueeze(0) + jitter
            batch_logits = self._gate(batch_feats)
            probs = torch.softmax(batch_logits, dim=-1)
            f_i = torch.full(
                (self._NUM_EXPERTS,), 1.0 / self._NUM_EXPERTS, dtype=probs.dtype
            )
            p_i = probs.mean(dim=0)
            aux = float(self._NUM_EXPERTS) * (f_i * p_i).sum()
        else:
            aux = torch.tensor(0.0, dtype=ce_loss.dtype)

        loss = weight * ce_loss + load_balance_weight * aux
        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self._opt.step()
        return float(loss.detach())


class SparseMoE:
    def __init__(self, top_k: int = 3, use_fine_grained: bool = True) -> None:
        self._use_fine_grained = use_fine_grained
        if use_fine_grained:
            self._fine_grained = FineGrainedSparseMoE()
            self._router = self._fine_grained  # compatible train_toward interface
            self._top_k = FineGrainedSparseMoE._TOP_K
            self._expert_names = self._fine_grained.expert_names
        else:
            self._router = MoERouter()
            self._top_k = top_k
            self._expert_names = [
                "time", "space", "social",
                "element", "risk", "narrative",
            ]

    @property
    def router(self) -> MoERouter | FineGrainedSparseMoE:
        return self._router

    @property
    def expert_names(self) -> list[str]:
        return list(self._expert_names)

    @property
    def load_balance_loss(self) -> float:
        if self._use_fine_grained:
            return self._fine_grained.load_balance_loss
        return 0.0

    def evaluate(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
        target_palace: int,
        preference_weights: dict[str, float] | None = None,
        day_gan: str | None = None,
    ) -> float:
        if self._use_fine_grained:
            return self._fine_grained.evaluate(
                h, grid, time_phase, target_palace, preference_weights, day_gan
            )

        weights = self._router.route(h, grid, time_phase)
        if preference_weights is not None:
            pref = np.array(
                [preference_weights.get(name, 0.0) for name in self._expert_names],
                dtype=np.float32,
            )
            # Multiplicatively bias the learned router by the agent's
            # accumulated expert preferences, then renormalise.
            weights = weights * (1.0 + pref)
            total = weights.sum()
            if total > 1e-10:
                weights = weights / total
        threshold = np.sort(weights)[-self._top_k]
        mask = weights >= threshold

        # P1-4: Derive context element from day_gan (天干五行) rather than
        # the hexagram's own lower trigram element. This gives element_expert
        # genuine discriminative power — the "scene context element" is the
        # element of the day stem, not the hexagram's self-same element.
        context_element = None
        if day_gan is not None:
            from zwm.core.constants import TIAN_GAN_ELEMENTS
            context_element = TIAN_GAN_ELEMENTS.get(day_gan)
        if context_element is None:
            context_element = h.lower_trigram.element

        scores = np.zeros(6, dtype=np.float32)
        if mask[0]:
            scores[0] = time_expert(h, time_phase)
        if mask[1]:
            scores[1] = space_expert(h, target_palace)
        if mask[2]:
            scores[2] = social_expert(h, grid, target_palace)
        if mask[3]:
            scores[3] = element_expert(h, context_element)
        if mask[4]:
            scores[4] = risk_expert(h)
        if mask[5]:
            scores[5] = narrative_expert(h)

        active_weights = weights * mask.astype(np.float32)
        if active_weights.sum() < 1e-10:
            return float(np.mean(scores))
        return float(np.dot(active_weights, scores) / active_weights.sum())

    def active_experts(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
    ) -> list[str]:
        if self._use_fine_grained:
            return self._fine_grained.active_experts(h, grid, time_phase)

        weights = self._router.route(h, grid, time_phase)
        threshold = np.sort(weights)[-self._top_k]
        return [
            self._expert_names[i]
            for i in range(6)
            if weights[i] >= threshold
        ]
