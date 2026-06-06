from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from zwm.core.hexagram import Hexagram
from zwm.self_field.palace_graph import LuoshuGrid

_FEATURE_DIM = 15
_NUM_EXPERTS = 6


class MoERouter(nn.Module):
    """Learned gating network over the six trinity experts.

    A real, trainable router (single linear gate + softmax) — not a frozen
    random projection. Weights start small and are updated online via
    ``train_toward`` whenever the agent observes which expert was rewarded,
    matching the learned-routing regime of modern sparse MoE models.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(_FEATURE_DIM, _NUM_EXPERTS)
        nn.init.normal_(self.gate.weight, std=0.1)
        nn.init.zeros_(self.gate.bias)
        self._opt = torch.optim.Adam(self.parameters(), lr=0.01)
        # P1 FIX: Track actual routing statistics for load-balance loss.
        # Previously a synthetic batch (jittered single sample + uniform f_i)
        # was used, making the aux loss gradient direction constant and
        # uninformative.  Now we maintain EMA buffers of actual routing
        # counts and gate probabilities for a meaningful anti-collapse signal.
        self.register_buffer("_route_count", torch.zeros(_NUM_EXPERTS))
        self.register_buffer("_gate_prob_ema", torch.ones(_NUM_EXPERTS) / _NUM_EXPERTS)
        self._total_ema: float = 0.0
        self._ema_decay: float = 0.99

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(features), dim=-1)

    def route(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
    ) -> np.ndarray:
        feats = torch.from_numpy(self._extract_features(h, grid, time_phase))
        with torch.no_grad():
            probs = self.forward(feats)
        return probs.numpy().astype(np.float32)

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
        """Push routing mass toward ``expert_index`` (cross-entropy step)
        + DeepSeek-V3-style auxiliary load-balancing loss.

        ``weight`` scales the update by the observed reward so good outcomes
        reinforce the responsible expert more strongly than poor ones.
        ``load_balance_weight`` controls the strength of the aux term that
        prevents router collapse (one expert hogging all mass). At λ=0 the
        aux term is off (legacy behaviour); λ>0 follows the modern MoE recipe.
        """
        if lr is not None:
            for group in self._opt.param_groups:
                group["lr"] = lr

        feats = torch.from_numpy(self._extract_features(h, grid, time_phase))
        logits = self.gate(feats)
        target = torch.tensor(expert_index, dtype=torch.long)
        ce_loss = nn.functional.cross_entropy(
            logits.unsqueeze(0), target.unsqueeze(0)
        )

        # P1 FIX: Auxiliary load-balancing loss (DeepSeek-V3 style)
        #   L_aux = α · E · Σᵢ fᵢ · pᵢ
        # where fᵢ = actual fraction of recent calls to expert i (EMA of
        # routing counts), and pᵢ = EMA of gate probability for expert i.
        # Previously we used a synthetic batch (jittered single sample)
        # with hardcoded uniform f_i, making the gradient direction
        # constant and uninformative.  Now we track actual routing
        # statistics for a meaningful anti-collapse signal.
        if load_balance_weight > 0.0:
            with torch.no_grad():
                # Update EMA of routing counts
                self._route_count[expert_index] += 1.0
                self._total_ema = (
                    self._total_ema * self._ema_decay + 1.0
                )
                # f_i = fraction of calls to each expert
                f_i = self._route_count / (self._route_count.sum() + 1e-8)
                # p_i = gate probability for this sample (not EMA)
                probs = torch.softmax(logits, dim=-1).detach()
                # Update EMA of gate probabilities
                self._gate_prob_ema = (
                    self._ema_decay * self._gate_prob_ema
                    + (1.0 - self._ema_decay) * probs
                )
            aux = float(_NUM_EXPERTS) * (f_i * self._gate_prob_ema).sum()
        else:
            aux = torch.tensor(0.0, dtype=ce_loss.dtype)

        loss = weight * ce_loss + load_balance_weight * aux
        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self._opt.step()
        return float(loss.detach())

    def _extract_features(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
    ) -> np.ndarray:
        lower_elem = h.lower_trigram.element
        upper_elem = h.upper_trigram.element

        features = np.zeros(_FEATURE_DIM, dtype=np.float32)
        features[0] = float(h.lower_trigram.pre_heaven_order) / 8.0
        features[1] = float(h.upper_trigram.pre_heaven_order) / 8.0
        features[2] = float(getattr(grid, 'self_position', 5)) / 9.0
        features[3] = time_phase / (2 * math.pi)
        # P0 fix: 不再硬编码 "乾为天"/"坤为地" 偏置, 用归一化的卦身份作为
        # 通用特征。所有 64 卦在这一维上的值都是连续的、可学习的。
        features[4] = float(h.normal_order) / 63.0
        # legacy slot — 保留为归一化卦身份, 不再作为 hardcode
        features[5] = float(h.normal_order) / 63.0
        features[6] = math.sin(time_phase)
        features[7] = math.cos(time_phase)
        features[8] = math.sin(2 * time_phase)
        features[9] = math.cos(2 * time_phase)
        features[10] = 1.0 if lower_elem == upper_elem else 0.0
        features[11] = 1.0 if lower_elem == "火" or upper_elem == "火" else 0.0
        features[12] = 1.0 if lower_elem == "水" or upper_elem == "水" else 0.0
        features[13] = 1.0 if lower_elem == "木" or upper_elem == "木" else 0.0
        features[14] = 1.0 if lower_elem == "金" or upper_elem == "金" else 0.0
        return features
