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
    ) -> float:
        """Push routing mass toward ``expert_index`` (cross-entropy step).

        ``weight`` scales the update by the observed reward so good outcomes
        reinforce the responsible expert more strongly than poor ones.
        """
        if lr is not None:
            for group in self._opt.param_groups:
                group["lr"] = lr

        feats = torch.from_numpy(self._extract_features(h, grid, time_phase))
        logits = self.gate(feats)
        target = torch.tensor(expert_index, dtype=torch.long)
        loss = weight * nn.functional.cross_entropy(
            logits.unsqueeze(0), target.unsqueeze(0)
        )
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
        features[2] = float(grid.self_position) / 9.0
        features[3] = time_phase / (2 * math.pi)
        features[4] = 1.0 if h.name in ("乾为天", "坤为地") else 0.0
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
