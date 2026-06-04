from __future__ import annotations

import numpy as np

from zwm.core.hexagram import Hexagram
from zwm.self_field.palace_graph import LuoshuGrid


class MoERouter:
    def __init__(self) -> None:
        rng = np.random.default_rng(42)
        self._w = rng.normal(0, 0.1, (15, 6)).astype(np.float32)
        self._b = np.zeros(6, dtype=np.float32)

    def route(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
    ) -> np.ndarray:
        features = self._extract_features(h, grid, time_phase)
        logits = features @ self._w + self._b
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def _extract_features(
        self,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float,
    ) -> np.ndarray:
        import math

        lower_elem = h.lower_trigram.element
        upper_elem = h.upper_trigram.element

        features = np.zeros(15, dtype=np.float32)
        features[0] = float(h.lower_trigram.pre_heaven_order) / 8.0
        features[1] = float(h.upper_trigram.pre_heaven_order) / 8.0
        features[2] = float(grid.self_position) / 9.0
        features[3] = time_phase / (2 * math.pi)
        features[4] = 1.0 if h.name in ("乾", "坤") else 0.0
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
