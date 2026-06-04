from __future__ import annotations

from abc import ABC, abstractmethod

from zwm.core.hexagram import Hexagram


class HexagramEncoder(ABC):
    @abstractmethod
    def encode(self, sensor_data: dict) -> Hexagram:
        ...

    @abstractmethod
    def feature_dim(self) -> int:
        ...


class RuleBasedEncoder(HexagramEncoder):
    def __init__(self) -> None:
        self._feature_mapping: dict[str, tuple[int, callable]] = {
            "temperature": (0, self._temp_to_yang),
            "terrain": (1, self._terrain_to_yang),
            "social_proximity": (2, self._social_to_yang),
            "resource_level": (3, self._resource_to_yang),
            "momentum": (4, self._momentum_to_yang),
            "overall_favorability": (5, self._favorability_to_yang),
        }

    def encode(self, sensor_data: dict) -> Hexagram:
        from zwm.core.yao import YANG, YIN
        from zwm.core.hexagram import Hexagram

        lines = [YIN] * 6
        for feature, (yao_idx, mapper) in self._feature_mapping.items():
            if feature in sensor_data:
                lines[yao_idx] = YANG if mapper(sensor_data[feature]) else YIN

        return Hexagram(*lines)

    def feature_dim(self) -> int:
        return 6

    @staticmethod
    def _temp_to_yang(temp: float) -> bool:
        return temp > 20.0

    @staticmethod
    def _terrain_to_yang(terrain: float) -> bool:
        return terrain > 0.5

    @staticmethod
    def _social_to_yang(proximity: float) -> bool:
        return proximity > 0.5

    @staticmethod
    def _resource_to_yang(level: float) -> bool:
        return level > 0.3

    @staticmethod
    def _momentum_to_yang(momentum: float) -> bool:
        return momentum > 0.0

    @staticmethod
    def _favorability_to_yang(favorability: float) -> bool:
        return favorability > 0.5
