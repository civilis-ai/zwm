from __future__ import annotations

import math
from dataclasses import dataclass

from zwm.core.constants import GANZHI_60, YUAN_HUI_YUN_SHI


@dataclass(frozen=True, slots=True)
class GanzhiTime:
    year_index: int = 0
    month_index: int = 0
    day_index: int = 0
    hour_index: int = 0

    @property
    def year_ganzhi(self) -> str:
        return GANZHI_60[self.year_index % 60]

    @property
    def month_ganzhi(self) -> str:
        return GANZHI_60[self.month_index % 60]

    @property
    def day_ganzhi(self) -> str:
        return GANZHI_60[self.day_index % 60]

    @property
    def hour_ganzhi(self) -> str:
        return GANZHI_60[self.hour_index % 60]

    def time_signal(self, t: float = 0.0) -> float:
        phases = [
            2 * math.pi * (self.year_index % 60) / 60,
            2 * math.pi * (self.month_index % 60) / 60,
            2 * math.pi * (self.day_index % 60) / 60,
            2 * math.pi * (self.hour_index % 60) / 60,
        ]
        signal = 0.0
        for n, phi in enumerate(phases, start=1):
            signal += math.cos(n * 2 * math.pi * t + phi) / n
        return signal

    def hexagram_phase_from_time(self) -> float:
        combined = (
            self.year_index * 525600
            + self.month_index * 43200
            + self.day_index * 1440
            + self.hour_index * 120
        )
        return 2 * math.pi * (combined % 129600) / 129600


class MultiScaleCalendar:
    def __init__(self, epoch_year: int = 0) -> None:
        self._epoch_year = epoch_year

    def time_layers(self, year: int, month: int = 1, day: int = 1, hour: int = 0) -> dict[str, float]:
        year_phase = 2 * math.pi * (year % 60) / 60
        month_phase = 2 * math.pi * ((year * 12 + month) % 60) / 60
        day_phase = 2 * math.pi * (day % 60) / 60

        return {
            "年": year_phase,
            "月": month_phase,
            "日": day_phase,
        }

    def cosmic_phases(self, year: int) -> dict[str, float]:
        yuan_progress = (year % YUAN_HUI_YUN_SHI["元"]) / YUAN_HUI_YUN_SHI["元"]
        hui_progress = (year % YUAN_HUI_YUN_SHI["会"]) / YUAN_HUI_YUN_SHI["会"]
        yun_progress = (year % YUAN_HUI_YUN_SHI["运"]) / YUAN_HUI_YUN_SHI["运"]
        shi_progress = (year % YUAN_HUI_YUN_SHI["世"]) / YUAN_HUI_YUN_SHI["世"]

        return {
            "元": 2 * math.pi * yuan_progress,
            "会": 2 * math.pi * hui_progress,
            "运": 2 * math.pi * yun_progress,
            "世": 2 * math.pi * shi_progress,
        }
