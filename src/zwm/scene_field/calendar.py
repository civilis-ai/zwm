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
    def day_gan(self) -> str:
        """日天干 — 六亲太极点."""
        tian_gan = ("甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸")
        return tian_gan[self.day_index % 10]

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

    @classmethod
    def from_date(cls, year: int, month: int = 1, day: int = 1, hour: int = 0) -> "GanzhiTime":
        """从公历日期计算四柱干支索引.

        日干支基准: 2026-01-01 = 乙巳日 (index 41).
        """
        y_gz = (year - 4) % 60

        _ref_2026_01_01_gz = 41
        _days_since = (year - 2026) * 365 + (year - 2025) // 4 + sum(
            [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30][:month - 1]
        ) + day - 1
        if year % 4 == 0 and month > 2:
            _days_since += 1
        d_gz = (_ref_2026_01_01_gz + _days_since) % 60

        d_gan = d_gz % 10
        m_gan_start = {0: 2, 1: 4, 2: 6, 3: 8, 4: 0, 5: 2, 6: 4, 7: 6, 8: 8, 9: 0}
        m_gz = (m_gan_start.get(y_gz % 10, 2) * 10 + (month - 1) * 2) % 60

        h_base = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 0, 6: 2, 7: 4, 8: 6, 9: 8}
        h_gz = (h_base.get(d_gan, 0) * 10 + hour // 2) % 60

        return cls(year_index=y_gz, month_index=m_gz, day_index=d_gz, hour_index=h_gz)

    def time_signal(self, t: float = 0.0) -> float:
        """干支时间信号 — 基于60甲子周期的相位编码."""
        idx = self.day_index % 60
        return (idx / 60.0) * 2 * math.pi

    def hexagram_phase_from_time(self) -> float:
        """从干支推导卦象相位 — 天干定上卦，地支定下卦."""
        gan_idx = self.year_index % 10
        zhi_idx = self.month_index % 12
        upper_phase = (gan_idx / 10.0) * 2 * math.pi
        lower_phase = (zhi_idx / 12.0) * 2 * math.pi
        return (upper_phase + lower_phase) / 2.0


class MultiScaleCalendar:
    def __init__(self, epoch_year: int = 0) -> None:
        self._epoch_year = epoch_year

    def time_layers(self, year: int, month: int = 1, day: int = 1, hour: int = 0) -> dict[str, float]:
        year_phase = 2 * math.pi * (year % 60) / 60
        month_phase = 2 * math.pi * ((year * 12 + month) % 60) / 60
        day_phase = 2 * math.pi * (day % 60) / 60
        hour_phase = 2 * math.pi * (hour % 60) / 60

        return {
            "年": year_phase,
            "月": month_phase,
            "日": day_phase,
            "时": hour_phase,
        }

    def calendar_context(
        self,
        year: int,
        month: int = 1,
        day: int = 1,
        hour: int = 0,
        cosmic_phases: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """返回所有时间信号 (年/月/日/时 + 元/会/运/世 宇宙相)."""
        layers = self.time_layers(year, month, day, hour)
        cosmic = cosmic_phases if cosmic_phases is not None else self.cosmic_phases(year)
        ctx: dict[str, float] = {}
        ctx.update(layers)
        ctx.update(cosmic)
        return ctx

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

    def time_context(self, year: int, month: int = 1, day: int = 1, hour: int = 0):
        """生成完整的 TimeContext (新 API — 推荐使用).

        这是与 TimeContext 模块的桥接点 — 一行代码获得
        元会运世/值年卦/纳甲/节气/六亲的所有信息。
        """
        from zwm.scene_field.time_context import TimeContext
        return TimeContext.compute(year, month, day, hour, calendar=self)
