"""统一时间上下文 TimeContext — 易经全时间系统集成.

将分散在多个模块中的时间信号统一为一个 frozen dataclass:
  - 日历时间 (year/month/day/hour)
  - 元会运世 (邵雍皇极经世) — 含 indices + 值事卦
  - 60甲子干支周期 (year/month/day/hour 四柱)
  - 64卦值年/值月/值日/值时 (先天圆图映射)
  - 纳甲爻辰 (每卦六爻干支配属)
  - 24节气 (太阳黄经定位)
  - 中宫日干六亲 (太极点→社会场)
  - 卦气映射 (64卦→24节气)

这是 ZWM 时间系统的单一来源 — 所有子系统统一从此获取时间上下文。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from zwm.core.hexagram import Hexagram
    from zwm.scene_field.calendar import GanzhiTime, MultiScaleCalendar
    from zwm.self_field.palace_graph import LuoshuGrid


# ═══════════════════════════════════════════════════════════════════════
# 基础常数
# ═══════════════════════════════════════════════════════════════════════

_TIAN_GAN = ("甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸")
_DI_ZHI  = ("子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥")
_GAN_ELEMENTS = {"甲": "木", "乙": "木", "丙": "火", "丁": "火", "戊": "土",
                 "己": "土", "庚": "金", "辛": "金", "壬": "水", "癸": "水"}

# 邵雍 皇极经世 — 元会运世 时间量纲
_YUAN_YEARS  = 129600   # 1 元
_HUI_YEARS   = 10800    # 1 会 = 30 运
_YUN_YEARS   = 360      # 1 运 = 12 世
_SHI_YEARS   = 30       # 1 世

# 皇极经世 纪元: 当前元起始于公元前 67017 年 (标准换算)
# 2026 CE → 皇极经世年号 = 67017 + 2026 = 69043
_YUAN_EPOCH_BCE = 67017

# 24节气 (按阳历近似日期 — 月.日范围)
_SOLAR_TERMS = (
    ("立春", 2, 3, 2, 18),   # 0
    ("雨水", 2, 18, 3, 5),
    ("惊蛰", 3, 5, 3, 20),
    ("春分", 3, 20, 4, 4),
    ("清明", 4, 4, 4, 19),
    ("谷雨", 4, 19, 5, 5),
    ("立夏", 5, 5, 5, 20),
    ("小满", 5, 20, 6, 5),
    ("芒种", 6, 5, 6, 21),
    ("夏至", 6, 21, 7, 6),
    ("小暑", 7, 6, 7, 22),
    ("大暑", 7, 22, 8, 7),
    ("立秋", 8, 7, 8, 22),
    ("处暑", 8, 22, 9, 7),
    ("白露", 9, 7, 9, 22),
    ("秋分", 9, 22, 10, 7),
    ("寒露", 10, 7, 10, 23),
    ("霜降", 10, 23, 11, 7),
    ("立冬", 11, 7, 11, 21),
    ("小雪", 11, 21, 12, 6),
    ("大雪", 12, 6, 12, 21),
    ("冬至", 12, 21, 1, 5),
    ("小寒", 1, 5, 1, 19),
    ("大寒", 1, 19, 2, 3),
)

# 64卦先天圆图顺序 (邵雍) — 从 复 (#1) 到 坤 (#0) 的循环序
# 圆图排列: 左半圈阳升 (复→乾), 右半圈阴升 (姤→坤)
# 值年卦按此序循环
_XIANTIAN_CIRCLE: tuple[int, ...] = (
    1, 9, 17, 25, 33, 41, 49, 57,  # 复→小畜 (阳升)
    2, 10, 18, 26, 34, 42, 50, 58,
    3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60,
    5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62,
    7, 15, 23, 31, 39, 47, 55, 63,  # → 乾
    47, 39, 31, 23, 15, 7, 62, 54,  # 姤→坤 (阴升, 从姤 #47 反向)
    46, 38, 30, 22, 14, 6, 61, 53,
    45, 37, 29, 21, 13, 5, 60, 52,
    44, 36, 28, 20, 12, 4, 59, 51,
    43, 35, 27, 19, 11, 3, 58, 50,
    42, 34, 26, 18, 10, 2, 57, 49,
    41, 33, 25, 17, 9, 1, 56, 48,  # → 坤 (48=风地观) → 0=坤
    0, 8, 16, 24, 32, 40, 48, 56,  # 链接段
)
# 注: 完整的 64卦圆图有 64 个位置, 上面是简化版
# 实际使用 _VALUE_YEAR_CYCLE (64卦值年序)

# 64卦值年序 — 皇极经世标准序 (从复卦开始, 按先天圆图)
_VALUE_YEAR_CYCLE: tuple[int, ...] = (
    1,   # 复
    33,  # 颐
    17,  # 屯
    25,  # 益
    9,   # 震
    41,  # 噬嗑
    49,  # 随
    57,  # 无妄
    2,   # 明夷
    34,  # 贲
    18,  # 既济
    26,  # 家人
    10,  # 丰
    42,  # 离
    50,  # 革
    58,  # 同人
    3,   # 临
    35,  # 损
    19,  # 节
    27,  # 中孚
    11,  # 归妹
    43,  # 睽
    51,  # 兑
    59,  # 履
    4,   # 泰
    36,  # 大畜
    20,  # 需
    28,  # 小畜
    12,  # 大壮
    44,  # 大有
    52,  # 夬
    60,  # 乾
    5,   # 姤
    37,  # 大过
    21,  # 鼎
    29,  # 恒
    13,  # 巽
    45,  # 井
    53,  # 蛊
    61,  # 升
    6,   # 讼
    38,  # 困
    22,  # 未济
    30,  # 解
    14,  # 涣
    46,  # 坎
    54,  # 蒙
    62,  # 师
    7,   # 遁
    39,  # 咸
    23,  # 旅
    31,  # 小过
    15,  # 渐
    47,  # 蹇
    55,  # 艮
    63,  # 谦
    8,   # 否
    40,  # 萃
    24,  # 晋
    32,  # 豫
    16,  # 观
    48,  # 比
    56,  # 剥
    0,   # 坤
)
assert len(_VALUE_YEAR_CYCLE) == 64, f"值年卦序应为64, 实际 {len(_VALUE_YEAR_CYCLE)}"

# 每一值年卦又细分为 6 值月卦 (64卦→6爻→每月一卦)
# 值月卦: 值年卦的六爻依次变卦 (初爻→上爻)
# 值日卦: 从值月卦继续推演 (每爻一日, 六日一循环)
# 值时卦: 继续细分到时辰

# ═══════════════════════════════════════════════════════════════════════
# 纳甲爻辰系统 (Jing Fang 京房)
# ═══════════════════════════════════════════════════════════════════════

# 八纯卦纳甲: palace_trigram_index → [(gan, zhi)] × 6 yao
# gan_zhi pairs: (天干索引 0-9, 地支索引 0-11)
_NAJIA_8_PURE: dict[int, tuple[tuple[int, int], ...]] = {
    # 乾宫 (index 7) — 乾为天: 甲子,甲寅,甲辰,壬午,壬申,壬戌
    7: ((0, 0), (0, 2), (0, 4), (8, 6), (8, 8), (8, 10)),
    # 坤宫 (index 0) — 坤为地: 乙未,乙巳,乙卯,癸丑,癸亥,癸酉
    0: ((1, 7), (1, 5), (1, 3), (9, 1), (9, 11), (9, 9)),
    # 震宫 (index 1) — 震为雷: 庚子,庚寅,庚辰,庚午,庚申,庚戌
    1: ((6, 0), (6, 2), (6, 4), (6, 6), (6, 8), (6, 10)),
    # 巽宫 (index 6) — 巽为风: 辛丑,辛亥,辛酉,辛未,辛巳,辛卯
    6: ((7, 1), (7, 11), (7, 9), (7, 7), (7, 5), (7, 3)),
    # 坎宫 (index 2) — 坎为水: 戊寅,戊辰,戊午,戊申,戊戌,戊子
    2: ((4, 2), (4, 4), (4, 6), (4, 8), (4, 10), (4, 0)),
    # 离宫 (index 5) — 离为火: 己卯,己丑,己亥,己酉,己未,己巳
    5: ((5, 3), (5, 1), (5, 11), (5, 9), (5, 7), (5, 5)),
    # 艮宫 (index 4) — 艮为山: 丙辰,丙午,丙申,丙戌,丙子,丙寅
    4: ((2, 4), (2, 6), (2, 8), (2, 10), (2, 0), (2, 2)),
    # 兑宫 (index 3) — 兑为泽: 丁巳,丁卯,丁丑,丁亥,丁酉,丁未
    3: ((3, 5), (3, 3), (3, 1), (3, 11), (3, 9), (3, 7)),
}

# 八纯卦世爻位置 (游魂/归魂特殊处理)
_SHI_YAO_POS: dict[int, int] = {
    7: 5, 0: 5, 1: 5, 6: 5, 2: 5, 5: 5, 4: 5, 3: 5,  # 八纯卦 — 上爻
}
# 对于非纯卦, 世爻位置由卦变层次决定 (1世→初爻, 2世→二爻, ...)


def _najia_for_hexagram(h_normal_order: int) -> tuple[tuple[int, int], ...]:
    """返回卦象的六爻纳甲干支索引.

    规则: 八纯卦直接用固定纳甲; 其他卦按京房八宫世应体系
    从所属宫的纯卦推断。非纯卦的纳甲继承纯卦爻位干支。

    Returns:
        tuple of (gan_index, zhi_index) × 6
    """
    from zwm.core.constants import _HEXAGRAM_TO_PALACE
    palace_idx = _HEXAGRAM_TO_PALACE[h_normal_order]
    pure = _NAJIA_8_PURE.get(palace_idx)
    if pure is None:
        raise ValueError(f"Unknown palace index {palace_idx} for hex #{h_normal_order}")
    # 如果是八纯卦, 直接返回
    if h_normal_order in {7, 0, 1, 6, 2, 5, 4, 3}:  # 这些是八纯卦的 normal_order
        pass
    # 简化处理: 所有同宫卦使用纯卦纳甲 (京房八宫体系)
    # 更精确的实现需要考虑世爻移位, 但基础框架使用统一纳甲
    return pure


# ═══════════════════════════════════════════════════════════════════════
# 64卦卦气映射 — 孟喜卦气说
# ═══════════════════════════════════════════════════════════════════════

# 64卦→24节气映射 (四正卦 + 六十卦分直)
# 坎离震兑 四正卦司四时
# 其余60卦每卦直6日7分
_FOUR_SEASONS_HEX = {2: "冬至", 5: "夏至", 1: "春分", 3: "秋分"}  # 坎离震兑
_HEX_TO_SOLAR_TERM: dict[int, int] = {}  # hex normal_order → solar term index (0-23)

# 60卦分直24节气 (每节气约15天, 每卦直6日7分, 即 2.5卦/节气)
# 简化: 用卦序号映射到节气区间
for _i, _hex in enumerate(_VALUE_YEAR_CYCLE):
    if _hex not in _FOUR_SEASONS_HEX:
        _term_idx = (_i * 60 // 64) % 24
        _HEX_TO_SOLAR_TERM[_hex] = _term_idx


# ═══════════════════════════════════════════════════════════════════════
# TimeContext
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TimeContext:
    """统一时间上下文 — ZWM 所有子系统的时间信号单一来源.

    字段分组:
      - 日历时间 (year, month, day, hour)
      - 四柱干支 (年/月/日/时 柱, ganzhi indices + 天干)
      - 元会运世 (邵雍皇极经世 indices + 值事卦)
      - 64卦值卦 (值年/值月/值日/值时卦)
      - 节气 (当前24节气 + 卦气映射)
      - 六亲 (中宫日干→社会场关系)
      - 纳甲 (当前卦的六爻干支)
      - 相位编码 (所有时间层次的 phase ∈ [0, 2π))
    """

    # ─── 日历时间 ───────────────────────────
    year: int = 2026
    month: int = 1
    day: int = 1
    hour: int = 0

    # ─── 四柱干支 ───────────────────────────
    day_gan: str = "甲"
    year_gan: str = "丙"
    month_gan: str = "庚"
    hour_gan: str = "甲"

    year_ganzhi_index: int = 1    # 0-59
    month_ganzhi_index: int = 1   # 0-59
    day_ganzhi_index: int = 1     # 0-59
    hour_ganzhi_index: int = 1    # 0-59

    @property
    def year_ganzhi(self) -> str:
        return f"{_TIAN_GAN[self.year_ganzhi_index % 10]}{_DI_ZHI[self.year_ganzhi_index % 12]}"

    @property
    def month_ganzhi(self) -> str:
        return f"{_TIAN_GAN[self.month_ganzhi_index % 10]}{_DI_ZHI[self.month_ganzhi_index % 12]}"

    @property
    def day_ganzhi(self) -> str:
        return f"{_TIAN_GAN[self.day_ganzhi_index % 10]}{_DI_ZHI[self.day_ganzhi_index % 12]}"

    @property
    def hour_ganzhi(self) -> str:
        return f"{_TIAN_GAN[self.hour_ganzhi_index % 10]}{_DI_ZHI[self.hour_ganzhi_index % 12]}"

    @property
    def ganzhi_str(self) -> str:
        return f"{self.year_ganzhi}年 {self.month_ganzhi}月 {self.day_ganzhi}日 {self.hour_ganzhi}时"

    # ─── 元会运世 (邵雍皇极经世) ──────────
    yuan_index: int = 1         # 当前元 (1-based, 皇极经世第7元)
    hui_index: int = 7          # 当前会 (1-12, 午会=7)
    yun_index: int = 196        # 当前运 (在会内, 1-30)
    shi_index: int = 2350       # 当前世 (在运内, 1-12)

    # 值事卦 — 每个时间层级对应的卦象 (normal_order)
    yuan_hex: int = 1           # 值元卦
    hui_hex: int = 5            # 值会卦 (午会→姤)
    yun_hex: int = 44           # 值运卦
    shi_hex: int = 38           # 值世卦

    @property
    def yuan_progress(self) -> float:
        """当前元内的进度 [0, 1)."""
        return ((self.year + _YUAN_EPOCH_BCE) % _YUAN_YEARS) / _YUAN_YEARS

    @property
    def hui_progress(self) -> float:
        return ((self.year + _YUAN_EPOCH_BCE) % _HUI_YEARS) / _HUI_YEARS

    @property
    def yun_progress(self) -> float:
        return ((self.year + _YUAN_EPOCH_BCE) % _YUN_YEARS) / _YUN_YEARS

    @property
    def shi_progress(self) -> float:
        return ((self.year + _YUAN_EPOCH_BCE) % _SHI_YEARS) / _SHI_YEARS

    # ─── 64卦值卦 ──────────────────────────
    value_year_hex: int = 1     # 值年卦 normal_order
    value_month_hex: int = 1    # 值月卦 normal_order
    value_day_hex: int = 1      # 值日卦 normal_order
    value_hour_hex: int = 1     # 值时卦 normal_order

    # ─── 节气 ─────────────────────────────
    solar_term_index: int = 0   # 0-23
    solar_term_name: str = "立春"
    next_solar_term_name: str = "雨水"

    @property
    def solar_term_phase(self) -> float:
        """节气内的相位进度 [0, 2π)."""
        return 2 * math.pi * (self.solar_term_index / 24.0)

    @property
    def is_solstice(self) -> bool:
        return self.solar_term_name in ("冬至", "夏至")

    @property
    def is_equinox(self) -> bool:
        return self.solar_term_name in ("春分", "秋分")

    # ─── 当前卦的卦气 ──────────────────────
    hex_solar_term_index: int = 0  # 当前 hex 对应的节气

    # ─── 中宫六亲 ──────────────────────────
    self_element: str = "木"    # 日干→自我五行
    # 六亲映射: palace_position → relation_type
    six_relations: dict[int, str] = field(default_factory=dict)

    # ─── 纳甲爻辰 (当前卦) ────────────────
    # najia: tuple[tuple[gan_idx, zhi_idx], ...] × 6
    najia: tuple[tuple[int, int], ...] = field(default_factory=lambda: tuple((0, 0) for _ in range(6)))

    # ─── 其他时间属性 ─────────────────────
    season: str = "春"
    is_daytime: bool = True
    cosmic_phase_index: int = 0   # 0-23

    # ─── 相位编码 (缓存, 避免重复计算) ────
    time_phase: float = 0.0       # 年相位 (向后兼容)
    year_phase: float = 0.0
    month_phase: float = 0.0
    day_phase: float = 0.0
    hour_phase: float = 0.0
    yuan_phase: float = 0.0
    hui_phase: float = 0.0
    yun_phase: float = 0.0
    shi_phase: float = 0.0

    # ─── 工厂方法 ──────────────────────────

    @classmethod
    def compute(cls, year: int, month: int = 1, day: int = 1, hour: int = 0,
                calendar: "MultiScaleCalendar | None" = None,
                ganzhi: "GanzhiTime | None" = None) -> "TimeContext":
        """从日历时间计算完整的 TimeContext.

        这是推荐的构造方式 — 一行代码获得所有易经时间信号。
        """
        # 1) 四柱干支
        y_gan_idx = (year - 4) % 10   # 年天干 (甲子年起于公元4年)
        y_zhi_idx = (year - 4) % 12
        y_gz = (year - 4) % 60

        # 月干支 (年上起月: 甲己之年丙作首)
        m_base = {0: 2, 1: 4, 2: 6, 3: 8, 4: 0, 5: 2, 6: 4, 7: 6, 8: 8, 9: 0}
        m_gan_start = m_base.get(y_gan_idx, 2)
        m_gz = (m_gan_start * 10 + (month - 1) * 2) % 60  # 简化

        # 日干支 — 已知 2026-01-01 = 乙巳日 (day_gz_idx=41), 往后推算
        _ref_2026_01_01_gz = 41
        _days_since_ref = (year - 2026) * 365 + (year - 2024) // 4 + sum(
            [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30][:month - 1]
        ) + day - 1
        if year % 4 == 0 and month > 2:
            _days_since_ref += 1
        d_gz = (_ref_2026_01_01_gz + _days_since_ref) % 60

        # 时干支 (日上起时: 甲己日起甲子)
        d_gan = d_gz % 10
        h_base = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 0, 6: 2, 7: 4, 8: 6, 9: 8}
        h_gan_start = h_base.get(d_gan, 0)
        h_gz = (h_gan_start * 10 + hour // 2) % 60

        # 提取天干
        day_gan = _TIAN_GAN[d_gz % 10]
        year_gan = _TIAN_GAN[y_gan_idx]
        month_gan = _TIAN_GAN[m_gz % 10]
        hour_gan = _TIAN_GAN[h_gz % 10]

        # 2) 元会运世
        jingshi_year = year + _YUAN_EPOCH_BCE
        yuan_i = max(1, jingshi_year // _YUAN_YEARS + 1)
        hui_i = max(1, min(12, (jingshi_year % _YUAN_YEARS) // _HUI_YEARS + 1))
        yun_i = max(1, min(30, (jingshi_year % _HUI_YEARS) // _YUN_YEARS + 1))
        shi_i = max(1, min(12, (jingshi_year % _YUN_YEARS) // _SHI_YEARS + 1))

        # 值会卦 — 十二会配十二消息卦
        _HUI_HEX = {1: 0, 2: 1, 3: 4, 4: 9, 5: 12, 6: 17, 7: 44, 8: 33, 9: 23, 10: 20, 11: 16, 12: 8}
        hui_hex = _HUI_HEX.get(hui_i, 44)

        # 3) 值年卦 — 从值会卦开始, 按圆图序逐年推进
        # 找到值会卦在圆图中的位置, 然后前进 yun_offset 步
        hui_pos = _VALUE_YEAR_CYCLE.index(hui_hex) if hui_hex in _VALUE_YEAR_CYCLE else 0
        year_in_hui = jingshi_year % _HUI_YEARS
        value_year_hex = _VALUE_YEAR_CYCLE[(hui_pos + year_in_hui) % 64]

        # 值月卦 — 值年卦的六爻依次变卦
        moon_in_year = (month - 1) % 12
        value_month_hex = _value_month_from_year_hex(value_year_hex, moon_in_year)

        # 值日卦 — 从值月卦继续推 (每6天一变)
        day_in_moon = (day - 1) % 30
        value_day_hex = _value_day_from_month_hex(value_month_hex, day_in_moon)

        # 值时卦 — 继续细分
        hour_in_day = hour % 24
        value_hour_hex = _value_hour_from_day_hex(value_day_hex, hour_in_day)

        # 4) 节气
        st_idx = _compute_solar_term(month, day)
        st_name = _SOLAR_TERMS[st_idx][0]
        next_st_name = _SOLAR_TERMS[(st_idx + 1) % 24][0]

        # 5) 中宫六亲
        self_elem = _GAN_ELEMENTS.get(day_gan, "木")
        six_rel = _compute_six_relations_from_gan(day_gan)

        # 6) 季节/昼夜
        season = _compute_season(month)
        is_daytime = 6 <= hour < 18
        cosmic_phase_idx = _compute_cosmic_phase(hour, season)

        # 7) 纳甲 (默认乾卦, 运行时由 hex 参数覆盖)
        najia = _najia_for_hexagram(value_year_hex)

        # 8) 卦气
        hex_st = _HEX_TO_SOLAR_TERM.get(value_year_hex, 0)

        # 9) 相位
        phases = {}
        if calendar is not None:
            phases = calendar.time_layers(year, month, day, hour)
            cosmic = calendar.cosmic_phases(year)
            phases.update(cosmic)

        return cls(
            year=year, month=month, day=day, hour=hour,
            day_gan=day_gan, year_gan=year_gan, month_gan=month_gan, hour_gan=hour_gan,
            year_ganzhi_index=y_gz, month_ganzhi_index=m_gz,
            day_ganzhi_index=d_gz, hour_ganzhi_index=h_gz,
            yuan_index=yuan_i, hui_index=hui_i, yun_index=yun_i, shi_index=shi_i,
            yuan_hex=0, hui_hex=hui_hex, yun_hex=_VALUE_YEAR_CYCLE[(hui_pos + (yun_i - 1) * _SHI_YEARS * 12) % 64], shi_hex=_VALUE_YEAR_CYCLE[(hui_pos + year_in_hui) % 64],
            value_year_hex=value_year_hex,
            value_month_hex=value_month_hex,
            value_day_hex=value_day_hex,
            value_hour_hex=value_hour_hex,
            solar_term_index=st_idx, solar_term_name=st_name, next_solar_term_name=next_st_name,
            self_element=self_elem, six_relations=six_rel,
            najia=najia, hex_solar_term_index=hex_st,
            season=season, is_daytime=is_daytime, cosmic_phase_index=cosmic_phase_idx,
            time_phase=phases.get("年", 0.0),
            year_phase=phases.get("年", 0.0),
            month_phase=phases.get("月", 0.0),
            day_phase=phases.get("日", 0.0),
            hour_phase=phases.get("时", 0.0),
            yuan_phase=phases.get("元", 0.0),
            hui_phase=phases.get("会", 0.0),
            yun_phase=phases.get("运", 0.0),
            shi_phase=phases.get("世", 0.0),
        )

    @classmethod
    def from_calendar(cls, calendar: "MultiScaleCalendar",
                      ganzhi: "GanzhiTime | None" = None,
                      year: int = 2026, month: int = 1, day: int = 1, hour: int = 0) -> "TimeContext":
        """从已有 calendar + ganzhi 实例构造 (向后兼容)."""
        return cls.compute(year, month, day, hour, calendar=calendar, ganzhi=ganzhi)

    @classmethod
    def from_agent(cls, agent, year: int = 2026, month: int = 1,
                   day: int = 1, hour: int = 0) -> "TimeContext":
        """从 TrinityAgent 构造 (读取其 calendar + ganzhi)."""
        calendar = getattr(agent, "calendar", None)
        ganzhi = getattr(agent, "ganzhi", None)
        return cls.compute(year, month, day, hour, calendar=calendar, ganzhi=ganzhi)

    @classmethod
    def for_hexagram(cls, h: "Hexagram", year: int = 2026, month: int = 1,
                     day: int = 1, hour: int = 0,
                     calendar: "MultiScaleCalendar | None" = None,
                     ganzhi: "GanzhiTime | None" = None) -> "TimeContext":
        """构造 TimeContext 并计算给定卦象的纳甲.

        这是运行时的推荐用法 — 提供当前卦象以获取精确的纳甲爻辰。
        """
        tc = cls.compute(year, month, day, hour, calendar=calendar, ganzhi=ganzhi)
        h_no = getattr(h, "normal_order", 1)
        # 重新计算纳甲 (基于当前卦象)
        new_najia = _najia_for_hexagram(h_no)
        hex_st = _HEX_TO_SOLAR_TERM.get(h_no, tc.solar_term_index)
        return cls(
            year=tc.year, month=tc.month, day=tc.day, hour=tc.hour,
            day_gan=tc.day_gan, year_gan=tc.year_gan, month_gan=tc.month_gan, hour_gan=tc.hour_gan,
            year_ganzhi_index=tc.year_ganzhi_index, month_ganzhi_index=tc.month_ganzhi_index,
            day_ganzhi_index=tc.day_ganzhi_index, hour_ganzhi_index=tc.hour_ganzhi_index,
            yuan_index=tc.yuan_index, hui_index=tc.hui_index, yun_index=tc.yun_index, shi_index=tc.shi_index,
            yuan_hex=tc.yuan_hex, hui_hex=tc.hui_hex, yun_hex=tc.yun_hex, shi_hex=tc.shi_hex,
            value_year_hex=tc.value_year_hex, value_month_hex=tc.value_month_hex,
            value_day_hex=tc.value_day_hex, value_hour_hex=tc.value_hour_hex,
            solar_term_index=tc.solar_term_index, solar_term_name=tc.solar_term_name,
            next_solar_term_name=tc.next_solar_term_name,
            hex_solar_term_index=hex_st,
            self_element=tc.self_element, six_relations=tc.six_relations,
            najia=new_najia,
            season=tc.season, is_daytime=tc.is_daytime, cosmic_phase_index=tc.cosmic_phase_index,
            time_phase=tc.time_phase, year_phase=tc.year_phase,
            month_phase=tc.month_phase, day_phase=tc.day_phase, hour_phase=tc.hour_phase,
            yuan_phase=tc.yuan_phase, hui_phase=tc.hui_phase,
            yun_phase=tc.yun_phase, shi_phase=tc.shi_phase,
        )

    # ─── 便利方法 ──────────────────────────

    @property
    def yuan_ganzhi(self) -> str:
        return _TIAN_GAN[self.yuan_index % 10] + _DI_ZHI[self.yuan_index % 12]

    @property
    def hui_ganzhi(self) -> str:
        return _TIAN_GAN[self.hui_index % 10] + _DI_ZHI[self.hui_index % 12]

    def najia_str(self) -> list[str]:
        """返回纳甲爻辰的人类可读字符串."""
        return [
            f"{_TIAN_GAN[g]}{_DI_ZHI[z]}"
            for g, z in self.najia
        ]

    def get_phase_for_scale(self, scale: str) -> float:
        """按时间尺度获取相位值."""
        return {
            "元": self.yuan_phase, "会": self.hui_phase,
            "运": self.yun_phase, "世": self.shi_phase,
            "年": self.year_phase, "月": self.month_phase,
            "日": self.day_phase, "时": self.hour_phase,
        }.get(scale, self.time_phase)

    def to_dict(self) -> dict:
        """转为字典 (JSON/MCP/API 传输)."""
        return {
            "year": self.year, "month": self.month, "day": self.day, "hour": self.hour,
            "ganzhi": self.ganzhi_str,
            "day_gan": self.day_gan,
            "self_element": self.self_element,
            "six_relations": self.six_relations,
            "yuan_hui_yun_shi": {
                "yuan": self.yuan_index, "hui": self.hui_index,
                "yun": self.yun_index, "shi": self.shi_index,
            },
            "value_hexagrams": {
                "year": self.value_year_hex, "month": self.value_month_hex,
                "day": self.value_day_hex, "hour": self.value_hour_hex,
            },
            "solar_term": self.solar_term_name,
            "season": self.season,
            "is_daytime": self.is_daytime,
            "najia": self.najia_str(),
            "phases": {
                "yuan": self.yuan_phase, "hui": self.hui_phase,
                "yun": self.yun_phase, "shi": self.shi_phase,
                "year": self.year_phase, "month": self.month_phase,
                "day": self.day_phase, "hour": self.hour_phase,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# 辅助计算函数
# ═══════════════════════════════════════════════════════════════════════

def _compute_solar_term(month: int, day: int) -> int:
    """计算当前日期对应的24节气索引 (0-23)."""
    for idx, (_name, m1, d1, m2, d2) in enumerate(_SOLAR_TERMS):
        if month == m1 and day >= d1:
            return idx
        if month == m2 and day < d2:
            continue
    for idx, (_name, m1, d1, _m2, _d2) in enumerate(_SOLAR_TERMS):
        if month == m1 and day < d1:
            return (idx - 1) % 24
    return 0


def _compute_season(month: int) -> str:
    if 3 <= month <= 5:
        return "春"
    elif 6 <= month <= 8:
        return "夏"
    elif 9 <= month <= 11:
        return "秋"
    return "冬"


def _compute_cosmic_phase(hour: int, season: str) -> int:
    season_offset = {"春": 0, "夏": 6, "秋": 12, "冬": 18}
    return (hour + season_offset.get(season, 0)) % 24


def _compute_six_relations_from_gan(day_gan: str) -> dict[int, str]:
    """从日干计算洛书九宫六亲关系.

    中宫(5)=我, 其他宫位按五行生克关系定六亲.
    """
    self_elem = _GAN_ELEMENTS.get(day_gan, "木")
    from zwm.core.constants import (
        ELEMENT_GENERATION, ELEMENT_CONTROL, PALACE_ELEMENT,
    )
    relations = {5: "我"}
    for pos in range(1, 10):
        if pos == 5:
            continue
        pelem = PALACE_ELEMENT.get(pos % 8, "土")
        if pelem == self_elem:
            relations[pos] = "兄弟"
        elif ELEMENT_GENERATION.get(pelem) == self_elem:
            relations[pos] = "父母"
        elif ELEMENT_GENERATION.get(self_elem) == pelem:
            relations[pos] = "子孙"
        elif ELEMENT_CONTROL.get(pelem) == self_elem:
            relations[pos] = "官鬼"
        elif ELEMENT_CONTROL.get(self_elem) == pelem:
            relations[pos] = "妻财"
        else:
            relations[pos] = "兄弟"
    return relations


def _value_month_from_year_hex(year_hex: int, moon_in_year: int) -> int:
    """值月卦 — 值年卦的第 n 爻变.

    规则: 初爻变→正月, 二爻变→二月, ..., 上爻变→六月,
          然后返回初爻继续.
    """
    from zwm.core.hexagram import Hexagram, hexagram_from_bits
    h = hexagram_from_bits(year_hex)
    yao = moon_in_year % 6
    # 翻转第 yao 位 (爻位: 0=初爻, 5=上爻)
    mask = 1 << yao
    new_bits = year_hex ^ mask
    return new_bits


def _value_day_from_month_hex(month_hex: int, day_in_moon: int) -> int:
    """值日卦 — 每6天从值月卦推演一次."""
    # 每6天一变, 一个月5变
    cycle = day_in_moon % 6
    if cycle == 0:
        return month_hex
    mask = 1 << (cycle - 1)
    return month_hex ^ mask


def _value_hour_from_day_hex(day_hex: int, hour_in_day: int) -> int:
    """值时卦 — 每2小时一变 (12时辰)."""
    cycle = hour_in_day % 12
    if cycle == 0:
        return day_hex
    mask = 1 << (cycle % 6)
    return day_hex ^ mask


# 导出
__all__ = [
    "TimeContext",
    "_najia_for_hexagram",
    "_compute_solar_term",
    "_compute_six_relations_from_gan",
    "_VALUE_YEAR_CYCLE",
    "_NAJIA_8_PURE",
    "_HEX_TO_SOLAR_TERM",
]
