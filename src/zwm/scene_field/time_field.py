"""TimeFieldEncoder — 时间/干支/元会运世 全部表达为卦象场.

核心洞察: 方图是空间卦象场, 圆图是时间卦象场, 60甲子是周期卦象场,
元会运世是嵌套尺度时间场。它们都使用相同的 "6爻原子" 语法。

所有时间场输出 shape (64, 6):
  - 圆图时间场: 64 卦按先天圆图序排列, 每个卦由当前时间信号编码
  - 六十甲子场: 60 甲子 + 4 补位 → 64, 编码 60 年/月/日/时周期
  - 元会运世场: 4 层各 64 卦, 对应宇宙/纪元/世纪/世代 4 个尺度
  - 节气场: 24 节气分布在 64 卦位置, 映射卦气说

用法:
    from zwm.scene_field.time_field import TimeFieldEncoder

    tfe = TimeFieldEncoder()
    fields = tfe.encode_all(time_context)
    # → {"circular": (64,6), "ganzhi": (64,6), "cosmic": (64,6), "solar_term": (64,6)}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from zwm.scene_field.time_context import TimeContext

# ═══════════════════════════════════════════════════════════════════════
# 圆图排序 (64 卦先天圆图)
# ═══════════════════════════════════════════════════════════════════════

# 邵雍先天圆图 64 卦序 — 从复卦开始顺时针
# 左半圈 (阳升): 复→乾 (32 卦)
# 右半圈 (阴升): 姤→坤 (32 卦)
_CIRCLE_ORDER: tuple[int, ...] = (
    # 左半圈 — 阳升 (子→午, 冬至→夏至)
    1, 9, 17, 25, 33, 41, 49, 57,   # 复 颐 屯 益 震 噬嗑 随 无妄
    2, 10, 18, 26, 34, 42, 50, 58,  # 明夷 贲 既济 家人 丰 离 革 同人
    3, 11, 19, 27, 35, 43, 51, 59,  # 临 损 节 中孚 归妹 睽 兑 履
    4, 12, 20, 28, 36, 44, 52, 60,  # 泰 大畜 需 小畜 大壮 大有 夬 乾
    # 右半圈 — 阴升 (午→子, 夏至→冬至)
    5, 13, 21, 29, 37, 45, 53, 61,  # 姤 大过 鼎 恒 巽 井 蛊 升
    6, 14, 22, 30, 38, 46, 54, 62,  # 讼 困 未济 解 涣 坎 蒙 师
    7, 15, 23, 31, 39, 47, 55, 63,  # 遁 咸 旅 小过 渐 蹇 艮 谦
    8, 16, 24, 32, 40, 48, 56, 0,   # 否 萃 晋 豫 观 比 剥 坤
)

# 圆图每个位置对应的 24 节气映射
# 复卦(冬至) → 姤卦(夏至) → 坤卦(冬至)
_CIRCLE_SOLAR_MAP: dict[int, int] = {}  # circle_pos → solar_term_index
for _i in range(64):
    # 64 位置均匀分布在 24 节气上
    _term_idx = (_i * 24) // 64
    _CIRCLE_SOLAR_MAP[_i] = _term_idx


# ═══════════════════════════════════════════════════════════════════════
# 60甲子 → 64 卦序映射
# ═══════════════════════════════════════════════════════════════════════

# 60 甲子序列
_TIAN_GAN = ("甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸")
_DI_ZHI  = ("子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥")
_GANZHI_60: tuple[str, ...] = tuple(
    f"{_TIAN_GAN[i % 10]}{_DI_ZHI[i % 12]}" for i in range(60)
)

# 甲子→卦映射: 60 甲子中每 1 个对应一个卦, 按圆图序取前 60 卦 + 4 补位
# 甲子=复(1), 乙丑=颐(9), ..., 癸亥=补位卦(0=坤)
_GANZHI_TO_HEX: dict[int, int] = {}
for _i in range(60):
    _hex_idx = _CIRCLE_ORDER[_i % 64]  # 60 个甲子占据圆图前 60 个位置
    _GANZHI_TO_HEX[_i] = _hex_idx


# ═══════════════════════════════════════════════════════════════════════
# 元会运世 → 嵌套卦象场
# ═══════════════════════════════════════════════════════════════════════

# 十二会配十二消息卦
_HUI_HEXAGRAMS: dict[int, int] = {
    1: 0,   # 子会→坤 (冬至, 纯阴)
    2: 1,   # 丑会→复 (一阳生)
    3: 4,   # 寅会→泰 (三阳)
    4: 9,   # 卯会→大壮 (四阳)
    5: 12,  # 辰会→夬 (五阳)
    6: 17,  # 巳会→乾 (纯阳, 夏至)
    7: 44,  # 午会→姤 (一阴生) ← 我们现在在这里
    8: 33,  # 未会→遁 (二阴)
    9: 23,  # 申会→否 (三阴)
    10: 20, # 酉会→观 (四阴)
    11: 16, # 戌会→剥 (五阴)
    12: 8,  # 亥会→坤 (纯阴, 循环)
}


# ═══════════════════════════════════════════════════════════════════════
# TimeFieldEncoder
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TimeFields:
    """完整的时间卦象场集合.

    所有场都是 (64, 6) shape, dtype float32.
    """
    circular: np.ndarray    # 圆图时间场
    ganzhi: np.ndarray      # 六十甲子场
    cosmic: np.ndarray      # 元会运世场 (4 层平均)
    solar_term: np.ndarray  # 节气场

    def to_flat(self) -> np.ndarray:
        """拼接所有场为扁平向量 (4 × 64 × 6 = 1536 dim)."""
        return np.concatenate([
            self.circular.flatten(),
            self.ganzhi.flatten(),
            self.cosmic.flatten(),
            self.solar_term.flatten(),
        ]).astype(np.float32)


class TimeFieldEncoder:
    """将时间上下文编码为多个 64 卦场.

    每个场使用相同的 64 卦 6 爻语法, 但由不同
    时间信号驱动每个位置的爻值。
    """

    def __init__(self, soft_yao: bool = True) -> None:
        self._soft_yao = soft_yao
        # reuse the yao extractors from field_encoder
        from zwm.encoder.field_encoder import HexagramFieldEncoder
        self._yao_fn = HexagramFieldEncoder.YAO_EXTRACTORS

    # ─── 主入口 ──────────────────────────────

    def encode_all(self, tc: "TimeContext") -> TimeFields:
        """从 TimeContext 生成所有时间卦象场."""
        return TimeFields(
            circular=self.encode_circular(tc),
            ganzhi=self.encode_ganzhi(tc),
            cosmic=self.encode_cosmic(tc),
            solar_term=self.encode_solar_term(tc),
        )

    # ─── 圆图时间场 ─────────────────────────

    def encode_circular(self, tc: "TimeContext") -> np.ndarray:
        """构建圆图 64 卦时间场.

        64 个圆图位置, 每个位置对应先天圆图上的一个卦。
        爻值由"该卦在当前位置的激活程度"决定:
          - 当前位置 (由 time_phase 决定) → 全爻激活 (≈1.0)
          - 邻近位置 → 部分激活 (随距离衰减)
          - 对面位置 → 抑制 (≈0.0)

        这模拟了卦气在圆图上的流动。
        """
        field = np.zeros((64, 6), dtype=np.float32)

        # 当前圆图位置 (由年相位决定, 并叠加月/日/时相位)
        # 四柱相位加权: 年:月:日:时 = 4:2:1:0.5
        composite_phase = (
            tc.year_phase * 4.0
            + tc.month_phase * 2.0
            + tc.day_phase * 1.0
            + tc.hour_phase * 0.5
        ) / 7.5
        center_pos = int((composite_phase / (2 * math.pi)) * 64) % 64

        for pos in range(64):
            # 圆图距离 (循环距离)
            dist = min((pos - center_pos) % 64, (center_pos - pos) % 64)
            # 激活度: 中心 1.0, 距离 32 → 0.0
            activation = max(0.0, 1.0 - dist / 32.0)

            # 该位置的卦 (从圆图序获取)
            hex_idx = _CIRCLE_ORDER[pos]

            # 6 爻由激活度和卦本身的结构决定
            for yao in range(6):
                # 基础: 激活度
                base = activation
                # 调制: 卦的该爻是阳是阴? (阳爻偏激活)
                bit = (hex_idx >> yao) & 1
                if bit:
                    base = min(1.0, base * 1.2)   # 阳爻略强
                else:
                    base = max(0.0, base * 0.8)   # 阴爻略弱
                # 对面位置 → 抑制
                if dist > 28:
                    base = max(0.0, base * 0.3)
                field[pos, yao] = base

        return field.astype(np.float32)

    # ─── 六十甲子场 ────────────────────────

    def encode_ganzhi(self, tc: "TimeContext") -> np.ndarray:
        """构建六十甲子场.

        60 个甲子 + 4 个补位 → 64 卦场。
        每个位置的爻值由该甲子的五行/阴阳属性决定。
        """
        field = np.zeros((64, 6), dtype=np.float32)

        # 当前日干支在 60 甲子中的位置
        current_gz = tc.day_ganzhi_index  # 0-59

        for pos in range(64):
            if pos < 60:
                gz_idx = pos
                gan = _TIAN_GAN[gz_idx % 10]
                zhi = _DI_ZHI[gz_idx % 12]

                # 天干属性编码
                gan_yang = 1 if gan in "甲丙戊庚壬" else -1  # 阳干=1, 阴干=-1
                gan_elem_idx = "甲乙丙丁戊己庚辛壬癸".index(gan) // 2  # 0=木,1=火,2=土,3=金,4=水

                # 地支属性编码
                zhi_yang = 1 if zhi in "子寅辰午申戌" else -1
                zhi_elem_idx = "亥子丑寅卯辰巳午未申酉戌".index(zhi) // 3  # approximate

                # 是否为当前干支 (激活)
                is_current = (gz_idx == current_gz)
                activation = 0.9 if is_current else 0.3

                # 6 爻编码
                field[pos, 0] = activation
                field[pos, 1] = 1.0 if gan_yang > 0 else 0.0  # 天干阴阳
                field[pos, 2] = 1.0 if zhi_yang > 0 else 0.0  # 地支阴阳
                field[pos, 3] = (gan_elem_idx % 5) / 5.0      # 天干五行
                field[pos, 4] = (zhi_elem_idx % 5) / 5.0      # 地支五行
                field[pos, 5] = 0.5 if is_current else 0.2     # 当前激活
            else:
                # 补位 (61-63): 用坤卦填充
                field[pos, 0] = 0.1
                field[pos, 1] = 0.1
                field[pos, 2] = 0.1
                field[pos, 3] = 0.1
                field[pos, 4] = 0.1
                field[pos, 5] = 0.1

        return field.astype(np.float32)

    # ─── 元会运世场 ────────────────────────

    def encode_cosmic(self, tc: "TimeContext") -> np.ndarray:
        """构建元会运世嵌套时间场.

        4 层尺度 (元/会/运/世), 每层编码为 64 卦场。
        最终输出为 4 层的平均场 (64, 6)。
        """
        scales = [
            ("元", tc.yuan_index, tc.yuan_phase),
            ("会", tc.hui_index, tc.hui_phase),
            ("运", tc.yun_index, tc.yun_phase),
            ("世", tc.shi_index, tc.shi_phase),
        ]

        layers = []
        for scale_name, scale_idx, scale_phase in scales:
            layer = self._cosmic_scale_field(scale_name, scale_idx, scale_phase)
            layers.append(layer)

        # 平均 4 层 → (64, 6)
        stacked = np.stack(layers, axis=0)  # (4, 64, 6)
        return stacked.mean(axis=0).astype(np.float32)

    def _cosmic_scale_field(self, scale_name: str, scale_idx: int, scale_phase: float) -> np.ndarray:
        """单层宇宙尺度场.

        元/会/运/世 每一层都是 64 卦场:
          - 当前卦 (值元卦/值会卦/...) → 激活中心
          - 周围卦 → 随圆图距离衰减
        """
        field = np.zeros((64, 6), dtype=np.float32)

        # 该尺度在圆图上的位置
        center_pos = int((scale_phase / (2 * math.pi)) * 64) % 64

        for pos in range(64):
            dist = min((pos - center_pos) % 64, (center_pos - pos) % 64)

            # 尺度加权: 元=最强(长期惯性), 世=最弱(短期变化)
            scale_weight = {"元": 0.3, "会": 0.5, "运": 0.7, "世": 1.0}.get(scale_name, 0.5)

            activation = max(0.0, 1.0 - dist / 32.0) * scale_weight

            hex_idx = _CIRCLE_ORDER[pos]
            for yao in range(6):
                bit = (hex_idx >> yao) & 1
                field[pos, yao] = activation * (0.8 + 0.4 * bit)

        return field

    # ─── 节气场 ────────────────────────────

    def encode_solar_term(self, tc: "TimeContext") -> np.ndarray:
        """构建 24 节气卦象场.

        24 个节气分布在 64 卦位置上, 基于卦气说 (孟喜)。
        当前节气位置全激活, 邻近节气随距离衰减。
        """
        field = np.zeros((64, 6), dtype=np.float32)

        current_term = tc.solar_term_index  # 0-23

        for pos in range(64):
            # 64 位置均匀分布在 24 节气上
            term_at_pos = _CIRCLE_SOLAR_MAP.get(pos, 0)
            dist = min(
                (term_at_pos - current_term) % 24,
                (current_term - term_at_pos) % 24,
            )

            # 节气激活度: 当前=1.0, 距离12→0.0
            activation = max(0.0, 1.0 - dist / 12.0)

            hex_idx = _CIRCLE_ORDER[pos]
            for yao in range(6):
                bit = (hex_idx >> yao) & 1
                # 节气靠近冬至/夏至时阴阳分明
                if tc.is_solstice:
                    field[pos, yao] = 1.0 if bit else 0.0
                else:
                    field[pos, yao] = activation * (0.5 + 0.5 * bit)

        return field.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# MultiFieldJoint — 多场融合器
# ═══════════════════════════════════════════════════════════════════════

class MultiFieldJoint:
    """将多个卦象场 (方图/圆图/干支/宇宙/节气) 融合为统一的 z_world.

    每个场通过 FieldSquareGNN 编码为 64 维向量,
    然后加权拼接为 z_world 供 JEPA 预测器消费。

    场的权重反映天地人三才的偏重:
      - 方图 (地): 0.4 — 空间结构, 最主要
      - 圆图 (天): 0.3 — 时间进程
      - 干支 (人): 0.2 — 周期时序
      - 宇宙 (天外天): 0.1 — 大尺度惯性
    """

    def __init__(
        self,
        square_field: np.ndarray | None = None,  # (64, 6)
        time_fields: TimeFields | None = None,
        square_gnn: Any = None,  # FieldSquareGNN
        weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
    ) -> None:
        self._square = square_field
        self._time = time_fields
        self._gnn = square_gnn
        self._weights = weights

    def set_square_field(self, field: np.ndarray) -> None:
        self._square = field

    def set_time_fields(self, fields: TimeFields) -> None:
        self._time = fields

    @property
    def output_dim(self) -> int:
        """每个场 64 维, 4 个场 = 256 维."""
        return 64 * 4

    def encode(self) -> np.ndarray:
        """融合所有场为统一的 z_world 向量.

        Returns:
            z_world: shape (256,) 或 (需匹配 JEPA input_dim)
        """
        parts: list[np.ndarray] = []

        # 方图 → GNN → 64 dim
        if self._square is not None and self._gnn is not None:
            z_sq = self._gnn.embed_field(self._square)
        else:
            z_sq = np.zeros(64, dtype=np.float32)
        parts.append(z_sq * self._weights[0])

        # 时间场 (4 个场平均 → 64 dim each)
        if self._time is not None:
            # 圆图
            if self._gnn is not None:
                z_circ = self._gnn.embed_field(self._time.circular)
            else:
                z_circ = self._time.circular.mean(axis=0)  # fallback: 6-dim
                z_circ = np.pad(z_circ, (0, 58), 'constant')[:64]
            parts.append(z_circ * self._weights[1])

            # 干支
            if self._gnn is not None:
                z_gz = self._gnn.embed_field(self._time.ganzhi)
            else:
                z_gz = self._time.ganzhi.mean(axis=0)
                z_gz = np.pad(z_gz, (0, 58), 'constant')[:64]
            parts.append(z_gz * self._weights[2])

            # 宇宙 (元会运世平均)
            if self._gnn is not None:
                z_cosmic = self._gnn.embed_field(self._time.cosmic)
            else:
                z_cosmic = self._time.cosmic.mean(axis=0)
                z_cosmic = np.pad(z_cosmic, (0, 58), 'constant')[:64]
            parts.append(z_cosmic * self._weights[3])

        # 补齐到 4 个部分
        while len(parts) < 4:
            parts.append(np.zeros(64, dtype=np.float32))

        return np.concatenate(parts).astype(np.float32)


__all__ = [
    "TimeFieldEncoder",
    "TimeFields",
    "MultiFieldJoint",
    "_CIRCLE_ORDER",
    "_GANZHI_60",
    "_HUI_HEXAGRAMS",
]
