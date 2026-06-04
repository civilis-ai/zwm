from __future__ import annotations

import functools
from typing import ClassVar

from zwm.core.trigram import (
    Trigram,
    _FUXI_SQUARE_COL_TRIGRAMS,
    _FUXI_SQUARE_ROW_TRIGRAMS,
    trigram_from_index,
)
from zwm.core.yao import YANG, YIN, YaoLine


@functools.total_ordering
class Hexagram:
    __slots__ = ("_lines",)
    _lines: tuple[YaoLine, YaoLine, YaoLine, YaoLine, YaoLine, YaoLine]

    _HEXAGRAM_NAMES: ClassVar[dict[int, str]] = {
        0: "坤", 1: "剥", 2: "比", 3: "观", 4: "豫", 5: "晋", 6: "萃", 7: "否",
        8: "谦", 9: "艮", 10: "蹇", 11: "渐", 12: "小过", 13: "旅", 14: "咸", 15: "遁",
        16: "师", 17: "蒙", 18: "坎", 19: "涣", 20: "解", 21: "未济", 22: "困", 23: "讼",
        24: "升", 25: "蛊", 26: "井", 27: "巽", 28: "恒", 29: "鼎", 30: "大过", 31: "姤",
        32: "复", 33: "颐", 34: "屯", 35: "益", 36: "震", 37: "噬嗑", 38: "随", 39: "无妄",
        40: "明夷", 41: "贲", 42: "既济", 43: "家人", 44: "丰", 45: "离", 46: "革", 47: "同人",
        48: "临", 49: "损", 50: "节", 51: "中孚", 52: "归妹", 53: "睽", 54: "兑", 55: "履",
        56: "泰", 57: "大畜", 58: "需", 59: "小畜", 60: "大壮", 61: "大有", 62: "夬", 63: "乾",
    }

    _HEXAGRAM_UNICODE: ClassVar[dict[int, str]] = {
        63: "䷀", 62: "䷁", 61: "䷂", 60: "䷃",
        59: "䷄", 58: "䷅", 57: "䷆", 56: "䷇",
        55: "䷈", 54: "䷉", 53: "䷊", 52: "䷋",
        51: "䷌", 50: "䷍", 49: "䷎", 48: "䷏",
        47: "䷐", 46: "䷑", 45: "䷒", 44: "䷓",
        43: "䷔", 42: "䷕", 41: "䷖", 40: "䷗",
        39: "䷘", 38: "䷙", 37: "䷚", 36: "䷛",
        35: "䷜", 34: "䷝", 33: "䷞", 32: "䷟",
        31: "䷠", 30: "䷡", 29: "䷢", 28: "䷣",
        27: "䷤", 26: "䷥", 25: "䷦", 24: "䷧",
        23: "䷨", 22: "䷩", 21: "䷪", 20: "䷫",
        19: "䷬", 18: "䷭", 17: "䷮", 16: "䷯",
        15: "䷰", 14: "䷱", 13: "䷲", 12: "䷳",
        11: "䷴", 10: "䷵",  9: "䷶",  8: "䷷",
         7: "䷸",  6: "䷹",  5: "䷺",  4: "䷻",
         3: "䷼",  2: "䷽",  1: "䷾",  0: "䷿",
    }

    _NAJIA_ELEMENTS: ClassVar[dict[int, tuple[str, str, str, str, str, str]]] = {
        0: ("土", "土", "木", "木", "水", "金"),
    }

    def __init__(
        self,
        line0: YaoLine,
        line1: YaoLine,
        line2: YaoLine,
        line3: YaoLine,
        line4: YaoLine,
        line5: YaoLine,
    ) -> None:
        object.__setattr__(self, "_lines", (line0, line1, line2, line3, line4, line5))

    def __setattr__(self, name, value):
        raise AttributeError("Hexagram is immutable")

    def __delattr__(self, name):
        raise AttributeError("Hexagram is immutable")

    @property
    def chu(self) -> YaoLine:
        return self._lines[0]

    @property
    def er(self) -> YaoLine:
        return self._lines[1]

    @property
    def san(self) -> YaoLine:
        return self._lines[2]

    @property
    def si(self) -> YaoLine:
        return self._lines[3]

    @property
    def wu(self) -> YaoLine:
        return self._lines[4]

    @property
    def shang(self) -> YaoLine:
        return self._lines[5]

    @property
    def lines(self) -> tuple[YaoLine, YaoLine, YaoLine, YaoLine, YaoLine, YaoLine]:
        return self._lines

    @property
    def lower_trigram(self) -> Trigram:
        return trigram_from_index(
            (int(self._lines[2]) << 2)
            | (int(self._lines[1]) << 1)
            | int(self._lines[0])
        )

    @property
    def upper_trigram(self) -> Trigram:
        return trigram_from_index(
            (int(self._lines[5]) << 2)
            | (int(self._lines[4]) << 1)
            | int(self._lines[3])
        )

    @property
    def normal_order(self) -> int:
        return (
            (int(self._lines[5]) << 5)
            | (int(self._lines[4]) << 4)
            | (int(self._lines[3]) << 3)
            | (int(self._lines[2]) << 2)
            | (int(self._lines[1]) << 1)
            | int(self._lines[0])
        )

    @property
    def fuxi_index(self) -> int:
        return self.lower_trigram.index * 8 + self.upper_trigram.index

    @property
    def name(self) -> str:
        return self._HEXAGRAM_NAMES[self.normal_order]

    @property
    def unicode(self) -> str:
        return self._HEXAGRAM_UNICODE[self.fuxi_index]

    @property
    def binary_str(self) -> str:
        return "".join(str(int(line)) for line in reversed(self._lines))

    @property
    def phase_vector(self) -> tuple[int, int, int, int, int, int]:
        return tuple(line.phase for line in self._lines)

    def complex_phase_diffs(self) -> tuple[complex, complex, complex, complex, complex, complex]:
        return tuple(line.complex_phase for line in self._lines)

    def mutate(self, mask: int) -> Hexagram:
        if not 0 <= mask <= 63:
            raise ValueError(f"Mutation mask must be 0-63, got {mask}")
        return hexagram_from_bits(self.normal_order ^ mask)

    def interlock(self) -> Hexagram:
        return Hexagram(
            self._lines[1], self._lines[2], self._lines[3],
            self._lines[2], self._lines[3], self._lines[4],
        )

    def reverse(self) -> Hexagram:
        return Hexagram(
            self._lines[5], self._lines[4], self._lines[3],
            self._lines[2], self._lines[1], self._lines[0],
        )

    def complement(self) -> Hexagram:
        return hexagram_from_bits(self.normal_order ^ 0b111111)

    def square_row(self) -> int:
        return 7 - self.lower_trigram.pre_heaven_order + 1

    def square_col(self) -> int:
        return 7 - self.upper_trigram.pre_heaven_order + 1

    def square_position(self) -> tuple[int, int]:
        return (self.square_row(), self.square_col())

    def circular_phase(self) -> int:
        return _CIRCULAR_ORDER.index(self.fuxi_index)

    def solar_term(self) -> str:
        from zwm.core.constants import SOLAR_TERMS
        phase_idx = self.circular_phase()
        return SOLAR_TERMS[phase_idx % 24]

    def hamming_distance(self, other: Hexagram) -> int:
        return (self.normal_order ^ other.normal_order).bit_count()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hexagram):
            return NotImplemented
        return self.normal_order == other.normal_order

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Hexagram):
            return NotImplemented
        return self.normal_order < other.normal_order

    def __hash__(self) -> int:
        return hash(self.normal_order)

    def __repr__(self) -> str:
        return f"Hexagram({self.name}, 0b{self.binary_str})"

    def __str__(self) -> str:
        return self.unicode


_CIRCULAR_ORDER: list[int] = [
    32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63,
    31, 30, 29, 28, 27, 26, 25, 24,
    23, 22, 21, 20, 19, 18, 17, 16,
    15, 14, 13, 12, 11, 10,  9,  8,
     7,  6,  5,  4,  3,  2,  1,  0,
]


_ALL_HEXAGRAMS: list[Hexagram] = []
for _idx in range(64):
    _lines_tuple = tuple(
        YANG if (_idx >> (5 - i)) & 1 else YIN for i in range(6)
    )
    _h = Hexagram(*_lines_tuple)
    _ALL_HEXAGRAMS.append(_h)


def hexagram_from_bits(bits: int) -> Hexagram:
    if not 0 <= bits <= 63:
        raise ValueError(f"Hexagram bits must be 0-63, got {bits}")
    return _ALL_HEXAGRAMS[bits]


def hexagram_from_trigrams(upper: Trigram, lower: Trigram) -> Hexagram:
    ll = lower.lower
    lm = lower.middle
    lu = lower.upper
    ul = upper.lower
    um = upper.middle
    uu = upper.upper
    return Hexagram(ll, lm, lu, ul, um, uu)


def hexagram_from_name(name: str) -> Hexagram:
    for h in _ALL_HEXAGRAMS:
        if h.name == name:
            return h
    raise ValueError(f"Unknown hexagram name: {name}")


def all_hexagrams() -> tuple[Hexagram, ...]:
    return tuple(_ALL_HEXAGRAMS)


def hexagram_from_phase_vector(phases: tuple[int, int, int, int, int, int]) -> Hexagram:
    bits = 0
    for i, p in enumerate(phases):
        if p == 0:
            bits |= 1 << (5 - i)
    return _ALL_HEXAGRAMS[bits]


def fuxi_square_hexagram(row: int, col: int) -> Hexagram:
    lower = _FUXI_SQUARE_ROW_TRIGRAMS[row]
    upper = _FUXI_SQUARE_COL_TRIGRAMS[col]
    return hexagram_from_trigrams(upper, lower)
