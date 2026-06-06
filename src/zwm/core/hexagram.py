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
    """六十四卦不可变值类型。

    Encoding convention (邵康节先天序):
      bit 5,4,3 = 上爻, 五爻, 四爻 (upper trigram / 左卦)
      bit 2,1,0 = 三爻, 二爻, 初爻 (lower trigram / 右卦)
      Read order: 上→初 (top to bottom, line5→line0)
      Yang = 1, Yin = 0
      normal_order = binary value = 先天序位置 (0=坤..63=乾)
    """
    __slots__ = ("_lines",)
    _lines: tuple[YaoLine, YaoLine, YaoLine, YaoLine, YaoLine, YaoLine]

    # 先天64卦序 names — indexed by normal_order (binary value).
    # no = upper_trigram_bits(5,4,3) << 3 | lower_trigram_bits(2,1,0)
    _HEXAGRAM_NAMES: ClassVar[dict[int, str]] = {
         0: "坤为地",  1: "地雷复",  2: "地水师",  3: "地泽临",
         4: "地山谦",  5: "地火明夷", 6: "地风升",  7: "地天泰",
         8: "雷地豫",  9: "雷为雷", 10: "雷水解", 11: "雷泽归妹",
        12: "雷山小过", 13: "雷火丰", 14: "雷风恒", 15: "雷天大壮",
        16: "水地比", 17: "水雷屯", 18: "水为水", 19: "水泽节",
        20: "水山蹇", 21: "水火既济", 22: "水风井", 23: "水天需",
        24: "泽地萃", 25: "泽雷随", 26: "泽水困", 27: "泽为泽",
        28: "泽山咸", 29: "泽火革", 30: "泽风大过", 31: "泽天夬",
        32: "山地剥", 33: "山雷颐", 34: "山水蒙", 35: "山泽损",
        36: "山为山", 37: "山火贲", 38: "山风蛊", 39: "山天大畜",
        40: "火地晋", 41: "火雷噬嗑", 42: "火水未济", 43: "火泽睽",
        44: "火山旅", 45: "火为火", 46: "火风鼎", 47: "火天大有",
        48: "风地观", 49: "风雷益", 50: "风水涣", 51: "风泽中孚",
        52: "风山渐", 53: "风火家人", 54: "风为风", 55: "风天小畜",
        56: "天地否", 57: "天雷无妄", 58: "天水讼", 59: "天泽履",
        60: "天山遁", 61: "天火同人", 62: "天风姤", 63: "乾为天",
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
    def lines(self) -> tuple[YaoLine, YaoLine, YaoLine, YaoLine, YaoLine, YaoLine]:
        return self._lines

    @property
    def lower_trigram(self) -> Trigram:
        """下卦 (右卦/阴): lines 0-2 = 初爻,二爻,三爻."""
        return trigram_from_index(
            (int(self._lines[2]) << 2)
            | (int(self._lines[1]) << 1)
            | int(self._lines[0])
        )

    @property
    def upper_trigram(self) -> Trigram:
        """上卦 (左卦/阳): lines 3-5 = 四爻,五爻,上爻."""
        return trigram_from_index(
            (int(self._lines[5]) << 2)
            | (int(self._lines[4]) << 1)
            | int(self._lines[3])
        )

    @property
    def normal_order(self) -> int:
        """先天序 index.  bit5=上爻(MSB) .. bit0=初爻(LSB).  0=坤..63=乾."""
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
        """伏羲方图 index = normal_order (先天64卦序)."""
        return self.normal_order

    @property
    def name(self) -> str:
        """卦名 (先天64卦序)."""
        return self._HEXAGRAM_NAMES[self.normal_order]

    @property
    def unicode(self) -> str:
        """Hexagram symbol: chr(0x4DC0 + normal_order)."""
        return chr(0x4DC0 + self.normal_order)

    @property
    def binary_str(self) -> str:
        """Top-to-bottom display: 上爻→初爻."""
        return "".join(str(int(line)) for line in reversed(self._lines))

    @property
    def phase_vector(self) -> tuple[int, int, int, int, int, int]:
        return tuple(line.phase for line in self._lines)

    def complex_phase_diffs(self) -> tuple[complex, complex, complex, complex, complex, complex]:
        return tuple(line.complex_phase for line in self._lines)

    def to_phase_vector(self) -> "tuple[complex, complex, complex, complex, complex, complex]":
        """Return the 6-line complex phase vector for spectrum analysis.

        P1-arch: this method inverts the spectrum → core dependency.
        Previously ``HexagramPhaseVector.from_hexagram(h)`` in
        ``spectrum/complex_phase.py`` lazily imported ``Hexagram``,
        which meant the frequency-domain module depended on the data
        model.  Now the data model exposes its phase representation
        directly, and spectrum only depends on primitive types.
        """
        return tuple(line.complex_phase for line in self._lines)

    def mutate(self, mask: int) -> Hexagram:
        """XOR mutation: flip bits indicated by mask."""
        if not 0 <= mask <= 63:
            raise ValueError(f"Mutation mask must be 0-63, got {mask}")
        return hexagram_from_bits(self.normal_order ^ mask)

    def interlock(self) -> Hexagram:
        """互卦: 取本卦 2-3-4 爻为下卦, 3-4-5 爻为上卦."""
        return Hexagram(
            self._lines[1], self._lines[2], self._lines[3],
            self._lines[2], self._lines[3], self._lines[4],
        )

    def reverse(self) -> Hexagram:
        """综卦: 上下颠倒 (flip line order)."""
        return Hexagram(
            self._lines[5], self._lines[4], self._lines[3],
            self._lines[2], self._lines[1], self._lines[0],
        )

    def complement(self) -> Hexagram:
        """错卦: 阴阳全部互换."""
        return hexagram_from_bits(self.normal_order ^ 0b111111)

    def square_row(self) -> int:
        return 7 - self.lower_trigram.pre_heaven_order + 1

    def square_col(self) -> int:
        return 7 - self.upper_trigram.pre_heaven_order + 1

    def square_position(self) -> tuple[int, int]:
        return (self.square_row(), self.square_col())

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


_ALL_HEXAGRAMS: list[Hexagram] = []
for _idx in range(64):
    # bit0→line0(初爻), bit1→line1(二爻), ..., bit5→line5(上爻)
    _lines_tuple = tuple(
        YANG if (_idx >> i) & 1 else YIN for i in range(6)
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


_NAME_TO_HEXAGRAM: dict[str, Hexagram] = {}

def hexagram_from_name(name: str) -> Hexagram:
    if not _NAME_TO_HEXAGRAM:
        for h in _ALL_HEXAGRAMS:
            _NAME_TO_HEXAGRAM[h.name] = h
    try:
        return _NAME_TO_HEXAGRAM[name]
    except KeyError as err:
        raise ValueError(f"Unknown hexagram name: {name}") from err


def all_hexagrams() -> tuple[Hexagram, ...]:
    return tuple(_ALL_HEXAGRAMS)


def hexagram_from_phase_vector(phases: tuple[int, int, int, int, int, int]) -> Hexagram:
    bits = 0
    for i, p in enumerate(phases):
        if p == 0:
            bits |= 1 << i
    return _ALL_HEXAGRAMS[bits]


def fuxi_square_hexagram(row: int, col: int) -> Hexagram:
    lower = _FUXI_SQUARE_ROW_TRIGRAMS[row]
    upper = _FUXI_SQUARE_COL_TRIGRAMS[col]
    return hexagram_from_trigrams(upper, lower)
