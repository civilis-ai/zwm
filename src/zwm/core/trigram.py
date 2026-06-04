from __future__ import annotations

import functools
from typing import ClassVar

from zwm.core.yao import YANG, YIN, YaoLine


@functools.total_ordering
class Trigram:
    __slots__ = ("_lower", "_middle", "_upper")
    _lower: YaoLine
    _middle: YaoLine
    _upper: YaoLine

    QIAN: ClassVar[Trigram]
    DUI: ClassVar[Trigram]
    LI: ClassVar[Trigram]
    ZHEN: ClassVar[Trigram]
    XUN: ClassVar[Trigram]
    KAN: ClassVar[Trigram]
    GEN: ClassVar[Trigram]
    KUN: ClassVar[Trigram]

    _PREDECESSOR_NAMES: ClassVar[dict[int, str]] = {
        7: "乾", 3: "兑", 5: "离", 1: "震",
        6: "巽", 2: "坎", 4: "艮", 0: "坤",
    }

    _ELEMENTS: ClassVar[dict[int, str]] = {
        7: "金", 3: "金",
        5: "火",
        1: "木", 6: "木",
        2: "水",
        4: "土", 0: "土",
    }

    _POST_HEAVEN_DIRECTIONS: ClassVar[dict[int, str]] = {
        7: "西北", 3: "西",
        5: "南",
        1: "东", 6: "东南",
        2: "北",
        4: "东北", 0: "西南",
    }

    _PRE_HEAVEN_ORDER: ClassVar[dict[int, int]] = {
        7: 1, 3: 2, 5: 3, 1: 4,
        6: 5, 2: 6, 4: 7, 0: 8,
    }

    def __init__(self, lower: YaoLine, middle: YaoLine, upper: YaoLine) -> None:
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_middle", middle)
        object.__setattr__(self, "_upper", upper)

    def __setattr__(self, name, value):
        raise AttributeError("Trigram is immutable")

    def __delattr__(self, name):
        raise AttributeError("Trigram is immutable")

    @property
    def lower(self) -> YaoLine:
        return self._lower

    @property
    def middle(self) -> YaoLine:
        return self._middle

    @property
    def upper(self) -> YaoLine:
        return self._upper

    @property
    def lines(self) -> tuple[YaoLine, YaoLine, YaoLine]:
        return (self._lower, self._middle, self._upper)

    @property
    def index(self) -> int:
        return (int(self._upper) << 2) | (int(self._middle) << 1) | int(self._lower)

    @property
    def pre_heaven_order(self) -> int:
        return self._PRE_HEAVEN_ORDER[self.index]

    @property
    def element(self) -> str:
        return self._ELEMENTS[self.index]

    @property
    def post_heaven_direction(self) -> str:
        return self._POST_HEAVEN_DIRECTIONS[self.index]

    @property
    def name(self) -> str:
        return self._PREDECESSOR_NAMES[self.index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trigram):
            return NotImplemented
        return self.index == other.index

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Trigram):
            return NotImplemented
        return self.index < other.index

    def __hash__(self) -> int:
        return hash(self.index)

    def __repr__(self) -> str:
        return f"Trigram({self.name})"

    def __str__(self) -> str:
        lines = [str(self._upper), str(self._middle), str(self._lower)]
        return "\n".join(lines)


Trigram.QIAN = Trigram(YANG, YANG, YANG)
Trigram.DUI = Trigram(YANG, YANG, YIN)
Trigram.LI = Trigram(YANG, YIN, YANG)
Trigram.ZHEN = Trigram(YANG, YIN, YIN)
Trigram.XUN = Trigram(YIN, YANG, YANG)
Trigram.KAN = Trigram(YIN, YANG, YIN)
Trigram.GEN = Trigram(YIN, YIN, YANG)
Trigram.KUN = Trigram(YIN, YIN, YIN)

_PREDECESSOR_TRIGRAMS: tuple[Trigram, ...] = (
    Trigram.KUN, Trigram.ZHEN, Trigram.KAN, Trigram.DUI,
    Trigram.GEN, Trigram.LI, Trigram.XUN, Trigram.QIAN,
)

_FUXI_SQUARE_ROW_TRIGRAMS: tuple[Trigram, ...] = _PREDECESSOR_TRIGRAMS
_FUXI_SQUARE_COL_TRIGRAMS: tuple[Trigram, ...] = _PREDECESSOR_TRIGRAMS


def trigram_from_index(index: int) -> Trigram:
    if not 0 <= index <= 7:
        raise ValueError(f"Trigram index must be 0-7, got {index}")
    return _PREDECESSOR_TRIGRAMS[index]


def trigram_from_lines(lower: YaoLine, middle: YaoLine, upper: YaoLine) -> Trigram:
    return trigram_from_index(
        (int(upper) << 2) | (int(middle) << 1) | int(lower)
    )
