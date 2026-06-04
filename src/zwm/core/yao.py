from __future__ import annotations

import functools


@functools.total_ordering
class YaoLine:
    __slots__ = ("_value",)
    _value: bool

    def __init__(self, value: bool) -> None:
        object.__setattr__(self, "_value", value)

    def __setattr__(self, name, value):
        raise AttributeError("YaoLine is immutable")

    def __delattr__(self, name):
        raise AttributeError("YaoLine is immutable")

    @property
    def is_yang(self) -> bool:
        return self._value

    @property
    def is_yin(self) -> bool:
        return not self._value

    def flip(self) -> YaoLine:
        return YaoLine(not self._value)

    @property
    def phase(self) -> int:
        return 0 if self._value else 1

    @property
    def complex_phase(self) -> complex:
        return 1 + 0j if self._value else -1 + 0j

    def __int__(self) -> int:
        return 1 if self._value else 0

    def __bool__(self) -> bool:
        return self._value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, YaoLine):
            return NotImplemented
        return self._value == other._value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, YaoLine):
            return NotImplemented
        return (not self._value) and bool(other._value)

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"YaoLine({'YANG' if self._value else 'YIN'})"

    def __str__(self) -> str:
        return "⚊" if self._value else "⚋"


YANG: YaoLine = YaoLine(True)
YIN: YaoLine = YaoLine(False)
