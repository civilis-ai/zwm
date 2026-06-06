"""SelfState — ZWM 的"我"：永远在中宫的第一人称锚点.

核心原则:
  "我"永远在中宫(5).  移动的是九宫拓扑的物理映射, 不是"我"的位置.
  日干决定自我五行.  六亲关系由日干固定, 不随"移动"而改变.
  八方关系以"我"为中心, 五行生克决定 — 这是不变的第一人称参考系.

用法:
    self_state = SelfState(day_gan="庚")  # 部署设定: 庚日→我属金
    print(self_state.self_element)       # "金"
    print(self_state.six_relations)      # {5: "我", 1: "妻财", 6: "妻财", ...}
    print(self_state.relation_to(3))     # "兄弟"
    target = self_state.next_to_explore() # 我最该去的下一宫
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zwm.core.constants import (
    ELEMENT_CONTROL,
    ELEMENT_GENERATION,
)

# 九宫→五行 (正确的后天文王八卦五行对应)
# 不是 trigram index, 而是 palace position → element
_PALACE_ELEMENT: dict[int, str] = {
    1: "水",  # 北·坎
    2: "土",  # 西南·坤
    3: "木",  # 东·震
    4: "木",  # 东南·巽
    5: "土",  # 中·中宫
    6: "金",  # 西北·乾
    7: "金",  # 西·兑
    8: "土",  # 东北·艮
    9: "火",  # 南·离
}

# 天干→五行
_GAN_ELEMENTS = {"甲":"木","乙":"木","丙":"火","丁":"火","戊":"土",
                 "己":"土","庚":"金","辛":"金","壬":"水","癸":"水"}

# 九宫方位名
_PALACE_NAMES = {
    1: "北·坎", 2: "西南·坤", 3: "东·震", 4: "东南·巽",
    5: "中·中宫", 6: "西北·乾", 7: "西·兑", 8: "东北·艮", 9: "南·离",
    10: "上·天", 11: "下·地",
}

# 关系和谐度
_RELATION_HARMONY = {"我": 1.0, "兄弟": 0.9, "父母": 0.7,
                     "子孙": 0.8, "妻财": 0.6, "官鬼": 0.3,
                     "天": 0.85, "地": 0.75}  # 天(上)/地(下) 的特殊关系

# 天地人三层
_LAYERS = {0: "天", 1: "人", 2: "地"}

# 关系和谐度 (以"我"为中心, 固定不变)
_RELATION_HARMONY = {"我": 1.0, "兄弟": 0.9, "父母": 0.7,
                     "子孙": 0.8, "妻财": 0.6, "官鬼": 0.3}


@dataclass
class SelfState:
    """ZWM 智能体的"我"——永远在中宫的第一人称锚点.

    "我"是中宫(5), 永不改变。八方的六亲关系由日干五行决定,
    也不随物理移动而改变。外部世界的空间映射变化时,
    agent 重新标定九宫拓扑, 但"我"始终在中心。

    Attributes:
        day_gan: 部署时设定的日干. 如 "庚"=金.
        self_element: 由日干推导的五行属性 (只读).
        six_relations: 以我为中心的八方六亲关系 (只读, 由日干固定).
        palace_visits: 八方访问历史, 驱动空间探索.
    """

    # ── 部署设定 ──
    day_gan: str = "甲"

    # ── 缓存 ──
    _six_relations: dict[int, str] = field(default_factory=dict, repr=False)

    # ── 经验 ──
    palace_visits: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.day_gan not in _GAN_ELEMENTS:
            raise ValueError(f"Unknown day_gan: '{self.day_gan}'")
        self._build_relations()

    # ── 只读属性 ──

    @property
    def self_element(self) -> str:
        """我的五行属性 (由日干决定)."""
        return _GAN_ELEMENTS[self.day_gan]

    @property
    def six_relations(self) -> dict[int, str]:
        """以我(中宫5)为中心的八方六亲关系.

        "我"永远在5. 八方关系由各地支宫的五行与"我"的生克决定,
        不随物理移动而改变。
        """
        return dict(self._six_relations)

    @property
    def total_visits(self) -> int:
        """去过多少个不同宫位."""
        return len([v for v in self.palace_visits.values() if v > 0])

    # ── 关系构建 ──

    def _build_relations(self) -> None:
        """构建以中宫5=我为中心的六亲 (八方 + 上/下).

        八方: 五行生克决定六亲.
        上(天): 乾金 — 与我(金)同元素 → 兄弟, 但天为尊 → "天"特殊关系.
        下(地): 坤土 — 土生金 → 父母 → "地"特殊关系.
        """
        self_elem = self.self_element
        relations = {5: "我"}
        for pos in [1, 2, 3, 4, 6, 7, 8, 9]:
            pelem = _PALACE_ELEMENT.get(pos, "土")
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
        # 上(天) / 下(地): 天地人三才的特殊关系
        relations[10] = "天"  # 上 — 天道
        relations[11] = "地"  # 下 — 地道
        self._six_relations = relations

    # ── 关系查询 ──

    def relation_to(self, palace: int) -> str:
        """查询我与某个宫位(含上/下)的关系."""
        return self._six_relations.get(palace, "兄弟")

    def palaces_of_relation(self, relation: str) -> list[int]:
        """查询具有某种关系的所有宫位."""
        return sorted(p for p, r in self._six_relations.items() if r == relation)

    def harmony_score(self, target_palace: int) -> float:
        """我与目标宫位的和谐度 (0-1).

        以"我"为中宫的固定关系度量, 不随物理移动改变。
        上(天)/下(地)有特殊和谐度。
        """
        if target_palace == 5:
            return 1.0
        rel = self.relation_to(target_palace)
        return _RELATION_HARMONY.get(rel, 0.5)

    @property
    def heaven_relation(self) -> str:
        """我与上(天)的关系."""
        return self._six_relations.get(10, "天")

    @property
    def earth_relation(self) -> str:
        """我与下(地)的关系."""
        return self._six_relations.get(11, "地")

    @property
    def layer_names(self) -> dict:
        """天地人三层."""
        return {0: "天·上", 1: "人·中", 2: "地·下"}

    # ── 探索 ──

    def next_to_explore(self) -> int:
        """我最该去的下一宫 (八方 + 上/下).

        优先未访问的宫位, 在同等条件下选五行最和谐的。
        """
        all_dirs = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]  # 八方 + 上下
        unvisited = [p for p in all_dirs if self.palace_visits.get(p, 0) == 0]
        if unvisited:
            return self._most_harmonic(unvisited)
        # 全访问过 → 选去得最少的
        counts = {p: self.palace_visits.get(p, 0) for p in all_dirs}
        min_count = min(counts.values())
        candidates = [p for p, c in counts.items() if c == min_count]
        return self._most_harmonic(candidates)

    def _most_harmonic(self, candidates: list[int]) -> int:
        """从候选宫位中选与"我"五行最和谐的."""
        best, best_score = candidates[0], -1.0
        for pos in candidates:
            score = self.harmony_score(pos)
            if score > best_score:
                best_score, best = score, pos
        return best

    def record_visit(self, palace: int, layer: int = 1) -> None:
        """记录一次宫位访问 (含上/下层)."""
        if palace != 5:  # 不记录中宫(自己)
            key = palace + layer * 100  # 0=天,1=人,2=地 → 区分不同层的访问
            self.palace_visits[palace] = self.palace_visits.get(palace, 0) + 1

    # ── 序列化 ──

    def to_dict(self) -> dict:
        return {
            "day_gan": self.day_gan,
            "self_element": self.self_element,
            "six_relations": self.six_relations,
            "palace_visits": dict(self.palace_visits),
        }

    def __repr__(self) -> str:
        rels = ",".join(f"宫{p}{r}" for p, r in sorted(self._six_relations.items())
                       if r != "兄弟" and p != 5)
        return (f"SelfState(日{self.day_gan}·{self.self_element}, "
                f"@中宫, 六亲:[{rels}], 访{self.total_visits}宫)")


__all__ = ["SelfState"]
