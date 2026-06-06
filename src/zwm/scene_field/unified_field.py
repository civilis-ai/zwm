from __future__ import annotations

import math
from dataclasses import dataclass

from zwm.core.hexagram import Hexagram
from zwm.self_field.palace_graph import LuoshuGrid
from zwm.scene_field.five_hexagrams import FiveHexagramChain
from zwm.scene_field.liuqin import determine_six_relations
from zwm.scene_field.wuxing import hexagram_element_profile

# 后天八卦洛书映射: 时间信号 → 宫位
# 年→坎1(水/冬/北), 月→艮8(山/东北), 日→震3(雷/东/春),
# 时→巽4(风/东南), 元→离9(火/夏/南), 会→坤2(地/西南),
# 运→兑7(泽/西/秋), 世→乾6(天/西北)
_CALENDAR_KEY_TO_PALACE: dict[str, int] = {
    "年": 1, "月": 8, "日": 3, "时": 4,
    "元": 9, "会": 2, "运": 7, "世": 6,
}

# 六亲角色 → 独热编码维度
_LIUQIN_ORDER = ["父母", "兄弟", "子孙", "妻财", "官鬼"]


@dataclass(frozen=True, slots=True)
class UnifiedField:
    hexagram: Hexagram
    five_chain: FiveHexagramChain
    grid: LuoshuGrid
    time_phase: float
    calendar_context: dict[str, float]
    six_relations: dict[int, str]
    element_profile: dict[str, float]
    luoshu_field: dict[int, float]

    @classmethod
    def snapshot(
        cls,
        h: Hexagram,
        grid: LuoshuGrid,
        time_phase: float = 0.0,
        calendar_context: dict | None = None,
        day_gan: str | None = None,
    ) -> UnifiedField:
        chain = FiveHexagramChain.from_current(h)
        relations = determine_six_relations(h, grid, day_gan=day_gan)
        elem_profile = hexagram_element_profile(h)

        # Derive time potentials from calendar_context if available;
        # otherwise use Luoshu self-field with harmonic defaults.
        # Maps Chinese calendar keys (年/月/日/时/元/会/运/世) to Luoshu
        # palace positions via the 后天八卦 correspondence.
        if calendar_context:
            time_pots = {p: 0.5 for p in range(1, 10)}
            for key, value in calendar_context.items():
                palace = _CALENDAR_KEY_TO_PALACE.get(key)
                if palace is not None:
                    # Normalize phase [0, 2π] → potential [0, 1]
                    time_pots[palace] = (value % (2 * math.pi)) / (2 * math.pi)
            # 中宫(5)取八宫均值
            time_pots[5] = sum(time_pots[p] for p in range(1, 10) if p != 5) / 8
        else:
            time_pots = {p: 0.5 for p in range(1, 10)}
        # P1-arch: compute the Luoshu self-field inline instead of
        # importing from self_field.harmony.  This removes the
        # scene_field → self_field dependency inversion (地 should
        # not depend on 人 — the scene should be agent-independent).
        # The harmony computation is imported lazily as before.
        from zwm.self_field.harmony import luoshu_harmony
        luoshu_field: dict[int, float] = {}
        for pos in range(1, 10):
            if pos == grid.self_position:
                luoshu_field[pos] = 1.0
            else:
                luoshu_field[pos] = luoshu_harmony(h, grid, pos) * time_pots.get(pos, 0.5)

        return cls(
            hexagram=h,
            five_chain=chain,
            grid=grid,
            time_phase=time_phase,
            calendar_context=calendar_context or {},
            six_relations=relations,
            element_profile=elem_profile,
            luoshu_field=luoshu_field,
        )

    def evolve(
        self,
        mutation_mask: int,
        new_time_phase: float | None = None,
        new_grid_position: int | None = None,
        day_gan: str | None = None,
    ) -> UnifiedField:
        chain = FiveHexagramChain.with_evolution(self.hexagram, mutation_mask)
        grid = self.grid
        if new_grid_position is not None:
            grid = grid.move_self(new_grid_position)

        new_time = new_time_phase if new_time_phase is not None else self.time_phase

        return UnifiedField(
            hexagram=chain.evolved,
            five_chain=chain,
            grid=grid,
            time_phase=new_time,
            calendar_context=self.calendar_context,
            six_relations=determine_six_relations(
                chain.evolved, grid,
                day_gan=day_gan,
            ),
            element_profile=hexagram_element_profile(chain.evolved),
            luoshu_field=self.luoshu_field,
        )

    @classmethod
    def encode(
        cls,
        h: Hexagram,
        grid: LuoshuGrid,
        calendar_context: dict | None = None,
        day_gan: str | None = None,
        time_phase: float = 0.0,
    ) -> tuple[UnifiedField, dict]:
        """Convenience factory: build a UnifiedField and return it
        together with a world-state dict suitable for downstream
        consumers (agent_phases, agent_priors).

        Returns:
            (uf, world_dict) where world_dict contains:
              - "unified_vec": the 29-dim to_tensor() output
              - "five_chain": the FiveHexagramChain
              - "six_relations": the six_relations dict
              - "element_profile": the element profile dict
              - "luoshu_field": the 9-palace field dict
        """
        uf = cls.snapshot(
            h, grid,
            time_phase=time_phase,
            calendar_context=calendar_context,
            day_gan=day_gan,
        )
        world = {
            "unified_vec": uf.to_tensor(),
            "five_chain": uf.five_chain,
            "six_relations": uf.six_relations,
            "element_profile": uf.element_profile,
            "luoshu_field": uf.luoshu_field,
        }
        return uf, world

    def to_tensor(self) -> list[float]:
        """Flatten the unified field into a 29-dim deterministic tensor.

        2026 P3-G de-redundancy + P3 calendar enrichment:  the 6-dim hexagram
        binary string was dropped because it is already represented in the
        12-dim ``hexagram_square_features`` consumed by ``LearnableSquareGNN``.
        Keeping it here would just double-count the hexagram identity
        and waste JEPA input capacity.

        P3: calendar context phases (年/月/日/时) are included so the
        world model can distinguish temporal contexts beyond the single
        time_phase scalar.  P3-C: 4 cosmic phases (元/会/运/世) — the
        129600/10800/360/30-year traditional cycles — provide
        civilization-scale context the world model can latch onto.

        UF-1: six-relations encoding now covers ALL 9 palaces (frequency
        vector) instead of only the self-position one-hot, so all palaces'
        六亲 information reaches the world model.

        UF-2: 值年卦 and 节气 signals from calendar_context are now
        consumed instead of silently discarded.  Calendar phases are
        compressed from 4 dims (年/月/日/时) to 2 dims (year+month,
        day+hour) to free 2 dims.

        Tensor breakdown:

          * 地: time_phase + grid self position              (2)
          * 地: Luoshu 9-palace field                       (9)
          * 地: 五行 (5 elements) weight                     (5)
          * 人: 六亲 frequency over 9 palaces                (5)
          * 天: calendar phases (year+month, day+hour)       (2)
          * 天: cosmic phases (元/会/运/世)                  (4)
          * 天: 值年卦 (hexagram normal_order / 64)          (1)
          * 天: 节气 (solar_term index / 24)                 (1)
                                                    ----------------
                                                    29 dims total
        """
        tensor: list[float] = []
        # 地: 时间相位 + 洛书宫位 (2维)
        tensor.append(self.time_phase)
        tensor.append(float(self.grid.self_position) / 9.0)
        # 地: 洛书场9宫 (9维)
        tensor.extend(self.luoshu_field.get(p, 0.0) for p in range(1, 10))
        # 地: 五行权重 (5维)
        tensor.extend(self.element_profile.get(e, 0.0) for e in ["金", "木", "水", "火", "土"])
        # 人: 六亲关系 (5维频率) — 所有9宫的六亲角色分布
        for role in _LIUQIN_ORDER:
            count = sum(1 for pos in range(9) if self.six_relations.get(pos) == role)
            tensor.append(count / 9.0)
        # 天: 日历上下文相位 (2维) — 年+月 / 日+时 压缩
        year_val = self.calendar_context.get("年", 0.0)
        month_val = self.calendar_context.get("月", 0.0)
        day_val = self.calendar_context.get("日", 0.0)
        hour_val = self.calendar_context.get("时", 0.0)
        tensor.append(((year_val + month_val) % (2 * math.pi)) / (2 * math.pi))
        tensor.append(((day_val + hour_val) % (2 * math.pi)) / (2 * math.pi))
        # 天: 宇宙相 (4维) — 元/会/运/世 129600/10800/360/30 年大周期
        for key in ("元", "会", "运", "世"):
            val = self.calendar_context.get(key, 0.0)
            tensor.append((val % (2 * math.pi)) / (2 * math.pi))
        # 天: 值年卦 (1维) — 64卦圆图位置
        zhi_nian_gua_val = self.calendar_context.get("值年卦", 0.0)
        tensor.append((zhi_nian_gua_val % (2 * math.pi)) / (2 * math.pi))
        # 天: 节气 (1维) — 24节气相位
        jie_qi_val = self.calendar_context.get("节气", 0.0)
        tensor.append((jie_qi_val % (2 * math.pi)) / (2 * math.pi))
        return tensor

    def bagua_directional_field(self) -> dict[str, float]:
        """后天八卦方向场 — 基于伏羲先天/文王后天八卦的方位力。

        每个卦象的上下卦在后天八卦中有固定方位，产生
        方向性的"力"，影响洛书宫位间的能量流动。
        """
        upper = self.hexagram.upper_trigram
        lower = self.hexagram.lower_trigram
        field = {}
        # Map trigrams to their Later Heaven positions
        trigram_positions = {
            "乾": 6, "兑": 7, "离": 9, "震": 3,
            "巽": 4, "坎": 1, "艮": 8, "坤": 2,
        }
        upper_pos = trigram_positions.get(upper.name, 5)
        lower_pos = trigram_positions.get(lower.name, 5)
        # Directional force: upper trigram pushes toward its palace
        for p in range(1, 10):
            dist = min(abs(p - upper_pos), 9 - abs(p - upper_pos))
            field[f"upper_{p}"] = 1.0 / (1.0 + dist)
            dist_l = min(abs(p - lower_pos), 9 - abs(p - lower_pos))
            field[f"lower_{p}"] = 0.5 / (1.0 + dist_l)
        return field
