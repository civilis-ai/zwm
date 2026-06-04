from __future__ import annotations

from dataclasses import dataclass

from zwm.core.hexagram import Hexagram
from zwm.self_field.palace_graph import LuoshuGrid
from zwm.scene_field.five_hexagrams import FiveHexagramChain
from zwm.scene_field.liuqin import determine_six_relations
from zwm.scene_field.wuxing import hexagram_element_profile


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
        from zwm.self_field.harmony import compute_self_field

        chain = FiveHexagramChain.from_current(h)
        relations = determine_six_relations(h, grid, day_gan=day_gan)
        elem_profile = hexagram_element_profile(h)

        # Derive time potentials from calendar_context if available;
        # otherwise use Luoshu self-field with harmonic defaults.
        if calendar_context:
            time_pots = {
                p: calendar_context.get(str(p), 0.5)
                for p in range(1, 10)
            }
        else:
            time_pots = {p: 0.5 for p in range(1, 10)}
        luoshu_field = compute_self_field(h, grid, time_pots)

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

    def to_tensor(self) -> list[float]:
        tensor: list[float] = []
        tensor.extend(float(b) for b in self.hexagram.binary_str)
        tensor.append(self.time_phase)
        tensor.append(float(self.grid.self_position) / 9.0)
        tensor.extend(self.luoshu_field.get(p, 0.0) for p in range(1, 10))
        tensor.extend(self.element_profile.get(e, 0.0) for e in ["金", "木", "水", "火", "土"])
        return tensor
