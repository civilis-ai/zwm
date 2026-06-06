"""FieldMutation — 卦象场级别的变异操作.

旧行动空间: 翻转当前卦的 1 爻 → 63 种行动 (6爻 ± 方向)
新行动空间: 翻转场中某个位置的某个爻 → 64×6 = 384 种原子行动

这利用了"卦象场"的全部信息容量:
  - 改变中心位置 (中宫=5) → 自我状态调整
  - 改变边缘位置 (1,9) → 环境探索
  - 改变特定行/列 → 批量调整 (同一元素家族)
  - 同时翻转多个位置 (macro-action) → 组合行动

FieldMutation 的输出可以被 MCTS 直接消费 — 每个变异是一个
MCTS 搜索分支, 384 个原子行动 + 可选的组合行动。
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# 洛书九宫 → 8×8 方图位置的映射
# ═══════════════════════════════════════════════════════════════════════

# 九宫在 8×8 网格中的中心位置 (row, col)
_PALACE_CENTER: dict[int, tuple[int, int]] = {
    1: (7, 1),  # 北 — 坎
    2: (7, 5),  # 西南 — 坤
    3: (3, 1),  # 东 — 震
    4: (3, 5),  # 东南 — 巽
    5: (3, 3),  # 中 — 中宫
    6: (1, 1),  # 西北 — 乾
    7: (1, 5),  # 西 — 兑
    8: (3, 7),  # 东北 — 艮
    9: (7, 7),  # 南 — 离
}

# 九宫在 8×8 中的影响范围 (每个宫覆盖 2×2 或 3×3 区域)
def _palace_positions(palace: int) -> list[int]:
    """返回九宫宫位在 8×8 方图中覆盖的位置列表."""
    if palace not in _PALACE_CENTER:
        return []
    cr, cc = _PALACE_CENTER[palace]
    positions = []
    # 每个宫覆盖中心周围 2×2 区域 + 中心
    for dr in range(-1, 1):
        for dc in range(-1, 1):
            r, c = cr + dr, cc + dc
            if 0 <= r < 8 and 0 <= c < 8:
                positions.append(r * 8 + c)
    return list(set(positions))


# 构建九宫→位置映射 (预计算)
PALACE_POSITIONS: dict[int, list[int]] = {
    p: _palace_positions(p) for p in range(1, 10)
}


# ═══════════════════════════════════════════════════════════════════════
# FieldMutation
# ═══════════════════════════════════════════════════════════════════════

class FieldMutation:
    """卦象场变异操作集合.

    三种粒度的行动:
      - ATOMIC:  单位置单爻翻转 (384 actions)
      - REGIONAL: 九宫区域的多爻翻转 (9 palaces × 6 yao = 54 actions)
      - MACRO:   整行/列/全场的批量翻转

    用法:
        fm = FieldMutation()
        actions = fm.action_list()  # → [(pos, yao_idx, name), ...]
        # MCTS 搜索时
        for pos, yao_idx, name in actions:
            new_field = fm.mutate(field, pos, yao_idx)
            score = evaluate(new_field)
    """

    # 384 原子行动元数据
    _ATOMIC_ACTIONS: list[tuple[int, int, str]] = []  # lazy-built

    def __init__(self, field_shape: tuple[int, int] = (64, 6)) -> None:
        self._n_positions, self._n_yao = field_shape

    @property
    def n_atomic_actions(self) -> int:
        return self._n_positions * self._n_yao  # 384

    @property
    def n_regional_actions(self) -> int:
        return 9 * self._n_yao  # 54

    def action_list(self, granularity: str = "atomic") -> list[dict]:
        """返回行动列表 (供 MCTS 搜索使用).

        Args:
            granularity: "atomic" | "regional" | "macro" | "all"

        Returns:
            list of dicts: {pos, yao_idx, name, granularity}
        """
        actions: list[dict] = []

        if granularity in ("atomic", "all"):
            for pos in range(64):
                row, col = pos // 8, pos % 8
                # 确定该位置属于哪个九宫
                palace = self._pos_to_palace(pos)
                for yao in range(6):
                    yao_names = ["初", "二", "三", "四", "五", "上"]
                    actions.append({
                        "pos": pos, "yao_idx": yao,
                        "name": f"P{pos}({row},{col})@{yao_names[yao]}爻",
                        "granularity": "atomic",
                        "palace": palace,
                        "row": row, "col": col,
                    })

        if granularity in ("regional", "all"):
            for palace in range(1, 10):
                positions = PALACE_POSITIONS.get(palace, [])
                for yao in range(6):
                    yao_names = ["初", "二", "三", "四", "五", "上"]
                    actions.append({
                        "pos": -1, "yao_idx": yao,
                        "palace_positions": positions,
                        "name": f"宫{palace}@{yao_names[yao]}爻",
                        "granularity": "regional",
                        "palace": palace,
                    })

        if granularity in ("macro", "all"):
            # 整行翻转 (8 rows × 6 yao)
            for row in range(8):
                for yao in range(6):
                    actions.append({
                        "pos": -1, "yao_idx": yao,
                        "row_positions": [row * 8 + c for c in range(8)],
                        "name": f"行{row}@爻{yao}",
                        "granularity": "macro",
                    })
            # 整列翻转
            for col in range(8):
                for yao in range(6):
                    actions.append({
                        "pos": -1, "yao_idx": yao,
                        "col_positions": [r * 8 + col for r in range(8)],
                        "name": f"列{col}@爻{yao}",
                        "granularity": "macro",
                    })

        return actions

    def mutate(
        self,
        field: np.ndarray,
        pos: int,
        yao_idx: int,
        mode: str = "flip",
    ) -> np.ndarray:
        """原子变异: 翻转场中某个位置的某个爻.

        Args:
            field: shape (64, 6) — 当前卦象场
            pos: 0-63 — 要变异的方图位置
            yao_idx: 0-5 — 要翻转的爻
            mode: "flip" (1-x), "set_1" (→1.0), "set_0" (→0.0)

        Returns:
            new_field: shape (64, 6) — 变异后的场 (新拷贝)
        """
        new_field = field.copy()
        if mode == "flip":
            new_field[pos, yao_idx] = 1.0 - new_field[pos, yao_idx]
        elif mode == "set_1":
            new_field[pos, yao_idx] = 1.0
        elif mode == "set_0":
            new_field[pos, yao_idx] = 0.0
        return new_field

    def mutate_regional(
        self,
        field: np.ndarray,
        palace: int,
        yao_idx: int,
        mode: str = "flip",
    ) -> np.ndarray:
        """区域变异: 翻转某个九宫的所有位置的某个爻.

        Args:
            palace: 1-9 — 目标宫位
            yao_idx: 0-5
        """
        positions = PALACE_POSITIONS.get(palace, [])
        new_field = field.copy()
        for pos in positions:
            if mode == "flip":
                new_field[pos, yao_idx] = 1.0 - new_field[pos, yao_idx]
            elif mode == "set_1":
                new_field[pos, yao_idx] = 1.0
            elif mode == "set_0":
                new_field[pos, yao_idx] = 0.0
        return new_field

    def mutate_row(self, field: np.ndarray, row: int, yao_idx: int) -> np.ndarray:
        """整行变异: 翻转某行的 8 个位置的某个爻."""
        new_field = field.copy()
        for col in range(8):
            pos = row * 8 + col
            new_field[pos, yao_idx] = 1.0 - new_field[pos, yao_idx]
        return new_field

    def mutate_col(self, field: np.ndarray, col: int, yao_idx: int) -> np.ndarray:
        """整列变异."""
        new_field = field.copy()
        for row in range(8):
            pos = row * 8 + col
            new_field[pos, yao_idx] = 1.0 - new_field[pos, yao_idx]
        return new_field

    def mutate_full(self, field: np.ndarray, yao_idx: int) -> np.ndarray:
        """全场变异: 翻转所有 64 个位置的某个爻 (等价于翻转向量全局)."""
        new_field = field.copy()
        new_field[:, yao_idx] = 1.0 - new_field[:, yao_idx]
        return new_field

    def classify(self, old_field: np.ndarray, new_field: np.ndarray) -> str:
        """分类变异类型: 判断是什么类型的变异."""
        diff = np.abs(new_field - old_field)
        n_changed = int(np.sum(diff > 0.1))
        if n_changed == 0:
            return "identity"
        elif n_changed == 1:
            return "atomic"
        elif n_changed <= 9:
            # 可能是一个宫位 (2-9 个位置)
            for palace in range(1, 10):
                positions = PALACE_POSITIONS.get(palace, [])
                changed_positions = set(
                    p for p in range(64) if np.any(diff[p] > 0.1)
                )
                if changed_positions.issubset(set(positions)):
                    return f"regional_p{palace}"
            return "small_multi"
        elif n_changed <= 8:
            return "row_or_col"
        elif n_changed == 64:
            return "global"
        return f"multi_{n_changed}"

    @staticmethod
    def _pos_to_palace(pos: int) -> int:
        """方图位置 → 九宫宫位."""
        row, col = pos // 8, pos % 8
        # 近似: 基于 8×8 → 3×3 的缩放
        pr = int(row * 3 / 8)  # 0,1,2
        pc = int(col * 3 / 8)  # 0,1,2
        # (pr, pc) → 九宫编号
        _map = {
            (0, 0): 6, (0, 1): 7, (0, 2): 8,
            (1, 0): 1, (1, 1): 5, (1, 2): 9,
            (2, 0): 4, (2, 1): 3, (2, 2): 2,
        }
        return _map.get((pr, pc), 5)

    def palace_mask(self, palace: int) -> np.ndarray:
        """返回九宫宫位在 64 位置上的布尔掩码.

        Returns:
            shape (64,) — True 表示该位置属于指定宫位
        """
        mask = np.zeros(64, dtype=bool)
        for pos in PALACE_POSITIONS.get(palace, []):
            mask[pos] = True
        return mask


__all__ = [
    "FieldMutation",
    "PALACE_POSITIONS",
    "_PALACE_CENTER",
]
