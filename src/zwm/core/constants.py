# ======================================================================
# Fail-loud guard — gate the gradual migration from ``except: pass``
# to strict mode.  Set ``ZWM_STRICT=1`` in CI / test to surface bugs
# that would otherwise be silently swallowed.
# ======================================================================
from __future__ import annotations

import os as _os


def fail_loud(context: str = "") -> bool:
    """Return True → caller must re-raise; False → log-and-continue.

    The 2026-06 audit found 55 ``except Exception: pass`` sites.
    This helper gradually gates them behind ``ZWM_STRICT`` so we
    stop hiding dimension mismatches and type errors in the OODA loop.

    Usage::

        try:
            ...
        except Exception as exc:
            if fail_loud(f"telemetry flush: {exc}"):
                raise
            _log.debug("telemetry flush failed: %s", exc)
    """
    return _os.environ.get("ZWM_STRICT") in ("1", "true", "yes")


# ======================================================================
# Architecture dimension constants — single source of truth
# ======================================================================
# World vector: 64 (square GNN) + 13 (circular phase) + 29 (unified field)
Z_WORLD_DIM: int = 106

# Size presets — scale the world-model Width / Depth / Latent to match
# the available hardware.  ``test`` stays tiny so CI is fast; ``base``
# is the default for GPU development; ``large`` is for production
# training and justifies the LoRA / 4-bit / FSDP2 subsystems.

SIZE_PRESETS = {
    "test": {
        "hidden_dim": 192,
        "latent_dim": 64,
        "action_embed_dim": 32,
        "mcts_iterations": 60,
        "replay_capacity": 256,
        "batch_size": 16,
        "n_particles": 4,
    },
    "base": {
        "hidden_dim": 256,
        "latent_dim": 128,
        "action_embed_dim": 64,
        "mcts_iterations": 200,
        "replay_capacity": 512,
        "batch_size": 32,
        "n_particles": 16,
    },
    "large": {
        "hidden_dim": 512,
        "latent_dim": 256,
        "action_embed_dim": 128,
        "mcts_iterations": 400,
        "replay_capacity": 1024,
        "batch_size": 64,
        "n_particles": 32,
    },
}

# Default: test preset values (safe for CI; override via ZWM_SIZE_PRESET).
_default = SIZE_PRESETS[_os.environ.get("ZWM_SIZE_PRESET", "test")]
LATENT_DIM: int = _default["latent_dim"]
HIDDEN_DIM: int = _default["hidden_dim"]
ACTION_EMBED_DIM: int = _default["action_embed_dim"]
# Predictor input dim when action conditioning is enabled
PREDICTOR_INPUT_DIM: int = LATENT_DIM + ACTION_EMBED_DIM
# Trainable VSA codebook output dim
VSA_DIM: int = 256

SOLAR_TERMS: tuple[str, ...] = (
    "冬至", "小寒", "大寒",
    "立春", "雨水", "惊蛰",
    "春分", "清明", "谷雨",
    "立夏", "小满", "芒种",
    "夏至", "小暑", "大暑",
    "立秋", "处暑", "白露",
    "秋分", "寒露", "霜降",
    "立冬", "小雪", "大雪",
)

LUOSHU_POSITIONS: dict[int, tuple[int, int]] = {
    4: (0, 0), 9: (0, 1), 2: (0, 2),
    3: (1, 0), 5: (1, 1), 7: (1, 2),
    8: (2, 0), 1: (2, 1), 6: (2, 2),
}

LUOSHU_GENERATION_PAIRS: set[tuple[int, int]] = {
    (1, 6), (6, 1),
    (2, 7), (7, 2),
    (3, 8), (8, 3),
    (4, 9), (9, 4),
    (5, 5),
}

LUOSHU_CONFLICT_PAIRS: set[tuple[int, int]] = {
    (1, 9), (9, 1),
    (2, 8), (8, 2),
    (3, 7), (7, 3),
    (4, 6), (6, 4),
}

LUOSHU_DIRECTION_NAMES: dict[int, str] = {
    1: "北", 2: "西南", 3: "东", 4: "东南",
    5: "中", 6: "西北", 7: "西", 8: "东北", 9: "南",
}

PALACE_POST_HEAVEN_BAGUA: dict[int, str] = {
    1: "坎", 2: "坤", 3: "震", 4: "巽",
    5: "中", 6: "乾", 7: "兑", 8: "艮", 9: "离",
}

ELEMENT_GENERATION: dict[str, str] = {
    "木": "火", "火": "土", "土": "金", "金": "水", "水": "木",
}

ELEMENT_CONTROL: dict[str, str] = {
    "木": "土", "土": "水", "水": "火", "火": "金", "金": "木",
}

ELEMENT_REVERSE_CONTROL: dict[str, str] = {
    "土": "木", "水": "土", "火": "水", "金": "火", "木": "金",
}

TRIGRAM_ELEMENTS: dict[int, str] = {
    7: "金", 3: "金",
    5: "火",
    1: "木", 6: "木",
    2: "水",
    4: "土", 0: "土",
}

_TIAN_GAN: tuple[str, ...] = (
    "甲", "乙", "丙", "丁", "戊",
    "己", "庚", "辛", "壬", "癸",
)

# 天干 → 五行映射 (甲乙→木, 丙丁→火, 戊己→土, 庚辛→金, 壬癸→水)
TIAN_GAN_ELEMENTS: dict[str, str] = {
    "甲": "木", "乙": "木",
    "丙": "火", "丁": "火",
    "戊": "土", "己": "土",
    "庚": "金", "辛": "金",
    "壬": "水", "癸": "水",
}

_DI_ZHI: tuple[str, ...] = (
    "子", "丑", "寅", "卯", "辰", "巳",
    "午", "未", "申", "酉", "戌", "亥",
)

GANZHI_60: tuple[str, ...] = tuple(
    f"{_TIAN_GAN[i % 10]}{_DI_ZHI[i % 12]}"
    for i in range(60)
)

YUAN_HUI_YUN_SHI: dict[str, int] = {
    "元": 129600,
    "会": 10800,
    "运": 360,
    "世": 30,
    "年": 1,
    "月": 1 / 12,
    "日": 1 / 365.25,
    "时": 1 / 4383,
}

CODON_TABLE: dict[int, str] = {
     0: "UUU",  1: "UUC",  2: "UUA",  3: "UUG",
     4: "CUU",  5: "CUC",  6: "CUA",  7: "CUG",
     8: "AUU",  9: "AUC", 10: "AUA", 11: "AUG",
    12: "GUU", 13: "GUC", 14: "GUA", 15: "GUG",
    16: "UCU", 17: "UCC", 18: "UCA", 19: "UCG",
    20: "CCU", 21: "CCC", 22: "CCA", 23: "CCG",
    24: "ACU", 25: "ACC", 26: "ACA", 27: "ACG",
    28: "GCU", 29: "GCC", 30: "GCA", 31: "GCG",
    32: "UAU", 33: "UAC", 34: "UAA", 35: "UAG",
    36: "CAU", 37: "CAC", 38: "CAA", 39: "CAG",
    40: "AAU", 41: "AAC", 42: "AAA", 43: "AAG",
    44: "GAU", 45: "GAC", 46: "GAA", 47: "GAG",
    48: "UGU", 49: "UGC", 50: "UGA", 51: "UGG",
    52: "CGU", 53: "CGC", 54: "CGA", 55: "CGG",
    56: "AGU", 57: "AGC", 58: "AGA", 59: "AGG",
    60: "GGU", 61: "GGC", 62: "GGA", 63: "GGG",
}

# 八宫 (8 Palace) mapping: normal_order → palace_trigram_index
# Each palace's pure trigram determines its element.
# Palace trigram indices match Trigram.index convention.
_HEXAGRAM_TO_PALACE: tuple[int, ...] = (
    0, 0, 2, 0, 3, 2, 1, 0,   #  0- 7: 坤 坤 坎 坤 兑 坎 震 坤
    1, 1, 1, 3, 3, 2, 1, 0,   #  8-15: 震 震 震 兑 兑 坎 震 坤
    0, 2, 2, 2, 3, 2, 1, 0,   # 16-23: 坤 坎 坎 坎 兑 坎 震 坤
    3, 1, 3, 3, 3, 2, 1, 0,   # 24-31: 兑 震 兑 兑 兑 坎 震 坤
    7, 6, 5, 4, 4, 4, 6, 4,   # 32-39: 乾 巽 离 艮 艮 艮 巽 艮
    7, 6, 5, 4, 5, 5, 5, 7,   # 40-47: 乾 巽 离 艮 离 离 离 乾
    7, 6, 5, 4, 4, 6, 6, 6,   # 48-55: 乾 巽 离 艮 艮 巽 巽 巽
    7, 6, 5, 4, 7, 5, 7, 7,   # 56-63: 乾 巽 离 艮 乾 离 乾 乾
)

# 卦宫五行: Palace trigram index → element
PALACE_ELEMENT: dict[int, str] = {
    7: "金", 3: "金",   # 乾, 兑
    5: "火",             # 离
    1: "木", 6: "木",   # 震, 巽
    2: "水",             # 坎
    4: "土", 0: "土",   # 艮, 坤
}
