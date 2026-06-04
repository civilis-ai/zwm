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

LUOSHU_NUMBERS: dict[int, int] = {
    1: 1, 2: 2, 3: 3, 4: 4,
    5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
}

LUOSHU_POSITIONS: dict[int, tuple[int, int]] = {
    4: (0, 0), 9: (0, 1), 2: (0, 2),
    3: (1, 0), 5: (1, 1), 7: (1, 2),
    8: (2, 0), 1: (2, 1), 6: (2, 2),
}

POSITION_TO_LUOSHU: dict[tuple[int, int], int] = {
    v: k for k, v in LUOSHU_POSITIONS.items()
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

FIVE_ELEMENTS: tuple[str, ...] = ("金", "木", "水", "火", "土")

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

YAO_WEIGHTS: tuple[float, ...] = (1.0, 0.9, 0.7, 0.5, 0.3, 0.2)

TIAN_GAN: tuple[str, ...] = (
    "甲", "乙", "丙", "丁", "戊",
    "己", "庚", "辛", "壬", "癸",
)

DI_ZHI: tuple[str, ...] = (
    "子", "丑", "寅", "卯", "辰", "巳",
    "午", "未", "申", "酉", "戌", "亥",
)

GANZHI_60: tuple[str, ...] = tuple(
    f"{TIAN_GAN[i % 10]}{DI_ZHI[i % 12]}"
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

MUTATION_TYPE_NAMES: dict[int, str] = {
    1: "初爻变", 2: "二爻变", 4: "三爻变",
    8: "四爻变", 16: "五爻变", 32: "上爻变",
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

# 日干 → 五行: Day Heavenly Stem to Element
# Based on the 十天干 (10 Heavenly Stems) elemental assignments
GAN_ELEMENT: dict[str, str] = {
    "甲": "木", "乙": "木",
    "丙": "火", "丁": "火",
    "戊": "土", "己": "土",
    "庚": "金", "辛": "金",
    "壬": "水", "癸": "水",
}

# 卦宫五行: Palace trigram index → element
PALACE_ELEMENT: dict[int, str] = {
    7: "金", 3: "金",   # 乾, 兑
    5: "火",             # 离
    1: "木", 6: "木",   # 震, 巽
    2: "水",             # 坎
    4: "土", 0: "土",   # 艮, 坤
}
