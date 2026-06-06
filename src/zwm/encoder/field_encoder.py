"""HexagramFieldEncoder — 传感器数据 → 8×8 卦象场.

核心洞察:
  1 个卦象 = 6-bit 原子信息单元 (类似 1 byte)
  1 张方图 = 8×8 = 64 卦象 = 384 bit 状态场 (类似 1 个内存页)
  传感器数据被分片/分区编码为 64 个不同卦象, 而非压缩为 1 个卦象

编码策略 (按传感器类型):
  - 空间传感器 (图像/LiDAR/网格): 8×8 空间分片
  - 时间传感器 (音频/时序): 8 时间窗 × 8 频带
  - 混合传感器 (多模态): 8 通道 × 8 量化桶
  - 通用传感器 (key-value dict): 8 组 × 8 特征

输出:
  field: np.ndarray shape (64, 6), dtype float32
    64 个位置, 每个位置 6 个 yao 信号 ∈ [0, 1]
    > 0.5 → YANG, ≤ 0.5 → YIN

此模块替代 RuleBasedEncoder 作为主要感知入口,
RuleBasedEncoder 保留为向后兼容的单卦编码器。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# 分片策略
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class FieldSlice:
    """方图场的一个分片 — 对应传感器数据的一个区域/方面."""
    position: int           # 0-63, 在 8×8 场中的位置
    row: int                # 0-7
    col: int                # 0-7
    data_slice: np.ndarray  # 该分片的传感器数据


def _row_col_to_pos(row: int, col: int) -> int:
    """(row, col) → 0-63 position (row-major)."""
    return row * 8 + col


def _pos_to_row_col(pos: int) -> tuple[int, int]:
    """0-63 → (row, col)."""
    return pos // 8, pos % 8


# ═══════════════════════════════════════════════════════════════════════
# Yao 信号映射器 — 将任意浮点值映射到 [0, 1]
# ═══════════════════════════════════════════════════════════════════════

def _sigmoid_yao(x: float, center: float = 0.5, steepness: float = 5.0) -> float:
    """Sigmoid 映射: 连续值 → yao 信号 ∈ (0, 1).

    比硬阈值提供更丰富的梯度信息, 适合端到端学习。
    """
    return float(1.0 / (1.0 + math.exp(-steepness * (x - center))))


def _linear_yao(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """线性映射: [lo, hi] → [0, 1], clamp 到边界."""
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _binary_yao(x: float, threshold: float = 0.5) -> float:
    """二值映射: x > threshold → 1.0 else 0.0 (硬阈值, 向后兼容)."""
    return 1.0 if x > threshold else 0.0


# ═══════════════════════════════════════════════════════════════════════
# HexagramFieldEncoder — 主编码器
# ═══════════════════════════════════════════════════════════════════════

class HexagramFieldEncoder:
    """将传感器数据编码为 64 卦 × 6 爻的连续场.

    用法:
        enc = HexagramFieldEncoder(strategy="spatial")
        field = enc.encode(sensor_data)  # shape (64, 6), dtype float32

    策略:
      - "spatial"  — 8×8 空间分片 (图像/网格传感器)
      - "temporal" — 8 时间窗 × 8 频带 (音频/时序)
      - "spectral" — 8 频率 × 8 通道 (多光谱/多传感器)
      - "adaptive" — 自动检测传感器类型选择最佳策略
      - "mixed"    — 8 传感器类型 × 8 统计量分桶
    """

    # 默认 6 爻对应的传感器分片函数
    # 每个 yao 有一个独立的特征提取器, 接收 (data_slice) → float ∈ [0, 1]
    YAO_EXTRACTORS: tuple[Callable, ...] = (
        # 初爻 — 均值 (整体水平)
        lambda d: _linear_yao(float(np.mean(d)), lo=-1.0, hi=1.0),
        # 二爻 — 标准差 (波动/变化)
        lambda d: _sigmoid_yao(float(np.std(d)), center=0.3, steepness=8.0),
        # 三爻 — 最大值 (峰值/极端)
        lambda d: _linear_yao(float(np.max(d) if len(d) > 0 else 0.5), lo=0.0, hi=1.0),
        # 四爻 — 最小值 (谷值/基础)
        lambda d: _linear_yao(float(np.min(d) if len(d) > 0 else 0.5), lo=0.0, hi=1.0),
        # 五爻 — 梯度 (趋势/方向)
        lambda d: _sigmoid_yao(
            float(np.mean(np.diff(d)) if len(d) > 1 else 0.0),
            center=0.0, steepness=10.0,
        ),
        # 上爻 — 熵 (复杂度/信息量)
        lambda d: _sigmoid_yao(_entropy_approx(d), center=0.5, steepness=3.0),
    )

    def __init__(self, strategy: str = "adaptive", soft_yao: bool = True) -> None:
        self._strategy = strategy
        self._soft_yao = soft_yao

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def output_dim(self) -> int:
        """输出总维度: 64 卦 × 6 爻."""
        return 64 * 6  # = 384

    # ─── 主入口 ──────────────────────────────

    def encode(self, sensor_data: dict | np.ndarray) -> np.ndarray:
        """编码传感器数据 → 64 卦场.

        Args:
            sensor_data: dict (key-value 传感器) 或 np.ndarray (原始数据)

        Returns:
            np.ndarray shape (64, 6), dtype float32 — 每个位置的 6 爻信号
        """
        strategy = self._strategy
        if strategy == "adaptive":
            strategy = self._detect_strategy(sensor_data)

        slices = self._slice_data(sensor_data, strategy)
        field = self._slices_to_field(slices)
        return field.astype(np.float32)

    def encode_flat(self, sensor_data: dict | np.ndarray) -> np.ndarray:
        """编码 → 384 维扁平向量 (向后兼容 106-dim 管道)."""
        field = self.encode(sensor_data)
        return field.flatten()  # shape (384,)

    # ─── 策略检测 ───────────────────────────

    @staticmethod
    def _detect_strategy(sensor_data: dict | np.ndarray) -> str:
        """根据传感器数据格式自动选择最佳分片策略."""
        if isinstance(sensor_data, np.ndarray):
            if sensor_data.ndim >= 3:
                return "spatial"     # 图像/视频 (H, W, C)
            elif sensor_data.ndim == 2:
                if sensor_data.shape[0] <= 64:
                    return "spectral"  # (channels, time)
                return "spatial"       # (H, W)
            else:
                return "temporal"      # 1D 时序
        elif isinstance(sensor_data, dict):
            n_keys = len(sensor_data)
            if n_keys <= 8:
                return "mixed"         # 少量传感器键
            return "spectral"           # 大量传感器键
        return "mixed"

    # ─── 数据分片 ───────────────────────────

    def _slice_data(self, sensor_data: dict | np.ndarray, strategy: str) -> list[FieldSlice]:
        """根据策略将传感器数据切分为 64 个分片."""
        if strategy == "spatial":
            return self._slice_spatial(sensor_data)
        elif strategy == "temporal":
            return self._slice_temporal(sensor_data)
        elif strategy == "spectral":
            return self._slice_spectral(sensor_data)
        elif strategy == "mixed":
            return self._slice_mixed(sensor_data)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _slice_spatial(self, data: np.ndarray | dict) -> list[FieldSlice]:
        """空间分片: 8×8 网格覆盖传感器空间.

        图像: (H, W, C) → 8×8 grid cells → 每 cell 内像素值
        LiDAR: (N, 3) → 投影到 8×8 俯视图 → 每 cell 内点云
        """
        slices: list[FieldSlice] = []
        if isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
            if arr.ndim >= 3:
                # 图像数据: 8×8 空间分片
                h, w = arr.shape[0], arr.shape[1]
                for row in range(8):
                    for col in range(8):
                        r0 = row * h // 8
                        r1 = (row + 1) * h // 8
                        c0 = col * w // 8
                        c1 = (col + 1) * w // 8
                        cell = arr[r0:r1, c0:c1].flatten()
                        slices.append(FieldSlice(
                            position=_row_col_to_pos(row, col),
                            row=row, col=col, data_slice=cell,
                        ))
            elif arr.ndim == 2:
                # 2D 矩阵: 8×8 子矩阵分片
                h, w = arr.shape[0], arr.shape[1]
                for row in range(8):
                    for col in range(8):
                        r0 = max(0, row * h // 8)
                        r1 = max(r0 + 1, (row + 1) * h // 8)
                        c0 = max(0, col * w // 8)
                        c1 = max(c0 + 1, (col + 1) * w // 8)
                        cell = arr[r0:r1, c0:c1].flatten()
                        slices.append(FieldSlice(
                            position=_row_col_to_pos(row, col),
                            row=row, col=col, data_slice=cell,
                        ))
            else:
                # 1D 数组: 分为 64 段
                n = len(arr)
                for pos in range(64):
                    i0 = pos * n // 64
                    i1 = (pos + 1) * n // 64
                    row, col = _pos_to_row_col(pos)
                    cell = arr[max(i0, 0):max(i1, 1)]
                    slices.append(FieldSlice(
                        position=pos, row=row, col=col,
                        data_slice=cell,
                    ))
        else:
            # dict: 每个 key 的值当作不同空间位置
            vals = list(data.values())
            for pos in range(64):
                row, col = _pos_to_row_col(pos)
                idx = pos % len(vals) if vals else 0
                v = vals[idx]
                if isinstance(v, (int, float)):
                    cell = np.array([float(v)], dtype=np.float32)
                elif isinstance(v, np.ndarray):
                    cell = v.flatten().astype(np.float32)
                else:
                    cell = np.array([0.5], dtype=np.float32)
                slices.append(FieldSlice(
                    position=pos, row=row, col=col, data_slice=cell,
                ))
        return slices

    def _slice_temporal(self, data: np.ndarray | dict) -> list[FieldSlice]:
        """时间分片: 8 时间窗 × 8 频带 (或 8 统计量).

        音频/时序数据: (T,) 或 (T, C) → 64 个时频块.
        """
        slices: list[FieldSlice] = []
        if isinstance(data, np.ndarray):
            arr = data.astype(np.float32).flatten()
        else:
            vals = [float(v) if isinstance(v, (int, float)) else 0.5
                    for v in data.values()]
            arr = np.array(vals, dtype=np.float32)

        n = len(arr)
        for pos in range(64):
            row, col = _pos_to_row_col(pos)
            # row → 时间窗 (8 个窗口)
            # col → 频带/统计量 (8 种)
            time_win = row
            freq_band = col

            # 时间窗口: 将数据分为 8 段
            t0 = time_win * n // 8
            t1 = (time_win + 1) * n // 8
            time_slice = arr[max(t0, 0):max(t1, 1)]

            if len(time_slice) == 0:
                slices.append(FieldSlice(
                    position=pos, row=row, col=col,
                    data_slice=np.array([0.5], dtype=np.float32),
                ))
                continue

            # 频带: 对时间窗口做 8 个不同频率的 bandpass (简化: 使用 FFT bin)
            if len(time_slice) > 1:
                fft = np.abs(np.fft.rfft(time_slice))
                n_bins = len(fft)
                f0 = freq_band * n_bins // 8
                f1 = (freq_band + 1) * n_bins // 8
                band = fft[max(f0, 0):max(f1, 1)]
                cell = band if len(band) > 0 else time_slice
            else:
                cell = time_slice

            slices.append(FieldSlice(
                position=pos, row=row, col=col, data_slice=cell,
            ))
        return slices

    def _slice_spectral(self, data: np.ndarray | dict) -> list[FieldSlice]:
        """频谱分片: 8 频率组 × 8 传感器通道.

        多光谱/多传感器: 每对 (freq_group, channel) 产生一个卦象.
        """
        slices: list[FieldSlice] = []
        if isinstance(data, np.ndarray):
            arr = data.astype(np.float32)
        else:
            vals = [float(v) if isinstance(v, (int, float)) else 0.5
                    for v in data.values()]
            arr = np.array(vals, dtype=np.float32)

        arr_flat = arr.flatten()
        n = len(arr_flat)

        for pos in range(64):
            row, col = _pos_to_row_col(pos)
            # row → 频率组, col → 通道
            # 将数据分为 8 个频率组 (频率域)
            # 8 个通道组 (传感器域)
            chunk_size = max(1, n // 64)
            i0 = pos * chunk_size
            i1 = min(n, (pos + 1) * chunk_size)
            cell = arr_flat[i0:i1] if i1 > i0 else arr_flat[:1]

            slices.append(FieldSlice(
                position=pos, row=row, col=col, data_slice=cell,
            ))
        return slices

    def _slice_mixed(self, data: dict | np.ndarray) -> list[FieldSlice]:
        """混合分片: 8 传感器类型 × 8 统计分桶.

        对于 key-value 传感器, 每个传感器 key 产生一个统计分布,
        64 个位置 = 8 sensor_keys × 8 statistic_buckets.
        """
        slices: list[FieldSlice] = []
        keys: list[str] = []
        values: list[np.ndarray] = []

        if isinstance(data, dict):
            for k, v in data.items():
                keys.append(k)
                if isinstance(v, (int, float)):
                    values.append(np.array([float(v)], dtype=np.float32))
                elif isinstance(v, np.ndarray):
                    values.append(v.flatten().astype(np.float32))
                elif isinstance(v, (list, tuple)):
                    values.append(np.array(v, dtype=np.float32))
                else:
                    values.append(np.array([0.5], dtype=np.float32))
        else:
            # Fallback
            arr = data if isinstance(data, np.ndarray) else np.array([float(data)], dtype=np.float32)
            return self._slice_spectral(arr)

        n_keys = max(len(keys), 1)
        n_values = max(len(values), 1)

        for pos in range(64):
            row, col = _pos_to_row_col(pos)
            # row (0-7) → 传感器 key index
            # col (0-7) → 统计 bucket
            key_idx = row % n_keys
            bucket = col  # 0-7: 8 种统计量

            val = values[key_idx % n_values]
            cell = _apply_stat_bucket(val, bucket)
            slices.append(FieldSlice(
                position=pos, row=row, col=col, data_slice=cell,
            ))
        return slices

    # ─── 分片 → 卦象场 ──────────────────────

    def _slices_to_field(self, slices: list[FieldSlice]) -> np.ndarray:
        """将 64 个分片编码为 (64, 6) yao 信号场."""
        field = np.zeros((64, 6), dtype=np.float32)
        for sl in slices:
            yao_signals = self._slice_to_yao(sl.data_slice)
            field[sl.position] = yao_signals
        return field

    def _slice_to_yao(self, data_slice: np.ndarray) -> np.ndarray:
        """将一个数据分片编码为 6 爻信号.

        每个爻由一个独立的特征提取器计算:
          初爻 = 均值 (整体水平)
          二爻 = 标准差 (波动/变化)
          三爻 = 最大值 (峰值)
          四爻 = 最小值 (谷值)
          五爻 = 梯度 (趋势)
          上爻 = 熵 (复杂度)
        """
        signals = np.zeros(6, dtype=np.float32)
        for yao_idx in range(6):
            try:
                signals[yao_idx] = self.YAO_EXTRACTORS[yao_idx](data_slice)
            except Exception:
                signals[yao_idx] = 0.5  # 不确定 → YIN/YANG 中间
        return signals


# ═══════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════

def _entropy_approx(data: np.ndarray) -> float:
    """近似熵 — 数据复杂度的快速度量."""
    if len(data) < 2:
        return 0.0
    # 归一化到 [0, 1]
    dmin, dmax = float(np.min(data)), float(np.max(data))
    if dmax - dmin < 1e-8:
        return 0.0
    norm = (data - dmin) / (dmax - dmin + 1e-8)
    # 计算分布 (10 bins)
    hist, _ = np.histogram(norm, bins=10, range=(0, 1))
    hist = hist.astype(np.float32) / max(len(data), 1)
    # Shannon entropy (normalized)
    entropy = 0.0
    for p in hist:
        if p > 0:
            entropy -= p * math.log(p)
    return min(entropy / math.log(10), 1.0)  # normalize to [0, 1]


def _apply_stat_bucket(data: np.ndarray, bucket: int) -> np.ndarray:
    """对数据应用 8 种统计桶之一.

    bucket 0 → 原始值
    bucket 1 → 平方
    bucket 2 → 差分
    bucket 3 → 累加
    bucket 4 → 绝对值
    bucket 5 → 符号
    bucket 6 → 排序
    bucket 7 → 归一化
    """
    d = data.astype(np.float32).flatten()
    if bucket == 0:
        return d
    elif bucket == 1:
        return d ** 2
    elif bucket == 2:
        return np.diff(d) if len(d) > 1 else d
    elif bucket == 3:
        return np.cumsum(d)
    elif bucket == 4:
        return np.abs(d)
    elif bucket == 5:
        return np.sign(d).astype(np.float32)
    elif bucket == 6:
        return np.sort(d)
    elif bucket == 7:
        dmin, dmax = float(np.min(d)), float(np.max(d))
        if dmax - dmin > 1e-8:
            return (d - dmin) / (dmax - dmin)
        return d
    return d


__all__ = [
    "HexagramFieldEncoder",
    "FieldSlice",
    "_sigmoid_yao",
    "_linear_yao",
    "_binary_yao",
    "_entropy_approx",
]
