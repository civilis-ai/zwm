"""FieldSquareGNN — 8×8 卦象场图神经网络.

与旧的 SquareGNN 的核心区别:

  旧: 64 个位置全用 fuxi_square_hexagram(row,col) 填入固定的先天方图,
      center 位置放当前卦象做 message passing → 只有 1 个卦是活的

  新: 64 个位置各自有从传感器数据独立编码的卦象,
      message passing 在所有 64 个活卦之间进行 → 64 个卦都是活的

架构:
  Input:  field ∈ R^(64×6) — 64 卦 × 6 爻信号
  Embed:  Linear(6 → hidden_dim) per position  (共享权重)
  GNN:    2-3 层 8-邻居 message passing
  Output:  (64, hidden_dim) node embeddings → pool → (64,) 方图嵌入

用法:
  gnn = FieldSquareGNN(hidden_dim=64, num_layers=3)
  z_sq = gnn.embed_field(field)  # (64, 6) → (64,)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 8×8 网格的邻居关系 (8-邻域: 上下左右 + 四角)
_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _neighbor_positions(pos: int) -> list[int]:
    """返回 0-63 位置的所有 8 邻域位置."""
    row, col = pos // 8, pos % 8
    neighbors = []
    for dr, dc in _NEIGHBOR_OFFSETS:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            neighbors.append(nr * 8 + nc)
    return neighbors


# 预计算邻居索引 (static, shared)
_NEIGHBOR_INDICES: tuple[list[int], ...] = tuple(
    _neighbor_positions(p) for p in range(64)
)


class FieldSquareGNN(nn.Module):
    """8×8 卦象场图神经网络.

    64 个节点 (每个方图位置), 8 邻域连接.
    每个节点嵌入 6 爻信号 → hidden_dim → 消息传递 → 64 维输出.
    """

    def __init__(
        self,
        yao_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self._yao_dim = yao_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._residual = residual

        # 爻嵌入: 6 → hidden_dim (64 个位置共享权重)
        self.yao_embed = nn.Sequential(
            nn.Linear(yao_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 位置编码: 64 个位置各有一个可学习的嵌入
        self.pos_embed = nn.Parameter(torch.randn(64, hidden_dim) * 0.02)

        # 消息传递层
        self.gnn_layers = nn.ModuleList([
            _FieldGNNLayer(hidden_dim, residual=residual)
            for _ in range(num_layers)
        ])

        # 输出投影: hidden_dim → 64
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def output_dim(self) -> int:
        return 64

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """前向传播: (batch, 64, 6) 或 (64, 6) → (batch, 64, hidden_dim).

        Args:
            field: shape (B, 64, 6) or (64, 6) — 64 卦的 6 爻信号

        Returns:
            node_embeddings: shape (B, 64, hidden_dim)
        """
        if field.dim() == 2:
            field = field.unsqueeze(0)  # (64, 6) → (1, 64, 6)
        B, N, D = field.shape
        assert N == 64, f"Field must have 64 positions, got {N}"
        assert D == self._yao_dim, f"Field must have {self._yao_dim} yao dims, got {D}"

        # 爻嵌入 + 位置编码
        x = self.yao_embed(field)  # (B, 64, hidden_dim)
        x = x + self.pos_embed.unsqueeze(0)  # 加位置编码

        # 消息传递
        for layer in self.gnn_layers:
            x = layer(x)  # (B, 64, hidden_dim)

        return x

    def embed_field(self, field: np.ndarray) -> np.ndarray:
        """编码 64 卦场 → 64 维方图向量.

        Args:
            field: shape (64, 6), dtype float32

        Returns:
            z_sq: shape (64,), dtype float32
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(field.astype(np.float32)).to(device)
        with torch.no_grad():
            node_emb = self.forward(x)          # (1, 64, hidden_dim)
            # Attention pooling: learn to focus on active positions
            pooled = node_emb.mean(dim=1)       # (1, hidden_dim)
            z_sq = self.output_proj(pooled)     # (1, 64)
        return z_sq.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_field_train(self, field: np.ndarray | torch.Tensor) -> torch.Tensor:
        """训练路径: 返回带梯度的 Tensor."""
        if isinstance(field, np.ndarray):
            device = next(self.parameters()).device
            field = torch.from_numpy(field.astype(np.float32)).to(device)
        node_emb = self.forward(field)
        pooled = node_emb.mean(dim=1)
        return self.output_proj(pooled)  # (B, 64)

    def node_embeddings(self, field: np.ndarray) -> np.ndarray:
        """获取 64 个节点的低维嵌入 (用于可视化/分析).

        Returns:
            shape (64, hidden_dim)
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(field.astype(np.float32)).to(device)
        with torch.no_grad():
            node_emb = self.forward(x)  # (1, 64, hidden_dim)
        return node_emb.squeeze(0).cpu().numpy()

    def attention_weights(self, field: np.ndarray) -> np.ndarray:
        """获取消息传递后的注意力分布 (哪几个位置最活跃).

        Returns:
            shape (64,) — 每个位置的 L2 norm (活跃度)
        """
        node_emb = self.node_embeddings(field)  # (64, hidden_dim)
        return np.linalg.norm(node_emb, axis=1)  # (64,)


class _FieldGNNLayer(nn.Module):
    """单层 8-邻域消息传递."""

    def __init__(self, hidden_dim: int, residual: bool = True) -> None:
        super().__init__()
        self._residual = residual
        # 消息函数: concat(node_i, node_j) → message
        self.msg_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # 聚合后更新: concat(node_i, aggregated) → updated
        self.update_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # 邻居权重 (可学习): 8 个邻居方向各有不同重要性
        self.neighbor_weight = nn.Parameter(torch.ones(8) / 8.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 64, hidden_dim) → updated: (B, 64, hidden_dim)."""
        B, N, D = x.shape
        device = x.device

        # 为每个位置的邻居构建消息
        # _NEIGHBOR_INDICES[pos] = [n1, n2, ..., nk] (k ≤ 8)
        updated = torch.zeros_like(x)

        for pos in range(N):
            nbrs = _NEIGHBOR_INDICES[pos]
            if not nbrs:
                updated[:, pos, :] = x[:, pos, :]
                continue

            # Gather neighbor embeddings
            nbr_emb = x[:, nbrs, :]  # (B, k, D)
            k = len(nbrs)

            # Self embedding expanded to match neighbors
            self_emb = x[:, pos:pos + 1, :].expand(-1, k, -1)  # (B, k, D)

            # Compute messages: msg_fn([self, neighbor])
            cat = torch.cat([self_emb, nbr_emb], dim=-1)  # (B, k, 2D)
            msgs = self.msg_fn(cat)  # (B, k, D)

            # Weighted aggregate (learnable direction weights)
            w = F.softmax(self.neighbor_weight[:k], dim=0)  # (k,)
            agg = (msgs * w.view(1, k, 1)).sum(dim=1)  # (B, D)

            # Update
            cat_update = torch.cat([x[:, pos, :], agg], dim=-1)  # (B, 2D)
            updated[:, pos, :] = self.update_fn(cat_update)

        if self._residual:
            updated = updated + x

        return updated


# ═══════════════════════════════════════════════════════════════════════
# FieldSquareCircularJoint — 方图场 + 圆图时间
# ═══════════════════════════════════════════════════════════════════════

class FieldSquareCircularJoint:
    """连接方图场 (64-dim) + 圆图相位 (13-dim) → 77-dim 世界向量.

    与旧 SquareCircularJoint 的接口兼容, 但方图部分来自
    FieldSquareGNN (处理 64 卦场) 而非单卦 GNN.
    """

    def __init__(self, field_gnn: FieldSquareGNN) -> None:
        self._field_gnn = field_gnn
        self._progression_angle: float = 0.0

    @property
    def field_gnn(self) -> FieldSquareGNN:
        return self._field_gnn

    @property
    def progression_angle(self) -> float:
        return self._progression_angle

    def encode(self, field, time_phase: float) -> np.ndarray:
        """编码 64 卦场 + 时间 → 77 维.

        Args:
            field: shape (64, 6) — 64 卦的 yao 信号, 或 Hexagram (回退为固定场)
            time_phase: float — 圆图时间相位

        Returns:
            z_world_sq_cp: shape (77,) — [z_sq(64), cp(13)]
        """
        from zwm.jepa.square_encoder import circular_phase_vector
        from zwm.core.hexagram import Hexagram

        # 回退: 单卦 → 重复填充 64 位置 (向后兼容)
        if isinstance(field, Hexagram):
            hex_bits = field.normal_order
            field_arr = np.zeros((64, 6), dtype=np.float32)
            for pos in range(64):
                for yao in range(6):
                    field_arr[pos, yao] = float((hex_bits >> yao) & 1)
            field = field_arr

        z_sq = self._field_gnn.embed_field(field)     # (64,)
        cp = circular_phase_vector(time_phase)         # (13,)
        self._progression_angle = time_phase
        return np.concatenate([z_sq, cp])              # (77,)

    def encode_train(self, field, time_phase: float) -> torch.Tensor:
        """训练路径: 返回带梯度的 77-dim Tensor.

        Args:
            field: torch.Tensor (64,6) 或 numpy (64,6) 或 Hexagram
        """
        from zwm.jepa.square_encoder import circular_phase_vector
        import torch
        from zwm.core.hexagram import Hexagram

        # 回退: Hexagram → 场
        if isinstance(field, Hexagram):
            hex_bits = field.normal_order
            field_arr = np.zeros((64, 6), dtype=np.float32)
            for pos in range(64):
                for yao in range(6):
                    field_arr[pos, yao] = float((hex_bits >> yao) & 1)
            field = field_arr
        if isinstance(field, np.ndarray):
            field = torch.from_numpy(field.astype(np.float32))

        z_sq = self._field_gnn.embed_field_train(field)  # (B, 64)
        cp = torch.from_numpy(circular_phase_vector(time_phase).astype(np.float32))
        if z_sq.device != cp.device:
            cp = cp.to(z_sq.device)
        if z_sq.dim() == 1:
            z_sq = z_sq.unsqueeze(0)
        if cp.dim() == 1:
            cp = cp.unsqueeze(0)
        self._progression_angle = time_phase
        return torch.cat([z_sq, cp], dim=-1)


__all__ = [
    "FieldSquareGNN",
    "FieldSquareCircularJoint",
    "_FieldGNNLayer",
    "_NEIGHBOR_INDICES",
]
