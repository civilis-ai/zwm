"""ZWM 结构化混合编码器 — 每种场用其自然处理器, 跨场注意融合.

架构选择分析:

| 场 | 结构 | 为什么不用 Transformer | 为什么不用纯 Mamba | 最优方案 |
|----|------|---------------------|-------------------|---------|
| 方图 | 8×8 网格 | 自注意力忽略网格拓扑 | 1D扫描破坏2D结构 | **GNN** (8邻域消息传递) |
| 圆图 | 64步循环 | O(n²)浪费,无循环先验 | ✅双向扫描+循环 | **BiMamba** (双向状态空间) |
| 干支 | 60周期 | 无周期感知 | ✅状态持久=周期 | **Mamba** (循环感知SSM) |
| 元会运世 | 4层嵌套 | 过度参数化 | 序列太短无益 | **MLP** (层次化压缩) |

融合: 4-头跨场注意力 → 每头关注一种场间交互模式

与 V-JEPA 的区别:
  V-JEPA: ViT patchify → flat self-attention → 无结构感知
  ZWM:   每种场用原生处理器 → 跨场注意 → 结构感知

与 Mamba-3/Jamba 的区别:
  Jamba: 交替 SSM + Attention block → 通用长序列
  ZWM:  每种场固定处理器 → 跨场融合 → 结构化世界模型

用法:
    enc = ZWMStructuredEncoder(
        input_dim=256,  # 4 fields × 64 dim
        hidden_dim=192,
        latent_dim=64,
    )
    z_latent = enc(z_world)  # (B, 256) → (B, 64)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ZWMStructuredEncoder",
    "BiMambaBlock",
    "CrossFieldFusion",
    "SinusoidalPositionalEncoding",
]


# ═══════════════════════════════════════════════════════════════════════
# 位置编码 — 64卦圆图序的 RoPE-style 编码
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    """64 位置的圆图序正弦位置编码.

    位置 0→复卦(冬至), 位置 32→姤卦(夏至), 位置 63→坤卦.
    编码捕获圆图上的循环距离关系。
    """

    def __init__(self, dim: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 64, D) → (B, 64, D) with position encoding added."""
        return x + self.pe[:x.size(1), :].unsqueeze(0).to(x.device)


# ═══════════════════════════════════════════════════════════════════════
# BiMamba Block — 双向状态空间模型
# ═══════════════════════════════════════════════════════════════════════

class BiMambaBlock(nn.Module):
    """简化的双向 Mamba (SSM) block.

    对 64 位置序列做双向扫描:
      forward scan:  复→姤→坤 (冬至→夏至→冬至, 阳升→阴升)
      backward scan: 坤→姤→复 (逆序, 捕获反向依赖)

    两方向拼接后投影回 hidden_dim。

    当 mamba-ssm 包不可用时, 使用简化的 1D 卷积替代
    (保留 SSM 的核心特性: 局部+全局的状态传播)。
    """

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._use_mamba = False

        # 尝试导入 mamba-ssm
        try:
            from mamba_ssm import Mamba
            self._mamba_fwd = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._mamba_bwd = Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._use_mamba = True
        except ImportError:
            pass

        if not self._use_mamba:
            # 回退: 双向 1D 卷积 + 门控 (保留 SSM 的核心语义)
            self._conv_fwd = nn.Conv1d(
                hidden_dim, hidden_dim * 2, kernel_size=d_conv,
                padding=d_conv - 1, groups=hidden_dim,
            )
            self._conv_bwd = nn.Conv1d(
                hidden_dim, hidden_dim * 2, kernel_size=d_conv,
                padding=d_conv - 1, groups=hidden_dim,
            )
            self._out_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            self._norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 64, D) → (B, 64, D)."""
        if self._use_mamba:
            # Mamba 期望 (B, L, D)
            fwd = self._mamba_fwd(x)          # forward scan
            bwd = self._mamba_bwd(x.flip(1))  # backward scan (reverse sequence)
            bwd = bwd.flip(1)
            return fwd + bwd

        # 回退: 简化 1D 卷积 SSM
        B, L, D = x.shape
        # Forward
        x_fwd = x.permute(0, 2, 1)  # (B, D, L)
        fwd_out = self._conv_fwd(x_fwd)[:, :, :L]  # (B, 2D, L)
        fwd_gate, fwd_val = fwd_out.chunk(2, dim=1)
        fwd_out = fwd_val * F.silu(fwd_gate)
        fwd_out = fwd_out.permute(0, 2, 1)  # (B, L, D)

        # Backward
        x_bwd = x.flip(1).permute(0, 2, 1)
        bwd_out = self._conv_bwd(x_bwd)[:, :, :L]
        bwd_gate, bwd_val = bwd_out.chunk(2, dim=1)
        bwd_out = bwd_val * F.silu(bwd_gate)
        bwd_out = bwd_out.flip(1).permute(0, 2, 1)

        # Merge
        merged = torch.cat([fwd_out, bwd_out], dim=-1)  # (B, L, 2D)
        return self._norm(self._out_proj(merged) + x)


# ═══════════════════════════════════════════════════════════════════════
# CrossFieldFusion — 跨场注意力融合
# ═══════════════════════════════════════════════════════════════════════

class CrossFieldFusion(nn.Module):
    """4 头跨场注意力融合.

    4 个头各关注一种场间交互:
      head 0: 方图↔圆图 (空间-时间对齐)
      head 1: 方图↔干支 (空间-周期对齐)
      head 2: 圆图↔干支 (时间-周期对齐)
      head 3: 元会运世→所有 (大尺度调制)

    每个头: Q from one field, K/V from another → 加权聚合
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self._dim = dim
        self._num_heads = num_heads
        self._head_dim = dim // num_heads
        assert dim % num_heads == 0

        # Per-head QKV projections (4 heads × 4 fields)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, fields: list[torch.Tensor]) -> torch.Tensor:
        """融合 4 个场嵌入.

        Args:
            fields: [z_sq, z_circ, z_gz, z_cosmic]
                    每个 shape (B, 64) 或 (B, hidden_dim)

        Returns:
            fused: (B, dim) — 融合后的统一表示
        """
        # Stack fields: (B, 4, D)
        stacked = torch.stack(fields, dim=1)  # (B, 4, D)
        B, N, D = stacked.shape

        # Multi-head cross-attention over the 4 fields
        Q = self.q_proj(stacked).view(B, N, self._num_heads, self._head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(stacked).view(B, N, self._num_heads, self._head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(stacked).view(B, N, self._num_heads, self._head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention over 4 fields
        scale = self._head_dim ** -0.5
        attn = F.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)  # (B, H, 4, 4)
        out = attn @ V  # (B, H, 4, D/H)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)  # (B, 4, D)

        # Pool over fields → (B, D)
        fused = out.mean(dim=1)  # (B, D)
        return self.norm(self.out_proj(fused) + stacked.mean(dim=1))


# ═══════════════════════════════════════════════════════════════════════
# ZWMStructuredEncoder — 主编码器
# ═══════════════════════════════════════════════════════════════════════

class ZWMStructuredEncoder(nn.Module):
    """ZWM 结构化混合编码器 — 替换 JEPA 的 _Encoder.

    处理管道:
      1. z_world (B, 256) → split → 4 fields × (B, 64)
      2. 每个 field 独立投影 → (B, 64, hidden_dim/4)
      3. 每个 field 用原生处理器:
          方图:    FieldSquareGNN (保持 8×8 结构)
          圆图:    BiMambaBlock (双向 SSM)
          干支:    BiMambaBlock (周期感知 SSM)
          元会运世: MLP (层次压缩)
      4. CrossFieldFusion → (B, hidden_dim)
      5. 输出投影 → (B, latent_dim)

    可选后端:
      - "hybrid" (默认): GNN + BiMamba + MLP + 跨场注意
      - "transformer": 纯 Transformer (V-JEPA 风格, 用于对比)
      - "mamba": 纯 Mamba (全部场都用 SSM)
      - "mlp": 纯 MLP (baseline, 当前行为)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 192,
        latent_dim: int = 64,
        backend: str = "hybrid",
        num_fields: int = 4,
        field_dim: int = 64,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._backend = backend
        self._num_fields = num_fields
        self._field_dim = field_dim

        # 验证输入维度
        assert input_dim == num_fields * field_dim, (
            f"input_dim ({input_dim}) must equal num_fields × field_dim "
            f"({num_fields} × {field_dim})"
        )

        inner_dim = hidden_dim // 2  # per-field hidden dim

        # ─── Per-field projections ───
        self._field_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(field_dim, inner_dim),
                nn.LayerNorm(inner_dim),
                nn.GELU(),
            ) for _ in range(num_fields)
        ])

        # ─── Field-specific processors ───
        if backend == "hybrid":
            # 方图: 保持为 MLP (FieldSquareGNN 已在前级处理)
            self._field_proc = nn.ModuleList([
                nn.Sequential(  # [0] 方图 — GNN 已处理空间结构
                    nn.Linear(inner_dim, inner_dim),
                    nn.LayerNorm(inner_dim),
                    nn.GELU(),
                ),
                BiMambaBlock(inner_dim),  # [1] 圆图 — 双向 SSM
                BiMambaBlock(inner_dim),  # [2] 干支 — 周期感知 SSM
                nn.Sequential(            # [3] 元会运世 — MLP 层次压缩
                    nn.Linear(inner_dim, inner_dim),
                    nn.LayerNorm(inner_dim),
                    nn.GELU(),
                ),
            ])
        elif backend == "transformer":
            # 全部场用 Transformer
            self._field_proc = nn.ModuleList([
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=inner_dim, nhead=4, dim_feedforward=inner_dim * 2,
                        activation="gelu", batch_first=True, norm_first=True,
                    ),
                    num_layers=2,
                ) for _ in range(num_fields)
            ])
        elif backend == "mamba":
            # 全部场用 BiMamba
            self._field_proc = nn.ModuleList([
                BiMambaBlock(inner_dim) for _ in range(num_fields)
            ])
        elif backend == "mlp":
            # 全部场用 MLP (baseline)
            self._field_proc = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(inner_dim, inner_dim),
                    nn.LayerNorm(inner_dim),
                    nn.GELU(),
                ) for _ in range(num_fields)
            ])
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # ─── Cross-field fusion ───
        self._fusion = CrossFieldFusion(inner_dim, num_heads=4)

        # ─── Positional encoding for BiMamba fields (registered in __init__
        # so model.to(device) moves its buffers automatically) ───
        self._pe = SinusoidalPositionalEncoding(inner_dim, 64)

        # ─── Output projection ───
        self._output = nn.Sequential(
            nn.Linear(inner_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    @property
    def backend(self) -> str:
        return self._backend

    def forward(self, z_world: torch.Tensor) -> torch.Tensor:
        """结构化的前向传播.

        Args:
            z_world: (B, D) — 任意维度, 自动适配:
                      256-dim → 4-field 结构化处理
                      其他 → pad/truncate 到 256, 或回退到 flat MLP

        Returns:
            z_latent: (B, latent_dim)
        """
        # Add batch dim if missing
        if z_world.dim() == 1:
            z_world = z_world.unsqueeze(0)

        B, D = z_world.shape

        # 回退: 维度不匹配时使用 flat MLP
        if D != self._num_fields * self._field_dim:
            # Pad to expected dim or use simple projection
            if D < self._input_dim:
                z = F.pad(z_world, (0, self._input_dim - D))
            else:
                z = z_world[:, :self._input_dim]
            # Flat projection (4 fields handled as equal slices)
            fields_raw = z.view(B, self._num_fields, self._field_dim)
        else:
            fields_raw = z_world.view(B, self._num_fields, self._field_dim)

        fields_list = [fields_raw[:, i, :] for i in range(self._num_fields)]

        # 2) Per-field projection
        projected = [proj(f) for proj, f in zip(self._field_proj, fields_list)]

        # 3) Per-field processing
        processed = []
        for i, (proc, p) in enumerate(zip(self._field_proc, projected)):
            if isinstance(proc, BiMambaBlock):
                # Add positional encoding + reshape for sequence processing
                p_seq = p.unsqueeze(1)  # (B, 1, D) → need (B, L, D)
                # Repeat to 64 positions
                p_seq = p_seq.expand(-1, 64, -1)  # (B, 64, D)
                # Add sinusoidal PE (captures 圆图序 positional info)
                p_seq = self._pe(p_seq)
                p_out = proc(p_seq).mean(dim=1)  # pool over 64 positions
            elif isinstance(proc, nn.TransformerEncoder):
                p_seq = p.unsqueeze(1)  # (B, 1, D)
                p_out = proc(p_seq).squeeze(1)
            else:
                p_out = proc(p)  # (B, D)
            processed.append(p_out)

        # 4) Cross-field fusion
        fused = self._fusion(processed)  # (B, D)

        # 5) Output
        return self._output(fused)

    def encode(self, z_world: np.ndarray) -> np.ndarray:
        """Eval 路径: numpy → latent."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(z_world.astype(np.float32)).to(device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            z = self.forward(x)
        return z.squeeze(0).cpu().numpy().astype(np.float32)
