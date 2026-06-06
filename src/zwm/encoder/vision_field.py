"""ZWM 原生视觉编码器 — 图像直接映射为 64 卦场.

核心理念: 视觉不是外部嫁接的, 而是世界模型的原生感知通道。
图像被分片为 8×8 网格, 每个 grid cell 独立编码为 6 爻信号,
输出 (64,6) 的卦象场 — 与 FieldSquareGNN 无缝对接, 与 JEPA 端到端训练。

架构选项:
  - "hexvit"  — 轻量 ViT, 8×8 patches, 6-dim yao head
  - "convhex" — CNN, 逐层 stride 到 8×8, 6-channel yao head
  - "swinhex" — Swin Transformer, 分层 8×8, 窗口注意
  - "hybrid"  — CNN stem + Transformer body (最优: 效率+表示)

所有架构输出统一: (B, 64, 6) — 64 卦 × 6 爻信号

与现有 vision_backbone.py 的区别:
  vision_backbone:  图像 → 外部特征的通用投影 (通用视觉)
  vision_field:     图像 → 64卦场 (ZWM 原生视觉)

用法:
    vf = ZWMVisionField(backbone="hybrid", img_size=224)
    hex_field = vf(image)  # (B, 64, 6) — 直接喂给 FieldSquareGNN
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)

__all__ = ["ZWMVisionField", "HexViT", "ConvHex", "SwinHexBlock"]


# ═══════════════════════════════════════════════════════════════════════
# 基础组件
# ═══════════════════════════════════════════════════════════════════════

class PatchEmbed8x8(nn.Module):
    """将图像分片为 8×8 grid, 每 patch → embed_dim.

    输入: (B, C, H, W) — 任意尺寸图像
    输出: (B, 64, embed_dim) — 64个patch嵌入
    """

    def __init__(self, img_size: int = 224, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self._img_size = img_size
        self._patch_size = img_size // 8
        self._proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=self._patch_size, stride=self._patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self._proj(x)  # (B, embed_dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, embed_dim)
        return x


# ═══════════════════════════════════════════════════════════════════════
# ConvHex — CNN → 64卦场
# ═══════════════════════════════════════════════════════════════════════

class ConvHex(nn.Module):
    """纯卷积视觉卦象场编码器.

    简单快速 — 3 层 stride-2 卷积 + 1×1 yao head,
    直接从像素空间映射到卦象场空间。
    """

    def __init__(self, img_size: int = 224, in_chans: int = 3, hidden: int = 64):
        super().__init__()
        self._img_size = img_size
        # 3 层 stride-2 → 将任意尺寸映射到 8×8 特征图
        self._stem = nn.Sequential(
            nn.Conv2d(in_chans, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        # 自适应池化到 8×8
        self._pool = nn.AdaptiveAvgPool2d((8, 8))
        # 每位置 → 6 爻信号
        self._yao_head = nn.Sequential(
            nn.Conv2d(hidden, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 6, 1),
            nn.Sigmoid(),  # yao ∈ [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, 64, 6)."""
        feat = self._stem(x)
        feat = self._pool(feat)                     # (B, hidden, 8, 8)
        yao = self._yao_head(feat)                  # (B, 6, 8, 8)
        yao = yao.flatten(2).transpose(1, 2)       # (B, 64, 6)
        return yao

    def encode(self, image: np.ndarray | torch.Tensor) -> np.ndarray:
        """numpy 图像 → (64, 6) 卦象场."""
        self.eval()
        device = next(self.parameters()).device
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = image.transpose(2, 0, 1)  # HWC → CHW
            x = torch.from_numpy(image.astype(np.float32)).to(device)
            if x.dim() == 3:
                x = x.unsqueeze(0)
        else:
            x = image.to(device)
        # Normalize to [0, 1] if not already
        if x.max() > 1.0:
            x = x / 255.0
        with torch.no_grad():
            field = self.forward(x)
        return field.squeeze(0).cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# HexViT — 轻量 ViT → 64卦场
# ═══════════════════════════════════════════════════════════════════════

class HexViT(nn.Module):
    """轻量 Vision Transformer 编码 64 卦场.

    架构: PatchEmbed → +PosEmbed → Transformer×N → YaoHead
    64 个 patch = 64 个 token → 每个 token → 6 爻

    对于 8×8=64 tokens, 自注意力的 O(n²) 完全可以忽略。
    重点在于: 每个 patch 可以关注任何其他 patch,
    全局上下文比 CNN 的局部感受野更适合世界状态编码。
    """

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self._img_size = img_size
        self._embed_dim = embed_dim

        # Patch embedding
        self._patch_embed = PatchEmbed8x8(img_size, in_chans, embed_dim)

        # Position embedding (64 positions, learnable)
        self._pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)

        # Transformer blocks (Pre-LN, GELU)
        self._blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Final LayerNorm
        self._norm = nn.LayerNorm(embed_dim)

        # 爻输出头: 每个位置的 embed_dim → 6 yao
        self._yao_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 6),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, 64, 6)."""
        x = self._patch_embed(x)                           # (B, 64, embed_dim)
        x = x + self._pos_embed                            # add position
        for blk in self._blocks:
            x = blk(x)                                     # (B, 64, embed_dim)
        x = self._norm(x)
        yao = self._yao_head(x)                            # (B, 64, 6)
        return yao

    def encode(self, image: np.ndarray | torch.Tensor) -> np.ndarray:
        """numpy → (64,6) field."""
        self.eval()
        device = next(self.parameters()).device
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = image.transpose(2, 0, 1)
            x = torch.from_numpy(image.astype(np.float32)).to(device)
            if x.dim() == 3:
                x = x.unsqueeze(0)
        else:
            x = image.to(device)
        if x.max() > 1.0:
            x = x / 255.0
        with torch.no_grad():
            field = self.forward(x)
        return field.squeeze(0).cpu().numpy().astype(np.float32)

    def get_attention_maps(self, image: torch.Tensor) -> torch.Tensor:
        """获取注意力图 (64,64) — 可视化哪些 patch 互相关注."""
        self.eval()
        device = next(self.parameters()).device
        x = self._patch_embed(image.to(device))
        x = x + self._pos_embed
        attn_maps = []
        for blk in self._blocks:
            B, N, D = x.shape
            qkv = blk._attn._qkv(x).reshape(B, N, 3, blk._attn._num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * (blk._attn._head_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            attn_maps.append(attn.mean(1))  # average over heads
            x = blk(x)
        return torch.stack(attn_maps)  # (depth, B, 64, 64)


# ═══════════════════════════════════════════════════════════════════════
# SwinHex Block
# ═══════════════════════════════════════════════════════════════════════

class SwinHexBlock(nn.Module):
    """简化 Swin Transformer block — 窗口注意力 + 移位窗口.

    适合: 8×8 小网格 → 4×4 窗口 → 跨窗口交互.
    """

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 4,
                 shift: bool = False, mlp_ratio: float = 2.0):
        super().__init__()
        self._dim = dim
        self._num_heads = num_heads
        self._window_size = window_size
        self._shift = shift
        self._head_dim = dim // num_heads

        self._norm1 = nn.LayerNorm(dim)
        self._attn_qkv = nn.Linear(dim, dim * 3)
        self._attn_proj = nn.Linear(dim, dim)

        self._norm2 = nn.LayerNorm(dim)
        self._mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 64, D) — 8×8 grid flattened."""
        B, N, D = x.shape
        H = W = 8

        # Reshape to (B, H, W, D)
        x_2d = x.view(B, H, W, D)

        shortcut = x_2d

        if self._shift:
            # Cyclic shift by window_size//2
            shift = self._window_size // 2
            x_2d = torch.roll(x_2d, shifts=(-shift, -shift), dims=(1, 2))

        # Window partition: (B, H, W, D) → (B*num_windows, window_size², D)
        ws = self._window_size
        x_windows = x_2d.view(B, H // ws, ws, W // ws, ws, D)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_windows = x_windows.view(-1, ws * ws, D)

        # Self-attention within windows
        qkv = self._attn_qkv(self._norm1(x_windows))
        qkv = qkv.view(-1, ws * ws, 3, self._num_heads, self._head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v) if hasattr(F, 'scaled_dot_product_attention') \
               else (F.softmax(q @ k.transpose(-2, -1) * (self._head_dim ** -0.5), dim=-1)) @ v
        attn = attn.transpose(1, 2).contiguous().view(-1, ws * ws, D)
        attn = self._attn_proj(attn)

        # Merge windows back
        attn = attn.view(B, H // ws, W // ws, ws, ws, D)
        attn = attn.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, D)

        if self._shift:
            attn = torch.roll(attn, shifts=(shift, shift), dims=(1, 2))

        x = shortcut + attn
        x = x + self._mlp(self._norm2(x)).view(B, H, W, D)

        return x.view(B, N, D)


# ═══════════════════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════════════════

class _TransformerBlock(nn.Module):
    """Pre-LN Transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self._norm1 = nn.LayerNorm(dim)
        self._attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self._norm2 = nn.LayerNorm(dim)
        self._mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self._norm1(x), self._norm1(x), self._norm1(x))[0]
        x = x + self._mlp(self._norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
# ZWMVisionField — 统一视觉封装
# ═══════════════════════════════════════════════════════════════════════

class ZWMVisionField(nn.Module):
    """ZWM 原生视觉编码器 — 图像 → (64,6) 卦象场.

    封装多个后端, 统一输出格式:
      - "hexvit":  ViT-base (4层, 96dim, 全局注意) — 最推荐
      - "convhex": CNN-base (3层stride, 极快)    — 速度优先
      - "swinhex": Swin-base (窗口注意)          — 大图优先
      - "hybrid":  CNN stem + Transformer body  — 均衡

    输出直接对接 FieldSquareGNN — 端到端梯度流动。
    """

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        backbone: str = "hexvit",
        **kwargs,
    ):
        super().__init__()
        self._img_size = img_size
        self._backbone_name = backbone

        if backbone == "hexvit":
            self._encoder = HexViT(img_size=img_size, in_chans=in_chans, **kwargs)
        elif backbone == "convhex":
            self._encoder = ConvHex(img_size=img_size, in_chans=in_chans, **kwargs)
        elif backbone == "hybrid":
            # CNN stem + 2-layer Transformer
            self._stem = nn.Sequential(
                nn.Conv2d(in_chans, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 96, 3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.GELU(),
            )
            self._pool = nn.AdaptiveAvgPool2d((8, 8))
            self._transformer = nn.Sequential(*[
                _TransformerBlock(96, num_heads=4, mlp_ratio=2.0)
                for _ in range(kwargs.get("depth", 2))
            ])
            self._norm = nn.LayerNorm(96)
            self._yao_head = nn.Sequential(
                nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 6), nn.Sigmoid(),
            )
            self._use_hybrid = True
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self._use_hybrid = backbone == "hybrid"

    @property
    def backbone(self) -> str:
        return self._backbone_name

    @property
    def encoder(self) -> nn.Module:
        return self._encoder if not self._use_hybrid else self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """图像 → (B, 64, 6) 卦象场."""
        if self._use_hybrid:
            B = x.shape[0]
            feat = self._stem(x)
            feat = self._pool(feat)                         # (B, 96, 8, 8)
            feat = feat.flatten(2).transpose(1, 2)          # (B, 64, 96)
            feat = self._transformer(feat)                  # Transformer on patches
            feat = self._norm(feat)
            return self._yao_head(feat)                     # (B, 64, 6)
        return self._encoder(x)

    def encode(self, image: np.ndarray | torch.Tensor) -> np.ndarray:
        """numpy 图像 → (64,6) 卦象场 (eval 路径)."""
        return self._encoder.encode(image)

    def encode_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """批量图像 → (B, 64, 6)."""
        self.eval()
        device = next(self.parameters()).device
        batch = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.ndim == 3:
                    img = img.transpose(2, 0, 1)
                t = torch.from_numpy(img.astype(np.float32))
            else:
                t = img
            if t.max() > 1.0:
                t = t / 255.0
            batch.append(t)
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            return self.forward(x).cpu().numpy().astype(np.float32)
