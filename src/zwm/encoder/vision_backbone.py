"""P3b — 内置视觉编码器 (ViT/CLIP/DINOv2).

提供无需外部特征提取器的视觉编码能力:
  - ViTBackbone       — Vision Transformer (timm 或 torchvision)
  - CLIPVisionBackbone — OpenAI CLIP 视觉塔
  - DINOv2Backbone    — Meta DINOv2 自监督视觉模型

与 VisionEncoder 的关系:
  - VisionEncoder (multimodal.py) 负责 视觉特征 → 6 爻信号 的投影
  - 新的 Backbone 类负责 图像 → 视觉特征 的提取
  - 两者组合: Image → Backbone → VisionEncoder → Hexagram

用法:
    from zwm.encoder.vision_backbone import CLIPVisionBackbone

    backbone = CLIPVisionBackbone()
    features = backbone.encode_image("path/to/image.jpg")
    # features.shape == (512,) — ready for VisionEncoder
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

__all__ = [
    "CLIPVisionBackbone",
    "DINOv2Backbone",
    "ViTBackbone",
    "auto_vision_backbone",
]


# ─── 抽象基类 ─────────────────────────────────────────

class _VisionBackbone:
    """视觉编码器基类."""
    name: str = "base"
    feature_dim: int = 512

    def encode_image(self, image: np.ndarray | str | Any) -> np.ndarray:
        """编码图像 → 特征向量.

        Args:
            image: numpy array (H, W, C), 文件路径, 或 PIL Image

        Returns:
            float32 numpy array of shape (feature_dim,)
        """
        raise NotImplementedError

    def encode_batch(self, images: list[Any]) -> np.ndarray:
        """编码批量图像 → 特征矩阵.

        Returns:
            float32 numpy array of shape (batch, feature_dim)
        """
        feats = [self.encode_image(img) for img in images]
        return np.stack(feats, axis=0).astype(np.float32)


# ─── CLIP 视觉编码器 ──────────────────────────────────

class CLIPVisionBackbone(_VisionBackbone):
    """OpenAI CLIP 视觉编码器.

    支持:
      - openai/clip-vit-base-patch32 (512-dim, 默认)
      - openai/clip-vit-large-patch14 (768-dim)

    依赖: ``pip install transformers`` (可选 — 有纯 numpy 回退)
    """

    name = "clip"
    feature_dim = 512

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Try transformers (CLIP)
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            self._model = CLIPModel.from_pretrained(self._model_name)
            self._model.to(self._device)
            self._model.eval()
            self._processor = CLIPProcessor.from_pretrained(self._model_name)
            # Determine actual feature dim
            cfg = self._model.config
            if hasattr(cfg, "vision_config"):
                self.feature_dim = cfg.vision_config.hidden_size
            else:
                self.feature_dim = cfg.projection_dim if hasattr(cfg, "projection_dim") else 512
            _log.info("CLIPVisionBackbone: loaded %s (dim=%d)", self._model_name, self.feature_dim)
            return
        except ImportError:
            _log.debug("transformers not available; trying open_clip")
        except Exception as exc:
            _log.warning("transformers CLIP load failed: %s", exc)

        # Try open_clip (lighter alternative)
        try:
            import open_clip
            import torch
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            self._model = model
            self._model.to(self._device)
            self._model.eval()
            self._processor = preprocess
            self.feature_dim = model.visual.output_dim
            _log.info("CLIPVisionBackbone: loaded open_clip ViT-B-32 (dim=%d)", self.feature_dim)
            return
        except ImportError:
            _log.debug("open_clip not available; using fallback")
        except Exception as exc:
            _log.warning("open_clip load failed: %s", exc)

        # 最后回退: 使用随机投影 (保持维度兼容)
        self._init_fallback()

    def _init_fallback(self) -> None:
        """随机投影回退 — 当无可用 CLIP 库时使用."""
        _log.warning(
            "CLIPVisionBackbone: neither transformers nor open_clip available. "
            "Using random projection fallback. Install with: pip install transformers"
        )
        self._fallback_proj = np.random.RandomState(42).randn(3 * 224 * 224, self.feature_dim)
        self._fallback_proj = self._fallback_proj / np.linalg.norm(
            self._fallback_proj, axis=0, keepdims=True
        )

    def encode_image(self, image: np.ndarray | str | Any) -> np.ndarray:
        self._lazy_init()

        # 加载图像
        img_array = self._load_image(image)

        if self._model is not None and self._processor is not None:
            import torch
            try:
                # CLIP processor expects PIL Image
                from PIL import Image
                if isinstance(img_array, np.ndarray):
                    pil_img = Image.fromarray(img_array.astype(np.uint8))
                else:
                    pil_img = img_array
                inputs = self._processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    # CLIPModel returns image features from vision_model
                    if hasattr(self._model, "get_image_features"):
                        feats = self._model.get_image_features(**inputs)
                    else:
                        vision_outputs = self._model.vision_model(**inputs)
                        feats = vision_outputs.pooler_output
                    return feats.cpu().numpy().astype(np.float32).flatten()
            except Exception as exc:
                _log.warning("CLIP forward failed: %s; using fallback", exc)

        # 回退: 随机投影
        flat = img_array.astype(np.float32).flatten()[:3 * 224 * 224]
        if len(flat) < 3 * 224 * 224:
            flat = np.pad(flat, (0, 3 * 224 * 224 - len(flat)))
        return (flat @ self._fallback_proj[:len(flat)]).astype(np.float32)

    def _load_image(self, image: np.ndarray | str | Any) -> np.ndarray:
        """加载图像为 numpy array (H, W, C)."""
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, str):
            # 文件路径
            try:
                from PIL import Image
                return np.array(Image.open(image).convert("RGB"))
            except Exception as exc:
                _log.warning("Failed to load image %s: %s", image, exc)
                return np.zeros((224, 224, 3), dtype=np.uint8)
        # PIL Image or other
        try:
            return np.array(image)
        except Exception:
            return np.zeros((224, 224, 3), dtype=np.uint8)


# ─── DINOv2 编码器 ─────────────────────────────────────

class DINOv2Backbone(_VisionBackbone):
    """Meta DINOv2 自监督视觉编码器.

    2026 SOTA 自监督视觉模型 — 无需标签训练, 特征质量优于 CLIP 视觉塔。

    支持:
      - dinov2_vits14 (384-dim)
      - dinov2_vitb14 (768-dim, 默认)
      - dinov2_vitl14 (1024-dim)
      - dinov2_vitg14 (1536-dim)

    依赖: ``pip install transformers`` (或 ``torch.hub``)
    """

    name = "dinov2"
    feature_dim = 768

    MODEL_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self.feature_dim = self.MODEL_DIMS.get(model_name, 768)
        self._model = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Try torch.hub (Meta official)
        try:
            import torch
            self._model = torch.hub.load(
                "facebookresearch/dinov2", self._model_name,
            )
            self._model.to(self._device)
            self._model.eval()
            _log.info("DINOv2Backbone: loaded %s via torch.hub (dim=%d)",
                      self._model_name, self.feature_dim)
            return
        except Exception as exc:
            _log.debug("torch.hub DINOv2 load failed: %s", exc)

        # Try transformers
        try:
            from transformers import AutoModel, AutoImageProcessor
            import torch
            hf_name = f"facebook/{self._model_name}"
            self._model = AutoModel.from_pretrained(hf_name)
            self._model.to(self._device)
            self._model.eval()
            self._processor = AutoImageProcessor.from_pretrained(hf_name)
            _log.info("DINOv2Backbone: loaded %s via transformers (dim=%d)",
                      hf_name, self.feature_dim)
            return
        except Exception as exc:
            _log.warning("transformers DINOv2 load failed: %s", exc)
            self._model = None

    def encode_image(self, image: np.ndarray | str | Any) -> np.ndarray:
        self._lazy_init()
        import torch

        # 加载图像
        if isinstance(image, str):
            from PIL import Image
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            from PIL import Image
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = image

        if self._model is not None:
            try:
                if hasattr(self, "_processor"):
                    # transformers path
                    inputs = self._processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                        feats = outputs.pooler_output
                else:
                    # torch.hub path — need to preprocess
                    import torchvision.transforms as T
                    preprocess = T.Compose([
                        T.Resize(224),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    img_t = preprocess(img).unsqueeze(0).to(self._device)
                    with torch.no_grad():
                        feats = self._model(img_t)
                return feats.cpu().numpy().astype(np.float32).flatten()
            except Exception as exc:
                _log.warning("DINOv2 forward failed: %s", exc)

        return np.zeros(self.feature_dim, dtype=np.float32)


# ─── ViT 通用编码器 (timm) ──────────────────────────────

class ViTBackbone(_VisionBackbone):
    """通用 Vision Transformer (timm) 编码器.

    支持任何 timm 支持的 ViT 变体:
      - vit_base_patch16_224 (768-dim)
      - vit_large_patch16_224 (1024-dim)
      - vit_huge_patch14_224 (1280-dim)

    依赖: ``pip install timm``
    """

    name = "vit"

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._initialized = False
        self.feature_dim = 768  # default for ViT-Base

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        try:
            import timm
            import torch
            self._model = timm.create_model(self._model_name, pretrained=True, num_classes=0)
            self._model.to(self._device)
            self._model.eval()
            # Get data config for preprocessing
            self._data_cfg = timm.data.resolve_data_config({}, model=self._model)
            self._transforms = timm.data.create_transform(**self._data_cfg, is_training=False)
            self.feature_dim = self._model.num_features
            _log.info("ViTBackbone: loaded %s via timm (dim=%d)",
                      self._model_name, self.feature_dim)
        except ImportError:
            _log.warning("timm not available; ViT backbone unavailable")
        except Exception as exc:
            _log.warning("timm ViT load failed: %s", exc)

    def encode_image(self, image: np.ndarray | str | Any) -> np.ndarray:
        self._lazy_init()
        if self._model is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        import torch
        from PIL import Image
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype(np.uint8))
        else:
            img = image

        img_t = self._transforms(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feats = self._model(img_t)
        return feats.cpu().numpy().astype(np.float32).flatten()


# ─── 自动检测 ──────────────────────────────────────────

def auto_vision_backbone(device: str = "cpu") -> _VisionBackbone:
    """自动检测最佳可用视觉编码器.

    检测顺序:
      1. CLIP (transformers) — 多模态对齐最好
      2. DINOv2 (torch.hub) — 自监督特征最好
      3. ViT (timm) — 通用分类 backbone

    Returns:
        可用的 _VisionBackbone 实例
    """
    # 尝试 CLIP
    try:
        import transformers
        bb = CLIPVisionBackbone(device=device)
        bb._lazy_init()
        if bb._model is not None:
            _log.info("auto_vision_backbone: using CLIP")
            return bb
    except Exception:
        pass

    # 尝试 DINOv2
    try:
        bb = DINOv2Backbone(device=device)
        bb._lazy_init()
        if bb._model is not None:
            _log.info("auto_vision_backbone: using DINOv2")
            return bb
    except Exception:
        pass

    # 回退: CLIP (至少它有随机投影回退)
    _log.info("auto_vision_backbone: using CLIP (with fallback projection)")
    return CLIPVisionBackbone(device=device)
