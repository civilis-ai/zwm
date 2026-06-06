"""Multimodal encoder — vision and language input channels.

Extends the sensor-only RuleBasedEncoder with:
  - VisionEncoder: encodes image features into hexagram space
  - LanguageEncoder: encodes text descriptions into hexagram space
  - MultimodalEncoder: fuses sensor + vision + language into a unified hexagram
  - LanguageBackbone: text → embedding (sentence-transformers or TF-IDF fallback)

These encoders bridge the gap between raw multimodal observations and the
hexagram-based world model, enabling the agent to process visual scenes
and natural language instructions alongside traditional sensor readings.

P0-2: LanguageBackbone provides a language model interface for the
MultimodalEncoder, replacing the empty-shell language_features parameter
with real text-to-embedding capability.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from zwm.core.hexagram import Hexagram
from zwm.encoder.base import HexagramEncoder


class LanguageBackbone:
    """Text → embedding vector producer.

    P0-2: Provides a real language model interface for the
    MultimodalEncoder.  Tries sentence-transformers first (lightweight,
    local), falls back to TF-IDF on the built-in vocabulary.

    Usage::

        backbone = LanguageBackbone()
        embedding = backbone.encode("天气晴朗，适合出行")
        # embedding.shape == (384,) or (512,) depending on backend
    """

    _EMBEDDING_DIM: int = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        import logging
        self._log = logging.getLogger(__name__)
        self._model = None
        self._model_name = model_name
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._dim = self._EMBEDDING_DIM
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazy-load the best available language model backend."""
        if self._initialized:
            return
        self._initialized = True
        # Try sentence-transformers.
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            self._log.info("LanguageBackbone: using sentence-transformers/%s (dim=%d)",
                           self._model_name, self._dim)
            return
        except ImportError:
            self._log.debug("sentence-transformers not available; trying TF-IDF fallback")
        except Exception as exc:
            self._log.debug("sentence-transformers load failed: %s", exc)

        # Fallback: simple TF-IDF on the hexagram domain vocabulary.
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build a small domain vocabulary for TF-IDF fallback."""
        import math
        # Hexagram-related Chinese terms.
        terms = [
            "乾", "坤", "震", "巽", "坎", "离", "艮", "兑",
            "天", "地", "雷", "风", "水", "火", "山", "泽",
            "吉", "凶", "元", "亨", "利", "贞",
            "阳", "阴", "刚", "柔", "动", "静",
            "进", "退", "升", "降", "聚", "散",
            "时间", "空间", "社会", "元素", "风险", "叙事",
            "和谐", "冲突", "探索", "利用", "学习", "适应",
            "前进", "后退", "左转", "右转", "停止",
            "观察", "预测", "评估", "行动", "学习",
            "传感器", "视觉", "语言", "图像", "文本",
            "天气", "晴朗", "阴雨", "温度", "湿度",
            "洛书", "河图", "太极", "八卦", "六十四卦",
        ]
        # Build a simple vocabulary with index.
        for i, term in enumerate(terms):
            self._vocab[term] = i
        # IDF: 1.0 for common terms, log(N) for rare ones.
        N = len(terms)
        for i, term in enumerate(terms):
            self._idf[term] = math.log(N / (i + 1)) + 1.0
        self._dim = self._EMBEDDING_DIM
        self._log.info("LanguageBackbone: using TF-IDF fallback (dim=%d, vocab=%d)",
                       self._dim, len(self._vocab))

    def encode(self, text: str) -> np.ndarray:
        """Encode text into an embedding vector.

        Returns a float32 numpy array of shape (embedding_dim,).
        """
        self._lazy_init()
        if self._model is not None:
            # sentence-transformers path.
            emb = self._model.encode(text, convert_to_numpy=True)
            return emb.astype(np.float32)

        # TF-IDF fallback.
        import math
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        for ch in text:
            if ch in self._vocab:
                idx = self._vocab[ch]
                vec[idx] += self._idf.get(ch, 1.0)
        # Also match bigrams.
        for i in range(len(text) - 1):
            bigram = text[i:i + 2]
            if bigram in self._vocab:
                idx = self._vocab[bigram]
                vec[idx] += self._idf.get(bigram, 1.0) * 0.5
        # Normalize.
        norm = float(np.linalg.norm(vec)) + 1e-8
        vec = vec / norm
        # Pad or truncate to _EMBEDDING_DIM.
        if len(vec) < self._EMBEDDING_DIM:
            padded = np.zeros(self._EMBEDDING_DIM, dtype=np.float32)
            padded[:len(vec)] = vec
            return padded
        return vec[:self._EMBEDDING_DIM].astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        self._lazy_init()
        return self._dim


class VisionEncoder(nn.Module):
    """Encodes image feature vectors into 6-dimensional yao signals.

    Takes a pre-extracted image feature vector (e.g., from a CLIP vision
    encoder or a CNN backbone) and projects it to 6 continuous values,
    each thresholded into yin/yang for a hexagram.

    P3b — 内置视觉 backbone 集成:
      当 ``visual_backbone`` 参数非 None 时, VisionEncoder 可以直接
      从图像路径或 numpy array 提取特征, 无需外部特征提取器。

      支持的 backbone:
        - "clip"    → CLIPVisionBackbone (默认, 512-dim)
        - "dinov2"  → DINOv2Backbone (768-dim)
        - "vit"     → ViTBackbone (768-dim)

      用法:
        # 无 backbone: 需要外部提供 visual_features
        enc = VisionEncoder(visual_dim=512)

        # 有 backbone: 可直接处理图像
        enc = VisionEncoder(visual_backbone="clip")
        h = enc.encode_image("path/to/image.jpg")  # → Hexagram
    """

    def __init__(
        self,
        visual_dim: int = 512,
        hidden_dim: int = 64,
        trainable: bool = True,
        visual_backbone: str | None = None,
    ) -> None:
        super().__init__()
        self._visual_dim = visual_dim
        self._backbone = None
        self._backbone_name = visual_backbone

        # P3b: lazy-init visual backbone
        if visual_backbone is not None:
            try:
                from zwm.encoder.vision_backbone import (
                    auto_vision_backbone, CLIPVisionBackbone,
                    DINOv2Backbone, ViTBackbone,
                )
                if visual_backbone == "clip":
                    self._backbone = CLIPVisionBackbone()
                elif visual_backbone == "dinov2":
                    self._backbone = DINOv2Backbone()
                elif visual_backbone == "vit":
                    self._backbone = ViTBackbone()
                else:
                    self._backbone = auto_vision_backbone()
                # 用实际特征维度更新 visual_dim
                self._visual_dim = self._backbone.feature_dim
                import logging
                logging.getLogger(__name__).info(
                    "VisionEncoder: using %s backbone (dim=%d)",
                    self._backbone.name, self._visual_dim,
                )
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Vision backbone init failed: %s; requires external features", exc,
                )

        self.projection = nn.Sequential(
            nn.Linear(self._visual_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
            nn.Sigmoid(),  # output in [0, 1], threshold at 0.5 for yin/yang
        )
        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)
        self._opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    @property
    def visual_dim(self) -> int:
        return self._visual_dim

    @property
    def has_backbone(self) -> bool:
        """P3b: 是否已加载视觉 backbone."""
        return self._backbone is not None

    @property
    def backbone_name(self) -> str | None:
        """P3b: 返回 backbone 名称."""
        if self._backbone is not None:
            return self._backbone.name
        return self._backbone_name

    def encode_image(self, image: np.ndarray | str) -> np.ndarray:
        """P3b: 直接从图像编码为 6-dim yao 信号.

        Args:
            image: numpy array (H, W, C), 文件路径, 或 PIL Image

        Returns:
            yao_signals: 6-dim float32 array in [0, 1]
        """
        if self._backbone is None:
            raise RuntimeError(
                "VisionEncoder.encode_image() requires a vision backbone. "
                "Construct with visual_backbone='clip' or similar."
            )
        features = self._backbone.encode_image(image)
        return self.encode_features(features)

    def encode_features(self, visual_features: np.ndarray) -> np.ndarray:
        """Project visual features to 6-dim yao signals [0, 1]."""
        x = torch.from_numpy(np.asarray(visual_features, dtype=np.float32))
        with torch.no_grad():
            yao_signals = self.projection(x)
        return yao_signals.numpy()

    def encode(self, visual_features: np.ndarray) -> Hexagram:
        """Encode visual features directly into a hexagram."""
        from zwm.core.yao import YANG, YIN
        signals = self.encode_features(visual_features)
        lines = [YANG if s > 0.5 else YIN for s in signals]
        return Hexagram(*lines)

    def train_step(
        self,
        visual_features: np.ndarray,
        target_hex: Hexagram,
        lr: float | None = None,
    ) -> float:
        """Train the projection to map visual features toward a target hexagram.

        Target is encoded as 6 binary values (1.0 for YANG, 0.0 for YIN).
        Uses binary cross-entropy loss.
        """
        if lr is not None:
            for group in self._opt.param_groups:
                group["lr"] = lr

        x = torch.from_numpy(np.asarray(visual_features, dtype=np.float32))
        target = torch.tensor(
            [1.0 if line.is_yang else 0.0 for line in target_hex.lines],
            dtype=torch.float32,
        )

        yao_signals = self.projection(x)
        loss = nn.functional.binary_cross_entropy(yao_signals, target)

        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self._opt.step()
        return float(loss.detach())


class LanguageEncoder(nn.Module):
    """Encodes text embedding vectors into 6-dimensional yao signals.

    Takes a pre-extracted text embedding (e.g., from a sentence transformer
    or CLIP text encoder) and projects it to 6 continuous values for
    hexagram construction.
    """

    def __init__(
        self,
        text_dim: int = 512,
        hidden_dim: int = 64,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self._text_dim = text_dim
        self.projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
            nn.Sigmoid(),
        )
        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)
        self._opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    @property
    def text_dim(self) -> int:
        return self._text_dim

    def encode_features(self, text_embedding: np.ndarray) -> np.ndarray:
        """Project text embedding to 6-dim yao signals [0, 1]."""
        x = torch.from_numpy(np.asarray(text_embedding, dtype=np.float32))
        with torch.no_grad():
            yao_signals = self.projection(x)
        return yao_signals.numpy()

    def encode(self, text_embedding: np.ndarray) -> Hexagram:
        """Encode text embedding directly into a hexagram."""
        from zwm.core.yao import YANG, YIN
        signals = self.encode_features(text_embedding)
        lines = [YANG if s > 0.5 else YIN for s in signals]
        return Hexagram(*lines)

    def train_step(
        self,
        text_embedding: np.ndarray,
        target_hex: Hexagram,
        lr: float | None = None,
    ) -> float:
        """Train the projection to map text embeddings toward a target hexagram."""
        if lr is not None:
            for group in self._opt.param_groups:
                group["lr"] = lr

        x = torch.from_numpy(np.asarray(text_embedding, dtype=np.float32))
        target = torch.tensor(
            [1.0 if line.is_yang else 0.0 for line in target_hex.lines],
            dtype=torch.float32,
        )

        yao_signals = self.projection(x)
        loss = nn.functional.binary_cross_entropy(yao_signals, target)

        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self._opt.step()
        return float(loss.detach())


class MultimodalEncoder(HexagramEncoder, nn.Module):
    """Fuses sensor + vision + language inputs into a unified hexagram.

    Combines three input channels:
      1. Sensor data (6 features via RuleBasedEncoder)
      2. Visual features (projected via VisionEncoder)
      3. Language embeddings (projected via LanguageEncoder)

    Each channel produces 6 yao signals in [0, 1]. The signals are
    fused via learned channel weights, then thresholded at 0.5 for
    the final yin/yang decision per yao position.

    The fusion weights are a *trainable* softmax-parameterised tensor
    (``self.channel_weights``) so the agent can adapt the relative
    importance of each modality from preference feedback via
    ``train_fusion_weights``.

    P0-2: ``language_backbone`` parameter accepts a ``LanguageBackbone``
    instance for real text-to-embedding conversion.  When provided,
    ``encode_text()`` converts raw text strings to embeddings that
    the LanguageEncoder can consume.
    """

    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 384,
        sensor_weight: float = 0.5,
        vision_weight: float = 0.3,
        language_weight: float = 0.2,
        language_backbone: LanguageBackbone | None = None,
    ) -> None:
        HexagramEncoder.__init__(self)
        nn.Module.__init__(self)
        from zwm.encoder.base import RuleBasedEncoder
        self._sensor_encoder = RuleBasedEncoder()
        self._vision_encoder = VisionEncoder(visual_dim)
        self._language_encoder = LanguageEncoder(text_dim)
        # P0-2: Language model backbone for real text encoding.
        self._language_backbone = language_backbone
        # Trainable channel weights — softmax-parameterised so they
        # remain a valid distribution after each gradient step.
        raw = np.array(
            [sensor_weight, vision_weight, language_weight],
            dtype=np.float32,
        )
        raw = np.log(np.maximum(raw, 1e-6))
        self._raw_weights = nn.Parameter(
            torch.from_numpy(raw), requires_grad=True
        )
        self._opt = torch.optim.Adam([self._raw_weights], lr=1e-3)

    @property
    def channel_weights(self) -> np.ndarray:
        """Return the current softmax-normalised channel weights."""
        with torch.no_grad():
            w = torch.softmax(self._raw_weights, dim=0).cpu().numpy()
        return w.astype(np.float32)

    @property
    def language_backbone(self) -> LanguageBackbone | None:
        """P0-2: The language model backbone (if configured)."""
        return self._language_backbone

    @language_backbone.setter
    def language_backbone(self, lb: LanguageBackbone | None) -> None:
        """P2: Set the language model backbone."""
        self._language_backbone = lb

    def encode_text(self, text: str) -> np.ndarray | None:
        """P0-2: Convert raw text to an embedding via the language backbone.

        Returns a float32 numpy array of shape (embedding_dim,) or None
        if no backbone is configured.
        """
        if self._language_backbone is None:
            return None
        return self._language_backbone.encode(text)

    def train_fusion_weights(
        self,
        sensor_data: dict | None = None,
        visual_features: np.ndarray | None = None,
        text_embedding: np.ndarray | None = None,
        target_hex: Hexagram | None = None,
        reward: float = 0.0,
    ) -> float:
        """One gradient step on the fusion weights.

        Maximises agreement between the fused signal and the hexagram
        truth (when ``target_hex`` is provided), modulated by ``reward``.
        Used by the agent's per-tick learning loop to adapt the relative
        importance of each modality from observed outcomes.
        """
        if target_hex is None:
            return 0.0
        target = torch.tensor(
            [1.0 if line.is_yang else 0.0 for line in target_hex.lines],
            dtype=torch.float32,
        )
        with torch.no_grad():
            sensor_s = self._signal_from_sensor(sensor_data)
            vision_s = self._signal_from_vision(visual_features)
            lang_s = self._signal_from_language(text_embedding)
            s_t = torch.from_numpy(sensor_s)
            v_t = torch.from_numpy(vision_s)
            l_t = torch.from_numpy(lang_s)
        w = torch.softmax(self._raw_weights, dim=0)
        fused = w[0] * s_t + w[1] * v_t + w[2] * l_t
        # Cross-entropy loss + reward shaping: reward > 0 lowers the
        # loss, reward < 0 raises it.  The sign is the standard RLHF
        # shaping trick.
        bce = nn.functional.binary_cross_entropy(fused, target)
        loss = bce * (1.0 - max(min(float(reward), 1.0), -1.0))

        self._opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([self._raw_weights], max_norm=1.0)
        self._opt.step()
        return float(loss.detach())

    def _signal_from_sensor(self, sensor_data: dict | None) -> np.ndarray | None:
        """Encode sensor → 6-dim yao signal.

        AUDIT-S7: returns ``None`` when no sensor data is available,
        instead of a hard-coded ``np.full(6, 0.5)``.  The fusion path
        (``encode_multimodal``) interprets ``None`` as "this channel
        is missing" and re-normalises the channel weights — the
        previous code, by contrast, fed a perfect 0.5 vector that
        lived at the exact ``> 0.5`` threshold of the yao classifier,
        forcing every missing channel to be encoded as YIN.
        """
        if sensor_data is not None:
            h_sensor = self._sensor_encoder.encode(sensor_data)
            return np.array(
                [1.0 if line.is_yang else 0.0 for line in h_sensor.lines],
                dtype=np.float32,
            )
        return None

    def _signal_from_vision(self, visual_features: np.ndarray | None) -> np.ndarray | None:
        """Encode vision features → 6-dim yao signal.

        AUDIT-S7: same convention as ``_signal_from_sensor`` —
        ``None`` on missing input.
        """
        if visual_features is not None:
            return self._vision_encoder.encode_features(visual_features)
        return None

    def _signal_from_language(self, text_embedding: np.ndarray | None) -> np.ndarray | None:
        """Encode language features → 6-dim yao signal.

        AUDIT-S7: same convention as ``_signal_from_sensor`` —
        ``None`` on missing input.
        """
        if text_embedding is not None:
            return self._language_encoder.encode_features(text_embedding)
        return None

    def encode(self, sensor_data: dict) -> Hexagram:
        """Encode sensor data only (backward compatible with RuleBasedEncoder)."""
        return self._sensor_encoder.encode(sensor_data)

    def encode_multimodal(
        self,
        sensor_data: dict | None = None,
        visual_features: np.ndarray | None = None,
        text_embedding: np.ndarray | None = None,
    ) -> Hexagram:
        """Encode multimodal inputs into a fused hexagram.

        At least one input channel must be provided. Missing channels
        are *skipped* (their signal is ``None``), and the fusion
        weights are re-normalised over the channels that did produce
        data — see AUDIT-S7.
        """
        from zwm.core.yao import YANG, YIN

        sensor_signals = self._signal_from_sensor(sensor_data)
        vision_signals = self._signal_from_vision(visual_features)
        language_signals = self._signal_from_language(text_embedding)

        # AUDIT-S7: re-normalise the channel weights to drop missing
        # channels instead of feeding a 0.5 stub.  The previous code
        # multiplied by the original weight (0.33) on a 0.5 vector,
        # dragging the fused signal toward 0.5 and biasing every
        # yao classifier toward YIN (since ``> 0.5 → YANG``).
        raw_w = self.channel_weights
        signals = [sensor_signals, vision_signals, language_signals]
        active = [s is not None for s in signals]
        if not any(active):
            raise ValueError(
                "encode_multimodal: at least one of sensor_data, "
                "visual_features, text_embedding must be provided"
            )
        active_w = raw_w * np.array([1.0 if a else 0.0 for a in active], dtype=np.float32)
        active_w = active_w / (active_w.sum() + 1e-8)

        fused = (
            active_w[0] * (sensor_signals if sensor_signals is not None else np.zeros(6, dtype=np.float32))
            + active_w[1] * (vision_signals if vision_signals is not None else np.zeros(6, dtype=np.float32))
            + active_w[2] * (language_signals if language_signals is not None else np.zeros(6, dtype=np.float32))
        )

        lines = [YANG if s > 0.5 else YIN for s in fused]
        return Hexagram(*lines)

    def feature_dim(self) -> int:
        return 6

    @property
    def vision_encoder(self) -> VisionEncoder:
        return self._vision_encoder

    @property
    def language_encoder(self) -> LanguageEncoder:
        return self._language_encoder

    def contrastive_loss(
        self,
        vision_features: np.ndarray,
        language_features: np.ndarray,
        temperature: float = 0.07,
    ) -> float:
        """InfoNCE contrastive loss between vision and language channels.

        Aligns the 天(vision) and 地(language) channels so that
        corresponding observations have similar representations.
        This is the CLIP/SigLIP training objective.
        """
        v = torch.from_numpy(vision_features).float()
        l = torch.from_numpy(language_features).float()

        # Normalize
        v_norm = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        l_norm = l / (l.norm(dim=-1, keepdim=True) + 1e-8)

        # Cosine similarity matrix
        sim = torch.mm(v_norm, l_norm.t()) / temperature

        # InfoNCE: cross-entropy with diagonal as target
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = (torch.nn.functional.cross_entropy(sim, labels) +
                torch.nn.functional.cross_entropy(sim.t(), labels)) / 2.0
        return float(loss)

    def train_contrastive(
        self,
        vision_features: np.ndarray,
        language_features: np.ndarray,
    ) -> float:
        """Train vision and language projections with contrastive loss.

        P1-3 (audit): proper InfoNCE / CLIP-style contrastive loss
        with a single, fully-functional optimizer step.  Previous
        version had a confusing mix of ``requires_grad_(True)`` on
        the input tensors (which are not parameters — those calls
        were dead code) and the residual loss.  Now we:
          1. zero the optimizers (cleaning any accumulated grads)
          2. compute the symmetric InfoNCE loss
          3. backward
          4. step the dedicated per-encoder optimizer
        """
        # 1) zero gradients
        for enc in [self.vision_encoder, self.language_encoder]:
            enc._opt.zero_grad()

        # 2) forward through projections (leaf tensors — no manual
        # requires_grad_ needed because the projection layers hold
        # the learnable parameters).
        v = torch.from_numpy(np.asarray(vision_features, dtype=np.float32))
        l = torch.from_numpy(np.asarray(language_features, dtype=np.float32))
        # P2 FIX: .project → .projection (the actual attribute name)
        v_proj = self.vision_encoder.projection(v)
        l_proj = self.language_encoder.projection(l)
        v_norm = v_proj / (v_proj.norm(dim=-1, keepdim=True) + 1e-8)
        l_norm = l_proj / (l_proj.norm(dim=-1, keepdim=True) + 1e-8)
        # CLIP-style symmetric InfoNCE
        sim = torch.mm(v_norm, l_norm.t()) / 0.07
        labels = torch.arange(sim.size(0))
        cl_loss = (
            torch.nn.functional.cross_entropy(sim, labels)
            + torch.nn.functional.cross_entropy(sim.t(), labels)
        ) / 2.0

        # 3) combined loss — the contrastive loss IS the training signal.
        #    Earlier versions referenced an undefined ``loss`` variable that
        #    crashed at runtime; the only loss that should be back-propagated
        #    is the symmetric InfoNCE computed in step 2.
        total_loss = cl_loss
        total_loss.backward()

        # 4) gradient clip + step
        for enc in [self.vision_encoder, self.language_encoder]:
            nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1.0)
            enc._opt.step()

        return float(total_loss)
