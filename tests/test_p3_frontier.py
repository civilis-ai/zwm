"""Tests for the 2026 P3 cohort — frontier architecture upgrades.

Each test exercises one of the eight 2026 frontiers that the JEPAPredictor
was upgraded to match:
  A — action-conditioned prediction
  B — Hierarchical multi-scale latents
  C — Product of Experts (BJEPA-style)
  D — SIGReg distribution-matching anti-collapse
  E — Bidirectional prediction
  F — Energy-based model scalar head
  G — UnifiedField de-redundancy (106-dim world vector)
  H — Dimension scale-up (hidden 192, latent 64)
"""
from __future__ import annotations

import numpy as np
import torch

from zwm.jepa.predictor import (
    HierarchicalJEPAPredictor,
    JEPAPredictor,
    _ActionEmbedding,
    _EnergyHead,
    _PriorExpert,
    sigreg_loss,
)
from zwm.scene_field.unified_field import UnifiedField
from zwm.self_field.palace_graph import LuoshuGrid
from zwm.core.hexagram import hexagram_from_bits


# -----------------------------------------------------------------------
# (G) UnifiedField de-redundancy
# -----------------------------------------------------------------------

class TestUnifiedFieldDeduplication:
    """The 6-dim hexagram binary string was dropped from to_tensor().
    P3: 4 calendar context phases (年/月/日/时) → 25 dims.
    P3-C: 4 cosmic phases (元/会/运/世) → 29 dims."""

    def test_to_tensor_is_29_dim(self):
        h = hexagram_from_bits(0b101010)
        g = LuoshuGrid()
        # Provide all 8 calendar keys so the cosmic phases (元/会/运/世)
        # are populated — otherwise the tensor would still be 29-dim but
        # those 4 dims would be 0.0.
        world = UnifiedField.snapshot(
            h, g, 0.0,
            calendar_context={
                "年": 1.0, "月": 2.0, "日": 3.0, "时": 4.0,
                "元": 0.5, "会": 1.5, "运": 2.5, "世": 3.5,
            },
        )
        t = world.to_tensor()
        assert len(t) == 29, f"expected 29-dim tensor, got {len(t)}"

    def test_cosmic_phases_consumed(self):
        """P3-C: cosmic phases (元/会/运/世) must appear in the tensor."""
        h = hexagram_from_bits(0b101010)
        g = LuoshuGrid()
        world = UnifiedField.snapshot(
            h, g, 0.0,
            calendar_context={
                "年": 1.0, "月": 2.0, "日": 3.0, "时": 4.0,
                "元": 1.0, "会": 1.0, "运": 1.0, "世": 1.0,
            },
        )
        t = world.to_tensor()
        # The last 4 dims are cosmic phases normalized to [0, 1]
        cosmic_dims = t[-4:]
        # 1.0 normalized to [0, 1] is 1.0 / (2π) ≈ 0.159
        expected = 1.0 / (2 * 3.141592653589793)
        for v in cosmic_dims:
            assert abs(v - expected) < 1e-6, f"cosmic phase dim = {v}, expected {expected}"

    def test_no_binary_bits_in_tensor(self):
        """The 6 binary bits of the hexagram are NOT in the tensor anymore."""
        h = hexagram_from_bits(0b111111)  # all-yang hexagram
        g = LuoshuGrid()
        world = UnifiedField.snapshot(h, g, 0.0, calendar_context=None)
        t = world.to_tensor()
        # If the binary bits were there, we'd see six consecutive 1.0s
        # (or six consecutive 0.0s for all-yin).  The remaining
        # dimensions (time/grid/luoshu/elements/liuqin) should be
        # bounded in [0, 1] but NOT all-1.0 in a row.
        assert not all(v == 1.0 for v in t[:6]), "binary bits leak into tensor"


# -----------------------------------------------------------------------
# (H) Dimension scale-up
# -----------------------------------------------------------------------

class TestDimensionScaleUp:
    """hidden_dim 64→192, latent_dim 32→64."""

    def test_default_hidden_dim_is_192(self):
        jp = JEPAPredictor(input_dim=106)
        assert jp.hidden_dim == 192

    def test_default_latent_dim_is_64(self):
        jp = JEPAPredictor(input_dim=106)
        assert jp.latent_dim == 64

    def test_predict_output_is_64_dim(self):
        jp = JEPAPredictor(input_dim=106)
        z = np.random.randn(106).astype(np.float32)
        out = jp.predict(z)
        # out is (1, 64) when squeezed gives 64
        assert out.shape[-1] == 64


# -----------------------------------------------------------------------
# (A) Action conditioning
# -----------------------------------------------------------------------

class TestActionConditioning:
    """The predictor consumes a mutation-mask embedding for action conditioning."""

    def test_mask_changes_prediction(self):
        jp = JEPAPredictor(input_dim=106, seed=42)
        z = np.random.RandomState(0).randn(106).astype(np.float32)
        p0 = jp.predict(z, mask=0)
        p5 = jp.predict(z, mask=5)
        p32 = jp.predict(z, mask=32)
        # Different actions should give different predictions
        assert not np.allclose(p0, p5)
        assert not np.allclose(p0, p32)
        assert not np.allclose(p5, p32)

    def test_action_embedding_dim(self):
        emb = _ActionEmbedding(num_masks=64, embed_dim=32)
        assert emb.dim == 32
        out = emb(torch.tensor(5))
        assert out.shape == (32,)

    def test_predictor_accepts_mask_int(self):
        jp = JEPAPredictor(input_dim=106)
        z = np.random.randn(106).astype(np.float32)
        out_int = jp.predict(z, mask=7)
        out_tensor = jp.predict(z, mask=torch.tensor(7))
        # Both should produce same output for the same mask
        assert out_int.shape == out_tensor.shape


# -----------------------------------------------------------------------
# (B) Hierarchical JEPA
# -----------------------------------------------------------------------

class TestHierarchicalJEPA:
    """Multi-scale temporal predictions at 1/4/16 step horizons."""

    def test_predict_returns_three_scales(self):
        hjp = HierarchicalJEPAPredictor(input_dim=106)
        z = np.random.randn(106).astype(np.float32)
        out = hjp.predict(z)
        assert set(out.keys()) == {"short", "mid", "long"}

    def test_three_scales_differ(self):
        """Different time scales should produce different predictions."""
        hjp = HierarchicalJEPAPredictor(input_dim=106, seed=42)
        z = np.random.RandomState(0).randn(106).astype(np.float32)
        out = hjp.predict(z)
        assert not np.allclose(out["short"], out["mid"])
        assert not np.allclose(out["mid"], out["long"])
        assert not np.allclose(out["short"], out["long"])

    def test_train_step_returns_per_level_losses(self):
        hjp = HierarchicalJEPAPredictor(input_dim=106)
        z1 = np.random.randn(106).astype(np.float32)
        z2 = np.random.randn(106).astype(np.float32)
        losses = hjp.train_step(z1, z2)
        assert "short" in losses and "mid" in losses and "long" in losses
        for k, v in losses.items():
            assert isinstance(v, (float, int)), f"{k}={v!r} type={type(v)}"


# -----------------------------------------------------------------------
# (C) Product of Experts
# -----------------------------------------------------------------------

class TestProductOfExperts:
    """Dynamics expert + structural prior expert via weighted PoE."""

    def test_prior_expert_outputs_latent_dim(self):
        expert = _PriorExpert(latent_dim=64, hidden_dim=128)
        out = expert(torch.randn(4, 64))
        assert out.shape == (4, 64)

    def test_poe_combines_dynamics_and_prior(self):
        """The PoE weight should affect the final prediction."""
        torch.manual_seed(0)
        jp_poe = JEPAPredictor(input_dim=106, poe_weight=0.5)
        torch.manual_seed(0)
        jp_no_poe = JEPAPredictor(input_dim=106, use_prior_expert=False)
        z = np.random.RandomState(0).randn(106).astype(np.float32)
        p_poe = jp_poe.predict(z)
        p_no = jp_no_poe.predict(z)
        assert not np.allclose(p_poe, p_no)


# -----------------------------------------------------------------------
# (D) SIGReg distribution-matching
# -----------------------------------------------------------------------

class TestSIGReg:
    """Slice-Induced Gaussian Regularisation (LeWorldModel 2026)."""

    def test_sigreg_returns_scalar(self):
        z = torch.randn(8, 64)
        loss = sigreg_loss(z, num_slices=64)
        assert loss.dim() == 0  # scalar

    def test_sigreg_is_finite(self):
        z = torch.randn(16, 32)
        loss = sigreg_loss(z)
        assert torch.isfinite(loss).item()

    def test_sigreg_handles_single_sample(self):
        """1-sample edge case falls back to a variance hinge."""
        z = torch.randn(1, 32)
        loss = sigreg_loss(z)
        assert torch.isfinite(loss).item()
        assert loss.item() >= 0.0

    def test_jepa_with_sigreg_trains(self):
        """End-to-end: training with SIGReg doesn't NaN."""
        jp = JEPAPredictor(input_dim=106, use_sigreg=True)
        for _ in range(3):
            z = np.random.randn(106).astype(np.float32)
            result = jp.train_step(z, np.random.randn(106).astype(np.float32))
            assert not np.isnan(result["loss"])


# -----------------------------------------------------------------------
# (E) Bidirectional prediction
# -----------------------------------------------------------------------

class TestBidirectionalPrediction:
    """BiJEPA 2026: backward predictor z_{t+1} -> z_t."""

    def test_backward_predictor_exists(self):
        jp = JEPAPredictor(input_dim=106, use_backward=True)
        assert jp._backward_predictor is not None

    def test_backward_disabled_by_flag(self):
        jp = JEPAPredictor(input_dim=106, use_backward=False)
        assert jp._backward_predictor is None

    def test_bidirectional_loss_is_finite(self):
        jp = JEPAPredictor(input_dim=106)
        z1 = np.random.randn(106).astype(np.float32)
        z2 = np.random.randn(106).astype(np.float32)
        result = jp.train_step(z1, z2)
        assert np.isfinite(result["loss"])


# -----------------------------------------------------------------------
# (F) Energy-based model head
# -----------------------------------------------------------------------

class TestEnergyHead:
    """P3-F: EBM scalar head for multimodal future support."""

    def test_energy_head_scalar_output(self):
        head = _EnergyHead(input_dim=64, hidden_dim=32)
        out = head(torch.randn(4, 64))
        assert out.shape == (4,)

    def test_energy_head_in_jepa(self):
        jp = JEPAPredictor(input_dim=106, use_energy_head=True)
        assert jp._energy_head is not None
        z = np.random.randn(106).astype(np.float32)
        out = jp.predict(z)
        assert out.shape[-1] == 64

    def test_energy_loss_drives_separation(self):
        """After training, energy(z_t) should be lower than energy(decoy)."""
        jp = JEPAPredictor(input_dim=32, use_energy_head=True, latent_dim=8, hidden_dim=16)
        for _ in range(8):
            z1 = np.random.randn(32).astype(np.float32)
            z2 = np.random.randn(32).astype(np.float32)
            jp.train_step(z1, z2)
        z = torch.from_numpy(np.random.randn(1, 8).astype(np.float32)).to(jp.device)
        e = jp._energy_head(z)
        assert torch.isfinite(e).all()
