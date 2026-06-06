"""JEPA — Joint-Embedding Predictive Architecture (world model)."""
from zwm.jepa.circular_encoder import CircularEncoder
from zwm.jepa.predictor import (
    HierarchicalJEPAPredictor,
    JEPAPredictor,
    wrap_fsdp2,
    wrap_fsdp2_hierarchical,
)
from zwm.jepa.square_encoder import (
    FixedWeightSquareGNN,
    LearnableSquareGNN,
    SquareCircularJoint,
)
from zwm.jepa.vq import VQCodebook

__all__ = [
    "CircularEncoder",
    "HierarchicalJEPAPredictor",
    "JEPAPredictor",
    "wrap_fsdp2",
    "wrap_fsdp2_hierarchical",
    "FixedWeightSquareGNN",
    "LearnableSquareGNN",
    "SquareCircularJoint",
    "VQCodebook",
]
