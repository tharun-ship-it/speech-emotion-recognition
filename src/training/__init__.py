"""
Training utilities for Speech Emotion Recognition.

Includes training loop, callbacks, loss functions,
and learning rate schedulers.
"""

from src.training.trainer import SERTrainer
from src.training.callbacks import (
    WarmupCosineDecay,
    GradientAccumulator,
    MetricsLogger,
    EMACallback
)
from src.training.losses import (
    FocalLoss,
    CombinedLoss,
    focal_loss,
    weighted_cross_entropy,
    label_smoothing_loss,
    center_loss,
    concordance_correlation_coefficient,
    get_class_weights
)

__all__ = [
    'SERTrainer',
    'WarmupCosineDecay',
    'GradientAccumulator',
    'MetricsLogger',
    'EMACallback',
    'FocalLoss',
    'CombinedLoss',
    'focal_loss',
    'weighted_cross_entropy',
    'label_smoothing_loss',
    'center_loss',
    'concordance_correlation_coefficient',
    'get_class_weights',
]
