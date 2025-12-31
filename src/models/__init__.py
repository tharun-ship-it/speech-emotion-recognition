"""
Model architectures for Speech Emotion Recognition.

Includes SER-Net and its components: attention mechanisms,
custom layers, and utility functions.
"""

from src.models.ser_net import SERNet, create_ser_model
from src.models.attention import (
    MultiHeadSelfAttention,
    TemporalAttentionPooling,
    PositionalEncoding
)
from src.models.layers import (
    DilatedConvBlock,
    ResidualBlock,
    TemporalConvLayer,
    SqueezeExcitation,
    FeatureFusion
)

__all__ = [
    'SERNet',
    'create_ser_model',
    'MultiHeadSelfAttention',
    'TemporalAttentionPooling',
    'PositionalEncoding',
    'DilatedConvBlock',
    'ResidualBlock',
    'TemporalConvLayer',
    'SqueezeExcitation',
    'FeatureFusion',
]
