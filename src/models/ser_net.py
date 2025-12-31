"""
SER-Net: Speech Emotion Recognition Network

Multi-scale temporal convolutional architecture for robust emotion 
classification from acoustic features.

Author: Tharun Ponnam
Email: tharunponnam007@gmail.com
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from typing import Tuple, Optional, List
import numpy as np

from .attention import MultiHeadSelfAttention, TemporalAttentionPooling
from .layers import DilatedConvBlock, ResidualBlock


class SERNet(Model):
    """
    Speech Emotion Recognition Network.
    
    A deep convolutional architecture that processes acoustic features
    through multi-scale temporal convolutions followed by attention-based
    pooling for utterance-level emotion classification.
    
    Architecture:
        Input -> Conv Stem -> [Dilated Conv Blocks] x N -> 
        Attention Pooling -> Classification Head -> Output
    
    Args:
        num_classes: Number of emotion categories
        conv_channels: List of channel dimensions for conv blocks
        dilation_rates: Dilation rates for multi-scale convolutions
        attention_heads: Number of attention heads in pooling
        attention_dim: Dimension of attention layer
        dropout_rate: Dropout probability for regularization
        l2_weight: L2 regularization weight
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        conv_channels: List[int] = [64, 128, 256],
        dilation_rates: List[int] = [1, 2, 4],
        attention_heads: int = 8,
        attention_dim: int = 256,
        dropout_rate: float = 0.3,
        l2_weight: float = 1e-4,
        **kwargs
    ):
        super(SERNet, self).__init__(**kwargs)
        
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.dilation_rates = dilation_rates
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.dropout_rate = dropout_rate
        self.l2_weight = l2_weight
        
        # Weight regularizer
        self.regularizer = regularizers.l2(l2_weight)
        
        # Input normalization
        self.input_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Convolutional stem
        self.conv_stem = self._build_conv_stem()
        
        # Multi-scale temporal blocks
        self.temporal_blocks = self._build_temporal_blocks()
        
        # Attention pooling
        self.attention_pool = TemporalAttentionPooling(
            num_heads=attention_heads,
            key_dim=attention_dim // attention_heads,
            dropout=dropout_rate
        )
        
        # Classification head
        self.classifier = self._build_classifier()
        
    def _build_conv_stem(self) -> tf.keras.Sequential:
        """Build initial convolutional feature extractor."""
        return tf.keras.Sequential([
            layers.Conv1D(
                filters=self.conv_channels[0],
                kernel_size=7,
                strides=1,
                padding='same',
                kernel_regularizer=self.regularizer,
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling1D(pool_size=2, strides=2),
            layers.Dropout(self.dropout_rate / 2)
        ], name='conv_stem')
    
    def _build_temporal_blocks(self) -> List[DilatedConvBlock]:
        """Build multi-scale dilated convolutional blocks."""
        blocks = []
        in_channels = self.conv_channels[0]
        
        for i, out_channels in enumerate(self.conv_channels):
            block = DilatedConvBlock(
                filters=out_channels,
                kernel_size=3,
                dilation_rates=self.dilation_rates,
                dropout_rate=self.dropout_rate,
                use_residual=(in_channels == out_channels),
                regularizer=self.regularizer
            )
            blocks.append(block)
            in_channels = out_channels
            
        return blocks
    
    def _build_classifier(self) -> tf.keras.Sequential:
        """Build classification head."""
        return tf.keras.Sequential([
            layers.Dense(
                256,
                kernel_regularizer=self.regularizer,
                use_bias=True
            ),
            layers.LayerNormalization(epsilon=1e-6),
            layers.ReLU(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(
                128,
                kernel_regularizer=self.regularizer,
                use_bias=True
            ),
            layers.LayerNormalization(epsilon=1e-6),
            layers.ReLU(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(
                self.num_classes,
                activation='softmax',
                kernel_regularizer=self.regularizer
            )
        ], name='classifier')
    
    def call(
        self, 
        inputs: tf.Tensor, 
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through the network.
        
        Args:
            inputs: Acoustic features with shape (batch, time, features)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Emotion class probabilities with shape (batch, num_classes)
        """
        # Input normalization
        x = self.input_norm(inputs)
        
        # Convolutional stem
        x = self.conv_stem(x, training=training)
        
        # Multi-scale temporal processing
        for block in self.temporal_blocks:
            x = block(x, training=training)
        
        # Attention-based temporal pooling
        x = self.attention_pool(x, training=training)
        
        # Classification
        logits = self.classifier(x, training=training)
        
        return logits
    
    def get_embeddings(
        self, 
        inputs: tf.Tensor,
        layer: str = 'attention'
    ) -> tf.Tensor:
        """
        Extract intermediate representations for analysis.
        
        Args:
            inputs: Acoustic features
            layer: Which layer to extract from ('conv', 'temporal', 'attention')
            
        Returns:
            Feature embeddings at specified layer
        """
        x = self.input_norm(inputs)
        x = self.conv_stem(x, training=False)
        
        if layer == 'conv':
            return x
        
        for block in self.temporal_blocks:
            x = block(x, training=False)
            
        if layer == 'temporal':
            return x
        
        x = self.attention_pool(x, training=False)
        return x
    
    def get_config(self) -> dict:
        """Return model configuration."""
        config = super(SERNet, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'conv_channels': self.conv_channels,
            'dilation_rates': self.dilation_rates,
            'attention_heads': self.attention_heads,
            'attention_dim': self.attention_dim,
            'dropout_rate': self.dropout_rate,
            'l2_weight': self.l2_weight
        })
        return config
    
    @classmethod
    def from_config(cls, config: dict) -> 'SERNet':
        """Create model from configuration dictionary."""
        return cls(**config)
    
    def summary_table(self) -> str:
        """Generate detailed layer summary."""
        lines = [
            "=" * 70,
            f"{'Layer':<30} {'Output Shape':<20} {'Parameters':>15}",
            "=" * 70
        ]
        
        total_params = 0
        for layer in self.layers:
            params = layer.count_params()
            total_params += params
            lines.append(
                f"{layer.name:<30} {str(layer.output_shape):<20} {params:>15,}"
            )
        
        lines.append("=" * 70)
        lines.append(f"{'Total Parameters:':<50} {total_params:>15,}")
        lines.append("=" * 70)
        
        return "\n".join(lines)


class SERNetLarge(SERNet):
    """
    Larger variant of SER-Net with increased capacity.
    
    Recommended for high-resource scenarios where additional
    model capacity can improve performance.
    """
    
    def __init__(self, num_classes: int = 8, **kwargs):
        super(SERNetLarge, self).__init__(
            num_classes=num_classes,
            conv_channels=[128, 256, 512],
            dilation_rates=[1, 2, 4, 8],
            attention_heads=16,
            attention_dim=512,
            dropout_rate=0.4,
            **kwargs
        )


def build_ser_net(
    input_shape: Tuple[int, int],
    num_classes: int = 8,
    variant: str = 'base',
    pretrained_weights: Optional[str] = None
) -> SERNet:
    """
    Factory function to build SER-Net model.
    
    Args:
        input_shape: Shape of input features (time_steps, feature_dim)
        num_classes: Number of emotion classes
        variant: Model variant ('base' or 'large')
        pretrained_weights: Path to pretrained weights file
        
    Returns:
        Compiled SER-Net model
    """
    model_cls = SERNetLarge if variant == 'large' else SERNet
    
    model = model_cls(num_classes=num_classes)
    
    # Build model with dummy input
    dummy_input = tf.zeros((1,) + input_shape)
    model(dummy_input)
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)
        print(f"Loaded pretrained weights from: {pretrained_weights}")
    
    return model


if __name__ == '__main__':
    # Quick sanity check
    model = build_ser_net(input_shape=(300, 180), num_classes=8)
    model.summary()
    
    # Test forward pass
    x = np.random.randn(4, 300, 180).astype(np.float32)
    output = model(x, training=False)
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 8), "Output shape mismatch!"
    print("Model test passed!")
