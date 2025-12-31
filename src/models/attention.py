"""
Attention Mechanisms for Speech Emotion Recognition

Implements multi-head self-attention and temporal attention pooling
for aggregating frame-level features into utterance representations.

Author: Tharun Ponnam
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-head self-attention mechanism.
    
    Allows the model to attend to different positions and capture
    long-range dependencies in the temporal sequence.
    
    Args:
        num_heads: Number of parallel attention heads
        key_dim: Dimension of key/query projections per head
        dropout: Dropout rate for attention weights
    """
    
    def __init__(
        self,
        num_heads: int = 8,
        key_dim: int = 32,
        dropout: float = 0.1,
        **kwargs
    ):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
            use_bias=True
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)
        
    def call(self, inputs, training=False, attention_mask=None):
        """
        Apply multi-head self-attention.
        
        Args:
            inputs: Input tensor of shape (batch, seq_len, dim)
            training: Whether in training mode
            attention_mask: Optional mask for padded positions
            
        Returns:
            Attended features with residual connection
        """
        # Self-attention
        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training,
            attention_mask=attention_mask
        )
        
        # Dropout and residual connection
        attention_output = self.dropout(attention_output, training=training)
        output = self.layer_norm(inputs + attention_output)
        
        return output
    
    def get_attention_weights(self, inputs):
        """Extract attention weights for visualization."""
        return self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            return_attention_scores=True
        )[1]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout_rate
        })
        return config


class TemporalAttentionPooling(layers.Layer):
    """
    Temporal attention pooling for utterance-level representation.
    
    Instead of simple averaging, learns to weight different time
    frames based on their importance for emotion recognition.
    
    This addresses the challenge that emotional cues may be concentrated
    in specific portions of an utterance rather than being uniform.
    
    Args:
        num_heads: Number of attention heads
        key_dim: Dimension per head
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_heads: int = 8,
        key_dim: int = 32,
        dropout: float = 0.1,
        **kwargs
    ):
        super(TemporalAttentionPooling, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        
        # Learnable query for attention pooling
        self.global_query = self.add_weight(
            name='global_query',
            shape=(1, 1, feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Attention layers
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        
        # Post-attention projection
        self.dense = layers.Dense(feature_dim, use_bias=False)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        
        super().build(input_shape)
        
    def call(self, inputs, training=False, mask=None):
        """
        Pool temporal sequence into fixed-size representation.
        
        Args:
            inputs: Frame-level features (batch, time, features)
            training: Training mode flag
            mask: Optional temporal mask
            
        Returns:
            Utterance-level representation (batch, features)
        """
        batch_size = tf.shape(inputs)[0]
        
        # Expand query for batch dimension
        query = tf.tile(self.global_query, [batch_size, 1, 1])
        
        # Attend to temporal sequence
        attended = self.attention(
            query=query,
            key=inputs,
            value=inputs,
            training=training,
            attention_mask=mask
        )
        
        # Remove singleton time dimension
        pooled = tf.squeeze(attended, axis=1)
        
        # Project and normalize
        output = self.dense(pooled)
        output = self.layer_norm(output)
        
        return output
    
    def get_frame_importance(self, inputs):
        """
        Get per-frame importance scores for interpretability.
        
        Returns attention weights showing which frames were most
        influential for the final prediction.
        """
        batch_size = tf.shape(inputs)[0]
        query = tf.tile(self.global_query, [batch_size, 1, 1])
        
        _, attention_weights = self.attention(
            query=query,
            key=inputs,
            value=inputs,
            return_attention_scores=True
        )
        
        # Average across heads and squeeze
        importance = tf.reduce_mean(attention_weights, axis=1)
        importance = tf.squeeze(importance, axis=1)
        
        return importance
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout_rate
        })
        return config


class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding for temporal modeling.
    
    Adds position information to input features, enabling the model
    to utilize temporal ordering in the attention mechanism.
    """
    
    def __init__(self, max_length: int = 1000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_length = max_length
        
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        
        # Compute positional encodings
        positions = np.arange(self.max_length)[:, np.newaxis]
        dimensions = np.arange(feature_dim)[np.newaxis, :]
        
        angles = positions / np.power(
            10000, 
            (2 * (dimensions // 2)) / feature_dim
        )
        
        # Apply sin to even indices, cos to odd indices
        encodings = np.zeros((self.max_length, feature_dim))
        encodings[:, 0::2] = np.sin(angles[:, 0::2])
        encodings[:, 1::2] = np.cos(angles[:, 1::2])
        
        self.positional_encoding = tf.constant(
            encodings[np.newaxis, :, :],
            dtype=tf.float32
        )
        
        super().build(input_shape)
        
    def call(self, inputs):
        """Add positional encoding to input features."""
        seq_len = tf.shape(inputs)[1]
        return inputs + self.positional_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({'max_length': self.max_length})
        return config


class SqueezedAttention(layers.Layer):
    """
    Lightweight attention variant for resource-constrained settings.
    
    Uses a simpler single-head attention with reduced computational
    overhead while still providing frame weighting capability.
    """
    
    def __init__(self, hidden_dim: int = 128, **kwargs):
        super(SqueezedAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        
        self.attention_weights = tf.keras.Sequential([
            layers.Dense(self.hidden_dim, activation='tanh'),
            layers.Dense(1, use_bias=False)
        ])
        
        super().build(input_shape)
        
    def call(self, inputs, mask=None):
        """
        Apply squeezed attention pooling.
        
        Args:
            inputs: (batch, time, features)
            mask: Optional temporal mask
            
        Returns:
            Pooled representation (batch, features)
        """
        # Compute attention scores
        scores = self.attention_weights(inputs)
        scores = tf.squeeze(scores, axis=-1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (1.0 - tf.cast(mask, tf.float32)) * -1e9
        
        # Normalize with softmax
        attention = tf.nn.softmax(scores, axis=-1)
        attention = tf.expand_dims(attention, axis=-1)
        
        # Weighted sum
        output = tf.reduce_sum(inputs * attention, axis=1)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'hidden_dim': self.hidden_dim})
        return config
