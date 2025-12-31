"""
Custom Neural Network Layers for SER-Net

Implements dilated convolutional blocks, residual connections,
and other specialized layers for temporal feature processing.

Author: Tharun Ponnam
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from typing import List, Optional, Tuple


class DilatedConvBlock(layers.Layer):
    """
    Multi-scale dilated convolutional block.
    
    Applies parallel convolutions with different dilation rates to 
    capture patterns at multiple temporal scales simultaneously.
    
    This is particularly effective for emotion recognition where
    emotional cues manifest across varying time spans.
    
    Args:
        filters: Number of output filters
        kernel_size: Convolutional kernel size
        dilation_rates: List of dilation rates for parallel branches
        dropout_rate: Dropout probability
        use_residual: Whether to add residual connection
        regularizer: Kernel regularizer
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dilation_rates: List[int] = [1, 2, 4],
        dropout_rate: float = 0.2,
        use_residual: bool = True,
        regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs
    ):
        super(DilatedConvBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.regularizer = regularizer
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Parallel dilated convolutions
        self.dilated_convs = []
        for rate in self.dilation_rates:
            conv = layers.Conv1D(
                filters=self.filters // len(self.dilation_rates),
                kernel_size=self.kernel_size,
                dilation_rate=rate,
                padding='same',
                kernel_regularizer=self.regularizer,
                use_bias=False
            )
            self.dilated_convs.append(conv)
        
        # Batch normalization and activation
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.ReLU()
        self.dropout = layers.Dropout(self.dropout_rate)
        
        # Residual projection if dimensions differ
        if self.use_residual and input_dim != self.filters:
            self.residual_proj = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding='same',
                kernel_regularizer=self.regularizer,
                use_bias=False
            )
        else:
            self.residual_proj = None
            
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        """
        Forward pass through dilated conv block.
        
        Args:
            inputs: Input tensor (batch, time, features)
            training: Training mode flag
            
        Returns:
            Processed features with multi-scale information
        """
        # Apply parallel dilated convolutions
        branch_outputs = []
        for conv in self.dilated_convs:
            branch = conv(inputs)
            branch_outputs.append(branch)
        
        # Concatenate multi-scale features
        x = tf.concat(branch_outputs, axis=-1)
        
        # Normalize and activate
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        
        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(inputs)
            else:
                residual = inputs
            x = x + residual
            
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rates': self.dilation_rates,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual
        })
        return config


class ResidualBlock(layers.Layer):
    """
    Standard residual block with pre-activation design.
    
    Uses the pre-activation variant (BN-ReLU-Conv) which has been
    shown to improve gradient flow in deep networks.
    
    Args:
        filters: Number of filters
        kernel_size: Convolution kernel size
        dropout_rate: Dropout probability
        regularizer: Kernel regularizer
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs
    ):
        super(ResidualBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.regularizer = regularizer
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Pre-activation layers
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.ReLU()
        self.conv1 = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_regularizer=self.regularizer,
            use_bias=False
        )
        
        self.bn2 = layers.BatchNormalization()
        self.activation2 = layers.ReLU()
        self.conv2 = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_regularizer=self.regularizer,
            use_bias=False
        )
        
        self.dropout = layers.Dropout(self.dropout_rate)
        
        # Shortcut projection
        if input_dim != self.filters:
            self.shortcut = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding='same',
                kernel_regularizer=self.regularizer,
                use_bias=False
            )
        else:
            self.shortcut = None
            
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        """Apply residual transformation."""
        # First conv block
        x = self.bn1(inputs, training=training)
        x = self.activation1(x)
        x = self.conv1(x)
        
        # Second conv block
        x = self.bn2(x, training=training)
        x = self.activation2(x)
        x = self.dropout(x, training=training)
        x = self.conv2(x)
        
        # Shortcut connection
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
        else:
            shortcut = inputs
            
        return x + shortcut
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate
        })
        return config


class TemporalConvLayer(layers.Layer):
    """
    Temporal convolutional layer with causal padding option.
    
    Provides flexibility for both causal (real-time) and
    non-causal (batch processing) applications.
    
    Args:
        filters: Number of output filters
        kernel_size: Size of convolutional kernel
        dilation_rate: Dilation factor for expanding receptive field
        causal: If True, use causal (left-only) padding
        activation: Activation function name
        regularizer: Kernel regularizer
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        causal: bool = False,
        activation: str = 'relu',
        regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs
    ):
        super(TemporalConvLayer, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.causal = causal
        self.activation_name = activation
        self.regularizer = regularizer
        
    def build(self, input_shape):
        # Calculate padding for same output length
        self.pad_size = (self.kernel_size - 1) * self.dilation_rate
        
        self.conv = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='valid',  # We'll handle padding manually
            kernel_regularizer=self.regularizer,
            use_bias=True
        )
        
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation(self.activation_name)
        
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        """Apply temporal convolution with appropriate padding."""
        if self.causal:
            # Causal padding: pad only on the left
            x = tf.pad(inputs, [[0, 0], [self.pad_size, 0], [0, 0]])
        else:
            # Symmetric padding for non-causal mode
            pad_left = self.pad_size // 2
            pad_right = self.pad_size - pad_left
            x = tf.pad(inputs, [[0, 0], [pad_left, pad_right], [0, 0]])
        
        x = self.conv(x)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'causal': self.causal,
            'activation': self.activation_name
        })
        return config


class SqueezeExcitation(layers.Layer):
    """
    Squeeze-and-Excitation block for channel recalibration.
    
    Learns to emphasize informative feature channels and suppress
    less useful ones through global context aggregation.
    
    Args:
        reduction_ratio: Factor to reduce channels in bottleneck
    """
    
    def __init__(self, reduction_ratio: int = 8, **kwargs):
        super(SqueezeExcitation, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_ratio, 1)
        
        self.squeeze = layers.GlobalAveragePooling1D()
        self.excitation = tf.keras.Sequential([
            layers.Dense(reduced_channels, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])
        
        super().build(input_shape)
        
    def call(self, inputs):
        """Apply squeeze-and-excitation recalibration."""
        # Squeeze: global context
        squeezed = self.squeeze(inputs)
        
        # Excitation: channel weights
        weights = self.excitation(squeezed)
        weights = tf.expand_dims(weights, axis=1)
        
        # Scale channels
        return inputs * weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


class FeatureFusion(layers.Layer):
    """
    Multi-modal feature fusion layer.
    
    Combines features from different acoustic representations
    (MFCC, mel-spectrogram, prosodic) using learned weights.
    
    Args:
        output_dim: Dimension of fused representation
        fusion_type: 'concat', 'attention', or 'gated'
    """
    
    def __init__(
        self,
        output_dim: int,
        fusion_type: str = 'gated',
        **kwargs
    ):
        super(FeatureFusion, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
    def build(self, input_shape):
        # Assume input is a list of feature tensors
        if isinstance(input_shape, list):
            total_dim = sum(shape[-1] for shape in input_shape)
            num_sources = len(input_shape)
        else:
            total_dim = input_shape[-1]
            num_sources = 1
        
        if self.fusion_type == 'gated':
            # Gated fusion with learned importance
            self.gate_dense = layers.Dense(num_sources, activation='softmax')
            self.projection = layers.Dense(self.output_dim)
        else:
            # Simple projection after concatenation
            self.projection = layers.Dense(self.output_dim)
            
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        
        super().build(input_shape)
        
    def call(self, inputs):
        """Fuse multiple feature streams."""
        if isinstance(inputs, list):
            if self.fusion_type == 'gated':
                # Compute gates from concatenated features
                concat = tf.concat(inputs, axis=-1)
                gates = self.gate_dense(concat)
                
                # Weighted combination
                stacked = tf.stack(inputs, axis=-1)
                gates = tf.expand_dims(gates, axis=-2)
                fused = tf.reduce_sum(stacked * gates, axis=-1)
            else:
                fused = tf.concat(inputs, axis=-1)
        else:
            fused = inputs
            
        # Project to output dimension
        output = self.projection(fused)
        output = self.layer_norm(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'fusion_type': self.fusion_type
        })
        return config
