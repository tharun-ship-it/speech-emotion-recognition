"""
Loss Functions for Speech Emotion Recognition

Implements specialized loss functions for handling class imbalance
and improving emotion classification performance.

Author: Tharun Ponnam
"""

import tensorflow as tf
from typing import Optional, Dict
import numpy as np


def focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    from_logits: bool = False
) -> tf.Tensor:
    """
    Focal Loss for addressing class imbalance.
    
    Down-weights easy examples and focuses training on hard negatives.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    References:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    
    Args:
        y_true: Ground truth labels (sparse integers)
        y_pred: Predicted probabilities
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class balance weight
        from_logits: If True, apply softmax to y_pred
        
    Returns:
        Focal loss value
    """
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    # Clip predictions for numerical stability
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Convert sparse labels to one-hot
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(
        tf.cast(y_true, tf.int32), 
        depth=num_classes
    )
    
    # Compute cross entropy
    cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
    
    # Compute focal weight
    p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
    focal_weight = tf.pow(1.0 - p_t, gamma)
    
    # Apply alpha if provided
    if alpha is not None:
        focal_weight = alpha * focal_weight
    
    # Compute focal loss
    focal_loss = focal_weight * tf.reduce_sum(cross_entropy, axis=-1)
    
    return tf.reduce_mean(focal_loss)


def weighted_cross_entropy(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    weights: Dict[int, float],
    from_logits: bool = False
) -> tf.Tensor:
    """
    Weighted cross entropy loss for class imbalance.
    
    Applies different weights to each class based on inverse
    frequency or custom weights.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        weights: Dictionary mapping class index to weight
        from_logits: If True, apply softmax to y_pred
        
    Returns:
        Weighted cross entropy loss
    """
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    num_classes = len(weights)
    y_true_one_hot = tf.one_hot(
        tf.cast(y_true, tf.int32),
        depth=num_classes
    )
    
    # Create weight tensor
    weight_tensor = tf.constant(
        [weights.get(i, 1.0) for i in range(num_classes)],
        dtype=tf.float32
    )
    
    # Compute weighted cross entropy
    cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
    weighted_ce = cross_entropy * weight_tensor
    
    return tf.reduce_mean(tf.reduce_sum(weighted_ce, axis=-1))


def label_smoothing_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smoothing: float = 0.1,
    from_logits: bool = False
) -> tf.Tensor:
    """
    Cross entropy loss with label smoothing.
    
    Prevents overconfident predictions and improves generalization
    by softening the target distribution.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        smoothing: Label smoothing factor (0 = no smoothing)
        from_logits: If True, apply softmax to y_pred
        
    Returns:
        Smoothed cross entropy loss
    """
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    y_true_one_hot = tf.one_hot(
        tf.cast(y_true, tf.int32),
        depth=tf.shape(y_pred)[-1]
    )
    
    # Apply label smoothing
    y_true_smooth = y_true_one_hot * (1.0 - smoothing) + smoothing / num_classes
    
    # Compute cross entropy
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true_smooth * tf.math.log(y_pred)
    
    return tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=-1))


def center_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    centers: tf.Variable,
    alpha: float = 0.5
) -> tf.Tensor:
    """
    Center loss for feature learning.
    
    Encourages features of the same class to be close to
    their class center, improving intra-class compactness.
    
    References:
        Wen et al., "A Discriminative Feature Learning Approach
        for Deep Face Recognition", ECCV 2016
    
    Args:
        embeddings: Feature embeddings from model
        labels: Class labels
        centers: Learnable class centers (tf.Variable)
        alpha: Center update rate
        
    Returns:
        Center loss value
    """
    # Get centers for each sample
    centers_batch = tf.gather(centers, labels)
    
    # Compute loss
    loss = tf.reduce_sum(tf.square(embeddings - centers_batch)) / 2.0
    
    # Update centers
    diff = centers_batch - embeddings
    unique_labels, unique_idx = tf.unique(labels)
    
    # Aggregate gradients for each center
    delta = tf.math.unsorted_segment_mean(
        diff, unique_idx, tf.shape(unique_labels)[0]
    )
    
    # Update centers
    centers_update = tf.tensor_scatter_nd_sub(
        centers,
        tf.expand_dims(unique_labels, 1),
        alpha * delta
    )
    centers.assign(centers_update)
    
    return loss / tf.cast(tf.shape(labels)[0], tf.float32)


def concordance_correlation_coefficient(
    y_true: tf.Tensor,
    y_pred: tf.Tensor
) -> tf.Tensor:
    """
    Concordance Correlation Coefficient loss.
    
    Useful for dimensional emotion recognition (valence/arousal).
    Measures agreement between predictions and ground truth.
    
    CCC = 2 * ρ * σ_x * σ_y / (σ_x^2 + σ_y^2 + (μ_x - μ_y)^2)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        1 - CCC (as loss, lower is better)
    """
    mean_true = tf.reduce_mean(y_true)
    mean_pred = tf.reduce_mean(y_pred)
    
    var_true = tf.reduce_mean(tf.square(y_true - mean_true))
    var_pred = tf.reduce_mean(tf.square(y_pred - mean_pred))
    
    covariance = tf.reduce_mean(
        (y_true - mean_true) * (y_pred - mean_pred)
    )
    
    ccc = (2 * covariance) / (
        var_true + var_pred + tf.square(mean_true - mean_pred) + 1e-8
    )
    
    return 1.0 - ccc


class FocalLoss(tf.keras.losses.Loss):
    """Keras Loss wrapper for focal loss."""
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        name: str = 'focal_loss'
    ):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
        
    def call(self, y_true, y_pred):
        return focal_loss(y_true, y_pred, self.gamma, self.alpha)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config


class CombinedLoss(tf.keras.losses.Loss):
    """
    Combined loss function with multiple objectives.
    
    Combines cross-entropy with auxiliary losses for
    improved training dynamics.
    
    Args:
        focal_weight: Weight for focal loss
        label_smooth_weight: Weight for label smoothing loss
        focal_gamma: Gamma for focal loss
        smoothing: Label smoothing factor
    """
    
    def __init__(
        self,
        focal_weight: float = 1.0,
        label_smooth_weight: float = 0.0,
        focal_gamma: float = 2.0,
        smoothing: float = 0.1,
        name: str = 'combined_loss'
    ):
        super().__init__(name=name)
        self.focal_weight = focal_weight
        self.label_smooth_weight = label_smooth_weight
        self.focal_gamma = focal_gamma
        self.smoothing = smoothing
        
    def call(self, y_true, y_pred):
        loss = 0.0
        
        if self.focal_weight > 0:
            loss += self.focal_weight * focal_loss(
                y_true, y_pred, gamma=self.focal_gamma
            )
            
        if self.label_smooth_weight > 0:
            loss += self.label_smooth_weight * label_smoothing_loss(
                y_true, y_pred, smoothing=self.smoothing
            )
            
        return loss


def get_class_weights(
    labels: np.ndarray,
    num_classes: int,
    method: str = 'inverse'
) -> Dict[int, float]:
    """
    Compute class weights from label distribution.
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        method: 'inverse' or 'effective' weighting
        
    Returns:
        Dictionary of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    if method == 'inverse':
        # Inverse frequency weighting
        weights = {
            int(c): total / (num_classes * count)
            for c, count in zip(unique, counts)
        }
    elif method == 'effective':
        # Effective number weighting
        beta = 0.9999
        weights = {
            int(c): (1 - beta) / (1 - beta ** count)
            for c, count in zip(unique, counts)
        }
    else:
        weights = {i: 1.0 for i in range(num_classes)}
        
    # Fill missing classes
    for i in range(num_classes):
        if i not in weights:
            weights[i] = 1.0
            
    return weights
