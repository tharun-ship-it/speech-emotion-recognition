"""
Custom Training Callbacks

Implements specialized callbacks for learning rate scheduling,
gradient accumulation, and metrics logging.

Author: Tharun Ponnam
"""

import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List


class WarmupCosineDecay:
    """
    Learning rate schedule with linear warmup and cosine decay.
    
    Starts with linear warmup from 0 to initial_lr, then applies
    cosine decay to min_lr.
    
    Args:
        initial_lr: Peak learning rate after warmup
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr: Minimum learning rate
    """
    
    def __init__(
        self,
        initial_lr: float = 1e-3,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-7
    ):
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        
    def __call__(self, epoch: int) -> float:
        """Compute learning rate for given epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay


class WarmupCosineDecayCallback(callbacks.Callback):
    """Keras callback wrapper for WarmupCosineDecay."""
    
    def __init__(
        self,
        initial_lr: float = 1e-3,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-7,
        verbose: bool = True
    ):
        super().__init__()
        self.schedule = WarmupCosineDecay(
            initial_lr, warmup_epochs, total_epochs, min_lr
        )
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.schedule(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        if self.verbose:
            print(f'\nLearning rate: {lr:.2e}')


class GradientAccumulator(callbacks.Callback):
    """
    Gradient accumulation callback for effective larger batch sizes.
    
    Accumulates gradients over multiple mini-batches before
    applying updates, enabling larger effective batch sizes
    on memory-constrained hardware.
    
    Args:
        accum_steps: Number of steps to accumulate
    """
    
    def __init__(self, accum_steps: int = 4):
        super().__init__()
        self.accum_steps = accum_steps
        self.step_count = 0
        self.gradient_accumulator = None
        
    def on_train_begin(self, logs=None):
        # Initialize gradient accumulator
        self.gradient_accumulator = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in self.model.trainable_variables
        ]
        
    def on_train_batch_begin(self, batch, logs=None):
        if self.step_count == 0:
            # Reset accumulated gradients
            for accum in self.gradient_accumulator:
                accum.assign(tf.zeros_like(accum))
                
    def on_train_batch_end(self, batch, logs=None):
        self.step_count = (self.step_count + 1) % self.accum_steps


class MetricsLogger(callbacks.Callback):
    """
    Enhanced metrics logging callback.
    
    Logs detailed per-class metrics and additional evaluation
    statistics during training.
    
    Args:
        log_dir: Directory for log files
        class_names: List of class names
    """
    
    def __init__(
        self,
        log_dir: str = 'logs',
        class_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or []
        self.epoch_logs = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Add timestamp
        import datetime
        logs['timestamp'] = datetime.datetime.now().isoformat()
        logs['epoch'] = epoch
        
        self.epoch_logs.append(logs)
        
        # Save to JSON
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.epoch_logs, f, indent=2)
            
    def on_train_end(self, logs=None):
        # Save final summary
        summary = {
            'total_epochs': len(self.epoch_logs),
            'best_val_accuracy': max(
                log.get('val_accuracy', 0) 
                for log in self.epoch_logs
            ),
            'final_train_loss': self.epoch_logs[-1].get('loss', 0),
            'final_val_loss': self.epoch_logs[-1].get('val_loss', 0)
        }
        
        with open(self.log_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


class ConfusionMatrixCallback(callbacks.Callback):
    """
    Callback to compute and log confusion matrix periodically.
    
    Args:
        validation_data: Validation dataset
        class_names: List of class names
        log_dir: Directory for saving matrices
        frequency: How often to compute (in epochs)
    """
    
    def __init__(
        self,
        validation_data: tf.data.Dataset,
        class_names: List[str],
        log_dir: str = 'logs',
        frequency: int = 5
    ):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.log_dir = Path(log_dir)
        self.frequency = frequency
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency != 0:
            return
            
        all_preds = []
        all_labels = []
        
        for features, labels in self.validation_data:
            preds = self.model(features, training=False)
            all_preds.extend(np.argmax(preds.numpy(), axis=-1))
            all_labels.extend(labels.numpy())
        
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Save
        np.save(
            self.log_dir / f'confusion_matrix_epoch_{epoch + 1}.npy',
            cm
        )


class EMACallback(callbacks.Callback):
    """
    Exponential Moving Average of model weights.
    
    Maintains an EMA of model weights for improved
    generalization during inference.
    
    Args:
        decay: EMA decay rate (higher = slower update)
    """
    
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None
        
    def on_train_begin(self, logs=None):
        self.ema_weights = [
            tf.Variable(w) for w in self.model.get_weights()
        ]
        
    def on_batch_end(self, batch, logs=None):
        for ema_w, model_w in zip(
            self.ema_weights, 
            self.model.get_weights()
        ):
            ema_w.assign(
                self.decay * ema_w + (1 - self.decay) * model_w
            )
            
    def apply_ema_weights(self):
        """Apply EMA weights to model."""
        self.original_weights = self.model.get_weights()
        self.model.set_weights([w.numpy() for w in self.ema_weights])
        
    def restore_original_weights(self):
        """Restore original weights."""
        self.model.set_weights(self.original_weights)


class GradientNormCallback(callbacks.Callback):
    """
    Monitor gradient norms during training.
    
    Useful for debugging training dynamics and detecting
    exploding/vanishing gradients.
    """
    
    def __init__(self, log_frequency: int = 100):
        super().__init__()
        self.log_frequency = log_frequency
        self.step = 0
        self.gradient_norms = []
        
    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        
        if self.step % self.log_frequency == 0:
            # Get gradient norms from optimizer
            if hasattr(self.model.optimizer, '_gradients'):
                norms = [
                    tf.norm(g).numpy() 
                    for g in self.model.optimizer._gradients
                    if g is not None
                ]
                avg_norm = np.mean(norms) if norms else 0
                self.gradient_norms.append(avg_norm)
                
                if avg_norm > 10:
                    print(f'\nWarning: High gradient norm: {avg_norm:.2f}')
