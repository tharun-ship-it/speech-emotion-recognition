"""
Training Pipeline for SER-Net

Implements training loop, callbacks, and optimization strategies
for speech emotion recognition.

Author: Tharun Ponnam
"""

import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.mixed_precision import set_global_policy
from typing import Optional, Dict, Tuple, List
import numpy as np
from pathlib import Path
import json
import time

from .losses import focal_loss, weighted_cross_entropy
from .callbacks import (
    WarmupCosineDecay,
    GradientAccumulator,
    MetricsLogger
)


class SERTrainer:
    """
    Trainer class for Speech Emotion Recognition.
    
    Handles training loop, validation, checkpointing, and
    learning rate scheduling.
    
    Args:
        model: SER-Net model instance
        train_dataset: Training tf.data.Dataset
        val_dataset: Validation tf.data.Dataset
        config: Training configuration dictionary
        output_dir: Directory for checkpoints and logs
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        config: Dict,
        output_dir: str = 'outputs'
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract config values
        self.epochs = config.get('epochs', 100)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.use_focal_loss = config.get('use_focal_loss', True)
        self.focal_gamma = config.get('focal_gamma', 2.0)
        self.class_weights = config.get('class_weights', None)
        self.mixed_precision = config.get('mixed_precision', False)
        
        # Setup training
        self._setup_training()
        
    def _setup_training(self):
        """Initialize optimizer, loss, and metrics."""
        # Mixed precision
        if self.mixed_precision:
            set_global_policy('mixed_float16')
            
        # Optimizer with weight decay
        self.optimizer = optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Loss function
        if self.use_focal_loss:
            self.loss_fn = lambda y, p: focal_loss(
                y, p, 
                gamma=self.focal_gamma
            )
        else:
            if self.class_weights:
                self.loss_fn = lambda y, p: weighted_cross_entropy(
                    y, p, 
                    weights=self.class_weights
                )
            else:
                self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False
                )
        
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy'
        )
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='val_accuracy'
        )
        
        # Learning rate schedule
        self.lr_schedule = WarmupCosineDecay(
            initial_lr=self.learning_rate,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.epochs
        )
        
    def _get_callbacks(self) -> List[callbacks.Callback]:
        """Create training callbacks."""
        callback_list = [
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / 'best_model.h5'),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                mode='min',
                restore_best_weights=True,
                verbose=1
            ),
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=str(self.output_dir / 'logs'),
                histogram_freq=1,
                update_freq='epoch'
            ),
            # CSV logging
            callbacks.CSVLogger(
                str(self.output_dir / 'training_log.csv'),
                append=True
            ),
            # Reduce LR on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callback_list
    
    @tf.function
    def _train_step(
        self, 
        features: tf.Tensor, 
        labels: tf.Tensor
    ) -> tf.Tensor:
        """Execute single training step."""
        with tf.GradientTape() as tape:
            predictions = self.model(features, training=True)
            loss = self.loss_fn(labels, predictions)
            
            # Add regularization loss
            if self.model.losses:
                loss += tf.add_n(self.model.losses)
                
            # Scale loss for mixed precision
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)
                
        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
            
        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        
        # Update metrics
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)
        
        return loss
    
    @tf.function
    def _val_step(
        self, 
        features: tf.Tensor, 
        labels: tf.Tensor
    ) -> tf.Tensor:
        """Execute single validation step."""
        predictions = self.model(features, training=False)
        loss = self.loss_fn(labels, predictions)
        
        self.val_loss.update_state(loss)
        self.val_accuracy.update_state(labels, predictions)
        
        return loss
    
    def train(self) -> Dict:
        """
        Execute full training loop.
        
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\nStarting training for {self.epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Update learning rate
            lr = self.lr_schedule(epoch)
            self.optimizer.learning_rate.assign(lr)
            
            # Reset metrics
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            self.val_loss.reset_state()
            self.val_accuracy.reset_state()
            
            # Training loop
            for step, (features, labels) in enumerate(self.train_dataset):
                self._train_step(features, labels)
                
                if step % 50 == 0:
                    print(
                        f"\rEpoch {epoch + 1}/{self.epochs} - "
                        f"Step {step} - "
                        f"Loss: {self.train_loss.result():.4f} - "
                        f"Acc: {self.train_accuracy.result():.4f}",
                        end=""
                    )
            
            # Validation loop
            for features, labels in self.val_dataset:
                self._val_step(features, labels)
            
            # Get metrics
            train_loss = self.train_loss.result().numpy()
            train_acc = self.train_accuracy.result().numpy()
            val_loss = self.val_loss.result().numpy()
            val_acc = self.val_accuracy.result().numpy()
            
            # Update history
            history['train_loss'].append(float(train_loss))
            history['train_accuracy'].append(float(train_acc))
            history['val_loss'].append(float(val_loss))
            history['val_accuracy'].append(float(val_acc))
            history['learning_rate'].append(float(lr))
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(
                f"\rEpoch {epoch + 1}/{self.epochs} - "
                f"{epoch_time:.1f}s - "
                f"loss: {train_loss:.4f} - "
                f"acc: {train_acc:.4f} - "
                f"val_loss: {val_loss:.4f} - "
                f"val_acc: {val_acc:.4f} - "
                f"lr: {lr:.2e}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model.save_weights(
                    str(self.output_dir / 'best_model.h5')
                )
                patience_counter = 0
                print(f"  ✓ New best model saved (val_acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                
            # Early stopping check
            if patience_counter >= 15:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
                
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.model.save_weights(
                    str(self.output_dir / f'checkpoint_epoch_{epoch + 1}.h5')
                )
        
        # Save final model
        self.model.save_weights(str(self.output_dir / 'final_model.h5'))
        
        # Save training history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
            
        print("\n" + "=" * 60)
        print(f"Training complete!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Models saved to: {self.output_dir}")
        
        return history
    
    def evaluate(
        self, 
        test_dataset: tf.data.Dataset
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Returns:
            Dictionary with evaluation metrics
        """
        test_loss = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
        all_predictions = []
        all_labels = []
        
        for features, labels in test_dataset:
            predictions = self.model(features, training=False)
            loss = self.loss_fn(labels, predictions)
            
            test_loss.update_state(loss)
            test_accuracy.update_state(labels, predictions)
            
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        pred_classes = np.argmax(all_predictions, axis=-1)
        
        # Compute additional metrics
        from sklearn.metrics import (
            classification_report, 
            confusion_matrix,
            f1_score
        )
        
        # Unweighted Average Recall
        uar = np.mean([
            np.sum((all_labels == c) & (pred_classes == c)) / 
            np.sum(all_labels == c)
            for c in np.unique(all_labels)
        ])
        
        results = {
            'test_loss': float(test_loss.result()),
            'test_accuracy': float(test_accuracy.result()),
            'uar': float(uar),
            'macro_f1': float(f1_score(all_labels, pred_classes, average='macro'))
        }
        
        print("\n" + "=" * 60)
        print("Evaluation Results:")
        print(f"  Loss: {results['test_loss']:.4f}")
        print(f"  Accuracy: {results['test_accuracy']:.4f}")
        print(f"  UAR: {results['uar']:.4f}")
        print(f"  Macro-F1: {results['macro_f1']:.4f}")
        
        return results


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    config: Dict,
    output_dir: str = 'outputs'
) -> Tuple[tf.keras.Model, Dict]:
    """
    Convenience function for training.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        output_dir: Output directory
        
    Returns:
        Trained model and history
    """
    trainer = SERTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        output_dir=output_dir
    )
    
    history = trainer.train()
    
    return model, history
