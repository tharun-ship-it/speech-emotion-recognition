#!/usr/bin/env python3
"""
Training Script for Speech Emotion Recognition

Trains SER-Net on the MSP-Podcast corpus with configurable
hyperparameters and logging.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --data_dir /path/to/data --epochs 100

Author: Tharun Ponnam
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import yaml
import tensorflow as tf
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_ser_model
from src.data import SERDataset, AudioFeatureExtractor
from src.training import SERTrainer, FocalLoss, get_class_weights
from src.training.callbacks import WarmupCosineDecay, MetricsLogger


def setup_logging(log_dir: str) -> logging.Logger:
    """Configure logging for training."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def setup_gpu():
    """Configure GPU memory growth."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Speech Emotion Recognition model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/train_config.yaml',
        help='Path to configuration file'
    )
    
    # Data paths
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        help='Path to MSP-Podcast data directory'
    )
    parser.add_argument(
        '--labels_file',
        type=str,
        help='Path to labels CSV file'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate', '-lr',
        type=float,
        help='Initial learning rate'
    )
    
    # Model parameters
    parser.add_argument(
        '--model_variant',
        type=str,
        choices=['base', 'large'],
        help='Model variant to use'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        help='Number of emotion classes'
    )
    
    # Output paths
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='outputs',
        help='Directory for outputs (checkpoints, logs)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='ser_experiment',
        help='Name for this experiment'
    )
    
    # Misc
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        help='Use mixed precision training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config and merge with command line args
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['audio_dir'] = args.data_dir
    if args.labels_file:
        config['data']['labels_file'] = args.labels_file
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.model_variant:
        config['model']['variant'] = args.model_variant
    if args.num_classes:
        config['model']['num_classes'] = args.num_classes
    
    # Setup
    setup_gpu()
    set_seed(args.seed)
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(
        args.output_dir,
        f"{args.experiment_name}_{timestamp}"
    )
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting training experiment: {args.experiment_name}")
    logger.info(f"Output directory: {experiment_dir}")
    
    # Save config
    config_save_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to {config_save_path}")
    
    # Mixed precision
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("Mixed precision training enabled")
    
    # Create feature extractor
    feature_config = config.get('features', {})
    feature_extractor = AudioFeatureExtractor(
        sample_rate=feature_config.get('sample_rate', 16000),
        n_mfcc=feature_config.get('n_mfcc', 40),
        n_mels=feature_config.get('n_mels', 128),
        hop_length=feature_config.get('hop_length', 512)
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    data_config = config['data']
    
    train_dataset = SERDataset(
        audio_dir=data_config['audio_dir'],
        labels_file=data_config['labels_file'],
        split='train',
        feature_extractor=feature_extractor,
        augment=True
    )
    
    val_dataset = SERDataset(
        audio_dir=data_config['audio_dir'],
        labels_file=data_config['labels_file'],
        split='val',
        feature_extractor=feature_extractor,
        augment=False
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create TF datasets
    train_config = config['training']
    batch_size = train_config['batch_size']
    
    train_tf_dataset = train_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle=True
    )
    val_tf_dataset = val_dataset.to_tf_dataset(
        batch_size=batch_size,
        shuffle=False
    )
    
    # Compute class weights
    train_labels = train_dataset.get_labels()
    num_classes = config['model']['num_classes']
    class_weights = get_class_weights(train_labels, num_classes)
    logger.info(f"Class weights: {class_weights}")
    
    # Create model
    model_config = config['model']
    input_shape = (
        feature_config.get('max_length', 300),
        feature_config.get('feature_dim', 180)
    )
    
    model = create_ser_model(
        input_shape=input_shape,
        num_classes=num_classes,
        variant=model_config.get('variant', 'base')
    )
    
    logger.info(f"Model created with {model.count_params():,} parameters")
    
    # Create trainer
    trainer = SERTrainer(
        model=model,
        train_dataset=train_tf_dataset,
        val_dataset=val_tf_dataset,
        config=train_config,
        class_weights=class_weights,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    history = trainer.train()
    
    # Save final model
    final_model_path = os.path.join(experiment_dir, 'final_model')
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Log final metrics
    logger.info("Training completed!")
    logger.info(f"Best validation UAR: {trainer.best_val_uar:.4f}")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    return history


if __name__ == '__main__':
    main()
