#!/usr/bin/env python3
"""
Evaluation Script for Speech Emotion Recognition

Evaluates trained models on test set with comprehensive metrics
and generates analysis reports.

Usage:
    python scripts/evaluate.py --model_path outputs/best_model --data_dir /path/to/data
    python scripts/evaluate.py --config configs/eval_config.yaml

Author: Tharun Ponnam
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_ser_model
from src.data import SERDataset, AudioFeatureExtractor
from src.inference import SERPredictor


EMOTION_LABELS = [
    'Angry', 'Happy', 'Sad', 'Neutral',
    'Fear', 'Disgust', 'Surprise', 'Contempt'
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate Speech Emotion Recognition model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model_path', '-m',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        required=True,
        help='Path to test data directory'
    )
    parser.add_argument(
        '--labels_file',
        type=str,
        required=True,
        help='Path to labels CSV file'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='eval_results',
        help='Directory for evaluation outputs'
    )
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=8,
        help='Number of emotion classes'
    )
    
    return parser.parse_args()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    # Unweighted Average Recall (UAR)
    uar = balanced_accuracy_score(y_true, y_pred)
    
    # Weighted Average Recall (WAR)
    war = np.mean(y_true == y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes)
    )
    
    # Macro and weighted F1
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    metrics = {
        'uar': float(uar),
        'war': float(war),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    save_path: str,
    normalize: bool = True
):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        save_path: Path to save figure
        normalize: Whether to normalize values
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(
    metrics: Dict,
    labels: List[str],
    save_path: str
):
    """
    Plot per-class precision, recall, and F1 scores.
    
    Args:
        metrics: Dictionary containing per-class metrics
        labels: Class labels
        save_path: Path to save figure
    """
    per_class = metrics['per_class']
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, per_class['precision'], width, label='Precision')
    ax.bar(x, per_class['recall'], width, label='Recall')
    ax.bar(x + width, per_class['f1'], width, label='F1-Score')
    
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(
    metrics: Dict,
    labels: List[str],
    output_path: str
):
    """
    Generate markdown evaluation report.
    
    Args:
        metrics: Evaluation metrics
        labels: Class labels
        output_path: Path to save report
    """
    report = f"""# Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Metrics

| Metric | Value |
|--------|-------|
| Unweighted Average Recall (UAR) | {metrics['uar']:.4f} |
| Weighted Average Recall (WAR) | {metrics['war']:.4f} |
| Macro F1-Score | {metrics['macro_f1']:.4f} |
| Weighted F1-Score | {metrics['weighted_f1']:.4f} |

## Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
"""
    
    per_class = metrics['per_class']
    for i, label in enumerate(labels):
        report += f"| {label} | {per_class['precision'][i]:.4f} | "
        report += f"{per_class['recall'][i]:.4f} | "
        report += f"{per_class['f1'][i]:.4f} | "
        report += f"{per_class['support'][i]} |\n"
    
    report += """
## Figures

- Confusion Matrix: `confusion_matrix.png`
- Per-Class Metrics: `per_class_metrics.png`

## Analysis

"""
    
    # Find best and worst performing classes
    recalls = per_class['recall']
    best_idx = np.argmax(recalls)
    worst_idx = np.argmin(recalls)
    
    report += f"- **Best performing emotion**: {labels[best_idx]} "
    report += f"(Recall: {recalls[best_idx]:.4f})\n"
    report += f"- **Most challenging emotion**: {labels[worst_idx]} "
    report += f"(Recall: {recalls[worst_idx]:.4f})\n"
    
    with open(output_path, 'w') as f:
        f.write(report)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'eval_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {args.model_path}")
    
    # Initialize predictor
    predictor = SERPredictor(
        model_path=args.model_path,
        num_classes=args.num_classes
    )
    
    # Load test dataset
    print("Loading test dataset...")
    feature_extractor = AudioFeatureExtractor()
    
    test_dataset = SERDataset(
        audio_dir=args.data_dir,
        labels_file=args.labels_file,
        split='test',
        feature_extractor=feature_extractor,
        augment=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Get predictions
    print("Running inference...")
    all_preds = []
    all_labels = []
    
    test_tf_dataset = test_dataset.to_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False
    )
    
    for features, labels in test_tf_dataset:
        predictions = predictor.model.predict(features, verbose=0)
        pred_labels = np.argmax(predictions, axis=-1)
        
        all_preds.extend(pred_labels.tolist())
        all_labels.extend(labels.numpy().tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    print("Computing metrics...")
    labels = EMOTION_LABELS[:args.num_classes]
    metrics = compute_metrics(all_labels, all_preds, args.num_classes)
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"UAR:        {metrics['uar']:.4f}")
    print(f"WAR:        {metrics['war']:.4f}")
    print(f"Macro F1:   {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("=" * 50)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Generate plots
    cm = np.array(metrics['confusion_matrix'])
    
    plot_confusion_matrix(
        cm, labels,
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    print("Confusion matrix saved")
    
    plot_per_class_metrics(
        metrics, labels,
        os.path.join(output_dir, 'per_class_metrics.png')
    )
    print("Per-class metrics plot saved")
    
    # Generate report
    generate_report(
        metrics, labels,
        os.path.join(output_dir, 'report.md')
    )
    print(f"Report saved to {output_dir}/report.md")
    
    return metrics


if __name__ == '__main__':
    main()
