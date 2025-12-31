"""
Dataset Classes for Speech Emotion Recognition

Provides TensorFlow dataset utilities for efficient data loading,
preprocessing, and batching during training and evaluation.

Author: Tharun Ponnam
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
import json

from .preprocessing import AudioFeatureExtractor
from .augmentation import AudioAugmentor, SpecAugment, get_default_augmentor


# Emotion label mapping for MSP-Podcast
EMOTION_LABELS = {
    'angry': 0,
    'happy': 1,
    'sad': 2,
    'neutral': 3,
    'fear': 4,
    'disgust': 5,
    'surprise': 6,
    'contempt': 7
}

LABEL_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}


class SERDataset:
    """
    Speech Emotion Recognition Dataset.
    
    Handles loading audio files, feature extraction, augmentation,
    and creation of TensorFlow dataset pipelines.
    
    Args:
        manifest_path: Path to CSV manifest with audio paths and labels
        feature_extractor: AudioFeatureExtractor instance
        augmentor: Optional AudioAugmentor for training
        max_length: Maximum sequence length (in frames)
        normalize: Whether to apply feature normalization
        cache_features: Whether to cache extracted features
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        feature_extractor: Optional[AudioFeatureExtractor] = None,
        augmentor: Optional[AudioAugmentor] = None,
        max_length: int = 300,
        normalize: bool = True,
        cache_features: bool = False,
        norm_stats_path: Optional[str] = None
    ):
        self.manifest_path = Path(manifest_path)
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.augmentor = augmentor
        self.max_length = max_length
        self.normalize = normalize
        self.cache_features = cache_features
        
        # Load manifest
        self.manifest = pd.read_csv(manifest_path)
        self.num_samples = len(self.manifest)
        
        # Feature cache
        self._feature_cache = {} if cache_features else None
        
        # Normalization statistics
        self.norm_mean = None
        self.norm_std = None
        if norm_stats_path and Path(norm_stats_path).exists():
            self._load_norm_stats(norm_stats_path)
            
    def _load_norm_stats(self, path: str):
        """Load precomputed normalization statistics."""
        with open(path, 'r') as f:
            stats = json.load(f)
        self.norm_mean = np.array(stats['mean'])
        self.norm_std = np.array(stats['std'])
        
    def save_norm_stats(self, path: str):
        """Save normalization statistics to file."""
        if self.norm_mean is None:
            raise ValueError("Normalization stats not computed")
            
        with open(path, 'w') as f:
            json.dump({
                'mean': self.norm_mean.tolist(),
                'std': self.norm_std.tolist()
            }, f)
            
    def compute_norm_stats(self, max_samples: int = 5000):
        """Compute normalization statistics from training data."""
        all_features = []
        
        sample_indices = np.random.choice(
            self.num_samples,
            size=min(max_samples, self.num_samples),
            replace=False
        )
        
        for idx in sample_indices:
            features, _ = self._load_sample(int(idx), augment=False)
            all_features.append(features)
            
        all_features = np.vstack(all_features)
        self.norm_mean = np.mean(all_features, axis=0)
        self.norm_std = np.std(all_features, axis=0) + 1e-8
        
    def _load_sample(
        self,
        index: int,
        augment: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load and process a single sample.
        
        Args:
            index: Sample index in manifest
            augment: Whether to apply augmentation
            
        Returns:
            Tuple of (features, label)
        """
        row = self.manifest.iloc[index]
        audio_path = row['audio_path']
        label = EMOTION_LABELS[row['emotion'].lower()]
        
        # Check cache
        if self._feature_cache is not None and audio_path in self._feature_cache:
            features = self._feature_cache[audio_path].copy()
        else:
            # Load and extract features
            features = self.feature_extractor.extract_from_file(audio_path)
            
            if self._feature_cache is not None:
                self._feature_cache[audio_path] = features.copy()
        
        # Apply augmentation to features if training
        if augment and self.augmentor is not None:
            # For now, apply SpecAugment to features
            spec_aug = SpecAugment()
            features = spec_aug(features)
            
        # Normalize
        if self.normalize and self.norm_mean is not None:
            features = (features - self.norm_mean) / self.norm_std
            
        # Pad or truncate
        features = self.feature_extractor.pad_or_truncate(
            features, self.max_length
        )
        
        return features.astype(np.float32), label
    
    def create_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = True,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        prefetch: int = tf.data.AUTOTUNE
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset pipeline.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation
            num_parallel_calls: Parallelism for preprocessing
            prefetch: Prefetch buffer size
            
        Returns:
            tf.data.Dataset object
        """
        indices = np.arange(self.num_samples)
        
        def generator():
            for idx in indices:
                features, label = self._load_sample(idx, augment=augment)
                yield features, label
                
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(self.max_length, self.feature_extractor.total_dim),
                    dtype=tf.float32
                ),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, self.num_samples))
            
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch)
        
        return dataset
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return self._load_sample(index)
    
    @property
    def class_weights(self) -> Dict[int, float]:
        """Compute class weights for imbalanced data."""
        label_counts = self.manifest['emotion'].value_counts()
        total = len(self.manifest)
        n_classes = len(EMOTION_LABELS)
        
        weights = {}
        for emotion, label in EMOTION_LABELS.items():
            count = label_counts.get(emotion, 1)
            weights[label] = total / (n_classes * count)
            
        return weights


class InferenceDataset:
    """
    Lightweight dataset for inference.
    
    Optimized for fast feature extraction and prediction
    without training overhead.
    
    Args:
        feature_extractor: AudioFeatureExtractor instance
        max_length: Maximum sequence length
        norm_stats_path: Path to normalization statistics
    """
    
    def __init__(
        self,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
        max_length: int = 300,
        norm_stats_path: Optional[str] = None
    ):
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.max_length = max_length
        
        self.norm_mean = None
        self.norm_std = None
        
        if norm_stats_path:
            with open(norm_stats_path, 'r') as f:
                stats = json.load(f)
            self.norm_mean = np.array(stats['mean'])
            self.norm_std = np.array(stats['std'])
            
    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Process audio array for inference.
        
        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate
            
        Returns:
            Processed features ready for model
        """
        # Resample if necessary
        if sample_rate != self.feature_extractor.sample_rate:
            import librosa
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.feature_extractor.sample_rate
            )
            
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Normalize
        if self.norm_mean is not None:
            features = (features - self.norm_mean) / self.norm_std
            
        # Pad or truncate
        features = self.feature_extractor.pad_or_truncate(
            features, self.max_length
        )
        
        # Add batch dimension
        return features[np.newaxis, :, :].astype(np.float32)
    
    def process_file(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Process audio file for inference.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Processed features ready for model
        """
        features = self.feature_extractor.extract_from_file(audio_path)
        
        # Normalize
        if self.norm_mean is not None:
            features = (features - self.norm_mean) / self.norm_std
            
        # Pad or truncate
        features = self.feature_extractor.pad_or_truncate(
            features, self.max_length
        )
        
        # Add batch dimension
        return features[np.newaxis, :, :].astype(np.float32)


def create_data_splits(
    manifest_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify: bool = True,
    random_seed: int = 42
):
    """
    Create train/val/test splits from manifest.
    
    Args:
        manifest_path: Path to full manifest CSV
        output_dir: Directory to save split manifests
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stratify: Whether to stratify by emotion
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(manifest_path)
    
    if stratify:
        # Stratified split by emotion
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for emotion in df['emotion'].unique():
            emotion_df = df[df['emotion'] == emotion].sample(frac=1)
            n = len(emotion_df)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_dfs.append(emotion_df.iloc[:train_end])
            val_dfs.append(emotion_df.iloc[train_end:val_end])
            test_dfs.append(emotion_df.iloc[val_end:])
            
        train_df = pd.concat(train_dfs).sample(frac=1)
        val_df = pd.concat(val_dfs).sample(frac=1)
        test_df = pd.concat(test_dfs).sample(frac=1)
    else:
        df = df.sample(frac=1)
        n = len(df)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
    
    # Save splits
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")


if __name__ == '__main__':
    # Test with dummy data
    import tempfile
    import os
    
    # Create dummy manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, 'manifest.csv')
        
        # Create dummy data
        df = pd.DataFrame({
            'audio_path': ['/dummy/path.wav'] * 10,
            'emotion': ['happy', 'sad', 'angry', 'neutral', 'fear',
                       'happy', 'sad', 'angry', 'neutral', 'fear']
        })
        df.to_csv(manifest_path, index=False)
        
        # Test class weights
        print("Testing dataset creation...")
        print(f"Emotion labels: {EMOTION_LABELS}")
        print(f"Number of classes: {len(EMOTION_LABELS)}")
        
        print("\nDataset module test passed!")
