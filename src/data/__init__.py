"""
Data processing utilities for Speech Emotion Recognition.

Includes audio preprocessing, feature extraction, data augmentation,
and dataset loading pipelines.
"""

from src.data.preprocessing import AudioFeatureExtractor, AudioPreprocessor
from src.data.augmentation import (
    AudioAugmentor,
    SpecAugment,
    RoomSimulator,
    MixupAugmentation
)
from src.data.dataset import (
    SERDataset,
    InferenceDataset,
    create_tf_dataset,
    load_msp_podcast
)

__all__ = [
    'AudioFeatureExtractor',
    'AudioPreprocessor',
    'AudioAugmentor',
    'SpecAugment',
    'RoomSimulator',
    'MixupAugmentation',
    'SERDataset',
    'InferenceDataset',
    'create_tf_dataset',
    'load_msp_podcast',
]
