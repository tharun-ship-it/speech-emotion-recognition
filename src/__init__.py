"""
Speech Emotion Recognition Package

A production-ready framework for speech emotion recognition using
deep learning with TensorFlow/Keras and Librosa.

Author: Tharun Ponnam
"""

__version__ = '1.0.0'
__author__ = 'Tharun Ponnam'
__email__ = 'tharunponnam007@gmail.com'

from src.inference import SERPredictor, RealTimePredictor

__all__ = [
    'SERPredictor',
    'RealTimePredictor',
    '__version__',
]
