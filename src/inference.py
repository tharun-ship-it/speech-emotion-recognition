"""
Inference Pipeline for Speech Emotion Recognition

Provides high-level API for emotion prediction from audio files
or streams, with support for batch processing and real-time analysis.

Author: Tharun Ponnam
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import json
import time
import librosa

from src.data.preprocessing import AudioFeatureExtractor
from src.models.ser_net import build_ser_net


# Emotion labels
EMOTION_LABELS = [
    'angry', 'happy', 'sad', 'neutral',
    'fear', 'disgust', 'surprise', 'contempt'
]


class SERPredictor:
    """
    Speech Emotion Recognition Predictor.
    
    High-level interface for emotion prediction from audio.
    
    Args:
        model_path: Path to saved model weights
        config_path: Path to model configuration
        norm_stats_path: Path to normalization statistics
        device: Device to use ('cpu' or 'gpu')
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.model_path = Path(model_path)
        self.device = device
        
        # Set device
        if device == 'cpu':
            tf.config.set_visible_devices([], 'GPU')
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize feature extractor
        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=self.config.get('sample_rate', 16000),
            n_mfcc=self.config.get('n_mfcc', 40),
            n_mels=self.config.get('n_mels', 128),
            hop_length=self.config.get('hop_length', 512)
        )
        
        # Load normalization stats
        self.norm_mean = None
        self.norm_std = None
        if norm_stats_path:
            self._load_norm_stats(norm_stats_path)
        
        # Build and load model
        self.model = self._load_model()
        
        # Sequence length
        self.max_length = self.config.get('max_length', 300)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load model configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_norm_stats(self, path: str):
        """Load normalization statistics."""
        with open(path, 'r') as f:
            stats = json.load(f)
        self.norm_mean = np.array(stats['mean'])
        self.norm_std = np.array(stats['std'])
        
    def _load_model(self) -> tf.keras.Model:
        """Build and load model weights."""
        input_shape = (
            self.max_length,
            self.feature_extractor.total_dim
        )
        
        model = build_ser_net(
            input_shape=input_shape,
            num_classes=len(EMOTION_LABELS),
            variant=self.config.get('variant', 'base')
        )
        
        model.load_weights(str(self.model_path))
        
        return model
    
    def _preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Preprocess audio for prediction.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate of input audio
            
        Returns:
            Preprocessed feature tensor
        """
        # Resample if needed
        target_sr = self.config.get('sample_rate', 16000)
        if sample_rate != target_sr:
            audio = librosa.resample(
                audio, 
                orig_sr=sample_rate, 
                target_sr=target_sr
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
    
    def predict(
        self,
        audio_input: Union[str, np.ndarray],
        sample_rate: int = 16000,
        return_probs: bool = True
    ) -> Dict:
        """
        Predict emotion from audio.
        
        Args:
            audio_input: Path to audio file or audio array
            sample_rate: Sample rate (if array input)
            return_probs: Whether to return all class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Load audio if path
        if isinstance(audio_input, (str, Path)):
            audio, sr = librosa.load(
                audio_input, 
                sr=self.config.get('sample_rate', 16000)
            )
            sample_rate = sr
        else:
            audio = audio_input
        
        # Preprocess
        features = self._preprocess(audio, sample_rate)
        
        # Predict
        predictions = self.model(features, training=False)
        probs = predictions.numpy()[0]
        
        # Get top prediction
        top_idx = np.argmax(probs)
        top_emotion = EMOTION_LABELS[top_idx]
        confidence = float(probs[top_idx])
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            'emotion': top_emotion,
            'confidence': confidence,
            'processing_time_ms': processing_time
        }
        
        if return_probs:
            result['probabilities'] = {
                emotion: float(prob)
                for emotion, prob in zip(EMOTION_LABELS, probs)
            }
        
        return result
    
    def predict_batch(
        self,
        audio_paths: List[str],
        return_probs: bool = True
    ) -> List[Dict]:
        """
        Predict emotions for multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            return_probs: Whether to return probabilities
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for path in audio_paths:
            try:
                result = self.predict(
                    path, 
                    return_probs=return_probs
                )
                result['file'] = str(path)
                result['status'] = 'success'
            except Exception as e:
                result = {
                    'file': str(path),
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        return results
    
    def get_embeddings(
        self,
        audio_input: Union[str, np.ndarray],
        sample_rate: int = 16000,
        layer: str = 'attention'
    ) -> np.ndarray:
        """
        Extract feature embeddings from audio.
        
        Useful for visualization, clustering, or downstream tasks.
        
        Args:
            audio_input: Audio file path or array
            sample_rate: Sample rate
            layer: Which layer to extract from
            
        Returns:
            Feature embedding array
        """
        # Load audio if path
        if isinstance(audio_input, (str, Path)):
            audio, sr = librosa.load(
                audio_input,
                sr=self.config.get('sample_rate', 16000)
            )
            sample_rate = sr
        else:
            audio = audio_input
        
        # Preprocess
        features = self._preprocess(audio, sample_rate)
        
        # Get embeddings
        embeddings = self.model.get_embeddings(features, layer=layer)
        
        return embeddings.numpy()[0]
    
    def get_frame_importance(
        self,
        audio_input: Union[str, np.ndarray],
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get per-frame importance scores for interpretability.
        
        Args:
            audio_input: Audio file path or array
            sample_rate: Sample rate
            
        Returns:
            Tuple of (time_axis, importance_scores)
        """
        # Load audio if path
        if isinstance(audio_input, (str, Path)):
            audio, sr = librosa.load(
                audio_input,
                sr=self.config.get('sample_rate', 16000)
            )
            sample_rate = sr
        else:
            audio = audio_input
        
        # Preprocess
        features = self._preprocess(audio, sample_rate)
        
        # Get importance scores from attention
        importance = self.model.attention_pool.get_frame_importance(
            self.model.temporal_blocks[-1](
                self.model.conv_stem(features)
            )
        )
        
        importance = importance.numpy()[0]
        
        # Create time axis
        hop_length = self.config.get('hop_length', 512)
        target_sr = self.config.get('sample_rate', 16000)
        time_axis = np.arange(len(importance)) * hop_length / target_sr
        
        return time_axis, importance


class RealTimePredictor:
    """
    Real-time emotion prediction from audio stream.
    
    Maintains a sliding window buffer for continuous
    emotion analysis with minimal latency.
    
    Args:
        predictor: SERPredictor instance
        window_size: Analysis window size in seconds
        hop_size: Hop between windows in seconds
    """
    
    def __init__(
        self,
        predictor: SERPredictor,
        window_size: float = 3.0,
        hop_size: float = 0.5
    ):
        self.predictor = predictor
        self.window_size = window_size
        self.hop_size = hop_size
        
        self.sample_rate = predictor.config.get('sample_rate', 16000)
        self.window_samples = int(window_size * self.sample_rate)
        self.hop_samples = int(hop_size * self.sample_rate)
        
        # Audio buffer
        self.buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.buffer_position = 0
        
        # Prediction history
        self.history = []
        
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """
        Process incoming audio chunk.
        
        Args:
            audio_chunk: New audio samples
            
        Returns:
            Prediction result if window is full, else None
        """
        chunk_len = len(audio_chunk)
        
        # Add to buffer
        if self.buffer_position + chunk_len < self.window_samples:
            self.buffer[
                self.buffer_position:self.buffer_position + chunk_len
            ] = audio_chunk
            self.buffer_position += chunk_len
            return None
        
        # Buffer full, make prediction
        result = self.predictor.predict(
            self.buffer,
            sample_rate=self.sample_rate
        )
        
        # Slide buffer
        self.buffer = np.roll(self.buffer, -self.hop_samples)
        self.buffer_position = self.window_samples - self.hop_samples
        
        # Add remaining chunk
        remaining = audio_chunk[:(chunk_len - self.hop_samples)]
        self.buffer[-len(remaining):] = remaining
        
        # Store in history
        self.history.append(result)
        
        return result
    
    def get_dominant_emotion(
        self,
        num_windows: int = 5
    ) -> str:
        """
        Get dominant emotion over recent windows.
        
        Args:
            num_windows: Number of recent windows to consider
            
        Returns:
            Most frequent emotion
        """
        if not self.history:
            return 'unknown'
        
        recent = self.history[-num_windows:]
        emotions = [r['emotion'] for r in recent]
        
        # Get mode
        from collections import Counter
        return Counter(emotions).most_common(1)[0][0]
    
    def reset(self):
        """Reset buffer and history."""
        self.buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.buffer_position = 0
        self.history = []


def load_predictor(
    model_dir: str,
    device: str = 'cpu'
) -> SERPredictor:
    """
    Convenience function to load predictor from model directory.
    
    Expects directory structure:
        model_dir/
            best_model.h5
            config.json (optional)
            norm_stats.json (optional)
    
    Args:
        model_dir: Path to model directory
        device: Device to use
        
    Returns:
        Configured SERPredictor
    """
    model_dir = Path(model_dir)
    
    model_path = model_dir / 'best_model.h5'
    config_path = model_dir / 'config.json'
    norm_stats_path = model_dir / 'norm_stats.json'
    
    return SERPredictor(
        model_path=str(model_path),
        config_path=str(config_path) if config_path.exists() else None,
        norm_stats_path=str(norm_stats_path) if norm_stats_path.exists() else None,
        device=device
    )


if __name__ == '__main__':
    print("Inference module loaded successfully!")
    print(f"Supported emotions: {EMOTION_LABELS}")
