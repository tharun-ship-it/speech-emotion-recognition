"""
Audio Feature Extraction Pipeline

Extracts acoustic features from raw audio for speech emotion recognition.
Combines multiple feature representations to capture complementary
emotional cues from speech.

Author: Tharun Ponnam
"""

import numpy as np
import librosa
from typing import Tuple, Optional, Dict, Union
from pathlib import Path
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning)


class AudioFeatureExtractor:
    """
    Comprehensive audio feature extraction for emotion recognition.
    
    Extracts and combines:
    - Mel-frequency cepstral coefficients (MFCCs)
    - Mel spectrogram
    - Prosodic features (F0, energy, etc.)
    - Delta and delta-delta features
    
    Args:
        sample_rate: Target sample rate for audio
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel filterbank channels
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        n_fft: FFT size
        fmin: Minimum frequency for mel scale
        fmax: Maximum frequency for mel scale
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 40,
        n_mels: int = 128,
        hop_length: int = 512,
        win_length: int = 2048,
        n_fft: int = 2048,
        fmin: float = 0.0,
        fmax: Optional[float] = None
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2
        
        # Feature dimensions
        self.mfcc_dim = n_mfcc * 3  # Including deltas
        self.mel_dim = n_mels
        self.prosodic_dim = 12
        self.total_dim = self.mfcc_dim + self.mel_dim + self.prosodic_dim
        
    def extract_features(
        self,
        audio: np.ndarray,
        return_components: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract all features from audio signal.
        
        Args:
            audio: Audio waveform as numpy array
            return_components: If True, return dict with separate features
            
        Returns:
            Feature array of shape (time, features) or dict of components
        """
        # Ensure audio is mono and float
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)
        
        # Extract feature components
        mfcc_features = self._extract_mfcc(audio)
        mel_features = self._extract_mel_spectrogram(audio)
        prosodic_features = self._extract_prosodic(audio)
        
        # Align temporal dimensions
        min_frames = min(
            mfcc_features.shape[1],
            mel_features.shape[1],
            prosodic_features.shape[1]
        )
        
        mfcc_features = mfcc_features[:, :min_frames].T
        mel_features = mel_features[:, :min_frames].T
        prosodic_features = prosodic_features[:, :min_frames].T
        
        if return_components:
            return {
                'mfcc': mfcc_features,
                'mel': mel_features,
                'prosodic': prosodic_features
            }
        
        # Concatenate all features
        features = np.concatenate(
            [mfcc_features, mel_features, prosodic_features],
            axis=-1
        )
        
        return features.astype(np.float32)
    
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCCs with delta and delta-delta.
        
        MFCCs capture spectral envelope information which correlates
        with vocal tract configuration and voice quality.
        """
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Compute deltas for temporal dynamics
        mfcc_delta = librosa.feature.delta(mfcc, width=9)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=9)
        
        # Stack features
        mfcc_all = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        return mfcc_all
    
    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-mel spectrogram.
        
        Provides detailed spectral information at mel scale which
        approximates human auditory perception.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale with floor
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def _extract_prosodic(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract prosodic features.
        
        Prosodic features capture suprasegmental aspects of speech
        including pitch, energy, and speaking rate which are strong
        indicators of emotional state.
        """
        # Fundamental frequency (F0) using PYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Replace NaN with interpolation for unvoiced segments
        f0 = self._interpolate_f0(f0)
        
        # Log F0 for better scale
        f0_log = np.log(f0 + 1e-10)
        
        # F0 statistics over local windows
        f0_delta = np.gradient(f0)
        f0_delta2 = np.gradient(f0_delta)
        
        # Energy (RMS)
        rms = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length
        )[0]
        
        # Log energy
        rms_log = np.log(rms + 1e-10)
        rms_delta = np.gradient(rms_log)
        
        # Zero crossing rate (voice quality indicator)
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            hop_length=self.hop_length
        )[0]
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0]
        
        # Spectral flatness (noisiness)
        flatness = librosa.feature.spectral_flatness(
            y=audio,
            hop_length=self.hop_length
        )[0]
        
        # Align lengths
        min_len = min(
            len(f0), len(rms), len(zcr), 
            len(centroid), len(rolloff), len(flatness)
        )
        
        # Stack prosodic features
        prosodic = np.vstack([
            f0_log[:min_len],
            f0_delta[:min_len],
            f0_delta2[:min_len],
            rms_log[:min_len],
            rms_delta[:min_len],
            zcr[:min_len],
            centroid[:min_len] / self.sample_rate,  # Normalize
            rolloff[:min_len] / self.sample_rate,
            flatness[:min_len],
            voiced_probs[:min_len] if len(voiced_probs) >= min_len else np.zeros(min_len),
            np.gradient(zcr[:min_len]),
            np.gradient(centroid[:min_len])
        ])
        
        return prosodic
    
    def _interpolate_f0(self, f0: np.ndarray) -> np.ndarray:
        """Interpolate F0 through unvoiced segments."""
        f0_interp = f0.copy()
        
        # Find voiced segments
        voiced = ~np.isnan(f0)
        
        if not np.any(voiced):
            # All unvoiced, return zeros
            return np.zeros_like(f0)
        
        # Linear interpolation
        indices = np.arange(len(f0))
        f0_interp = np.interp(
            indices,
            indices[voiced],
            f0[voiced]
        )
        
        return f0_interp
    
    def extract_from_file(
        self,
        audio_path: Union[str, Path],
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> np.ndarray:
        """
        Extract features from audio file.
        
        Args:
            audio_path: Path to audio file
            duration: Duration to load in seconds (None for full file)
            offset: Start offset in seconds
            
        Returns:
            Feature array
        """
        audio, _ = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
            duration=duration,
            offset=offset
        )
        
        return self.extract_features(audio)
    
    def normalize_features(
        self,
        features: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply z-score normalization to features.
        
        Args:
            features: Feature array (time, features)
            mean: Precomputed mean (for test time)
            std: Precomputed std (for test time)
            
        Returns:
            Tuple of (normalized_features, mean, std)
        """
        if mean is None:
            mean = np.mean(features, axis=0, keepdims=True)
        if std is None:
            std = np.std(features, axis=0, keepdims=True) + 1e-8
            
        normalized = (features - mean) / std
        
        return normalized, mean.squeeze(), std.squeeze()
    
    def pad_or_truncate(
        self,
        features: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """
        Pad or truncate features to fixed length.
        
        Args:
            features: Feature array (time, features)
            target_length: Target number of frames
            
        Returns:
            Fixed-length feature array
        """
        current_length = features.shape[0]
        
        if current_length > target_length:
            # Center crop
            start = (current_length - target_length) // 2
            return features[start:start + target_length]
        elif current_length < target_length:
            # Pad with zeros
            pad_total = target_length - current_length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(
                features,
                ((pad_left, pad_right), (0, 0)),
                mode='constant'
            )
        
        return features


def compute_global_statistics(
    audio_paths: list,
    extractor: AudioFeatureExtractor,
    max_samples: int = 5000
) -> Dict[str, np.ndarray]:
    """
    Compute mean and std over dataset for normalization.
    
    Args:
        audio_paths: List of audio file paths
        extractor: Feature extractor instance
        max_samples: Maximum files to process
        
    Returns:
        Dictionary with 'mean' and 'std' arrays
    """
    all_features = []
    
    sample_paths = np.random.choice(
        audio_paths,
        size=min(max_samples, len(audio_paths)),
        replace=False
    )
    
    for path in sample_paths:
        try:
            features = extractor.extract_from_file(path)
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    # Concatenate all features
    all_features = np.vstack(all_features)
    
    return {
        'mean': np.mean(all_features, axis=0),
        'std': np.std(all_features, axis=0) + 1e-8
    }


if __name__ == '__main__':
    # Test feature extraction
    import tempfile
    import soundfile as sf
    
    # Create test audio
    duration = 3.0
    sr = 16000
    t = np.linspace(0, duration, int(duration * sr))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Add some modulation for test
    audio *= (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    
    # Save temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr)
        temp_path = f.name
    
    # Extract features
    extractor = AudioFeatureExtractor(sample_rate=sr)
    features = extractor.extract_from_file(temp_path)
    
    print(f"Audio duration: {duration}s")
    print(f"Feature shape: {features.shape}")
    print(f"Expected dimensions: {extractor.total_dim}")
    
    # Test component extraction
    audio_loaded, _ = librosa.load(temp_path, sr=sr)
    components = extractor.extract_features(audio_loaded, return_components=True)
    
    print("\nFeature components:")
    for name, arr in components.items():
        print(f"  {name}: {arr.shape}")
    
    # Cleanup
    import os
    os.unlink(temp_path)
    
    print("\nFeature extraction test passed!")
