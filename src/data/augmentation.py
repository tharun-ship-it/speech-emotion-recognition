"""
Audio Data Augmentation for Speech Emotion Recognition

Implements various augmentation techniques to improve model
robustness and generalization to real-world conditions.

Author: Tharun Ponnam
"""

import numpy as np
from typing import Tuple, Optional, Callable
import librosa


class AudioAugmentor:
    """
    Audio augmentation pipeline for training data.
    
    Applies various transformations to audio signals to increase
    training data diversity and improve model robustness.
    
    Args:
        sample_rate: Audio sample rate
        noise_prob: Probability of adding noise
        noise_snr_range: Range of SNR values for noise
        pitch_shift_prob: Probability of pitch shifting
        pitch_shift_range: Range of semitones for pitch shift
        time_stretch_prob: Probability of time stretching
        time_stretch_range: Range of stretch factors
        volume_prob: Probability of volume change
        volume_range: Range of volume scaling factors
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_prob: float = 0.5,
        noise_snr_range: Tuple[float, float] = (5, 20),
        pitch_shift_prob: float = 0.3,
        pitch_shift_range: Tuple[float, float] = (-3, 3),
        time_stretch_prob: float = 0.3,
        time_stretch_range: Tuple[float, float] = (0.8, 1.2),
        volume_prob: float = 0.3,
        volume_range: Tuple[float, float] = (0.7, 1.3)
    ):
        self.sample_rate = sample_rate
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.volume_prob = volume_prob
        self.volume_range = volume_range
        
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to audio.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Augmented audio waveform
        """
        augmented = audio.copy()
        
        # Apply augmentations with specified probabilities
        if np.random.random() < self.noise_prob:
            augmented = self.add_noise(augmented)
            
        if np.random.random() < self.pitch_shift_prob:
            augmented = self.pitch_shift(augmented)
            
        if np.random.random() < self.time_stretch_prob:
            augmented = self.time_stretch(augmented)
            
        if np.random.random() < self.volume_prob:
            augmented = self.volume_change(augmented)
            
        return augmented
    
    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise at random SNR."""
        snr = np.random.uniform(*self.noise_snr_range)
        
        # Compute signal power
        signal_power = np.mean(audio ** 2)
        
        # Compute noise power for target SNR
        noise_power = signal_power / (10 ** (snr / 10))
        
        # Generate and add noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        
        return audio + noise.astype(audio.dtype)
    
    def pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Shift pitch by random number of semitones."""
        semitones = np.random.uniform(*self.pitch_shift_range)
        
        return librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=semitones
        )
    
    def time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Stretch or compress audio in time domain."""
        rate = np.random.uniform(*self.time_stretch_range)
        
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def volume_change(self, audio: np.ndarray) -> np.ndarray:
        """Scale audio volume."""
        factor = np.random.uniform(*self.volume_range)
        
        return audio * factor


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for ASR
    
    Applies frequency and time masking to spectrograms as
    described in Park et al. (2019).
    
    Args:
        freq_mask_param: Maximum number of frequency channels to mask
        time_mask_param: Maximum number of time steps to mask
        n_freq_masks: Number of frequency masks to apply
        n_time_masks: Number of time masks to apply
    """
    
    def __init__(
        self,
        freq_mask_param: int = 30,
        time_mask_param: int = 50,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        
    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram (time, freq)
            
        Returns:
            Augmented spectrogram
        """
        augmented = spectrogram.copy()
        time_steps, freq_channels = augmented.shape
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, freq_channels))
            f0 = np.random.randint(0, freq_channels - f)
            augmented[:, f0:f0 + f] = 0
            
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, min(self.time_mask_param, time_steps))
            t0 = np.random.randint(0, time_steps - t)
            augmented[t0:t0 + t, :] = 0
            
        return augmented


class RoomSimulator:
    """
    Simple room impulse response simulation.
    
    Applies convolution with synthetic room impulse responses
    to simulate different acoustic environments.
    
    Args:
        sample_rate: Audio sample rate
        rt60_range: Range of reverberation times (RT60)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        rt60_range: Tuple[float, float] = (0.1, 0.5)
    ):
        self.sample_rate = sample_rate
        self.rt60_range = rt60_range
        
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply room simulation to audio."""
        rt60 = np.random.uniform(*self.rt60_range)
        
        # Generate simple exponentially decaying impulse response
        rir_length = int(rt60 * self.sample_rate)
        t = np.arange(rir_length) / self.sample_rate
        
        # Exponential decay
        decay = np.exp(-6.9 * t / rt60)  # -60dB at RT60
        
        # Random phase
        noise = np.random.randn(rir_length)
        rir = noise * decay
        
        # Normalize
        rir = rir / np.max(np.abs(rir))
        
        # Convolve
        reverbed = np.convolve(audio, rir, mode='same')
        
        # Mix with original (early reflections)
        mix = 0.7 * audio + 0.3 * reverbed
        
        return mix.astype(audio.dtype)


class ComposeAugmentation:
    """
    Compose multiple augmentations.
    
    Args:
        augmentations: List of augmentation callables
        p: Probability of applying each augmentation
    """
    
    def __init__(
        self,
        augmentations: list,
        p: float = 0.5
    ):
        self.augmentations = augmentations
        self.p = p
        
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Apply composed augmentations."""
        for aug in self.augmentations:
            if np.random.random() < self.p:
                audio = aug(audio)
        return audio


class MixupAugmentation:
    """
    Mixup augmentation for audio and labels.
    
    Linearly interpolates between pairs of examples to create
    new training samples.
    
    Args:
        alpha: Beta distribution parameter for mixing coefficient
    """
    
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        
    def __call__(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        label1: np.ndarray,
        label2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup to audio pair.
        
        Args:
            audio1, audio2: Audio waveforms
            label1, label2: One-hot encoded labels
            
        Returns:
            Mixed audio and interpolated labels
        """
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Ensure same length
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # Mix
        mixed_audio = lam * audio1 + (1 - lam) * audio2
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_audio, mixed_label


def get_default_augmentor(sample_rate: int = 16000) -> AudioAugmentor:
    """Get default augmentor with recommended settings."""
    return AudioAugmentor(
        sample_rate=sample_rate,
        noise_prob=0.5,
        noise_snr_range=(5, 20),
        pitch_shift_prob=0.3,
        pitch_shift_range=(-3, 3),
        time_stretch_prob=0.3,
        time_stretch_range=(0.8, 1.2),
        volume_prob=0.3,
        volume_range=(0.7, 1.3)
    )


if __name__ == '__main__':
    # Test augmentations
    import matplotlib.pyplot as plt
    
    # Generate test audio
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(duration * sr))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Apply augmentations
    augmentor = get_default_augmentor(sr)
    augmented = augmentor(audio)
    
    print(f"Original shape: {audio.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print(f"Original range: [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")
    
    # Test SpecAugment
    spec = np.random.randn(300, 128)
    spec_aug = SpecAugment()(spec)
    
    print(f"\nSpectrogram shape: {spec.shape}")
    print(f"Augmented spec shape: {spec_aug.shape}")
    print(f"Masked ratio: {(spec_aug == 0).sum() / spec_aug.size:.2%}")
    
    print("\nAugmentation tests passed!")
