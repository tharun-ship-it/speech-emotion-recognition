# API Reference

Complete API documentation for the Speech Emotion Recognition package.

---

## Table of Contents

- [Models](#models)
- [Data Processing](#data-processing)
- [Training](#training)
- [Inference](#inference)

---

## Models

### `src.models.SERNet`

Main model class for Speech Emotion Recognition.

```python
class SERNet(tf.keras.Model):
    """
    Speech Emotion Recognition Network.
    
    Multi-scale dilated convolutions with attention pooling
    for emotion classification from acoustic features.
    """
```

#### Constructor

```python
SERNet(
    num_classes: int = 8,
    filters: List[int] = [64, 64, 64],
    dilation_rates: List[int] = [1, 2, 4],
    num_residual_blocks: int = 3,
    attention_heads: int = 4,
    dense_units: int = 256,
    dropout_rate: float = 0.3
)
```

**Parameters:**
- `num_classes`: Number of emotion categories
- `filters`: Filter sizes for dilated convolution branches
- `dilation_rates`: Dilation rates for multi-scale processing
- `num_residual_blocks`: Number of residual blocks after fusion
- `attention_heads`: Number of attention heads
- `dense_units`: Units in classification head
- `dropout_rate`: Dropout probability

#### Methods

```python
def call(self, x, training=False):
    """
    Forward pass.
    
    Args:
        x: Input features (batch, time, features)
        training: Whether in training mode
        
    Returns:
        Logits (batch, num_classes)
    """
```

---

### `src.models.create_ser_model`

Factory function for creating SER-Net models.

```python
def create_ser_model(
    input_shape: Tuple[int, int],
    num_classes: int = 8,
    variant: str = 'base'
) -> tf.keras.Model:
    """
    Create a SER-Net model.
    
    Args:
        input_shape: (time_steps, feature_dim)
        num_classes: Number of output classes
        variant: 'base' or 'large'
        
    Returns:
        Compiled Keras model
    """
```

**Example:**
```python
from src.models import create_ser_model

model = create_ser_model(
    input_shape=(300, 180),
    num_classes=8,
    variant='base'
)
model.summary()
```

---

### `src.models.attention.MultiHeadSelfAttention`

Multi-head self-attention layer.

```python
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Multi-head self-attention for temporal sequences.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
    """
```

---

### `src.models.attention.TemporalAttentionPooling`

Attention-based temporal pooling layer.

```python
class TemporalAttentionPooling(tf.keras.layers.Layer):
    """
    Weighted temporal pooling using learned attention.
    
    Reduces (batch, time, features) to (batch, features)
    with learned importance weights.
    
    Args:
        units: Hidden units for attention network
    """
```

---

## Data Processing

### `src.data.AudioFeatureExtractor`

Comprehensive audio feature extraction.

```python
class AudioFeatureExtractor:
    """
    Extract acoustic features from audio signals.
    
    Combines MFCCs, mel spectrograms, and prosodic features
    into a unified representation.
    """
```

#### Constructor

```python
AudioFeatureExtractor(
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    max_length: int = 300
)
```

**Parameters:**
- `sample_rate`: Target sample rate in Hz
- `n_mfcc`: Number of MFCC coefficients
- `n_mels`: Number of mel filter banks
- `hop_length`: Hop length for STFT
- `n_fft`: FFT window size
- `max_length`: Maximum sequence length (frames)

#### Methods

```python
def extract(self, audio: np.ndarray) -> np.ndarray:
    """
    Extract features from audio signal.
    
    Args:
        audio: Audio waveform (1D numpy array)
        
    Returns:
        Features array (time, feature_dim)
    """

def extract_from_file(self, audio_path: str) -> np.ndarray:
    """
    Extract features from audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Features array (time, feature_dim)
    """
```

**Example:**
```python
from src.data import AudioFeatureExtractor
import librosa

extractor = AudioFeatureExtractor(sample_rate=16000)

# From file
features = extractor.extract_from_file('audio.wav')

# From array
audio, sr = librosa.load('audio.wav', sr=16000)
features = extractor.extract(audio)

print(f"Feature shape: {features.shape}")  # (T, 180)
```

---

### `src.data.AudioAugmentor`

Audio data augmentation utilities.

```python
class AudioAugmentor:
    """
    Audio augmentation for training robustness.
    
    Includes noise injection, time stretching,
    pitch shifting, and room simulation.
    """
```

#### Methods

```python
def add_noise(
    self, 
    audio: np.ndarray, 
    snr_db: float = 20
) -> np.ndarray:
    """Add Gaussian noise at specified SNR."""

def time_stretch(
    self, 
    audio: np.ndarray, 
    rate: float
) -> np.ndarray:
    """Time stretch audio by rate factor."""

def pitch_shift(
    self, 
    audio: np.ndarray, 
    sr: int, 
    n_steps: float
) -> np.ndarray:
    """Shift pitch by n_steps semitones."""

def augment(
    self, 
    audio: np.ndarray, 
    sr: int
) -> np.ndarray:
    """Apply random augmentation combination."""
```

**Example:**
```python
from src.data import AudioAugmentor

augmentor = AudioAugmentor()

# Single augmentation
noisy = augmentor.add_noise(audio, snr_db=15)
stretched = augmentor.time_stretch(audio, rate=1.1)

# Random augmentation
augmented = augmentor.augment(audio, sr=16000)
```

---

### `src.data.SERDataset`

Dataset class for loading and preprocessing data.

```python
class SERDataset:
    """
    Dataset for Speech Emotion Recognition.
    
    Handles loading audio files, extracting features,
    and creating TensorFlow datasets.
    """
```

#### Constructor

```python
SERDataset(
    audio_dir: str,
    labels_file: str,
    split: str = 'train',
    feature_extractor: AudioFeatureExtractor = None,
    augment: bool = False,
    cache_features: bool = True
)
```

#### Methods

```python
def __len__(self) -> int:
    """Return number of samples."""

def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
    """Get sample by index."""

def to_tf_dataset(
    self, 
    batch_size: int = 32, 
    shuffle: bool = True
) -> tf.data.Dataset:
    """Convert to TensorFlow dataset."""

def get_labels(self) -> np.ndarray:
    """Return all labels."""
```

**Example:**
```python
from src.data import SERDataset, AudioFeatureExtractor

extractor = AudioFeatureExtractor()

dataset = SERDataset(
    audio_dir='data/audio',
    labels_file='data/labels.csv',
    split='train',
    feature_extractor=extractor,
    augment=True
)

# Create TF dataset
tf_dataset = dataset.to_tf_dataset(batch_size=32)

for features, labels in tf_dataset.take(1):
    print(f"Batch shape: {features.shape}")
```

---

## Training

### `src.training.SERTrainer`

Training orchestration class.

```python
class SERTrainer:
    """
    Trainer for Speech Emotion Recognition models.
    
    Handles training loop, validation, checkpointing,
    and logging.
    """
```

#### Constructor

```python
SERTrainer(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    config: Dict,
    class_weights: Dict[int, float] = None,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs'
)
```

#### Methods

```python
def train(self) -> Dict:
    """
    Run training loop.
    
    Returns:
        Training history dictionary
    """

def evaluate(self, dataset: tf.data.Dataset) -> Dict:
    """
    Evaluate model on dataset.
    
    Returns:
        Evaluation metrics
    """

def load_checkpoint(self, path: str):
    """Load model from checkpoint."""

def save_checkpoint(self, path: str):
    """Save model checkpoint."""
```

**Example:**
```python
from src.training import SERTrainer

trainer = SERTrainer(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    config={
        'epochs': 100,
        'learning_rate': 1e-3,
        'early_stopping_patience': 10
    }
)

history = trainer.train()
```

---

### `src.training.losses.FocalLoss`

Focal loss for handling class imbalance.

```python
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss implementation.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Balance parameter (default: 0.25)
    """
```

**Example:**
```python
from src.training.losses import FocalLoss

loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']
)
```

---

### `src.training.callbacks.WarmupCosineDecay`

Learning rate schedule with warmup.

```python
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup and cosine decay.
    
    Args:
        initial_learning_rate: Peak learning rate
        warmup_steps: Steps for linear warmup
        total_steps: Total training steps
        min_learning_rate: Minimum learning rate
    """
```

**Example:**
```python
from src.training.callbacks import WarmupCosineDecay

schedule = WarmupCosineDecay(
    initial_learning_rate=1e-3,
    warmup_steps=1000,
    total_steps=10000
)

optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
```

---

## Inference

### `src.inference.SERPredictor`

Production inference class.

```python
class SERPredictor:
    """
    Speech Emotion Recognition predictor.
    
    Provides easy-to-use interface for emotion prediction
    from audio files or waveforms.
    """
```

#### Constructor

```python
SERPredictor(
    model_path: str,
    num_classes: int = 8,
    device: str = 'auto'
)
```

#### Methods

```python
def predict(self, audio_path: str) -> Dict:
    """
    Predict emotion from audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with 'emotion', 'confidence', 'probabilities'
    """

def predict_from_array(
    self, 
    audio: np.ndarray, 
    sample_rate: int = 16000
) -> Dict:
    """
    Predict emotion from audio array.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate of audio
        
    Returns:
        Prediction dictionary
    """

def predict_batch(
    self, 
    features: np.ndarray
) -> List[Dict]:
    """
    Predict emotions for batch of features.
    
    Args:
        features: Batch of features (batch, time, features)
        
    Returns:
        List of prediction dictionaries
    """
```

**Example:**
```python
from src.inference import SERPredictor

predictor = SERPredictor(
    model_path='models/best_model',
    num_classes=8
)

# Single file prediction
result = predictor.predict('audio.wav')
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")

# From array
audio, sr = librosa.load('audio.wav', sr=16000)
result = predictor.predict_from_array(audio, sr)
```

---

### `src.inference.RealTimePredictor`

Real-time streaming prediction.

```python
class RealTimePredictor:
    """
    Real-time emotion prediction from audio stream.
    
    Maintains a sliding window buffer and provides
    continuous predictions with configurable update rate.
    """
```

#### Constructor

```python
RealTimePredictor(
    model_path: str,
    window_size: float = 3.0,
    hop_size: float = 0.5,
    sample_rate: int = 16000
)
```

#### Methods

```python
def process_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
    """
    Process audio chunk and return prediction if ready.
    
    Args:
        audio_chunk: New audio samples
        
    Returns:
        Prediction dict or None if buffer not full
    """

def reset(self):
    """Reset internal buffer."""
```

**Example:**
```python
from src.inference import RealTimePredictor

predictor = RealTimePredictor(
    model_path='models/best_model',
    window_size=3.0,
    hop_size=0.5
)

# Simulated streaming
for chunk in audio_stream:
    result = predictor.process_chunk(chunk)
    if result is not None:
        print(f"Detected: {result['emotion']}")
```

---

## REST API

### Endpoints

#### `POST /predict`

Predict emotion from uploaded audio file.

**Request:**
```bash
curl -X POST \
  -F "file=@audio.wav" \
  http://localhost:8000/predict
```

**Response:**
```json
{
  "emotion": "Happy",
  "confidence": 0.847,
  "probabilities": {
    "Angry": 0.023,
    "Happy": 0.847,
    "Sad": 0.012,
    "Neutral": 0.089,
    "Fear": 0.008,
    "Disgust": 0.005,
    "Surprise": 0.011,
    "Contempt": 0.005
  },
  "processing_time_ms": 45.2
}
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_AUDIO` | Audio file cannot be processed |
| `DURATION_ERROR` | Audio too short or too long |
| `MODEL_ERROR` | Model inference failed |
| `FEATURE_ERROR` | Feature extraction failed |

---

*For additional support, contact: tharunponnam007@gmail.com*
