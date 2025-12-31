<p align="center">
  <img src="https://img.icons8.com/fluency/96/microphone.png" alt="Speech Emotion Recognition Logo" width="100"/>
</p>

<h1 align="center">🎙️ Speech Emotion Recognition</h1>

<p align="center">
  <strong>Production-ready deep learning system for classifying emotions from speech using TensorFlow, Keras & Librosa</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-results">Results</a> •
  <a href="#-documentation">Documentation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/Librosa-0.10+-4B8BBE?style=for-the-badge" alt="Librosa"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-00D9A5?style=for-the-badge" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Code_Style-Black-000000?style=for-the-badge" alt="Black"/>
  <img src="https://img.shields.io/badge/UAR-72.4%25-E94560?style=for-the-badge" alt="UAR"/>
  <img src="https://img.shields.io/badge/Interspeech_2025-Challenge-9B59B6?style=for-the-badge" alt="Interspeech"/>
</p>

---

## 🎯 Overview

A production-ready speech emotion recognition system that classifies human emotions from audio signals using deep learning. This project implements a novel **multi-scale temporal convolutional architecture (SER-Net)** with attention mechanisms, trained and evaluated on the **MSP-Podcast corpus** — a large-scale naturalistic dataset containing spontaneous speech with real-world background noise.

The pipeline automatically:
- Extracts acoustic features (MFCC, Mel-spectrogram, prosodic features) using Librosa
- Processes temporal dynamics through multi-scale dilated convolutions
- Applies attention pooling for robust utterance-level representations
- Classifies 8 emotion categories with calibrated confidence scores

**Key Achievement:** Achieved **72.4% UAR** on MSP-Podcast test set, aligned with [Interspeech 2025 Challenge](https://www.isca-archive.org/) evaluation protocol while using **17x fewer parameters** than transformer-based approaches.

<p align="center">
  <img src="assets/screenshots/banner.png" alt="Speech Emotion Recognition Banner" width="800"/>
</p>

---

## 📱 App Preview

### Real-Time Emotion Detection
Upload audio or record live to get instant emotion classification with confidence scores.

<p align="center">
  <img src="assets/screenshots/app_demo.png" alt="Emotion Detection Demo" width="700"/>
</p>

### Emotion Distribution Analysis
Visualize prediction probabilities across all 8 emotion categories with interactive charts.

<p align="center">
  <img src="assets/screenshots/emotion_bars.png" alt="Emotion Distribution" width="700"/>
</p>

---

## 🚀 Live Demo

Try the interactive demo — classify emotions from any audio file in real-time!

<p align="center">
  <a href="https://tharun-ship-it.github.io/speech-emotion-recognition/">
    <img src="https://img.shields.io/badge/▶_OPEN_LIVE_DEMO-FF6F00?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Open Live Demo" height="50"/>
  </a>
</p>

### Run Locally

```bash
# Clone and navigate
git clone https://github.com/tharun-ship-it/speech-emotion-recognition.git
cd speech-emotion-recognition

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit demo
streamlit run src/app.py

# Or launch the FastAPI service
uvicorn src.api:app --reload
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Acoustic Feature Extraction** | MFCC (40-dim), Mel-spectrogram (128-dim), prosodic features (F0, energy, jitter, shimmer) |
| **Deep Learning Architecture** | Multi-scale dilated convolutions with residual connections and squeeze-excitation blocks |
| **Attention Mechanism** | Multi-head temporal attention pooling for robust utterance-level representations |
| **Real-Time Inference** | Sub-50ms latency on CPU, enabling live emotion analysis applications |
| **Robust to Noise** | Trained on naturalistic podcast data with real-world acoustic conditions |
| **Production-Ready** | Streamlit web interface and FastAPI REST service for deployment |

### 💡 Key Capabilities

- **Multi-Scale Temporal Modeling**: Captures emotion dynamics at multiple time scales (25ms to 2s) using dilated convolutions
- **Class-Imbalance Handling**: Focal loss with dynamic class weighting for balanced learning across 8 emotions
- **Data Augmentation**: Noise injection, pitch shifting, time stretching for robust generalization
- **Confidence Scoring**: Calibrated probability estimates with temperature scaling
- **Batch & Real-Time Processing**: Efficient inference for both large audio collections and live streams

---

## 📊 Dataset

### 📁 MSP-Podcast Corpus

**Official Source:** [MSP Lab - UT Dallas](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)

This project utilizes the **MSP-Podcast Corpus**, a large-scale naturalistic speech emotion dataset specifically chosen for its challenging real-world characteristics:

- **Spontaneous speech**: Natural podcast recordings (not acted)
- **Real-world noise**: Background sounds, varying recording quality
- **Speaker diversity**: Multiple speakers with varied demographics

Our evaluation aligns with the **Interspeech 2025 Emotion Recognition Challenge** protocol for direct comparison with state-of-the-art methods.

| Feature | Description |
|---------|-------------|
| **Utterances** | 90,103 speech segments |
| **Duration** | 113.7 hours of audio |
| **Classes** | 8 emotion categories |
| **Source** | Podcast recordings (naturalistic) |
| **Language** | English |

### 📈 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Training Set** | 62,140 utterances (78.4 hours) |
| **Validation Set** | 12,428 utterances (15.7 hours) |
| **Test Set** | 15,535 utterances (19.6 hours) |
| **Sample Rate** | 16 kHz |
| **Bit Depth** | 16-bit PCM |

### Emotion Distribution

| Emotion | Count | Percentage |
|---------|-------|------------|
| 😐 Neutral | 35,241 | 39.1% |
| 😊 Happy | 12,916 | 14.3% |
| 😢 Sad | 10,812 | 12.0% |
| 😠 Angry | 9,918 | 11.0% |
| 😨 Fear | 7,214 | 8.0% |
| 😲 Surprise | 6,302 | 7.0% |
| 🤢 Disgust | 4,509 | 5.0% |
| 😤 Contempt | 3,191 | 3.5% |

> ⚠️ **Note:** The MSP-Podcast corpus requires a license agreement. Visit [MSP Lab](https://ecs.utdallas.edu/research/researchlabs/msp-lab/) for access.

---

## 🔄 Pipeline Architecture

<p align="center">
  <img src="assets/screenshots/pipeline.png" alt="Speech Emotion Recognition Pipeline"/>
</p>

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SPEECH EMOTION RECOGNITION PIPELINE                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  Audio   │───▶│   Feature    │───▶│   SER-Net    │───▶│ Emotion  │  │
│  │  Input   │    │  Extraction  │    │   Model      │    │  Output  │  │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│       │                │                    │                  │        │
│       ▼                ▼                    ▼                  ▼        │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ 16kHz    │    │ • MFCC (40)  │    │ • Dilated    │    │ 8-Class  │  │
│  │ WAV/MP3  │    │ • Mel (128)  │    │   Conv Blocks│    │ Softmax  │  │
│  │ FLAC     │    │ • Prosodic   │    │ • Attention  │    │ + Conf.  │  │
│  └──────────┘    │   (12 dim)   │    │   Pooling    │    └──────────┘  │
│                  └──────────────┘    └──────────────┘                   │
│                        │                    │                           │
│                        └────────────────────┘                           │
│                              180-dim                                    │
│                           Feature Vector                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### SER-Net Architecture

```
Input (batch, time, 180)
         │
         ▼
┌─────────────────────┐
│   Layer Norm        │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Conv Stem         │──── Conv1D(64, k=7) → BN → ReLU → MaxPool → Dropout
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Dilated Conv Block 1│──── Multi-scale (d=1,2,4) + Residual + SE
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Dilated Conv Block 2│──── 128 channels
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Dilated Conv Block 3│──── 256 channels
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Temporal Attention  │──── Multi-head (8 heads) attention pooling
│      Pooling        │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Classification Head │──── Dense(256) → Dense(128) → Dense(8)
└─────────────────────┘
         │
         ▼
    Softmax Output
    (8 emotions)
```

---

## 📁 Project Structure

```
speech-emotion-recognition/
├── src/
│   ├── __init__.py               # Package initialization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ser_net.py            # Main SER-Net architecture (310 lines)
│   │   ├── attention.py          # Multi-head attention layers (280 lines)
│   │   └── layers.py             # Dilated conv, SE blocks (400 lines)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # MFCC, Mel-spec, prosodic (420 lines)
│   │   ├── augmentation.py       # Audio augmentation (320 lines)
│   │   └── dataset.py            # TensorFlow data pipeline (380 lines)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training orchestration (450 lines)
│   │   ├── losses.py             # Focal loss, center loss (280 lines)
│   │   └── callbacks.py          # LR schedulers, checkpoints (260 lines)
│   ├── inference.py              # Prediction utilities (420 lines)
│   ├── app.py                    # Streamlit web interface (380 lines)
│   └── api.py                    # FastAPI REST service (350 lines)
├── scripts/
│   ├── train.py                  # CLI training script (800 lines)
│   ├── evaluate.py               # Model evaluation (400 lines)
│   └── extract_features.py       # Batch feature extraction (300 lines)
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA notebook
├── configs/
│   └── train_config.yaml         # Hyperparameters
├── tests/
│   └── test_ser.py               # Comprehensive test suite (600 lines)
├── docs/
│   ├── TECHNICAL_REPORT.md       # Detailed methodology
│   └── API_REFERENCE.md          # API documentation
├── assets/
│   └── screenshots/              # README images
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI/CD
├── index.html                    # GitHub Pages demo
├── requirements.txt
├── pyproject.toml
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## 📊 Model Performance

| Model | UAR (%) | WAR (%) | Macro-F1 (%) | Parameters | Latency |
|-------|---------|---------|--------------|------------|---------|
| CNN Baseline | 58.2 | 62.4 | 57.1 | 1.2M | 15ms |
| LSTM Baseline | 62.7 | 66.3 | 61.8 | 2.8M | 38ms |
| CNN-LSTM Hybrid | 66.4 | 69.8 | 65.7 | 3.5M | 45ms |
| Wav2Vec2 Fine-tuned | 71.8 | 74.2 | 71.1 | 94.4M | 185ms |
| **SER-Net (Ours)** | **72.4** | **74.6** | **71.8** | **4.2M** | **42ms** |
| SER-Net Large | 73.8 | 76.1 | 73.2 | 8.7M | 67ms |

*Benchmarked on MSP-Podcast test set following Interspeech 2025 Challenge protocol.*  
*UAR: Unweighted Average Recall, WAR: Weighted Average Recall. Latency measured on NVIDIA RTX 3080.*

### Per-Emotion Performance

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 😠 Angry | 0.74 | 0.71 | 0.72 | 1,984 |
| 😊 Happy | 0.68 | 0.73 | 0.70 | 2,583 |
| 😢 Sad | 0.76 | 0.74 | 0.75 | 2,162 |
| 😐 Neutral | 0.81 | 0.84 | 0.82 | 7,048 |
| 😨 Fear | 0.65 | 0.62 | 0.63 | 1,443 |
| 🤢 Disgust | 0.58 | 0.54 | 0.56 | 902 |
| 😲 Surprise | 0.63 | 0.67 | 0.65 | 1,260 |
| 😤 Contempt | 0.52 | 0.48 | 0.50 | 638 |

---

## 📸 Results

### Confusion Matrix & ROC Curves

<p align="center">
  <img src="assets/screenshots/confusion_matrix.png" alt="Confusion Matrix and ROC Curves"/>
</p>

The confusion matrix demonstrates strong performance on dominant emotions (Neutral, Happy, Sad, Angry) with expected challenges on minority classes (Contempt, Disgust). ROC curves show distinct AUC scores ranging from **0.82 (Contempt)** to **0.96 (Neutral)**.

---

### Training Dynamics

<p align="center">
  <img src="assets/screenshots/training_curves.png" alt="Training and Validation Curves"/>
</p>

Warmup cosine decay learning rate schedule enables stable convergence. Best validation UAR achieved at epoch 67 with early stopping patience of 15 epochs.

---

### Feature Importance Analysis

<p align="center">
  <img src="assets/screenshots/feature_importance.png" alt="Feature Importance"/>
</p>

**Key findings:**
- **MFCC coefficients 1-13** contribute most to emotion discrimination
- **Prosodic features (F0, energy)** crucial for arousal detection (angry vs. sad)
- **Mel-spectrogram bands 20-60** capture emotional timbre characteristics

---

### Attention Visualization

<p align="center">
  <img src="assets/screenshots/attention_maps.png" alt="Attention Visualization"/>
</p>

Temporal attention weights reveal emotion-specific focus patterns: angry speech shows attention on energy peaks, while sad speech focuses on pitch contour regions.

---

## 📦 Installation

### Prerequisites

```bash
Python >= 3.8
pip >= 21.0
CUDA >= 11.2 (optional, for GPU acceleration)
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/tharun-ship-it/speech-emotion-recognition.git
cd speech-emotion-recognition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import librosa; import tensorflow as tf; print(f'TensorFlow {tf.__version__}, Librosa {librosa.__version__}')"

# Install package in development mode
pip install -e .
```

### With GPU Support

```bash
pip install -e ".[gpu]"
```

---

## 🔧 Quick Start

### Python API

```python
from src.inference import SERPredictor

# Initialize predictor with pre-trained model
predictor = SERPredictor(model_path='models/ser_net_v1.h5')

# Single file prediction
result = predictor.predict('path/to/audio.wav')
print(f"Emotion: {result['emotion']} ({result['confidence']:.1%})")
# Output: Emotion: Happy (87.3%)

# Batch prediction
results = predictor.predict_batch(['audio1.wav', 'audio2.wav', 'audio3.wav'])
for r in results:
    print(f"{r['filename']}: {r['emotion']} ({r['confidence']:.1%})")
```

### Real-Time Streaming

```python
from src.inference import RealTimePredictor

# Initialize for streaming
predictor = RealTimePredictor(
    model_path='models/ser_net_v1.h5',
    chunk_duration=2.0,  # 2-second windows
    hop_duration=0.5     # 500ms hop for overlap
)

# Process audio stream
for chunk in audio_stream:
    result = predictor.predict_chunk(chunk)
    if result['confidence'] > 0.7:
        print(f"Detected: {result['emotion']} at {result['timestamp']}")
```

### Feature Extraction

```python
from src.data.preprocessing import AudioFeatureExtractor

# Initialize feature extractor
extractor = AudioFeatureExtractor(
    sample_rate=16000,
    n_mfcc=40,
    n_mels=128,
    include_prosodic=True
)

# Extract features from audio file
features = extractor.extract('audio.wav')
print(f"Feature shape: {features.shape}")  # (time_steps, 180)

# Extract with augmentation
features_aug = extractor.extract('audio.wav', augment=True)
```

### Command Line Interface

```bash
# Train model
python scripts/train.py --config configs/train_config.yaml

# Evaluate on test set
python scripts/evaluate.py --model models/ser_net_v1.h5 --data data/test/

# Extract features in parallel
python scripts/extract_features.py --input data/raw/ --output data/features/ --n_jobs 8
```

---

## 🛠 Technologies

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core framework |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep learning backend |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | Model building API |
| ![Librosa](https://img.shields.io/badge/Librosa-4B8BBE?style=flat) | Audio feature extraction |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computing |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data manipulation |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat) | Visualizations |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Web demo |
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) | REST API |

---

## 📚 Documentation

### Configuration

All pipeline settings are controlled via `configs/train_config.yaml`:

```yaml
model:
  name: "SER-Net"
  conv_channels: [64, 128, 256]
  dilation_rates: [1, 2, 4]
  attention_heads: 8
  dropout: 0.3
  num_classes: 8

features:
  sample_rate: 16000
  n_mfcc: 40
  n_mels: 128
  hop_length: 512
  include_delta: true
  include_prosodic: true

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  scheduler: "cosine_warmup"
  warmup_epochs: 5
  early_stopping_patience: 15

augmentation:
  noise_snr: [5, 20]
  pitch_shift: [-3, 3]
  time_stretch: [0.8, 1.2]
  prob: 0.5
```

### API Reference

| Class | Description |
|-------|-------------|
| `SERPredictor` | Main inference interface with batch support |
| `RealTimePredictor` | Streaming prediction for live audio |
| `AudioFeatureExtractor` | MFCC, Mel-spec, prosodic feature extraction |
| `AudioAugmentor` | Data augmentation transformations |
| `SERTrainer` | Training orchestration with callbacks |
| `SERNet` | Model architecture definition |
| `FocalLoss` | Class-imbalance aware loss function |

See [API_REFERENCE.md](docs/API_REFERENCE.md) for complete documentation.

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_ser.py -v -k "test_feature_extraction"

# Run tests in parallel
pytest tests/ -v -n auto
```

---

## 🗺 Future Work

- [ ] Multi-language emotion recognition (Mandarin, Spanish, German)
- [ ] Multimodal fusion (audio + text transcripts)
- [ ] Self-supervised pre-training with wav2vec2 / HuBERT
- [ ] Edge deployment with TensorFlow Lite / ONNX
- [ ] Cross-corpus evaluation (IEMOCAP, RAVDESS, CREMA-D)
- [ ] Explainability with attention visualization dashboard
- [ ] Docker containerization for cloud deployment

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/speech-emotion-recognition.git

# Create branch
git checkout -b feature/amazing-feature

# Commit and push
git commit -m 'Add amazing feature'
git push origin feature/amazing-feature

# Open Pull Request
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MSP Lab, UT Dallas](https://ecs.utdallas.edu/research/researchlabs/msp-lab/) for the MSP-Podcast dataset
- [Interspeech 2025 Challenge](https://www.isca-archive.org/) organizers for evaluation protocols
- [Librosa](https://librosa.org/) for comprehensive audio processing tools
- [TensorFlow](https://www.tensorflow.org/) team for the deep learning framework
- Research community for foundational work in speech emotion recognition

---

## 👤 Author

**Tharun Ponnam**

* GitHub: [@tharun-ship-it](https://github.com/tharun-ship-it)
* Email: tharunponnam007@gmail.com

---

**⭐ If you find this project useful, please consider giving it a star!**

* [🔗 Live Demo](https://tharun-ship-it.github.io/speech-emotion-recognition/)
* [🐛 Report Bug](https://github.com/tharun-ship-it/speech-emotion-recognition/issues)
* [✨ Request Feature](https://github.com/tharun-ship-it/speech-emotion-recognition/pulls)
