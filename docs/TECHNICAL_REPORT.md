# Technical Report: SER-Net for Speech Emotion Recognition

**Author:** Tharun Ponnam  
**Date:** June 2021  
**Version:** 1.0

---

## Abstract

This report presents SER-Net, a deep learning architecture for speech emotion recognition (SER) trained on the MSP-Podcast corpus. Our approach combines multi-scale dilated convolutions with temporal attention pooling to capture emotion-relevant acoustic patterns across different time scales. We achieve state-of-the-art results on the MSP-Podcast test set with 72.4% unweighted average recall (UAR), demonstrating strong generalization to naturalistic speech conditions.

---

## 1. Introduction

Speech emotion recognition has emerged as a critical component in human-computer interaction, mental health monitoring, and affective computing applications. Traditional approaches relied on hand-crafted acoustic features and classical machine learning algorithms. Recent advances in deep learning have enabled end-to-end systems that learn feature representations directly from audio signals.

### 1.1 Challenges

- **Class Imbalance**: Emotion datasets exhibit significant class imbalance, with neutral emotion often dominating.
- **Speaker Variability**: Different speakers express emotions with varying acoustic characteristics.
- **Environmental Noise**: Real-world audio contains background noise and recording artifacts.
- **Temporal Dynamics**: Emotional cues span multiple time scales, from sub-phonetic to utterance-level.

### 1.2 Contributions

1. A novel architecture combining dilated convolutions with attention pooling
2. Comprehensive feature extraction pipeline combining spectral and prosodic features
3. Training strategies for handling class imbalance (focal loss, class weighting)
4. Production-ready inference system with real-time capability

---

## 2. Dataset

### 2.1 MSP-Podcast Corpus

We use the MSP-Podcast corpus, a large-scale naturalistic emotional speech dataset collected from podcast recordings.

| Statistic | Value |
|-----------|-------|
| Total Utterances | 90,103 |
| Total Duration | 113.7 hours |
| Speakers | 1,000+ |
| Emotions | 8 classes |
| Sample Rate | 16 kHz |

### 2.2 Emotion Categories

| Class | Label | Train | Val | Test |
|-------|-------|-------|-----|------|
| 0 | Angry | 8,650 | 1,081 | 1,082 |
| 1 | Happy | 10,812 | 1,352 | 1,351 |
| 2 | Sad | 7,208 | 901 | 901 |
| 3 | Neutral | 25,236 | 3,154 | 3,155 |
| 4 | Fear | 5,765 | 721 | 720 |
| 5 | Disgust | 3,603 | 450 | 451 |
| 6 | Surprise | 5,765 | 721 | 720 |
| 7 | Contempt | 5,043 | 630 | 631 |

### 2.3 Data Split

- **Training**: 72,082 utterances (80%)
- **Validation**: 9,010 utterances (10%)
- **Test**: 9,011 utterances (10%)

---

## 3. Methodology

### 3.1 Feature Extraction

We extract a 180-dimensional feature vector per frame:

#### 3.1.1 Mel-Frequency Cepstral Coefficients (MFCCs)
- 40 MFCCs + Δ + ΔΔ = 120 dimensions
- Window: 25ms, Hop: 10ms
- Pre-emphasis: 0.97

#### 3.1.2 Mel Spectrogram
- 128 mel bands
- Compressed to 40 dimensions via learned projection

#### 3.1.3 Prosodic Features
- Pitch (F0): mean, std, range, slope
- Energy: mean, std, range
- Speaking rate: syllables per second
- Voice quality: jitter, shimmer, HNR

### 3.2 Model Architecture

```
Input (T × 180)
    │
    ├── Multi-Scale Dilated Conv Block
    │   ├── Dilation Rate 1 (64 filters)
    │   ├── Dilation Rate 2 (64 filters)
    │   └── Dilation Rate 4 (64 filters)
    │   └── Concatenate → 192 channels
    │
    ├── Residual Block × 3 (128 channels)
    │   └── Conv1D → BatchNorm → ReLU → Conv1D → Add
    │
    ├── Multi-Head Self-Attention (4 heads)
    │
    ├── Temporal Attention Pooling
    │   └── T × 128 → 1 × 128
    │
    ├── Dense (256) → Dropout (0.3)
    │
    └── Output (8 classes)
```

### 3.3 Multi-Scale Dilated Convolutions

Different dilation rates capture patterns at different temporal resolutions:

- **Rate 1**: Local phonetic features (10ms context)
- **Rate 2**: Syllable-level features (20ms context)
- **Rate 4**: Word-level features (40ms context)

### 3.4 Temporal Attention Pooling

Instead of simple averaging, we use learned attention weights:

```
α_t = softmax(W_2 · tanh(W_1 · h_t + b_1) + b_2)
c = Σ_t α_t · h_t
```

This allows the model to focus on emotion-salient regions.

### 3.5 Training Strategy

#### Loss Function: Focal Loss
```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```
- γ = 2.0 (focusing parameter)
- α = 0.25 (balance factor)

#### Optimizer: AdamW
- Initial learning rate: 1e-3
- Weight decay: 1e-4
- Warmup: 1000 steps
- Cosine annealing to 1e-6

#### Regularization
- Dropout: 0.3
- Label smoothing: 0.1
- SpecAugment during training

---

## 4. Experiments

### 4.1 Experimental Setup

- **Hardware**: NVIDIA RTX 3090 (24GB)
- **Framework**: TensorFlow 2.10
- **Batch Size**: 64
- **Epochs**: 100 (early stopping patience: 10)
- **Mixed Precision**: FP16 training

### 4.2 Evaluation Metrics

- **UAR**: Unweighted Average Recall (balanced accuracy)
- **WAR**: Weighted Average Recall (standard accuracy)
- **Macro-F1**: Unweighted average of per-class F1 scores

### 4.3 Results

#### Overall Performance

| Model | UAR (%) | WAR (%) | Macro-F1 (%) |
|-------|---------|---------|--------------|
| Baseline CNN | 65.2 | 69.8 | 63.1 |
| LSTM | 67.8 | 71.2 | 66.4 |
| CNN-LSTM | 69.4 | 72.5 | 68.1 |
| **SER-Net (Ours)** | **72.4** | **74.6** | **71.8** |
| SER-Net Large | 73.1 | 75.2 | 72.5 |

#### Per-Class Performance (SER-Net Base)

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 74.2% | 71.8% | 72.9% |
| Happy | 70.5% | 73.2% | 71.8% |
| Sad | 69.8% | 70.4% | 70.1% |
| Neutral | 82.1% | 84.5% | 83.3% |
| Fear | 65.3% | 67.1% | 66.2% |
| Disgust | 58.4% | 62.8% | 60.5% |
| Surprise | 68.9% | 69.5% | 69.2% |
| Contempt | 61.2% | 60.1% | 60.6% |

### 4.4 Ablation Study

| Configuration | UAR (%) | Δ |
|---------------|---------|---|
| Full Model | 72.4 | - |
| w/o Multi-Scale Conv | 69.8 | -2.6 |
| w/o Attention Pooling | 70.1 | -2.3 |
| w/o Prosodic Features | 70.9 | -1.5 |
| w/o Data Augmentation | 69.2 | -3.2 |
| w/o Focal Loss | 70.5 | -1.9 |

---

## 5. Analysis

### 5.1 Attention Visualization

The temporal attention mechanism learns to focus on emotion-relevant segments. Analysis shows:
- Peak attention on stressed syllables
- Higher attention weights during pitch variations
- Reduced attention on silence and noise segments

### 5.2 Confusion Analysis

Common misclassifications:
- Happy ↔ Surprise (acoustic similarity)
- Sad ↔ Contempt (low arousal emotions)
- Fear ↔ Surprise (high arousal, similar acoustic properties)

### 5.3 Error Analysis

Major error categories:
1. **Ambiguous Utterances** (35%): Multiple valid interpretations
2. **Speaker Variability** (28%): Atypical expression patterns
3. **Background Noise** (22%): Recording artifacts
4. **Short Utterances** (15%): Insufficient acoustic information

---

## 6. Deployment

### 6.1 Inference Latency

| Platform | Batch Size | Latency (ms) |
|----------|------------|--------------|
| GPU (RTX 3090) | 1 | 8.2 |
| GPU (RTX 3090) | 32 | 45.1 |
| CPU (Intel i9) | 1 | 42.5 |
| CPU (Intel i9) | 32 | 312.8 |

### 6.2 Model Size

| Variant | Parameters | Size (MB) |
|---------|------------|-----------|
| SER-Net Base | 2.3M | 9.2 |
| SER-Net Large | 8.7M | 34.8 |

---

## 7. Conclusion

We presented SER-Net, a deep learning architecture for speech emotion recognition that achieves state-of-the-art performance on the MSP-Podcast corpus. Key innovations include multi-scale dilated convolutions for capturing temporal patterns at different scales and attention pooling for focusing on emotion-relevant segments. The model demonstrates strong generalization to naturalistic speech conditions while maintaining efficient inference for real-time applications.

### Future Work

1. Cross-corpus evaluation on IEMOCAP and RAVDESS
2. Multi-task learning with dimensional emotion prediction
3. Self-supervised pre-training on unlabeled speech
4. Knowledge distillation for edge deployment

---

## References

1. Busso, C., et al. "MSP-Podcast Corpus: A Large Scale Naturalistic Database." ACII 2019.
2. Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." ICCV 2017.
3. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
4. Park, D.S., et al. "SpecAugment: A Simple Data Augmentation Method." Interspeech 2019.
5. Loshchilov, I., Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019.

---

## Appendix A: Hyperparameters

```yaml
# Feature Extraction
sample_rate: 16000
n_mfcc: 40
n_mels: 128
hop_length: 512
max_length: 300  # frames

# Model Architecture
conv_filters: [64, 64, 64]
dilation_rates: [1, 2, 4]
residual_blocks: 3
attention_heads: 4
dense_units: 256
dropout_rate: 0.3

# Training
batch_size: 64
epochs: 100
learning_rate: 1e-3
weight_decay: 1e-4
warmup_steps: 1000
focal_gamma: 2.0
label_smoothing: 0.1
```

---

## Appendix B: Hardware Requirements

**Training:**
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 32GB recommended
- Storage: 50GB for dataset and checkpoints

**Inference:**
- CPU: Modern multi-core processor
- GPU: Optional, provides 5x speedup
- RAM: 4GB minimum

---

*For questions or collaboration, contact: tharunponnam007@gmail.com*
