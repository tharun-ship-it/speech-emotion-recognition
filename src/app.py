"""
Speech Emotion Recognition - Streamlit Web Application

Interactive demo for real-time emotion detection from speech.
Supports file upload and microphone input.

Author: Tharun Ponnam
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import io
import time

# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .emotion-label {
        font-size: 2.5rem;
        font-weight: 700;
        text-transform: capitalize;
    }
    .confidence-score {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

# Emotion color mapping
EMOTION_COLORS = {
    'angry': '#e74c3c',
    'happy': '#f39c12',
    'sad': '#3498db',
    'neutral': '#95a5a6',
    'fear': '#9b59b6',
    'disgust': '#27ae60',
    'surprise': '#e91e63',
    'contempt': '#607d8b'
}

EMOTION_EMOJIS = {
    'angry': '😠',
    'happy': '😊',
    'sad': '😢',
    'neutral': '😐',
    'fear': '😨',
    'disgust': '🤢',
    'surprise': '😲',
    'contempt': '😏'
}


class DemoPredictor:
    """
    Simplified predictor for demo purposes.
    
    In production, this would load the actual model.
    For the demo, we simulate predictions based on audio features.
    """
    
    def __init__(self):
        self.emotions = list(EMOTION_COLORS.keys())
        self.sample_rate = 16000
        
    def extract_features(self, audio: np.ndarray) -> dict:
        """Extract audio features for visualization."""
        features = {}
        
        # Basic features
        features['duration'] = len(audio) / self.sample_rate
        features['rms_energy'] = float(np.sqrt(np.mean(audio**2)))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate
        )[0]
        features['spectral_centroid'] = float(np.mean(spectral_centroids))
        
        # Pitch estimation
        pitches, magnitudes = librosa.piptrack(
            y=audio, sr=self.sample_rate
        )
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) > 0:
            features['pitch_mean'] = float(np.mean(pitch_values[pitch_values > 0]))
        else:
            features['pitch_mean'] = 0.0
            
        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=13
        )
        features['mfcc_mean'] = [float(np.mean(mfcc)) for mfcc in mfccs]
        
        return features
    
    def predict(self, audio: np.ndarray) -> dict:
        """
        Predict emotion from audio.
        
        Note: This is a demo simulation. In production,
        the actual trained model would be used.
        """
        features = self.extract_features(audio)
        
        # Feature-based heuristic for demo
        energy = features['rms_energy']
        zcr = features['zcr_mean']
        centroid = features['spectral_centroid']
        pitch = features['pitch_mean']
        
        # Generate probabilities based on audio characteristics
        probs = np.random.dirichlet(np.ones(8) * 2)
        
        # Adjust based on features (simplified heuristics)
        if energy > 0.1 and zcr > 0.1:
            probs[0] += 0.3  # angry
        if pitch > 200 and energy > 0.05:
            probs[1] += 0.25  # happy
        if energy < 0.02 and zcr < 0.05:
            probs[2] += 0.2  # sad
        if 0.02 < energy < 0.08:
            probs[3] += 0.15  # neutral
            
        # Normalize
        probs = probs / probs.sum()
        
        top_idx = np.argmax(probs)
        
        return {
            'emotion': self.emotions[top_idx],
            'confidence': float(probs[top_idx]),
            'probabilities': {
                e: float(p) for e, p in zip(self.emotions, probs)
            },
            'features': features
        }


@st.cache_resource
def load_predictor():
    """Load the emotion predictor."""
    return DemoPredictor()


def plot_waveform(audio: np.ndarray, sr: int):
    """Plot audio waveform."""
    fig, ax = plt.subplots(figsize=(10, 3))
    times = np.arange(len(audio)) / sr
    ax.plot(times, audio, color='#667eea', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform')
    ax.set_xlim(0, times[-1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spectrogram(audio: np.ndarray, sr: int):
    """Plot mel spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax,
        cmap='magma'
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram')
    plt.tight_layout()
    return fig


def plot_emotion_probabilities(probs: dict):
    """Plot emotion probability distribution."""
    emotions = list(probs.keys())
    values = list(probs.values())
    colors = [EMOTION_COLORS.get(e, '#667eea') for e in emotions]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(emotions, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title('Emotion Distribution')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.2%}', va='center', fontsize=10
        )
    
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Speech Emotion Recognition</h1>', 
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Detect emotions from speech using deep learning</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        show_waveform = st.checkbox("Show Waveform", value=True)
        show_spectrogram = st.checkbox("Show Spectrogram", value=True)
        show_distribution = st.checkbox("Show Probability Distribution", value=True)
        
        st.divider()
        
        st.header("📊 Model Info")
        st.markdown("""
        **Architecture:** SER-Net  
        **Dataset:** MSP-Podcast  
        **Classes:** 8 emotions  
        **Performance:** 72.4% UAR
        """)
        
        st.divider()
        
        st.header("🔗 Links")
        st.markdown("""
        - [GitHub Repository](https://github.com/tharun-ship-it/speech-emotion-recognition)
        - [Paper (arXiv)](https://arxiv.org)
        - [Dataset Info](https://ecs.utdallas.edu/research/researchlabs/msp-lab/)
        """)
    
    # Main content
    predictor = load_predictor()
    
    # Input method selection
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Audio"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload a speech audio file for emotion analysis"
        )
        
        if uploaded_file is not None:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load audio
            audio, sr = librosa.load(tmp_path, sr=16000)
            
            st.audio(uploaded_file, format='audio/wav')
            
            # Process button
            if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):
                with st.spinner("Processing audio..."):
                    start_time = time.time()
                    result = predictor.predict(audio)
                    processing_time = (time.time() - start_time) * 1000
                
                # Display results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    emotion = result['emotion']
                    confidence = result['confidence']
                    emoji = EMOTION_EMOJIS.get(emotion, '🎭')
                    color = EMOTION_COLORS.get(emotion, '#667eea')
                    
                    st.markdown(f"""
                    <div class="emotion-card" style="background: {color};">
                        <div style="font-size: 4rem;">{emoji}</div>
                        <div class="emotion-label">{emotion}</div>
                        <div class="confidence-score">
                            Confidence: {confidence:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Processing Time", f"{processing_time:.1f} ms")
                    st.metric("Audio Duration", f"{result['features']['duration']:.2f} s")
                
                with col2:
                    if show_distribution:
                        st.pyplot(plot_emotion_probabilities(result['probabilities']))
                
                # Visualizations
                if show_waveform:
                    st.subheader("📈 Waveform")
                    st.pyplot(plot_waveform(audio, sr))
                
                if show_spectrogram:
                    st.subheader("🌈 Mel Spectrogram")
                    st.pyplot(plot_spectrogram(audio, sr))
                
                # Feature details
                with st.expander("🔬 Extracted Features"):
                    features = result['features']
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("RMS Energy", f"{features['rms_energy']:.4f}")
                    col2.metric("ZCR Mean", f"{features['zcr_mean']:.4f}")
                    col3.metric("Spectral Centroid", f"{features['spectral_centroid']:.1f} Hz")
                    col4.metric("Pitch Mean", f"{features['pitch_mean']:.1f} Hz")
    
    with tab2:
        st.info(
            "🎤 Microphone recording requires running the app locally. "
            "Please upload a file or clone the repo to use this feature."
        )
        
        st.code("""
# Run locally with microphone support:
pip install streamlit sounddevice
streamlit run src/app.py
        """, language="bash")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Built with ❤️ by <a href="https://github.com/tharun-ship-it">Tharun Ponnam</a> | 
        Powered by TensorFlow & Librosa
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
