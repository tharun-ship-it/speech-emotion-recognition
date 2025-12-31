"""
Unit Tests for Speech Emotion Recognition

Comprehensive test suite covering data processing, model architecture,
training utilities, and inference pipeline.

Author: Tharun Ponnam
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAudioFeatureExtractor(unittest.TestCase):
    """Tests for AudioFeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.data.preprocessing import AudioFeatureExtractor
        self.extractor = AudioFeatureExtractor(
            sample_rate=16000,
            n_mfcc=40,
            n_mels=128,
            hop_length=512
        )
        
        # Create synthetic audio signal (1 second at 16kHz)
        self.sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        # Generate a sine wave with some harmonics
        self.audio = (
            0.5 * np.sin(2 * np.pi * 440 * t) +
            0.3 * np.sin(2 * np.pi * 880 * t) +
            0.2 * np.sin(2 * np.pi * 1320 * t)
        ).astype(np.float32)
    
    def test_mfcc_extraction(self):
        """Test MFCC feature extraction."""
        mfcc = self.extractor._extract_mfcc(self.audio)
        
        self.assertIsInstance(mfcc, np.ndarray)
        self.assertEqual(mfcc.shape[0], 40)  # n_mfcc
        self.assertGreater(mfcc.shape[1], 0)  # time frames
    
    def test_mel_spectrogram_extraction(self):
        """Test mel spectrogram extraction."""
        mel = self.extractor._extract_mel_spectrogram(self.audio)
        
        self.assertIsInstance(mel, np.ndarray)
        self.assertEqual(mel.shape[0], 128)  # n_mels
        self.assertGreater(mel.shape[1], 0)
    
    def test_combined_features(self):
        """Test combined feature extraction."""
        features = self.extractor.extract(self.audio)
        
        self.assertIsInstance(features, np.ndarray)
        # Should have time x feature_dim shape
        self.assertEqual(len(features.shape), 2)
    
    def test_feature_normalization(self):
        """Test that features are normalized."""
        features = self.extractor.extract(self.audio)
        
        # Check that values are in reasonable range
        self.assertTrue(np.isfinite(features).all())


class TestDataAugmentation(unittest.TestCase):
    """Tests for audio augmentation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.data.augmentation import AudioAugmentor
        self.augmentor = AudioAugmentor()
        
        # Create synthetic audio
        self.sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        self.audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    def test_add_noise(self):
        """Test noise augmentation."""
        augmented = self.augmentor.add_noise(self.audio, snr_db=20)
        
        self.assertEqual(augmented.shape, self.audio.shape)
        # Should be different from original
        self.assertFalse(np.allclose(augmented, self.audio))
    
    def test_time_stretch(self):
        """Test time stretching."""
        rate = 1.2
        stretched = self.augmentor.time_stretch(self.audio, rate)
        
        # Length should change
        expected_length = int(len(self.audio) / rate)
        self.assertAlmostEqual(len(stretched), expected_length, delta=512)
    
    def test_pitch_shift(self):
        """Test pitch shifting."""
        shifted = self.augmentor.pitch_shift(
            self.audio, self.sample_rate, n_steps=2
        )
        
        self.assertEqual(shifted.shape, self.audio.shape)


class TestSERNet(unittest.TestCase):
    """Tests for SER-Net model architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.models import create_ser_model
        self.input_shape = (300, 180)  # time_steps, features
        self.num_classes = 8
        self.batch_size = 4
    
    def test_model_creation_base(self):
        """Test base model creation."""
        from src.models import create_ser_model
        model = create_ser_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            variant='base'
        )
        
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(
            model.input_shape[1:],
            self.input_shape
        )
        
        # Check output shape
        self.assertEqual(
            model.output_shape[-1],
            self.num_classes
        )
    
    def test_model_creation_large(self):
        """Test large model variant creation."""
        from src.models import create_ser_model
        model = create_ser_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            variant='large'
        )
        
        self.assertIsInstance(model, tf.keras.Model)
    
    def test_forward_pass(self):
        """Test forward pass through model."""
        from src.models import create_ser_model
        model = create_ser_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        # Create dummy input
        x = np.random.randn(self.batch_size, *self.input_shape).astype(np.float32)
        
        # Forward pass
        output = model(x, training=False)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
        # Check output is probability distribution
        probs = tf.nn.softmax(output)
        self.assertTrue(np.allclose(tf.reduce_sum(probs, axis=-1).numpy(), 1.0))
    
    def test_model_trainable(self):
        """Test that model is trainable."""
        from src.models import create_ser_model
        model = create_ser_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy data
        x = np.random.randn(self.batch_size, *self.input_shape).astype(np.float32)
        y = np.random.randint(0, self.num_classes, self.batch_size)
        
        # Train one step
        initial_weights = model.get_weights()[0].copy()
        model.fit(x, y, epochs=1, verbose=0)
        final_weights = model.get_weights()[0]
        
        # Weights should change
        self.assertFalse(np.allclose(initial_weights, final_weights))


class TestAttentionLayers(unittest.TestCase):
    """Tests for attention mechanism layers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_length = 100
        self.embed_dim = 64
    
    def test_multi_head_attention(self):
        """Test multi-head self-attention."""
        from src.models.attention import MultiHeadSelfAttention
        
        layer = MultiHeadSelfAttention(embed_dim=self.embed_dim, num_heads=4)
        x = tf.random.normal((self.batch_size, self.seq_length, self.embed_dim))
        
        output = layer(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_temporal_attention_pooling(self):
        """Test temporal attention pooling."""
        from src.models.attention import TemporalAttentionPooling
        
        layer = TemporalAttentionPooling(units=self.embed_dim)
        x = tf.random.normal((self.batch_size, self.seq_length, self.embed_dim))
        
        output = layer(x)
        
        # Should reduce sequence dimension
        self.assertEqual(output.shape, (self.batch_size, self.embed_dim))
    
    def test_positional_encoding(self):
        """Test positional encoding."""
        from src.models.attention import PositionalEncoding
        
        layer = PositionalEncoding(max_length=200)
        x = tf.random.normal((self.batch_size, self.seq_length, self.embed_dim))
        
        output = layer(x)
        
        self.assertEqual(output.shape, x.shape)
        # Should be different from input
        self.assertFalse(np.allclose(output.numpy(), x.numpy()))


class TestCustomLayers(unittest.TestCase):
    """Tests for custom layer implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_length = 100
        self.channels = 64
    
    def test_dilated_conv_block(self):
        """Test dilated convolution block."""
        from src.models.layers import DilatedConvBlock
        
        layer = DilatedConvBlock(filters=self.channels, dilation_rate=2)
        x = tf.random.normal((self.batch_size, self.seq_length, self.channels))
        
        output = layer(x)
        
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[-1], self.channels)
    
    def test_residual_block(self):
        """Test residual block."""
        from src.models.layers import ResidualBlock
        
        layer = ResidualBlock(filters=self.channels)
        x = tf.random.normal((self.batch_size, self.seq_length, self.channels))
        
        output = layer(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_squeeze_excitation(self):
        """Test squeeze-and-excitation block."""
        from src.models.layers import SqueezeExcitation
        
        layer = SqueezeExcitation(channels=self.channels)
        x = tf.random.normal((self.batch_size, self.seq_length, self.channels))
        
        output = layer(x)
        
        self.assertEqual(output.shape, x.shape)


class TestLossFunctions(unittest.TestCase):
    """Tests for custom loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.num_classes = 8
        
        # Create dummy predictions and labels
        self.y_pred = tf.nn.softmax(
            tf.random.normal((self.batch_size, self.num_classes))
        )
        self.y_true = tf.random.uniform(
            (self.batch_size,),
            minval=0,
            maxval=self.num_classes,
            dtype=tf.int32
        )
    
    def test_focal_loss(self):
        """Test focal loss computation."""
        from src.training.losses import focal_loss
        
        loss = focal_loss(self.y_true, self.y_pred, gamma=2.0)
        
        self.assertIsInstance(loss, tf.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.numpy(), 0)
    
    def test_label_smoothing_loss(self):
        """Test label smoothing loss."""
        from src.training.losses import label_smoothing_loss
        
        loss = label_smoothing_loss(
            self.y_true,
            self.y_pred,
            smoothing=0.1
        )
        
        self.assertIsInstance(loss, tf.Tensor)
        self.assertGreater(loss.numpy(), 0)
    
    def test_class_weights_computation(self):
        """Test class weight computation."""
        from src.training.losses import get_class_weights
        
        # Create imbalanced labels
        labels = np.concatenate([
            np.zeros(100),
            np.ones(50),
            np.full(25, 2)
        ])
        
        weights = get_class_weights(labels, num_classes=3)
        
        self.assertEqual(len(weights), 3)
        # More rare classes should have higher weights
        self.assertGreater(weights[2], weights[0])


class TestInference(unittest.TestCase):
    """Tests for inference pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_shape = (300, 180)
        self.num_classes = 8
    
    def test_predictor_initialization(self):
        """Test SERPredictor initialization with model."""
        from src.models import create_ser_model
        from src.inference import SERPredictor
        
        # Create and save a temporary model
        model = create_ser_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model')
            model.save(model_path)
            
            # Initialize predictor
            predictor = SERPredictor(
                model_path=model_path,
                num_classes=self.num_classes
            )
            
            self.assertIsNotNone(predictor.model)
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        from src.models import create_ser_model
        from src.inference import SERPredictor
        
        model = create_ser_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model')
            model.save(model_path)
            
            predictor = SERPredictor(
                model_path=model_path,
                num_classes=self.num_classes
            )
            
            # Create dummy features
            features = np.random.randn(4, *self.input_shape).astype(np.float32)
            
            predictions = predictor.predict_batch(features)
            
            self.assertEqual(len(predictions), 4)
            for pred in predictions:
                self.assertIn('emotion', pred)
                self.assertIn('confidence', pred)


class TestCallbacks(unittest.TestCase):
    """Tests for training callbacks."""
    
    def test_warmup_cosine_decay(self):
        """Test warmup cosine decay schedule."""
        from src.training.callbacks import WarmupCosineDecay
        
        initial_lr = 1e-3
        warmup_steps = 100
        total_steps = 1000
        
        schedule = WarmupCosineDecay(
            initial_learning_rate=initial_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        
        # Test warmup phase
        lr_at_0 = schedule(0)
        lr_at_50 = schedule(50)
        lr_at_100 = schedule(100)
        
        self.assertLess(lr_at_0, lr_at_50)
        self.assertLess(lr_at_50, lr_at_100)
        
        # Test decay phase
        lr_at_500 = schedule(500)
        lr_at_900 = schedule(900)
        
        self.assertGreater(lr_at_100, lr_at_500)
        self.assertGreater(lr_at_500, lr_at_900)


class TestDataset(unittest.TestCase):
    """Tests for dataset loading utilities."""
    
    def test_tf_dataset_creation(self):
        """Test TensorFlow dataset creation."""
        from src.data.dataset import create_tf_dataset
        
        # Create dummy data
        features = np.random.randn(100, 300, 180).astype(np.float32)
        labels = np.random.randint(0, 8, 100)
        
        dataset = create_tf_dataset(
            features,
            labels,
            batch_size=16,
            shuffle=True
        )
        
        # Check dataset properties
        for batch_features, batch_labels in dataset.take(1):
            self.assertEqual(batch_features.shape[0], 16)
            self.assertEqual(batch_labels.shape[0], 16)


if __name__ == '__main__':
    unittest.main(verbosity=2)
