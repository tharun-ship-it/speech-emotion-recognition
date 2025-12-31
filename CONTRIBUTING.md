# Contributing to Speech Emotion Recognition

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a branch for your changes
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/speech-emotion-recognition.git
cd speech-emotion-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black isort flake8 mypy

# Install package in editable mode
pip install -e .
```

### Verifying Installation

```bash
# Run tests
pytest tests/ -v

# Check code style
black --check src/
flake8 src/
```

## Making Changes

### Branching Strategy

- `main`: Stable release branch
- `develop`: Development branch
- `feature/*`: New features
- `fix/*`: Bug fixes
- `docs/*`: Documentation updates

### Creating a Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a new branch
git checkout -b feature/your-feature-name
```

### Commit Messages

Follow the conventional commits specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(models): add support for wav2vec features
fix(preprocessing): handle edge case in MFCC extraction
docs(readme): update installation instructions
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ser.py -v

# Run specific test
pytest tests/test_ser.py::TestSERNet::test_forward_pass -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Follow the existing test structure
- Aim for good coverage of new code

```python
def test_feature_extraction_shape():
    """Test that feature extractor returns correct shape."""
    extractor = AudioFeatureExtractor(n_mfcc=40)
    audio = np.random.randn(16000)  # 1 second at 16kHz
    
    features = extractor.extract(audio)
    
    assert features.ndim == 2
    assert features.shape[1] == 180  # Expected feature dim
```

## Submitting Changes

### Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Update the CHANGELOG if applicable
4. Submit the pull request

### Pull Request Template

```markdown
## Description
[Describe your changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

## Testing
- [ ] I have added tests for my changes
- [ ] All existing tests pass

## Checklist
- [ ] My code follows the project style guidelines
- [ ] I have updated the documentation
- [ ] I have added necessary tests
```

## Style Guidelines

### Python Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

### Docstrings

Use Google-style docstrings:

```python
def extract_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract acoustic features from audio signal.
    
    Args:
        audio: Audio waveform as numpy array.
        sample_rate: Sample rate of the audio in Hz.
        
    Returns:
        Feature array of shape (time_steps, feature_dim).
        
    Raises:
        ValueError: If audio is empty or invalid.
        
    Example:
        >>> features = extract_features(audio, sample_rate=16000)
        >>> print(features.shape)
        (100, 180)
    """
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Dict, Optional, Tuple

def predict_emotion(
    audio_path: str,
    model: tf.keras.Model,
    threshold: float = 0.5
) -> Dict[str, float]:
    ...
```

## Areas for Contribution

### Good First Issues

- Documentation improvements
- Test coverage improvements
- Bug fixes with clear reproduction steps
- Code style improvements

### Feature Requests

- Support for additional audio formats
- New model architectures
- Additional emotion datasets
- Visualization improvements
- Performance optimizations

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase

Thank you for contributing!
