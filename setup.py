"""
Speech Emotion Recognition Package Setup

Author: Tharun Ponnam
Email: tharunponnam007@gmail.com
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Core dependencies
install_requires = [
    "tensorflow>=2.10.0,<2.16.0",
    "librosa>=0.10.0",
    "numpy>=1.23.0,<2.0.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.2.0",
    "pandas>=1.5.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "soundfile>=0.12.0",
]

# Optional dependencies
extras_require = {
    "api": [
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "python-multipart>=0.0.6",
    ],
    "app": [
        "streamlit>=1.28.0",
    ],
    "viz": [
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "plotly>=5.13.0",
    ],
    "dev": [
        "pytest>=7.3.0",
        "pytest-cov>=4.1.0",
        "black>=23.3.0",
        "flake8>=6.0.0",
        "mypy>=1.3.0",
    ],
    "tracking": [
        "wandb>=0.15.0",
        "tensorboard>=2.12.0",
    ],
}

# Full installation
extras_require["full"] = sum(extras_require.values(), [])

setup(
    name="speech-emotion-recognition",
    version="1.0.0",
    author="Tharun Ponnam",
    author_email="tharunponnam007@gmail.com",
    description="Speech Emotion Recognition using Multi-Scale Temporal Convolutions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tharun-ship-it/speech-emotion-recognition",
    project_urls={
        "Bug Tracker": "https://github.com/tharun-ship-it/speech-emotion-recognition/issues",
        "Documentation": "https://github.com/tharun-ship-it/speech-emotion-recognition#readme",
        "Source": "https://github.com/tharun-ship-it/speech-emotion-recognition",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "ser-train=scripts.train:main",
            "ser-predict=scripts.predict:main",
            "ser-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "speech emotion recognition",
        "deep learning",
        "tensorflow",
        "audio processing",
        "affective computing",
        "speech processing",
        "emotion detection",
    ],
)
