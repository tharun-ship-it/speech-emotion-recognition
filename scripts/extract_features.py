#!/usr/bin/env python3
"""
Feature Extraction Script for Speech Emotion Recognition

Extracts acoustic features from audio files and saves them
for efficient training and inference.

Usage:
    python scripts/extract_features.py --input_dir /path/to/audio --output_dir /path/to/features
    python scripts/extract_features.py --input_dir data/audio --output_dir data/features --n_jobs 8

Author: Tharun Ponnam
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import AudioFeatureExtractor


def setup_logging(log_dir: str) -> logging.Logger:
    """Configure logging."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'extract_features_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract acoustic features from audio files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help='Directory to save extracted features'
    )
    parser.add_argument(
        '--labels_file',
        type=str,
        help='Path to labels CSV (optional, for filtering)'
    )
    
    # Feature extraction parameters
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Target sample rate'
    )
    parser.add_argument(
        '--n_mfcc',
        type=int,
        default=40,
        help='Number of MFCCs'
    )
    parser.add_argument(
        '--n_mels',
        type=int,
        default=128,
        help='Number of mel bands'
    )
    parser.add_argument(
        '--hop_length',
        type=int,
        default=512,
        help='Hop length for STFT'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=300,
        help='Maximum sequence length (frames)'
    )
    
    # Processing
    parser.add_argument(
        '--n_jobs', '-j',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--file_ext',
        type=str,
        default='.wav',
        help='Audio file extension'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing features'
    )
    
    return parser.parse_args()


def get_audio_files(
    input_dir: str,
    file_ext: str,
    labels_file: str = None
) -> List[str]:
    """
    Get list of audio files to process.
    
    Args:
        input_dir: Directory containing audio files
        file_ext: File extension to look for
        labels_file: Optional labels file for filtering
        
    Returns:
        List of audio file paths
    """
    input_path = Path(input_dir)
    
    # Get all audio files
    audio_files = list(input_path.rglob(f'*{file_ext}'))
    
    # Filter by labels file if provided
    if labels_file and os.path.exists(labels_file):
        df = pd.read_csv(labels_file)
        if 'file_id' in df.columns:
            valid_ids = set(df['file_id'].values)
            audio_files = [
                f for f in audio_files
                if f.stem in valid_ids
            ]
    
    return [str(f) for f in audio_files]


def extract_single_file(args: Tuple) -> Tuple[str, np.ndarray, str]:
    """
    Extract features from a single audio file.
    
    Args:
        args: Tuple of (audio_path, extractor_params)
        
    Returns:
        Tuple of (file_id, features, error_message)
    """
    audio_path, extractor_params = args
    
    try:
        # Create extractor (can't be pickled, so create per-process)
        extractor = AudioFeatureExtractor(**extractor_params)
        
        # Extract features
        features = extractor.extract_from_file(audio_path)
        
        file_id = Path(audio_path).stem
        return (file_id, features, None)
        
    except Exception as e:
        file_id = Path(audio_path).stem
        return (file_id, None, str(e))


def main():
    """Main feature extraction function."""
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs')
    logger = setup_logging(log_dir)
    
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Get audio files
    audio_files = get_audio_files(
        args.input_dir,
        args.file_ext,
        args.labels_file
    )
    logger.info(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        logger.error("No audio files found!")
        return
    
    # Filter already processed files
    if not args.overwrite:
        features_dir = os.path.join(args.output_dir, 'features')
        if os.path.exists(features_dir):
            existing = {
                Path(f).stem.replace('_features', '')
                for f in os.listdir(features_dir)
                if f.endswith('.npy')
            }
            original_count = len(audio_files)
            audio_files = [
                f for f in audio_files
                if Path(f).stem not in existing
            ]
            logger.info(
                f"Skipping {original_count - len(audio_files)} "
                "already processed files"
            )
    
    if len(audio_files) == 0:
        logger.info("All files already processed!")
        return
    
    # Extractor parameters (serializable)
    extractor_params = {
        'sample_rate': args.sample_rate,
        'n_mfcc': args.n_mfcc,
        'n_mels': args.n_mels,
        'hop_length': args.hop_length,
        'max_length': args.max_length
    }
    
    # Prepare arguments for parallel processing
    process_args = [(f, extractor_params) for f in audio_files]
    
    # Create output directories
    features_dir = os.path.join(args.output_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    
    # Process files
    successful = 0
    failed = 0
    errors = []
    
    logger.info(f"Processing with {args.n_jobs} workers...")
    
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {
            executor.submit(extract_single_file, arg): arg[0]
            for arg in process_args
        }
        
        with tqdm(total=len(audio_files), desc="Extracting features") as pbar:
            for future in as_completed(futures):
                file_id, features, error = future.result()
                
                if error is None and features is not None:
                    # Save features
                    output_path = os.path.join(
                        features_dir,
                        f'{file_id}_features.npy'
                    )
                    np.save(output_path, features)
                    successful += 1
                else:
                    failed += 1
                    errors.append((file_id, error))
                
                pbar.update(1)
    
    # Summary
    logger.info("=" * 50)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    # Save errors log
    if errors:
        errors_path = os.path.join(args.output_dir, 'extraction_errors.csv')
        pd.DataFrame(errors, columns=['file_id', 'error']).to_csv(
            errors_path, index=False
        )
        logger.info(f"Error log saved to {errors_path}")
    
    # Save metadata
    metadata = {
        'sample_rate': args.sample_rate,
        'n_mfcc': args.n_mfcc,
        'n_mels': args.n_mels,
        'hop_length': args.hop_length,
        'max_length': args.max_length,
        'total_files': successful + failed,
        'successful': successful,
        'failed': failed
    }
    
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")


if __name__ == '__main__':
    main()
