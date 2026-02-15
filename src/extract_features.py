"""
Feature Extraction for Piezoelectric Vibration Data
Extracts time-domain and frequency-domain features from raw sensor data.
"""

import numpy as np
import os
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq

def extract_time_features(data):
    """Extract time-domain statistical features."""
    features = {
        'mean': np.mean(data),
        'std': np.std(data),
        'max': np.max(data),
        'min': np.min(data),
        'peak_to_peak': np.ptp(data),
        'rms': np.sqrt(np.mean(data**2)),
        'energy': np.sum(data**2),
    }
    return features

def extract_frequency_features(data, sample_rate=100):
    """Extract frequency-domain features using FFT."""
    # Compute FFT
    n = len(data)
    fft_vals = fft(data)
    fft_freqs = fftfreq(n, 1/sample_rate)
    
    # Take only positive frequencies
    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    magnitudes = np.abs(fft_vals[pos_mask])
    
    # Find dominant frequency
    dominant_idx = np.argmax(magnitudes)
    dominant_freq = freqs[dominant_idx]
    dominant_magnitude = magnitudes[dominant_idx]
    
    # Spectral features
    total_power = np.sum(magnitudes**2)
    spectral_centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0
    
    features = {
        'dominant_frequency': dominant_freq,
        'dominant_magnitude': dominant_magnitude,
        'spectral_centroid': spectral_centroid,
        'total_spectral_power': total_power,
        'mean_magnitude': np.mean(magnitudes),
        'std_magnitude': np.std(magnitudes),
    }
    
    return features

def process_trial(filepath, sample_rate=100):
    """Process a single trial and extract all features."""
    # Load data
    data = np.load(filepath)
    
    # Extract features
    time_feats = extract_time_features(data)
    freq_feats = extract_frequency_features(data, sample_rate)
    
    # Combine all features
    all_features = {**time_feats, **freq_feats}
    
    return all_features

def main():
    """Main feature extraction pipeline."""
    print("=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    print()
    
    # Setup paths
    raw_dir = Path("data/raw")
    features_dir = Path("data/features")
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all data files
    data_files = sorted(list(raw_dir.glob("position_*_trial_*.npy")))
    
    if len(data_files) == 0:
        print("❌ No data files found in data/raw/")
        print("   Make sure you've run data collection first!")
        return
    
    print(f"Found {len(data_files)} data files")
    print()
    
    # Storage for all features and labels
    all_features = []
    all_labels = []
    feature_names = None
    
    # Process each file
    for i, filepath in enumerate(data_files):
        # Extract position and trial from filename
        # Format: position_X_trial_Y.npy
        parts = filepath.stem.split('_')
        position = int(parts[1])
        trial = int(parts[3])
        
        # Extract features
        features = process_trial(filepath, sample_rate=100)
        
        # Store feature names (from first file)
        if feature_names is None:
            feature_names = list(features.keys())
        
        # Convert to array
        feature_vector = np.array([features[name] for name in feature_names])
        
        all_features.append(feature_vector)
        all_labels.append(position)
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(data_files)} files...")
    
    print(f"✓ Processed all {len(data_files)} files")
    print()
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print()
    
    # Save processed features
    np.save(features_dir / "features.npy", X)
    np.save(features_dir / "labels.npy", y)
    
    # Save feature names
    with open(features_dir / "feature_names.txt", 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    print("✓ Saved features to data/features/")
    print()
    
    # Print summary statistics
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {len(feature_names)}")
    print()
    
    print("Features extracted:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name}")
    print()
    
    print("Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for pos, count in zip(unique, counts):
        print(f"  Position {pos}: {count} samples")
    print()
    
    print("=" * 60)
    print("✓ Feature extraction complete!")
    print("=" * 60)
    print()
    print("Next step: python src/train_classifier.py")

if __name__ == "__main__":
    main()
