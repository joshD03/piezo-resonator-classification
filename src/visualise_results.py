"""
Visualize Results for Piezoelectric Vibration Classifier
Creates confusion matrix, performance plots, and animations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_test, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[f"Pos {i}" for i in np.unique(y_test)]
    )

    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix - Vibration Position Classification', 
                 fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {save_path}")
    plt.close()

def plot_feature_importance(save_path):
    """Plot feature importance from trained model."""
    import pickle

    # Load model and feature names
    with open('models/classifier.pkl', 'rb') as f:
        clf = pickle.load(f)

    feature_names_file = Path("data/features/feature_names.txt")
    if not feature_names_file.exists():
        print("⚠ Feature names file not found, skipping importance plot")
        return

    with open(feature_names_file, 'r') as f:
        feature_names = [line.strip() for line in f]

    # Get importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title('Feature Importance - Random Forest Classifier', fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved feature importance to {save_path}")
    plt.close()

def plot_sample_signals(save_path):
    """Plot example raw signals from each position."""
    raw_dir = Path("data/raw")

    fig, axes = plt.subplots(5, 1, figsize=(12, 10))

    for pos in range(5):
        # Load first trial from each position
        filepath = raw_dir / f"position_{pos}_trial_0.npy"

        if not filepath.exists():
            continue

        data = np.load(filepath)
        time = np.arange(len(data)) * 0.01  # 100 Hz sampling = 0.01s

        axes[pos].plot(time, data, linewidth=0.8)
        axes[pos].set_ylabel('ADC Value', fontsize=10)
        axes[pos].set_title(f'Position {pos}', fontsize=11)
        axes[pos].grid(alpha=0.3)

        if pos == 4:
            axes[pos].set_xlabel('Time (s)', fontsize=11)

    fig.suptitle('Sample Vibration Signals - Each Position', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved sample signals to {save_path}")
    plt.close()

def create_vibration_animation(save_path):
    """Create animated GIF showing real vibration data for all positions."""
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("⚠ Animation requires matplotlib.animation, skipping GIF creation")
        return

    raw_dir = Path("data/raw")

    # Load first trial from each position
    signals = []
    positions = []
    freqs = []
    amps = []

    for pos in range(5):
        filepath = raw_dir / f"position_{pos}_trial_0.npy"
        if filepath.exists():
            data = np.load(filepath)
            signals.append(data)
            positions.append(pos)

            # Get frequency and amplitude for labels
            from scipy.fft import fft, fftfreq
            fft_vals = fft(data)
            fft_freqs = fftfreq(len(data), 0.01)
            pos_mask = fft_freqs > 0
            freqs_pos = fft_freqs[pos_mask]
            mags = np.abs(fft_vals[pos_mask])
            dominant_freq = freqs_pos[np.argmax(mags)]
            freqs.append(dominant_freq)
            amps.append(np.max(data))

    if len(signals) == 0:
        print("⚠ No signal data found, skipping animation")
        return

    # Animation parameters
    sample_rate = 100  # Hz
    duration = 0.5  # Show first 0.5 seconds
    n_frames = int(duration * sample_rate)

    # Create figure
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    fig.suptitle('Real-Time Vibration Signals - Piezoelectric Sensor Data', 
                 fontsize=14, fontweight='bold')

    lines = []
    time = np.arange(len(signals[0])) / sample_rate

    # Initialize plots
    for i, (ax, pos) in enumerate(zip(axes, positions)):
        line, = ax.plot([], [], 'b-', linewidth=1.5)
        lines.append(line)

        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1100)
        ax.set_ylabel('ADC', fontsize=9)
        ax.set_title(f'Position {pos} | Freq: {freqs[i]:.1f} Hz | Peak: {amps[i]:.0f}', 
                    fontsize=10, loc='left')
        ax.grid(alpha=0.3)

        if i == 4:
            ax.set_xlabel('Time (s)', fontsize=11)

    plt.tight_layout()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(frame):
        for i, line in enumerate(lines):
            t_data = time[:frame+1]
            y_data = signals[i][:frame+1]
            line.set_data(t_data, y_data)
        return lines

    print("Creating vibration animation (this may take 1-2 minutes)...")
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=n_frames, interval=10, 
                        blit=True, repeat=True)

    try:
        anim.save(save_path, writer='pillow', fps=30, dpi=100)
        print(f"✓ Saved vibration animation to {save_path}")
    except Exception as e:
        print(f"⚠ Could not save animation: {e}")
        print("  (Install pillow: pip install pillow)")

    plt.close()

def main():
    print("=" * 60)
    print("RESULTS VISUALIZATION")
    print("=" * 60)
    print()

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Load test predictions
    models_dir = Path("models")
    y_test = np.load(models_dir / "y_test.npy")
    y_pred = np.load(models_dir / "y_pred.npy")

    # Generate plots
    print("Generating visualizations...")
    print()

    plot_confusion_matrix(y_test, y_pred, 
                         results_dir / "confusion_matrix.png")

    plot_feature_importance(results_dir / "feature_importance.png")

    plot_sample_signals(results_dir / "sample_signals.png")

    # Create animation
    print()
    create_vibration_animation(results_dir / "vibration_animation.gif")

    print()
    print("=" * 60)
    print("✓ Visualization complete!")
    print("=" * 60)
    print()
    print(f"Results saved to {results_dir}/")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - sample_signals.png")
    print("  - vibration_animation.gif")

if __name__ == "__main__":
    main()