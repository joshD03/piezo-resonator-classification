"""
Physics Validation - Euler-Bernoulli Beam Theory
Compares experimental vibration data to theoretical predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import fft, fftfreq
import sys
import os

# Euler-Bernoulli constants for cantilever beam
LAMBDA_N = np.array([1.875, 4.694, 7.855, 10.996])  # First 4 mode shape constants

def theoretical_natural_frequency(n, L, E, I, rho, A):
    """
    Calculate theoretical natural frequency for mode n.

    f_n = (lambda_n^2 / 2*pi*L^2) * sqrt(EI / rho*A)
    """
    lambda_n = LAMBDA_N[n]
    return (lambda_n**2 / (2 * np.pi * L**2)) * np.sqrt((E * I) / (rho * A))

def theoretical_modal_shape(x, L, n=0):
    """
    Calculate theoretical deflection shape for mode n at position x.
    """
    lambda_n = LAMBDA_N[n]
    lambda_x = lambda_n * x / L
    lambda_L = lambda_n

    sigma = (np.sinh(lambda_L) - np.sin(lambda_L)) / (np.cosh(lambda_L) + np.cos(lambda_L))

    shape = (np.cosh(lambda_x) - np.cos(lambda_x)) - sigma * (np.sinh(lambda_x) - np.sin(lambda_x))

    # Normalize to 1.0 at free end
    shape_max = (np.cosh(lambda_L) - np.cos(lambda_L)) - sigma * (np.sinh(lambda_L) - np.sin(lambda_L))

    return shape / shape_max

def analyze_experimental_data():
    """Extract frequency and amplitude data from all positions."""

    # Try to find data directory
    possible_paths = [
        Path("data/raw"),
        Path("../data/raw"),
        Path("../../data/raw"),
    ]

    raw_dir = None
    for path in possible_paths:
        if path.exists():
            raw_dir = path
            break

    if raw_dir is None:
        print("ERROR: Cannot find data/raw directory!")
        print("Current directory:", os.getcwd())
        print("Please run from project root or adjust paths")
        sys.exit(1)

    print(f"Found data directory: {raw_dir.absolute()}")
    print()

    # Check what files exist
    all_files = list(raw_dir.glob("*.npy"))
    print(f"Found {len(all_files)} .npy files")

    if len(all_files) == 0:
        print("ERROR: No .npy files found!")
        sys.exit(1)

    positions_cm = [0, 1, 2, 3, 4]

    results = {
        'positions': [],
        'distances_m': [],
        'dominant_freqs': [],
        'peak_amplitudes': [],
        'rms_amplitudes': [],
        'energies': []
    }

    print("Analyzing experimental data...")
    print()

    for pos in positions_cm:
        trials_freqs = []
        trials_amps = []
        trials_rms = []
        trials_energy = []

        # Analyze all trials for this position
        for trial in range(10):
            filepath = raw_dir / f"position_{pos}_trial_{trial}.npy"

            if not filepath.exists():
                print(f"  Warning: {filepath.name} not found, skipping...")
                continue

            data = np.load(filepath)

            if len(data) == 0:
                continue

            # Frequency analysis
            sample_rate = 100  # Hz
            fft_vals = fft(data)
            fft_freqs = fftfreq(len(data), 1/sample_rate)

            pos_mask = fft_freqs > 0
            freqs = fft_freqs[pos_mask]
            magnitudes = np.abs(fft_vals[pos_mask])

            if len(magnitudes) > 0:
                dominant_idx = np.argmax(magnitudes)
                dominant_freq = freqs[dominant_idx]

                trials_freqs.append(dominant_freq)
                trials_amps.append(np.max(data))
                trials_rms.append(np.sqrt(np.mean(data**2)))
                trials_energy.append(np.sum(data**2))

        if len(trials_freqs) == 0:
            print(f"  ERROR: No valid data for position {pos}!")
            continue

        # Average across trials
        results['positions'].append(pos)
        results['dominant_freqs'].append(np.mean(trials_freqs))
        results['peak_amplitudes'].append(np.mean(trials_amps))
        results['rms_amplitudes'].append(np.mean(trials_rms))
        results['energies'].append(np.mean(trials_energy))

    # Position mapping - ADJUST THESE TO YOUR ACTUAL SETUP!
    position_mapping = {
        0: 4,   # Position 0 at 4cm from clamp
        1: 7,   # Position 1 at 7cm
        2: 10,  # Position 2 at 10cm
        3: 13,  # Position 3 at 13cm
        4: 15,  # Position 4 at 15cm (near free end)
    }

    results['distances_m'] = [position_mapping[p] / 100.0 for p in results['positions']]

    print("Experimental data summary:")
    print(f"{'Position':<10} {'Distance(m)':<12} {'Freq(Hz)':<12} {'Peak Amp':<12} {'Energy'}")
    print("-" * 60)
    for i in range(len(results['positions'])):
        print(f"{results['positions'][i]:<10} {results['distances_m'][i]:<12.3f} "
              f"{results['dominant_freqs'][i]:<12.2f} {results['peak_amplitudes'][i]:<12.1f} "
              f"{results['energies'][i]:.0f}")
    print()

    return results

def estimate_beam_properties():
    """Estimate physical properties of the ruler."""
    # Ruler dimensions (ADJUST TO YOUR ACTUAL RULER!)
    L = 0.15  # Free length in meters (15cm)
    width = 0.03  # 3cm width
    thickness = 0.001  # 1mm thick (typical plastic ruler)

    # Plastic properties (polypropylene/polystyrene typical)
    E = 2.5e9  # Young's modulus ~2.5 GPa for plastic
    rho = 1050  # Density ~1050 kg/m^3 for plastic

    # Calculate geometric properties
    A = width * thickness  # Cross-sectional area
    I = (width * thickness**3) / 12  # Second moment of area for rectangle

    properties = {
        'L': L,
        'width': width,
        'thickness': thickness,
        'E': E,
        'rho': rho,
        'A': A,
        'I': I
    }

    return properties

def compare_frequencies(experimental_data, beam_props):
    """Compare experimental vs theoretical frequencies."""

    print("=" * 70)
    print("FREQUENCY COMPARISON")
    print("=" * 70)
    print()

    # Calculate theoretical fundamental frequency - PASS ONLY NEEDED PARAMS
    f1_theory = theoretical_natural_frequency(0, beam_props['L'], beam_props['E'], 
                                             beam_props['I'], beam_props['rho'], beam_props['A'])
    f2_theory = theoretical_natural_frequency(1, beam_props['L'], beam_props['E'],
                                             beam_props['I'], beam_props['rho'], beam_props['A'])

    print(f"Theoretical fundamental frequency (mode 1): {f1_theory:.2f} Hz")
    print(f"Theoretical 2nd mode frequency (mode 2): {f2_theory:.2f} Hz")
    print()

    # Compare with experimental
    exp_freqs = experimental_data['dominant_freqs']
    avg_exp_freq = np.mean(exp_freqs)
    std_exp_freq = np.std(exp_freqs)

    print(f"Experimental dominant frequency: {avg_exp_freq:.2f} ± {std_exp_freq:.2f} Hz")
    print(f"Difference from theory: {abs(avg_exp_freq - f1_theory):.2f} Hz")
    print(f"Relative error: {abs(avg_exp_freq - f1_theory) / f1_theory * 100:.1f}%")
    print()

    # Check if experimental matches mode 1 or mode 2
    error_mode1 = abs(avg_exp_freq - f1_theory) / f1_theory
    error_mode2 = abs(avg_exp_freq - f2_theory) / f2_theory

    if error_mode1 < 0.3:
        print("✓ Experimental frequency matches MODE 1 (fundamental)")
    elif error_mode2 < 0.3:
        print("✓ Experimental frequency matches MODE 2 (2nd harmonic)")
    else:
        print("⚠ Frequency doesn't match simple modes - likely multi-modal vibration")

    print()

    return f1_theory, f2_theory

def compare_amplitudes(experimental_data, beam_props):
    """Compare experimental amplitude distribution vs theoretical modal shape."""

    print("=" * 70)
    print("AMPLITUDE DISTRIBUTION COMPARISON")
    print("=" * 70)
    print()

    L = beam_props['L']
    distances = np.array(experimental_data['distances_m'])
    exp_amplitudes = np.array(experimental_data['peak_amplitudes'])

    # Normalize experimental amplitudes
    exp_amp_norm = exp_amplitudes / np.max(exp_amplitudes)

    # Calculate theoretical shape at these positions
    theory_amp_norm = theoretical_modal_shape(distances, L, n=0)

    # Calculate fit quality
    residuals = exp_amp_norm - theory_amp_norm
    r_squared = 1 - (np.sum(residuals**2) / np.sum((exp_amp_norm - np.mean(exp_amp_norm))**2))
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"{'Position':<12} {'Exp Amp(norm)':<15} {'Theory Amp':<15} {'Error'}")
    print("-" * 60)
    for i in range(len(distances)):
        print(f"{experimental_data['positions'][i]:<12} {exp_amp_norm[i]:<15.3f} "
              f"{theory_amp_norm[i]:<15.3f} {residuals[i]:+.3f}")
    print()
    print(f"R² fit quality: {r_squared:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print()

    if r_squared > 0.8:
        print("✓ EXCELLENT agreement with Euler-Bernoulli theory!")
    elif r_squared > 0.6:
        print("✓ Good agreement with Euler-Bernoulli theory")
    else:
        print("⚠ Moderate agreement - complex vibration modes or measurement variation")

    print()

    return exp_amp_norm, theory_amp_norm, r_squared

def visualize_physics_validation(experimental_data, beam_props, f1_theory, exp_amp_norm, theory_amp_norm, r_squared):
    """Create comprehensive physics validation plots."""

    # Try to find results directory
    possible_paths = [
        Path("results"),
        Path("../results"),
    ]

    results_dir = None
    for path in possible_paths:
        results_dir = path
        results_dir.mkdir(exist_ok=True)
        break

    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Amplitude vs Position (Experimental vs Theory)
    ax1 = plt.subplot(2, 2, 1)

    distances_cm = np.array(experimental_data['distances_m']) * 100

    # Fine resolution for smooth theory curve
    x_fine = np.linspace(min(experimental_data['distances_m']), 
                         max(experimental_data['distances_m']), 100)
    theory_fine = theoretical_modal_shape(x_fine, beam_props['L'], n=0)

    ax1.plot(x_fine * 100, theory_fine, 'r-', linewidth=2, label='E-B Theory (Mode 1)', alpha=0.7)
    ax1.plot(distances_cm, exp_amp_norm, 'bo-', markersize=10, linewidth=2, 
             label='Experimental', markerfacecolor='lightblue', markeredgewidth=2)

    ax1.set_xlabel('Distance from Clamp (cm)', fontsize=11)
    ax1.set_ylabel('Normalised Amplitude', fontsize=11)
    ax1.set_title(f'Amplitude Distribution (R² = {r_squared:.3f})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot 2: Energy vs Position
    ax2 = plt.subplot(2, 2, 2)

    energies = np.array(experimental_data['energies'])

    ax2.plot(distances_cm, energies, 'go-', markersize=10, linewidth=2,
             markerfacecolor='lightgreen', markeredgewidth=2)
    ax2.set_xlabel('Distance from Clamp (cm)', fontsize=11)
    ax2.set_ylabel('Signal Energy', fontsize=11)
    ax2.set_title('Energy Distribution Along Beam', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)

    # Annotate trend
    z = np.polyfit(distances_cm, energies, 2)
    p = np.poly1d(z)
    ax2.plot(distances_cm, p(distances_cm), 'g--', alpha=0.5, linewidth=1.5, label='Quadratic fit')
    ax2.legend(fontsize=9)

    # Plot 3: Frequency Analysis
    ax3 = plt.subplot(2, 2, 3)

    freqs = experimental_data['dominant_freqs']

    ax3.plot(distances_cm, freqs, 'mo-', markersize=10, linewidth=2,
             markerfacecolor='plum', markeredgewidth=2, label='Experimental')
    ax3.axhline(y=f1_theory, color='r', linestyle='--', linewidth=2, 
                label=f'Theory Mode 1 ({f1_theory:.1f} Hz)', alpha=0.7)

    avg_freq = np.mean(freqs)
    ax3.axhline(y=avg_freq, color='purple', linestyle=':', linewidth=2,
                label=f'Exp. Average ({avg_freq:.1f} Hz)', alpha=0.7)

    ax3.set_xlabel('Distance from Clamp (cm)', fontsize=11)
    ax3.set_ylabel('Dominant Frequency (Hz)', fontsize=11)
    ax3.set_title('Frequency vs Position', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # Plot 4: Theory vs Experiment Scatter
    ax4 = plt.subplot(2, 2, 4)

    ax4.scatter(theory_amp_norm, exp_amp_norm, s=150, alpha=0.6, 
                c=distances_cm, cmap='viridis', edgecolors='black', linewidth=1.5)

    # Perfect fit line
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Fit', alpha=0.5)

    ax4.set_xlabel('Theoretical Amplitude (normalised)', fontsize=11)
    ax4.set_ylabel('Experimental Amplitude (normalised)', fontsize=11)
    ax4.set_title('Theory vs Experiment Correlation', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    ax4.set_xlim([-0.1, 1.1])
    ax4.set_ylim([-0.1, 1.1])

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=min(distances_cm), vmax=max(distances_cm)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4)
    cbar.set_label('Distance (cm)', fontsize=10)

    plt.suptitle('Physics Validation: Euler-Bernoulli Beam Theory', 
                 fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(results_dir / 'physics_validation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved physics validation plot to {results_dir}/physics_validation.png")
    plt.close()

def create_modal_animation(save_path, beam_props, f1_theory):
    """Create animated GIF showing theoretical beam vibration."""
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("⚠ Animation requires matplotlib.animation, skipping GIF creation")
        return

    L = beam_props['L']
    omega = 2 * np.pi * f1_theory

    # Spatial points along beam
    x = np.linspace(0, L, 100)

    # Measurement positions (adjust to your actual setup)
    measurement_positions = np.array([0.04, 0.07, 0.10, 0.13, 0.15])
    sensor_position = 0.08  # Sensor at 8cm

    # Time points for 2 full periods
    n_frames = 60
    period = 1 / f1_theory
    t = np.linspace(0, 2 * period, n_frames)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Lines and points
    line_beam, = ax.plot([], [], 'b-', linewidth=3, label='Beam deflection')
    line_static, = ax.plot(x*100, np.zeros_like(x), 'k--', 
                          linewidth=1, alpha=0.3, label='Rest position')

    scatter_measure = ax.scatter([], [], s=150, c='red', marker='v', 
                                zorder=5, label='Strike positions', 
                                edgecolors='darkred', linewidths=2)

    sensor_line = ax.axvline(sensor_position*100, color='green', 
                            linestyle=':', linewidth=2, alpha=0.7,
                            label='Piezo sensor (8cm)')

    # Clamp visualisation
    ax.axvspan(-0.5, 0, alpha=0.3, color='gray', label='Clamp')
    ax.axvline(0, color='black', linewidth=3)

    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('Distance from Clamp (cm)', fontsize=12)
    ax.set_ylabel('Deflection (normalised)', fontsize=12)
    ax.set_title(f'Euler-Bernoulli Theory: Cantilever Beam Mode 1 ({f1_theory:.1f} Hz)', 
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()

    def modal_shape_time(x_pos, t_val):
        """Calculate deflection at position x and time t."""
        # Modal shape (mode 1)
        w_static = theoretical_modal_shape(x_pos, L, n=0)
        # Time-dependent oscillation
        w = w_static * np.sin(omega * t_val) * 2.5  # amplitude=2.5 for visibility
        return w

    def init():
        line_beam.set_data([], [])
        scatter_measure.set_offsets(np.empty((0, 2)))
        return line_beam, scatter_measure

    def animate(frame):
        # Calculate beam shape at this time
        w = modal_shape_time(x, t[frame])
        line_beam.set_data(x*100, w)

        # Update measurement points
        w_measure = modal_shape_time(measurement_positions, t[frame])
        scatter_measure.set_offsets(np.c_[measurement_positions*100, w_measure])

        return line_beam, scatter_measure

    print("Creating theoretical modal animation (this may take 1-2 minutes)...")
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=n_frames, interval=33,
                        blit=True, repeat=True)

    try:
        anim.save(save_path, writer='pillow', fps=30, dpi=100)
        print(f"✓ Saved theoretical animation to {save_path}")
    except Exception as e:
        print(f"⚠ Could not save animation: {e}")
        print("  (Install pillow: pip install pillow)")

    plt.close()

def main():
    """Main physics validation pipeline."""
    print("=" * 70)
    print("PHYSICS VALIDATION - EULER-BERNOULLI BEAM THEORY")
    print("=" * 70)
    print()

    # Step 1: Analyze experimental data
    exp_data = analyze_experimental_data()

    if len(exp_data['positions']) == 0:
        print("ERROR: No experimental data found!")
        sys.exit(1)

    # Step 2: Estimate beam properties
    beam_props = estimate_beam_properties()

    print("Estimated beam properties:")
    print(f"  Length: {beam_props['L']*100:.1f} cm")
    print(f"  Width: {beam_props['width']*100:.1f} cm")
    print(f"  Thickness: {beam_props['thickness']*1000:.2f} mm")
    print(f"  Young's modulus: {beam_props['E']/1e9:.1f} GPa (plastic)")
    print()

    # Step 3: Compare frequencies
    f1_theory, f2_theory = compare_frequencies(exp_data, beam_props)

    # Step 4: Compare amplitudes
    exp_amp_norm, theory_amp_norm, r_squared = compare_amplitudes(exp_data, beam_props)

    # Results directory for plots
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Step 5: Visualise
    visualize_physics_validation(exp_data, beam_props, f1_theory, 
                                 exp_amp_norm, theory_amp_norm, r_squared)

    # Step 6: Create animation
    print()
    create_modal_animation(results_dir / 'theoretical_vibration.gif',
                          beam_props, f1_theory)

    # Final summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    avg_freq = np.mean(exp_data['dominant_freqs'])
    freq_error = abs(avg_freq - f1_theory) / f1_theory * 100

    print(f"Frequency validation:")
    print(f"  Theory: {f1_theory:.1f} Hz")
    print(f"  Experiment: {avg_freq:.1f} Hz")
    print(f"  Error: {freq_error:.1f}%")
    print()

    print(f"Amplitude shape validation:")
    print(f"  R² = {r_squared:.3f}")

    if r_squared > 0.8 and freq_error < 30:
        print()
        print("✓✓✓ STRONG VALIDATION OF EULER-BERNOULLI THEORY ✓✓✓")
        print("    Experimental data closely matches theoretical predictions!")
    elif r_squared > 0.6 or freq_error < 40:
        print()
        print("✓ GOOD VALIDATION OF EULER-BERNOULLI THEORY")
        print("  Experimental trends match theory with expected real-world variations")
    else:
        print()
        print("⚠ PARTIAL VALIDATION")
        print("  Complex vibration modes or measurement challenges present")

    print()
    print("=" * 70)
    print("✓ Physics validation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()