import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# --- Configuration ---
WT_TENSORS_FILE = 'ix_sep_tens_1A23A_fwd_wt.npy'
MT_TENSORS_FILE = 'ix_sep_tens_1A23A_fwd_mt.npy'
DEGREES_FILE = 'ix_sep_tens_1A23A_fwd_rotations.txt'

CHANNEL_LABELS = [
    "Hydrophobic",
    "Aromatic",
    "Hydrogen Bonding Donor",
    "Hydrogen Bond Acceptor",
    "Positive Ionizable",
    "Negative Ionizable",
    "Occupancy"
]
NUM_CHANNELS = 7

def load_and_extract_degrees(degrees_file):
    """Loads rotation data and extracts the X-axis rotation degrees."""
    if not os.path.exists(degrees_file):
        print(f"Error: Degrees file not found at '{degrees_file}'. Exiting.")
        return None
    
    try:
        # Load the data, skipping the header (skiprows=1) and using comma delimiter
        rotation_data = np.loadtxt(degrees_file, delimiter=',', skiprows=1)
        
        # Expecting N x 3 (x_rot, y_rot, z_rot). We extract x_rot (index 0).
        if rotation_data.ndim == 1:
             # Handle case where only one rotation is provided (e.g., one rotation angle)
            x_degrees = rotation_data.reshape(-1).astype(int)
        elif rotation_data.ndim == 2 and rotation_data.shape[1] >= 1:
            x_degrees = rotation_data[:, 0].astype(int) 
        else:
            raise ValueError("Rotation data format unexpected. Expected N x 1 or N x 3 columns.")
            
        return x_degrees

    except Exception as e:
        print(f"Error loading degrees file (check header and delimiter): {e}")
        return None

def calculate_mse_per_channel(tensors_file, x_degrees):
    """
    Calculates the Mean Squared Error (MSE) for each channel of all rotated 
    tensors against the reference tensor (the 0-degree orientation).
    """
    if not os.path.exists(tensors_file):
        print(f"Error: Tensor file not found at '{tensors_file}'. Skipping analysis.")
        return None
    
    print(f"\nProcessing {tensors_file}...")
    
    try:
        tensors = np.load(tensors_file)
        n_samples = tensors.shape[0]
        # Expected shape: (N, 16, 16, 16, 7)
        if tensors.ndim != 5 or tensors.shape[-1] != NUM_CHANNELS:
            print(f"Warning: Expected shape (N, 16, 16, 16, 7). Found {tensors.shape}.")

    except Exception as e:
        print(f"Error loading tensor file: {e}")
        return None

    if n_samples != len(x_degrees):
        print(f"Error: Mismatch between number of tensors ({n_samples}) and rotation degrees ({len(x_degrees)}).")
        return None

    # --- CRITICAL FIX: Find the 0-degree reference tensor ---
    try:
        # Find the index corresponding to the 0 degree rotation
        ref_idx = np.where(x_degrees == 0)[0][0]
        T_ref = tensors[ref_idx]
        print(f"  > Reference tensor (0°) found at index {ref_idx}.")
    except IndexError:
        print(f"Error: Could not find a 0 degree rotation in the degrees file for reference. Exiting analysis for {tensors_file}.")
        return None
    
    # Storage for MSE results: (7 channels x N rotations)
    mse_results = np.zeros((NUM_CHANNELS, n_samples))

    # The calculation proceeds in the file order (i) to maintain alignment with the rotation data
    for c in range(NUM_CHANNELS):
        # Extract the constant reference channel data and flatten
        ref_channel_data = T_ref[..., c].flatten()
        
        for i in range(n_samples):
            # Extract the current rotated channel data and flatten
            rotated_channel_data = tensors[i, ..., c].flatten()
            
            # Calculate MSE: mean((T_ref - T_rot)^2)
            mse = np.mean((ref_channel_data - rotated_channel_data)**2)
            mse_results[c, i] = mse
            
        print(f"  > Calculated MSE for Channel {c+1}: {CHANNEL_LABELS[c]}")

    return mse_results

def plot_results(mse_data, x_degrees, title_suffix):
    """
    Plots the 7 channels of MSE data against rotation degrees in a single figure,
    ensuring data is sorted by x_degrees before plotting.
    """
    
    # --- Sorting Data ---
    # Get the indices that would sort x_degrees
    sort_indices = np.argsort(x_degrees)
    
    # Apply the sorting indices to both x_degrees and the MSE data
    sorted_x_degrees = x_degrees[sort_indices]
    sorted_mse_data = mse_data[:, sort_indices]
    
    # --- Plotting ---
    
    # Create a 4x2 grid for 7 plots (+ 1 empty slot)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(18, 18))
    fig.suptitle(f'Mean Squared Error (MSE) vs. X-Axis Rotation: {title_suffix}', 
                 fontsize=20, fontweight='bold', y=1.02)
    
    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    for c in range(NUM_CHANNELS):
        ax = axes_flat[c]
        
        # Plot the sorted MSE values against the sorted degrees
        ax.plot(sorted_x_degrees, sorted_mse_data[c, :], marker='o', markersize=4, 
                linestyle='-', linewidth=2, color='darkblue')

        ax.set_title(f'Channel {c+1}: {CHANNEL_LABELS[c]}', fontsize=14)
        ax.set_xlabel('X-Axis Rotation Degree (°)', fontsize=12)
        ax.set_ylabel('MSE (vs. 0° Reference)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set clean limits for the X-axis
        ax.set_xlim(sorted_x_degrees.min(), sorted_x_degrees.max())
        
        # Optionally, set Y-limits based on the data for better comparison
        y_max = sorted_mse_data.max() * 1.05
        ax.set_ylim(0, y_max)
        
    # Hide the empty 8th subplot
    axes_flat[-1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout
    plt.show()

def main():
    """Main function to orchestrate loading, analysis, and plotting."""
    
    # 1. Load Rotation Degrees (shared by both WT and MT)
    x_degrees = load_and_extract_degrees(DEGREES_FILE)
    if x_degrees is None:
        return

    print(f"Loaded {len(x_degrees)} rotation steps (X-degrees).")

    # 2. Analyze and Plot WT Tensors
    wt_mse_results = calculate_mse_per_channel(WT_TENSORS_FILE, x_degrees)
    if wt_mse_results is not None:
        plot_results(wt_mse_results, x_degrees, "Wild Type (WT) Protein")

    # 3. Analyze and Plot MT Tensors
    mt_mse_results = calculate_mse_per_channel(MT_TENSORS_FILE, x_degrees)
    if mt_mse_results is not None:
        plot_results(mt_mse_results, x_degrees, "Mutant (MT) Protein")

    print("\nAnalysis complete. Two plot windows have been generated (or saved, depending on environment).")

if __name__ == '__main__':
    main()