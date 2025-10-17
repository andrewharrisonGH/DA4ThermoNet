import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
import os

# --- Configuration ---
TENSORS_FILE = 'ixiy_tensors_1A23A_fwd.npy'
DEGREES_FILE = 'ixiy_tensors_1A23A_fwd_rotations.txt'

GLOBAL_SEED = 42 

def run_isomap_and_plot():
    """
    Loads data, performs ISOMAP (3 components), and plots in 3D.
    Color is determined by a custom blend:
    - Low X/Y rotation: White
    - Increasing X-rotation: Towards Dark Blue
    - Increasing Y-rotation: Towards Dark Red
    - Max combined rotation: Indigo/Dark Purple
    All data points are of the same size.
    """
    np.random.seed(GLOBAL_SEED)
    
    # --- Data Loading and Check ---
    if not os.path.exists(TENSORS_FILE):
        print(f"Error: Tensor file not found at '{TENSORS_FILE}'. Please place your .npy file in the working directory. Exiting.")
        return
    if not os.path.exists(DEGREES_FILE):
        print(f"Error: Degrees file not found at '{DEGREES_FILE}'. Please place your .txt file in the working directory. Exiting.")
        return
    
    print(f"Loading data from {TENSORS_FILE} and {DEGREES_FILE}...")

    try:
        tensors = np.load(TENSORS_FILE)
        n_samples = tensors.shape[0]
        tensor_shape = tensors.shape[1:]
    except Exception as e:
        print(f"Error loading tensor file: {e}")
        return

    try:
        rotation_data = np.loadtxt(DEGREES_FILE, delimiter=',', skiprows=1)
        if rotation_data.ndim == 2 and rotation_data.shape[1] >= 2:
            x_degrees = rotation_data[:, 0].astype(int) 
            y_degrees = rotation_data[:, 1].astype(int)
        else:
            raise ValueError("Rotation data must contain at least two columns (x_rot and y_rot).")
    except Exception as e:
        print(f"Error loading degrees file: {e}")
        return

    if n_samples != len(x_degrees):
        print(f"Error: Mismatch between number of tensors ({n_samples}) and rotation pairs ({len(x_degrees)}).")
        return

    print(f"Loaded {n_samples} samples. Original tensor shape: {tensor_shape}")

    # --- Data Preparation for ISOMAP ---
    n_features = np.prod(tensor_shape)
    data_flat = tensors.reshape(n_samples, n_features)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_flat)
    
    print("Data standardized.")

    # --- ISOMAP Dimensionality Reduction (3 Components) ---
    n_neighbors = 15
    # ** Changed n_components to 3 for 3D visualization **
    isomap = Isomap(n_components=3, n_neighbors=n_neighbors)
    manifold_components = isomap.fit_transform(data_scaled)
    
    ic1 = manifold_components[:, 0]
    ic2 = manifold_components[:, 1]
    ic3 = manifold_components[:, 2] # Third component for Z-axis
    
    print(f"ISOMAP applied with n_components=3, n_neighbors={n_neighbors}. Data reduced to 3 components.")
    
    # --- Custom Color Mapping: White Base to Dark Red/Blue ---
    # Normalize X and Y degrees to a 0-1 range
    x_min, x_max = np.min(x_degrees), np.max(x_degrees)
    y_min, y_max = np.min(y_degrees), np.max(y_degrees)
    
    norm_x = (x_degrees - x_min) / (x_max - x_min if x_max - x_min > 0 else 1)
    norm_y = (y_degrees - y_min) / (y_max - y_min if y_max - y_min > 0 else 1)
    
    # Maximum intensity (darkness) for the primary color channels
    MAX_INTENSITY = 0.7 
    
    colors = np.zeros((n_samples, 3))
    
    # Luminance drop (L) from White (1.0) based on max rotation (pulls towards darker colors)
    L = 1.0 - MAX_INTENSITY * np.maximum(norm_x, norm_y) 
    
    # Calculate RGB components:
    # High Y adds Red (R), High X adds Blue (B). Green (G) is kept lower to enforce R/B contrast.
    R = L + MAX_INTENSITY * norm_y
    G = L 
    B = L + MAX_INTENSITY * norm_x
    
    # Combine and clip to [0, 1] range
    colors[:, 0] = np.clip(R, 0, 1)
    colors[:, 1] = np.clip(G, 0, 1)
    colors[:, 2] = np.clip(B, 0, 1)
    
    # --- Visualization (3D Plot with Custom Color) ---
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # All points are now a uniform size
    scatter_size = 100 
    scatter = ax.scatter(
        ic1, 
        ic2, 
        ic3, # Plotting the third component
        c=colors,       # Custom RGB colors
        s=scatter_size, # Uniform size
        alpha=0.7,
        edgecolors='k', 
        linewidth=0.5
    )

    ax.set_xlabel('Isomap Component 1 (IC1)', fontsize=14, labelpad=10)
    ax.set_ylabel('Isomap Component 2 (IC2)', fontsize=14, labelpad=10)
    ax.set_zlabel('Isomap Component 3 (IC3)', fontsize=14, labelpad=10)
    
    ax.set_title(
        f'3D ISOMAP of Protein Tensors (n_neighbors={n_neighbors}, Seed={GLOBAL_SEED})\nColor: Low Deg (White) to High X (Dark Blue) / High Y (Dark Red)', 
        fontsize=16, 
        fontweight='bold',
        y=1.02
    )
    
    # --- Custom Color Legend (2D Swatch) ---
    cmap_res = 100
    color_map_x = np.linspace(0, 1, cmap_res)
    color_map_y = np.linspace(0, 1, cmap_res)
    
    color_swatch = np.zeros((cmap_res, cmap_res, 3))
    
    for i in range(cmap_res): # y_norm (rows)
        for j in range(cmap_res): # x_norm (columns)
            nx = color_map_x[j]
            ny = color_map_y[i]
            
            L_swatch = 1.0 - MAX_INTENSITY * max(nx, ny)
            R_swatch = L_swatch + MAX_INTENSITY * ny
            G_swatch = L_swatch
            B_swatch = L_swatch + MAX_INTENSITY * nx
            
            color_swatch[i, j, 0] = np.clip(R_swatch, 0, 1)
            color_swatch[i, j, 1] = np.clip(G_swatch, 0, 1)
            color_swatch[i, j, 2] = np.clip(B_swatch, 0, 1)


    # Create a separate axis for the custom color legend
    ax_legend = fig.add_axes([0.8, 0.1, 0.08, 0.2]) # [left, bottom, width, height]
    ax_legend.imshow(color_swatch, origin='lower', aspect='auto')
    ax_legend.set_xlabel('X-Rot (Blue)', fontsize=10)
    ax_legend.set_ylabel('Y-Rot (Red)', fontsize=10)
    ax_legend.set_xticks([0, cmap_res-1])
    ax_legend.set_xticklabels([f'{x_min}°', f'{x_max}°'])
    ax_legend.set_yticks([0, cmap_res-1])
    ax_legend.set_yticklabels([f'{y_min}°', f'{y_max}°'])
    ax_legend.set_title('Color Legend', fontsize=12, pad=10)

    # Optional: Annotate key combined rotation points
    key_pairs = [(0, 0), (90, 90), (180, 180), (270, 270)]
    for x_deg, y_deg in key_pairs:
        try:
            idx = np.where((x_degrees == x_deg) & (y_degrees == y_deg))[0][0]
            ax.text(
                ic1[idx], 
                ic2[idx], 
                ic3[idx],
                f'({x_deg}°, {y_deg}°)', 
                fontsize=10, 
                fontweight='bold',
                color='darkred'
            )
        except IndexError:
            pass
            
    ax.view_init(elev=20, azim=120)
    
    plt.show()

if __name__ == '__main__':
    run_isomap_and_plot()