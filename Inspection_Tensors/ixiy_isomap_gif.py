import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation, PillowWriter # New imports for animation
import os

# --- Configuration ---
TENSORS_FILE = 'ixiy_tensors_1A23A_fwd.npy'
DEGREES_FILE = 'ixiy_tensors_1A23A_fwd_rotations.txt'

GLOBAL_SEED = 42 

def run_isomap_and_plot():
    """
    Loads data, performs ISOMAP (3 components), and generates a spinning 3D GIF.
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
    isomap = Isomap(n_components=3, n_neighbors=n_neighbors)
    manifold_components = isomap.fit_transform(data_scaled)
    
    ic1 = manifold_components[:, 0]
    ic2 = manifold_components[:, 1]
    ic3 = manifold_components[:, 2]
    
    print(f"ISOMAP applied with n_components=3, n_neighbors={n_neighbors}. Data reduced to 3 components.")
    
    # --- Custom Color Mapping: White Base to Dark Red/Blue ---
    x_min, x_max = np.min(x_degrees), np.max(x_degrees)
    y_min, y_max = np.min(y_degrees), np.max(y_degrees)
    
    norm_x = (x_degrees - x_min) / (x_max - x_min if x_max - x_min > 0 else 1)
    norm_y = (y_degrees - y_min) / (y_max - y_min if y_max - y_min > 0 else 1)
    
    MAX_INTENSITY = 0.7 
    
    # Calculate colors based on the requested White -> Dark Red/Blue scheme
    colors = np.zeros((n_samples, 3))
    L = 1.0 - MAX_INTENSITY * np.maximum(norm_x, norm_y) 
    
    R = L + MAX_INTENSITY * norm_y
    G = L 
    B = L + MAX_INTENSITY * norm_x
    
    colors[:, 0] = np.clip(R, 0, 1)
    colors[:, 1] = np.clip(G, 0, 1)
    colors[:, 2] = np.clip(B, 0, 1)
    
    # --- Visualization (3D Plot Setup) ---
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 12))
    # Note: Using fig.add_subplot(..., projection='3d') creates a compatible axes.
    ax = fig.add_subplot(111, projection='3d') 

    # All points are now a uniform size
    scatter_size = 100 
    scatter = ax.scatter(
        ic1, 
        ic2, 
        ic3, 
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
    
    for i in range(cmap_res):
        for j in range(cmap_res):
            nx = color_map_x[j]
            ny = color_map_y[i]
            
            L_swatch = 1.0 - MAX_INTENSITY * max(nx, ny)
            R_swatch = L_swatch + MAX_INTENSITY * ny
            G_swatch = L_swatch
            B_swatch = L_swatch + MAX_INTENSITY * nx
            
            color_swatch[i, j, 0] = np.clip(R_swatch, 0, 1)
            color_swatch[i, j, 1] = np.clip(G_swatch, 0, 1)
            color_swatch[i, j, 2] = np.clip(B_swatch, 0, 1)

    ax_legend = fig.add_axes([0.8, 0.1, 0.08, 0.2]) # [left, bottom, width, height]
    ax_legend.imshow(color_swatch, origin='lower', aspect='auto')
    ax_legend.set_xlabel('X-Rot (Blue)', fontsize=10)
    ax_legend.set_ylabel('Y-Rot (Red)', fontsize=10)
    ax_legend.set_xticks([0, cmap_res-1])
    ax_legend.set_xticklabels([f'{x_min}°', f'{x_max}°'])
    ax_legend.set_yticks([0, cmap_res-1])
    ax_legend.set_yticklabels([f'{y_min}°', f'{y_max}°'])
    ax_legend.set_title('Color Legend', fontsize=12, pad=10)

    # --- Animation Function ---
    def animate(i):
        # Update the azimuth angle (viewing rotation) by 1 degree per frame
        ax.view_init(elev=20, azim=i * 1) 
        return scatter,

    print("Generating animation (isomap_rotation.gif). This may take a moment...")
    
    # Create the animation object
    # frames=360 ensures a full 360-degree rotation (360 frames * 1 degree/frame)
    # interval=50 sets the delay between frames to 50ms (20 frames per second)
    anim = FuncAnimation(fig, animate, frames=360, interval=50, blit=False)
    
    # Save the animation as a GIF
    try:
        anim.save('isomap_rotation.gif', writer=PillowWriter(fps=20))
        print("Animation successfully saved to 'isomap_rotation.gif'.")
    except Exception as e:
        print(f"\nError saving GIF: {e}")
        print("Please ensure you have the Pillow library installed (`pip install pillow`)")
        
    plt.close(fig) # Close the figure after saving the animation

if __name__ == '__main__':
    run_isomap_and_plot()