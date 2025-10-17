import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# --- Configuration ---
# File paths for tensors and rotation degrees
TENSORS_FILE = 'ix_tensors_1A23A_fwd.npy'
DEGREES_FILE = 'ix_tensors_1A23A_fwd_rotations.txt'

def run_pca_and_plot():
    """
    Loads data from specified files, performs PCA, and plots the first two principal components.
    Handles the comma-delimited rotation file with a header row.
    """
    
    # --- Data Loading and Check ---
    
    if not os.path.exists(TENSORS_FILE):
        print(f"Error: Tensor file not found at '{TENSORS_FILE}'. Please place your .npy file in the working directory. Exiting.")
        return
    if not os.path.exists(DEGREES_FILE):
        print(f"Error: Degrees file not found at '{DEGREES_FILE}'. Please place your .txt file in the working directory. Exiting.")
        return
    
    print(f"Loading data from {TENSORS_FILE} and {DEGREES_FILE}...")

    # Load the tensors
    try:
        tensors = np.load(TENSORS_FILE)
        if tensors.ndim < 2:
            raise ValueError(".npy file content is not an array of tensors with multiple samples.")
        n_samples = tensors.shape[0]
        tensor_shape = tensors.shape[1:]
    except Exception as e:
        print(f"Error loading tensor file: {e}")
        print("Exiting. Please ensure your .npy file is valid.")
        return

    # Load the rotation data
    try:
        # Load the data, skipping the header (skiprows=1) and using comma delimiter
        rotation_data = np.loadtxt(DEGREES_FILE, delimiter=',', skiprows=1)
        
        # Assuming the data is N x 3 (x_rot, y_rot, z_rot) and the user wants x_rot (index 0)
        if rotation_data.ndim == 2 and rotation_data.shape[1] >= 1:
            degrees = rotation_data[:, 0].astype(int) 
        elif rotation_data.ndim == 1 and rotation_data.size > 0:
            # Fallback if loadtxt only outputs a 1D array (meaning only one column was present)
            degrees = rotation_data.astype(int)
        else:
            raise ValueError("Rotation data could not be parsed as a 1D or 2D array.")
            
    except Exception as e:
        print(f"Error loading degrees file (check header and delimiter): {e}")
        print("Exiting. Please ensure your .txt file is comma-delimited and includes the header 'x_rot,y_rot,z_rot'.")
        return

    if n_samples != len(degrees):
        print(f"Error: Mismatch between number of tensors ({n_samples}) and degrees ({len(degrees)}).")
        return

    print(f"Loaded {n_samples} samples. Original tensor shape: {tensor_shape}")

    # --- Data Preparation for PCA ---
    
    # 1. Reshape: Flatten each tensor into a single feature vector
    n_features = np.prod(tensor_shape)
    data_flat = tensors.reshape(n_samples, n_features)
    
    print(f"Data flattened to shape: ({n_samples}, {n_features})")

    # 2. Standardization: Scale data to have mean 0 and variance 1
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_flat)
    
    print("Data standardized.")

    # --- Principal Component Analysis (PCA) ---
    
    # 3. Apply PCA, keeping the first two principal components
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)
    
    # The result is an array where the first column is PC1 and the second is PC2
    pc1 = principal_components[:, 0]
    pc2 = principal_components[:, 1]
    
    # --- Visualization ---
    
    # 1. Plotting the results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a colormap to map the degrees (0 to 355) to a color gradient
    scatter = ax.scatter(
        pc1, 
        pc2, 
        c=degrees,  # Color by degree
        cmap='Blues', # Use a cyclical colormap (suitable for rotation)
        s=100,      # Marker size
        alpha=0.8,
        edgecolors='grey', # White edges for better visibility
        linewidth=0.5
    )

    # 2. Add labels and title
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
    ax.set_title(
        'PCA of Protein Tensors: PC2 vs PC1\nColor-coded by X-axis Rotation Degree', 
        fontsize=14, 
        fontweight='bold'
    )
    
    # 3. Add Colorbar for degree scale
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
    cbar.set_label('Rotation Degree (0° to 355°)', rotation=270, labelpad=15, fontsize=12)
    
    # 5. Optional: Annotate a few specific points (e.g., 0, 90, 180, 270)
    key_degrees = [0, 90, 180, 270]
    for deg in key_degrees:
        try:
            # Find the index corresponding to the specific degree
            idx = np.where(degrees == deg)[0][0]
            ax.annotate(
                f'{deg}°', 
                (pc1[idx], pc2[idx]), 
                textcoords="offset points", 
                xytext=(5, 5),
                fontsize=10, 
                fontweight='bold',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.2", color='gray')
            )
        except IndexError:
            pass # Degree not found
            
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_pca_and_plot()