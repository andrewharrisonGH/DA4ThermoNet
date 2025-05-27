import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# --- Load CSV ---
data = np.loadtxt("1A23_feats.csv", delimiter=",")
coords = data[:, :3]
features = data[:, 3:]

# Choose which feature to visualize
feature_index = 3
values = features[:, feature_index]  # Already normalized to [0, 1]

# Apply stronger transparency for small values
alphas = values**2  # enhance contrast: lower values more transparent

# RGBA colors from colormap + per-point alpha
colors = [(*plt.cm.viridis(v)[:3], a) for v, a in zip(values, alphas)]

# --- Plotting ---
fig = plt.figure(figsize=(10, 8), facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor='white')

# 3D scatter with transparency and no edges
sc = ax.scatter(
    coords[:, 0], coords[:, 1], coords[:, 2],
    c=colors,
    s=60,               # larger dots
    edgecolors='none'
)

# --- Axes styling ---
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title(f"Voxel Feature {feature_index}", fontsize=14, pad=20)
ax.grid(True, alpha=0.3)
ax.set_box_aspect([1, 1, 1])  # equal aspect

# --- Add colorbar ---
norm = Normalize(vmin=0, vmax=1)  # since already normalized
sm = ScalarMappable(norm=norm, cmap='viridis')
sm.set_array([])  # dummy mappable
cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
cb.set_label('Feature Intensity', fontsize=12)

plt.tight_layout()
plt.show()