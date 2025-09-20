import numpy as np

# Load forward dataset
fwd = np.load("test_gends2_fwd.npy")
print("Forward dataset shape:", fwd.shape)
print("Forward dataset dtype:", fwd.dtype)
print("Memory usage (MB):", fwd.nbytes / (1024 ** 2))

# Preview first tensor
print("First sample (forward):")
print(fwd[0])