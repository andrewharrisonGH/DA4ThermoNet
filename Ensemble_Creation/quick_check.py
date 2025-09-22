import numpy as np

# # Load the .npy file
# data = np.load("Q1744_tensors2_fwd.npy")

# # Print shape and dtype
# print("Shape:", data.shape)
# print("Data type:", data.dtype)

y_direct = np.loadtxt('ssym_tensors_fwd_ddg.txt')
y_inverse = -y_direct
print(y_direct[0:5])
print(y_inverse[0:5])