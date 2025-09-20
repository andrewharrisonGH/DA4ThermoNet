import h5py
import numpy as np
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_h5_to_npy.py <input_file.h5>")
        sys.exit(1)

    h5_file = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(h5_file))[0]

    # Open the HDF5 file and save forward/reverse datasets as .npy
    with h5py.File(h5_file, "r") as f:
        fwd_data = f["fwd"][:]
        rev_data = f["rev"][:]

        # Print shapes
        print(f"Forward dataset shape: {fwd_data.shape}")
        print(f"Reverse dataset shape: {rev_data.shape}")

        # Save to .npy
        np.save(f"{base_name}_fwd.npy", fwd_data)
        np.save(f"{base_name}_rev.npy", rev_data)

    print(f"Saved {base_name}_fwd.npy and {base_name}_rev.npy successfully.")

if __name__ == "__main__":
    main()