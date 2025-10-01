#!/usr/bin/env python3

import numpy as np
import sys
from argparse import ArgumentParser
import time
import os

# Define a safe chunk size. 1000 samples should result in chunks around 230 MB, 
# which is well within the safety margin for a 68GB system.
CHUNK_SIZE = 1000 

def parse_cmd_args():
    """
    Parses command line arguments for the feature pre-processing script.
    """
    parser = ArgumentParser(description='Pre-processes feature arrays in memory-safe chunks by moving the channel axis and casting to float32.')
    parser.add_argument('-i', '--input_file', dest='input_file', type=str, required=True,
                        help='Path to the input NumPy feature file (e.g., direct_features.npy).')
    parser.add_argument('-o', '--output_file', dest='output_file', type=str, required=True,
                        help='Path to save the pre-processed output file (e.g., direct_features_preprocessed.npy).')
    args = parser.parse_args()
    return args

def preprocess_and_save(input_path, output_path):
    """
    Loads features in chunks, moves axis (C to last dim), casts to float32, and saves 
    the result incrementally to a new memory-mapped file.
    """
    print(f"--- Starting Memory-Safe Chunked Pre-processing ---")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    # CRITICAL: Since the last run failed to produce a correct file, we force an overwrite.
    if os.path.exists(output_path):
        print(f"WARNING: Output file {output_path} exists and will be overwritten to apply fixes.")
        
    try:
        start_time = time.time()
        
        # 1. Load input data structure (read-only memory-map)
        print("Loading input data structure (memory-mapped)...")
        X_original = np.load(input_path, mmap_mode='r')
        
        # Original shape is expected to be (N, 14, 16, 16, 16) -> (N, C, D, H, W)
        n_samples = X_original.shape[0]
        
        # New shape will be (N, 16, 16, 16, 14) -> (N, D, H, W, C) with dtype float32
        new_shape = (n_samples, 16, 16, 16, 14)
        
        print(f"Original Shape: {X_original.shape} | New Shape: {new_shape} | Total Samples: {n_samples}")
        print(f"Processing {n_samples} samples in chunks of {CHUNK_SIZE}...")
        
        # 2. Create the output memory-mapped file (w+ forces creation/overwrite)
        X_out = np.lib.format.open_memmap(
            output_path, 
            mode='w+', 
            dtype=np.float32, 
            shape=new_shape
        )

        # 3. Iterate and process in chunks
        for i in range(0, n_samples, CHUNK_SIZE):
            end_idx = min(i + CHUNK_SIZE, n_samples)
            
            # Load the small chunk from the read-only memory map and process it in RAM
            X_chunk = X_original[i:end_idx] 
            
            # Perform the required transformation: (N, C, D, H, W) -> (N, D, H, W, C)
            # We use explicit transpose (0, 2, 3, 4, 1) to guarantee the axis order.
            X_processed_chunk = np.transpose(X_chunk, (0, 2, 3, 4, 1)).astype(np.float32)
            
            # Write the processed chunk directly to the output memory map
            X_out[i:end_idx] = X_processed_chunk
            
            # Print progress
            if (i // CHUNK_SIZE) % 100 == 0 or end_idx == n_samples:
                 progress = (end_idx / n_samples) * 100
                 print(f"  Processed {end_idx}/{n_samples} samples ({progress:.2f}%)")

        # 4. Finalize and Validate
        X_out.flush()
        del X_out # Important: close the file handle

        # Final validation check
        X_check = np.load(output_path, mmap_mode='r')
        if X_check.shape == new_shape and X_check.dtype == np.float32:
            print(f"SUCCESS: Validation passed. Output file is in the correct shape {new_shape} and dtype.")
        else:
            raise ValueError(f"Validation FAILED! Expected shape {new_shape} and dtype float32, but got {X_check.shape} and {X_check.dtype}.")
            
        end_time = time.time()
        print(f"--- Pre-processing complete in {end_time - start_time:.2f} seconds ---")
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        # Clean up partially created file if necessary
        if 'X_out' in locals():
            # Attempt to delete the partially created output file
            os.remove(output_path)
            print(f"Removed partially created file: {output_path}")
        sys.exit(1)


def main():
    args = parse_cmd_args()
    preprocess_and_save(args.input_file, args.output_file)

if __name__ == '__main__':
    # Increase the recursion limit for very large arrays.
    sys.setrecursionlimit(2000)
    main()