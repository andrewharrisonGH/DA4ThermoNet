#!/usr/bin/env python3
import os
import time
import gc
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
import numpy as np
from numpy.lib import format as npformat
from utils import pdb_utils


def parse_cmd():
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help='File containing a list of protein mutations.')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
                        help='Base filename to write the dataset.')
    parser.add_argument('-p', '--pdb_dir', dest='pdb_dir', type=str, required=True,
                        help='Directory where PDB files are stored.')
    parser.add_argument('--rotations', dest='rotations', type=str, required=True,
                        help='CSV file containing x, y, z rotations.')
    parser.add_argument('--boxsize', dest='boxsize', type=int,
                        help='Bounding box size around mutation site.')
    parser.add_argument('--voxelsize', dest='voxelsize', type=int,
                        help='Voxel size.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Print verbose messages from HTMD.')
    parser.add_argument('--reverse', dest='reverse', action='store_true',
                        help='If set, only the reverse dataset is created and ddGs are sign-flipped.')
    parser.add_argument('--ncores', dest='ncores', type=int, default=1,
                        help='Number of cores for multiprocessing.')
    return parser.parse_args()


def compute_features(task):
    """
    Computes voxelized features and returns separate tensors along with the rotation angles.
    """
    pdb_chain, pos, wt, mt, ddg, rot, pdb_dir, boxsize, voxelsize, verbose, reverse_flag = task
    x_deg, y_deg, z_deg = map(float, rot)
    res_num = int(pos)

    # Load WT and MT structures
    wt_path = os.path.join(pdb_dir, pdb_chain, f"{pdb_chain}_relaxed.pdb")
    mt_path = os.path.join(pdb_dir, pdb_chain, f"{pdb_chain}_relaxed_{wt}{pos}{mt}_relaxed.pdb")
    if not os.path.exists(wt_path) or not os.path.exists(mt_path):
        raise FileNotFoundError(f"Missing PDB files: {wt_path} or {mt_path}")

    # Convert rotation from degrees to radians
    rot_rad = np.radians([x_deg, y_deg, z_deg])

    features_wt = pdb_utils.compute_voxel_features(res_num, wt_path, boxsize=boxsize,
                                                   voxelsize=voxelsize, verbose=verbose,
                                                   rotations=rot_rad)
    features_mt = pdb_utils.compute_voxel_features(res_num, mt_path, boxsize=boxsize,
                                                   voxelsize=voxelsize, verbose=verbose,
                                                   rotations=rot_rad)

    # Remove property channel 6 from both
    features_wt = np.delete(features_wt, obj=6, axis=0)
    features_mt = np.delete(features_mt, obj=6, axis=0)

    # Assign WT/MT tensors based on reverse_flag
    if reverse_flag:
        tensor_wt = features_mt.astype('float32')
        tensor_mt = features_wt.astype('float32')
    else:
        tensor_wt = features_wt.astype('float32')
        tensor_mt = features_wt.astype('float32')

    # Transpose both tensors to (D, H, W, C) format
    tensor_wt = np.transpose(tensor_wt, (1, 2, 3, 0))
    tensor_mt = np.transpose(tensor_mt, (1, 2, 3, 0))

    del features_wt, features_mt
    gc.collect()
    
    # --- MODIFICATION START ---
    # Return the rotation tuple instead of the ddG value
    return {'wt_tensor': tensor_wt, 'mt_tensor': tensor_mt, 'rotation': rot}
    # --- MODIFICATION END ---


def generate_tasks(input_file, rotations_list, pdb_dir, boxsize, voxelsize, verbose, reverse_flag):
    """
    Yields tasks one by one instead of creating a list in memory.
    """
    with open(input_file, 'r') as f:
        for line in f:
            if 'pdb_id' in line:
                continue
            
            pdb_chain, pos, wt, mt, ddg = line.strip().split(',')
            pdb_chain = pdb_chain.upper()
            
            for rot in rotations_list:
                yield (pdb_chain, pos, wt, mt, ddg, rot,
                       pdb_dir, boxsize, voxelsize, verbose, reverse_flag)


def main():
    args = parse_cmd()
    pdb_dir = os.path.abspath(args.pdb_dir)

    # Load rotations
    rotations = []
    with open(args.rotations, 'r') as f:
        for line in f:
            rots = line.strip().split(',')
            if rots[0] != 'x_rot':
                rotations.append(tuple(rots))

    # 1. Probe shape using a temporary generator
    C = X = Y = Z = None 
    probe_generator = generate_tasks(args.input, rotations, pdb_dir, args.boxsize,
                                     args.voxelsize, args.verbose, args.reverse)
    for task in probe_generator:
        try:
            result = compute_features(task)
            if result is not None:
                D, H, W, C = result['wt_tensor'].shape 
                break
        except FileNotFoundError:
            continue
    
    if D is None:
        raise RuntimeError("Could not generate any valid features to determine the output shape.")

    # 2. Estimate total samples
    num_mutations = sum(1 for line in open(args.input)) - 1
    n_samples = num_mutations * len(rotations)

    # 3. Create the main generator for the multiprocessing pool
    tasks_for_pool = generate_tasks(args.input, rotations, pdb_dir, args.boxsize,
                                    args.voxelsize, args.verbose, args.reverse)
    
    # 4. Define separate output filenames
    dset_name = "rev" if args.reverse else "fwd"
    wt_npy_path = args.output + f"_{dset_name}_wt.npy"
    mt_npy_path = args.output + f"_{dset_name}_mt.npy"
    # --- MODIFICATION START ---
    txt_path = args.output + f"_{dset_name}_rotations.txt" # Changed filename
    # --- MODIFICATION END ---

    # 5. Preallocate memory-mapped files
    mmap_wt = npformat.open_memmap(
        wt_npy_path, mode="w+", dtype="float32", shape=(n_samples, D, H, W, C)
    )
    mmap_mt = npformat.open_memmap(
        mt_npy_path, mode="w+", dtype="float32", shape=(n_samples, D, H, W, C)
    )
    print(f"Allocated WT memory-mapped array at '{wt_npy_path}' with shape: {mmap_wt.shape}")
    print(f"Allocated MT memory-mapped array at '{mt_npy_path}' with shape: {mmap_mt.shape}")

    # --- MODIFICATION START ---
    # Open the rotations file and write a header
    with open(txt_path, "w") as rotations_file:
        rotations_file.write("x_rot,y_rot,z_rot\n") # Optional but good practice
    # --- MODIFICATION END ---
        idx = 0
        with Pool(min(args.ncores, cpu_count()), maxtasksperchild=1) as pool:
            for res in pool.imap_unordered(compute_features, tasks_for_pool, chunksize=2):
                if res is None:
                    continue
                
                if idx < n_samples:
                    mmap_wt[idx] = res['wt_tensor']
                    mmap_mt[idx] = res['mt_tensor']
                    # --- MODIFICATION START ---
                    # Write the rotation data to the file
                    rot_x, rot_y, rot_z = res['rotation']
                    rotations_file.write(f"{rot_x},{rot_y},{rot_z}\n")
                    # --- MODIFICATION END ---
                    idx += 1
                else:
                    print(f"Warning: generated more results than allocated space ({n_samples}).")
                
                del res
                gc.collect()

    # Flush both memory maps to ensure data is written to disk
    mmap_wt.flush()
    mmap_mt.flush()

    # Update final print statements
    print(f"\n✅ Saved {idx} WT tensors to {wt_npy_path}")
    print(f"✅ Saved {idx} MT tensors to {mt_npy_path}")
    # --- MODIFICATION START ---
    print(f"✅ Saved {idx} rotations to {txt_path}")
    # --- MODIFICATION END ---


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nDataset generation took {end_time - start_time:.2f} seconds")