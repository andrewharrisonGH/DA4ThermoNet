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
                        help='If set, only the reverse dataset is created.')
    parser.add_argument('--ncores', dest='ncores', type=int, default=1,
                        help='Number of cores for multiprocessing.')
    return parser.parse_args()


def compute_features(task):
    """
    Computes voxel features for a given mutation and rotation, 
    returning the concatenated tensor and the rotation used.
    """
    # rot is a tuple of strings (x_deg, y_deg, z_deg)
    pdb_chain, pos, wt, mt, ddg, rot, pdb_dir, boxsize, voxelsize, verbose, reverse_flag = task
    x_deg, y_deg, z_deg = rot
    res_num = int(pos)

    # Load WT and MT structures
    wt_path = os.path.join(pdb_dir, pdb_chain, f"{pdb_chain}_relaxed.pdb")
    mt_path = os.path.join(pdb_dir, pdb_chain, f"{pdb_chain}_relaxed_{wt}{pos}{mt}_relaxed.pdb")
    if not os.path.exists(wt_path) or not os.path.exists(mt_path):
        # We don't want to raise an error, just return None so the pool skips this task
        print(f"FileNotFoundError: Missing PDB files for {pdb_chain} {wt}{pos}{mt}. Skipping.")
        return None

    # Convert rotation from degrees to radians for compute_voxel_features
    # We keep the degrees tuple for saving later
    rot_rad = np.radians(list(map(float, rot))) 

    features_wt = pdb_utils.compute_voxel_features(res_num, wt_path, boxsize=boxsize,
                                                   voxelsize=voxelsize, verbose=verbose,
                                                   rotations=rot_rad)
    features_mt = pdb_utils.compute_voxel_features(res_num, mt_path, boxsize=boxsize,
                                                   voxelsize=voxelsize, verbose=verbose,
                                                   rotations=rot_rad)

    # Remove property channel 6
    features_wt = np.delete(features_wt, obj=6, axis=0)
    features_mt = np.delete(features_mt, obj=6, axis=0)

    # Combine channels (concatenation logic is kept)
    if reverse_flag:
        # MT vs WT
        tensor = np.concatenate((features_mt, features_wt), axis=0).astype('float32')
    else:
        # WT vs MT
        tensor = np.concatenate((features_wt, features_mt), axis=0).astype('float32')

    # Transpose to (D, H, W, C)
    tensor = np.transpose(tensor, (1, 2, 3, 0))

    del features_wt, features_mt
    gc.collect()
    
    # *** MODIFICATION: Return the rotation degrees instead of ddg ***
    # rot is the tuple of strings: (x_deg, y_deg, z_deg)
    return {'tensor': tensor, 'rotation': rot}


def generate_tasks(input_file, rotations_list, pdb_dir, boxsize, voxelsize, verbose, reverse_flag):
    """
    Yields tasks one by one instead of creating a list in memory.
    """
    with open(input_file, 'r') as f:
        for line in f:
            # Skip header line
            if 'pdb_id' in line:
                continue
            
            # pdb_chain, pos, wt, mt, ddg = line.strip().split(',')
            # We still read ddg but no longer use it in the final output file
            try:
                pdb_chain, pos, wt, mt, ddg = line.strip().split(',')
            except ValueError:
                # Handle lines with incorrect number of columns if necessary
                continue
                
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
            if rots[0].lower() not in ('x_rot', 'x'): # General header check
                # rot is stored as a tuple of strings (e.g., ('0.0', '0.0', '0.0'))
                rotations.append(tuple(rots)) 

    # 1. Probe shape using a temporary generator
    D = H = W = C = None 
    probe_generator = generate_tasks(args.input, rotations, pdb_dir, args.boxsize,
                                     args.voxelsize, args.verbose, args.reverse)
    for task in probe_generator:
        try:
            result = compute_features(task)
            if result is not None:
                # The shape returned by compute_features is now (D, H, W, C)
                D, H, W, C = result['tensor'].shape 
                break
        except Exception:
             # Catch all exceptions during probing to find the shape
            continue
    
    if D is None:
        raise RuntimeError("Could not generate any valid features to determine the output shape.")

    # 2. Estimate the total number of samples for the memory-mapped file
    num_mutations = sum(1 for line in open(args.input) if not line.startswith('pdb_id'))
    n_samples = num_mutations * len(rotations)

    # 3. Create the main generator that will be passed to the multiprocessing pool
    tasks_for_pool = generate_tasks(args.input, rotations, pdb_dir, args.boxsize,
                                     args.voxelsize, args.verbose, args.reverse)
    
    # Decide dataset name
    dset_name = "rev" if args.reverse else "fwd"
    npy_path = args.output + f"_{dset_name}.npy"
    # *** MODIFICATION: New filename for rotations ***
    rot_path = args.output + f"_{dset_name}_rotations.csv" 

    # Preallocate .npy file as a memory-mapped array with the final (N, D, H, W, C) shape
    mmap = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype="float32", shape=(n_samples, D, H, W, C)
    )
    print(f"Allocated memory-mapped array with shape: ({n_samples}, {D}, {H}, {W}, {C})")

    # *** MODIFICATION: Write header and rotations to a CSV file ***
    with open(rot_path, "w") as rot_file:
        rot_file.write("x_rot,y_rot,z_rot\n") # Write a header
        idx = 0
        
        # maxtasksperchild=1 is a great way to ensure memory from each task is fully released
        with Pool(min(args.ncores, cpu_count()), maxtasksperchild=1) as pool:
            # Pass the generator directly to the pool. It will pull tasks as needed.
            for res in pool.imap_unordered(compute_features, tasks_for_pool, chunksize=2):
                if res is None:
                    continue
                
                if idx < n_samples:
                    mmap[idx] = res['tensor']
                    # *** MODIFICATION: Write rotation to the file ***
                    x, y, z = res['rotation']
                    rot_file.write(f"{x},{y},{z}\n") 
                    idx += 1
                else:
                    print(f"Warning: generated more results than allocated space ({n_samples}). Increase allocated space if this is unexpected.")
                
                del res
                gc.collect()

    # Flush the memory map to ensure data is written to disk
    mmap.flush() 
    print(f"Saved {idx} tensors to {npy_path} with allocated shape {mmap.shape}")
    print(f"Saved rotations to {rot_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Dataset generation took {end_time - start_time:.2f} seconds")