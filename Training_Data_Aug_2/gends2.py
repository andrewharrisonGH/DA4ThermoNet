#!/usr/bin/env python3

import os
import sys
import time
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
import h5py
import numpy as np
from utils import pdb_utils


def parse_cmd():
    """Parse command-line arguments.

    Returns
    -------
    args : Namespace
        Command-line arguments.
    """
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
    """Compute WT and MT voxel features for a single mutation and rotation.

    Parameters
    ----------
    task : tuple
        (pdb_chain, pos, wt, mt, ddg, rotation, pdb_dir, boxsize, voxelsize, verbose, reverse_flag)

    Returns
    -------
    dict or None
        Dictionary with tensor and ddG if successful, else None.
    """
    pdb_chain, pos, wt, mt, ddg, rot, pdb_dir, boxsize, voxelsize, verbose, reverse_flag = task
    x, y, z = rot

    wt_path = os.path.join(pdb_dir, pdb_chain, 'Augmented', f"{pdb_chain}_wt_{pos}_{x}_{y}_{z}.pdb")
    mt_path = os.path.join(pdb_dir, pdb_chain, 'Augmented', f"{pdb_chain}_mt_{wt}{pos}{mt}_{x}_{y}_{z}.pdb")

    if not os.path.exists(wt_path) or not os.path.exists(mt_path):
        return None

    # Compute features
    features_wt = pdb_utils.compute_voxel_features(pos, wt_path, boxsize=boxsize, voxelsize=voxelsize, verbose=verbose)
    features_mt = pdb_utils.compute_voxel_features(pos, mt_path, boxsize=boxsize, voxelsize=voxelsize, verbose=verbose)

    # Remove property channel 6
    features_wt = np.delete(features_wt, obj=6, axis=0)
    features_mt = np.delete(features_mt, obj=6, axis=0)

    # Select dataset type
    if reverse_flag:
        tensor = np.concatenate((features_mt, features_wt), axis=0).astype('float32')
        ddg_val = -float(ddg)
    else:
        tensor = np.concatenate((features_wt, features_mt), axis=0).astype('float32')
        ddg_val = float(ddg)

    return {'tensor': tensor, 'ddg': ddg_val}


def main():
    """Main function to generate voxel dataset and ddG file."""
    args = parse_cmd()
    pdb_dir = os.path.abspath(args.pdb_dir)

    # Load rotations
    rotations = []
    with open(args.rotations, 'r') as f:
        for line in f:
            rots = line.strip().split(',')
            if rots[0] != 'x_rot':
                rotations.append(tuple(rots))

    # Collect tasks
    tasks = []
    with open(args.input, 'r') as f:
        for line in f:
            pdb_chain, pos, wt, mt, ddg = line.strip().split(',')
            if pdb_chain == 'pdb_id':
                continue
            pdb_chain = pdb_chain.upper()
            for rot in rotations:
                tasks.append((pdb_chain, pos, wt, mt, ddg, rot,
                              pdb_dir, args.boxsize, args.voxelsize, args.verbose, args.reverse))

    # Probe shape using first valid feature
    for task in tasks:
        result = compute_features(task)
        if result is not None:
            C, X, Y, Z = result['tensor'].shape
            break

    # Decide dataset name
    dset_name = "rev" if args.reverse else "fwd"

    # Create HDF5 file and ddG text
    h5_path = args.output + f'_{dset_name}.h5'
    txt_path = args.output + f'_{dset_name}_ddg.txt'
    with h5py.File(h5_path, 'w') as hf, open(txt_path, 'w') as ddg_file:
        dset = hf.create_dataset(dset_name, shape=(0, C, X, Y, Z), maxshape=(None, C, X, Y, Z),
                                 dtype='float32', chunks=True)

        batch_size = 50
        buffer_tensors, buffer_ddg = [], []
        idx = 0

        with Pool(min(args.ncores, cpu_count())) as pool:
            for res in pool.imap(compute_features, tasks):
                if res is None:
                    continue
                buffer_tensors.append(res['tensor'])
                buffer_ddg.append(res['ddg'])

                if len(buffer_tensors) >= batch_size:
                    dset.resize((idx + len(buffer_tensors), C, X, Y, Z))
                    dset[idx:idx + len(buffer_tensors)] = np.array(buffer_tensors)
                    for ddg_val in buffer_ddg:
                        ddg_file.write(f"{ddg_val}\n")

                    idx += len(buffer_tensors)
                    buffer_tensors, buffer_ddg = [], []

        # Write remaining
        if buffer_tensors:
            dset.resize((idx + len(buffer_tensors), C, X, Y, Z))
            dset[idx:idx + len(buffer_tensors)] = np.array(buffer_tensors)
            for ddg_val in buffer_ddg:
                ddg_file.write(f"{ddg_val}\n")

    # Save as a single .npy array
    npy_path = args.output + f"_{dset_name}.npy"
    with h5py.File(h5_path, 'r') as hf:
        dset = hf[dset_name][:]
        np.save(npy_path, dset)
        print(f"Saved {npy_path} with shape {dset.shape}")

    # Remove HDF5 file
    os.remove(h5_path)
    print(f"Removed temporary HDF5 file: {h5_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Dataset generation took {end_time - start_time:.2f} seconds")