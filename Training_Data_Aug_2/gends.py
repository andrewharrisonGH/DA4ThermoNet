#!/usr/bin/env python3

import os
import sys
from utils import pdb_utils
import time
from argparse import ArgumentParser
import numpy as np
import h5py
import datetime


def parse_cmd():
    """Parse command-line arguments.

    Returns
    -------
    Command-line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
                        help='File that contains a list of protein mutations.')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
                        help='File to which to write the dataset.')
    parser.add_argument('-p', '--pdb_dir', dest='pdb_dir', type=str, required=True,
                        help='Directory where PDB file are stored.')
    parser.add_argument('--rotations', dest='rotations', type=str, required=True,
                        help='File that contains the x, y, z rotations in degrees.')
    parser.add_argument('--boxsize', dest='boxsize', type=int,
                        help='Size of the bounding box around the mutation site.')
    parser.add_argument('--voxelsize', dest='voxelsize', type=int,
                        help='Size of the voxel.')
    parser.add_argument('-w', '--overwrite', dest='overwrite', action='store_true',
                        help='Whether to overwrite PDBQT files and mutant PDB files.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Whether to print verbose messages from HTMD function calls.')
    parser.add_argument('--reverse', dest='reverse', action='store_true',
                        help='If set, generate only reverse datasets (MT→WT). Otherwise forward (WT→MT).')
    args = parser.parse_args()
    # do any necessary argument checking here before returning
    return args


def h5_to_npy_iterative(h5_file, dataset_name, npy_file, batch_size=100):
    """
    Convert one dataset from HDF5 to NPY without loading all into memory.
    """
    with h5py.File(h5_file, "r") as f:
        dset = f[dataset_name]
        shape = dset.shape
        dtype = dset.dtype

        # Prepare memmap .npy file
        arr = np.lib.format.open_memmap(
            npy_file, mode="w+", dtype=dtype, shape=shape
        )

        # Write in chunks
        for start in range(0, shape[0], batch_size):
            end = min(start + batch_size, shape[0])
            arr[start:end] = dset[start:end]

        # Ensure data is written
        del arr

    print(f"Saved {npy_file} with shape {shape} and dtype {dtype}")


def main():
    """

    Returns
    -------

    """
    args = parse_cmd()

    # Load rotations
    rotations = []
    with open(args.rotations, 'r') as f:
        for l in f:
            rots = l.strip().split(',')
            if rots[0] != 'x_rot':
                rotations.append(tuple(rots))

    # Probe shape from first valid WT feature
    sample_feature = None
    pdb_dir = os.path.abspath(args.pdb_dir)

    with open(args.input, 'r') as f:
        for l in f:
            pdb_chain, pos, wt, mt, ddg = l.strip().split(',')
            if pdb_chain == 'pdb_id':
                continue
            pdb_chain = pdb_chain.upper()
            x, y, z = rotations[0]
            wt_pdb_path = os.path.join(pdb_dir, pdb_chain, 'Augmented',
                                       f"{pdb_chain}_wt_{pos}_{x}_{y}_{z}.pdb")
            if os.path.exists(wt_pdb_path):
                features = pdb_utils.compute_voxel_features(pos, wt_pdb_path,
                                                            boxsize=args.boxsize,
                                                            voxelsize=args.voxelsize,
                                                            verbose=args.verbose)
                features = np.delete(features, obj=6, axis=0)
                mt_features = features.copy()
                sample_feature = np.concatenate((features, mt_features), axis=0)
                break

    if sample_feature is None:
        print("Error: Could not find any valid PDB to probe feature shape.")
        sys.exit(1)

    C, X, Y, Z = sample_feature.shape  # feature dimensions

    # Create HDF5 datasets with maxshape=None for resizing
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_name = f"{args.output}_{timestamp}.h5"
    with h5py.File(h5_name, 'w') as hf:
        if args.reverse:
            dset = hf.create_dataset('rev', shape=(0, C, X, Y, Z),
                                     maxshape=(None, C, X, Y, Z),
                                     dtype='float32', chunks=True)
        else:
            dset = hf.create_dataset('fwd', shape=(0, C, X, Y, Z),
                                     maxshape=(None, C, X, Y, Z),
                                     dtype='float32', chunks=True)

        idx = 0

        with open(args.input, 'r') as f:
            for l in f:
                pdb_chain, pos, wt, mt, ddg = l.strip().split(',')
                if pdb_chain == 'pdb_id':
                    continue
                pdb_chain = pdb_chain.upper()

                for x, y, z in rotations:
                    wt_key = f"{pdb_chain}{pos}_{x}_{y}_{z}"
                    wt_pdb_path = os.path.join(pdb_dir, pdb_chain, 'Augmented',
                                               f"{pdb_chain}_wt_{pos}_{x}_{y}_{z}.pdb")
                    mt_pdb_path = os.path.join(pdb_dir, pdb_chain, 'Augmented',
                                               f"{pdb_chain}_mt_{wt}{pos}{mt}_{x}_{y}_{z}.pdb")

                    # WT features
                    if not os.path.exists(wt_pdb_path):
                        print('Missing WT PDB:', wt_pdb_path)
                        sys.exit(1)
                    features_wt = pdb_utils.compute_voxel_features(pos, wt_pdb_path,
                                                                boxsize=args.boxsize,
                                                                voxelsize=args.voxelsize,
                                                                verbose=args.verbose)
                    features_wt = np.delete(features_wt, obj=6, axis=0)
                        
                    # MT features
                    if not os.path.exists(mt_pdb_path):
                        print('Missing mutant PDB:', mt_pdb_path)
                        sys.exit(1)
                    features_mt = pdb_utils.compute_voxel_features(pos, mt_pdb_path,
                                                                   boxsize=args.boxsize,
                                                                   voxelsize=args.voxelsize,
                                                                   verbose=args.verbose)
                    features_mt = np.delete(features_mt, obj=6, axis=0)

                    # Forward & Reverse
                    if args.reverse:
                        combined = np.concatenate((features_mt, features_wt), axis=0).astype('float32')
                    else:
                        combined = np.concatenate((features_wt, features_mt), axis=0).astype('float32')

                    # Resize and add
                    dset.resize((idx + 1, C, X, Y, Z))
                    dset[idx] = combined
                    idx += 1

    print(f"Saved dataset: {h5_name}")

    # --- Convert to NPY automatically ---
    h5_file = h5_name
    base_name = args.output

    with h5py.File(h5_file, "r") as f:
        if "fwd" in f:
            h5_to_npy_iterative(h5_file, "fwd", f"{base_name}_fwd.npy")
        if "rev" in f:
            h5_to_npy_iterative(h5_file, "rev", f"{base_name}_rev.npy")

    # Delete .h5 file after conversion
    os.remove(h5_file)
    print(f"Deleted {h5_file} after successful conversion.")

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print('gends.py took', elapsed, 'seconds to generate the dataset.')
