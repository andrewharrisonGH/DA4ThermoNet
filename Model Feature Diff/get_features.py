from moleculekit.tools.voxeldescriptors import getVoxelDescriptors,getCenters, rotateCoordinates
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.molecule import Molecule
from Bio.PDB import PDBParser, NeighborSearch
import argparse
import os
import numpy as np


def compute_voxel_features(mutation_site, pdb_file, verbose=False, 
        boxsize=16, voxelsize=1):
    """Compute voxel features around the mutation site.

    Parameters
    ----------
    pdbqt_wt : AutoDock PDBQT file
        AutoDock PDBQT file of the wild-type protein.
    pdbqt_mt : AutoDock PDBQT file
        AutoDock PDBQT file of the mutant protein.
    mutation_site : int
        Residue sequence position where the mutation took place.
    rotations : list
        Rotation angles in radian around x, y, and z axes.

    Returns
    -------
    NumPy nd-array

    """
    mol = Molecule(pdb_file)
    prot = prepareProteinForAtomtyping(mol, verbose=verbose)
    center = mol.get('coords', 'resid ' + str(mutation_site) + ' and name CB')
    # center_wt = compute_interaction_center(pdb_file_wt, mutation_site)
    if center.size == 0:
        center = mol.get('coords', 'resid ' + str(mutation_site) + ' and name CA')
    #
    
    features, centers, _ = getVoxelDescriptors(prot, center=center.flatten(), 
        boxsize=[boxsize, boxsize, boxsize], voxelsize=voxelsize, validitychecks=False)
    
    # Reshape features to N_voxels × N_channels (flattened voxel grid)
    features = features.reshape(-1, features.shape[1])  # (N_voxels, 7)
    
    return features, centers  # both shape: (N_voxels, X)

def main():
    parser = argparse.ArgumentParser(description="Fetch features for a given PDB file around a specified residue")
    parser.add_argument("--pdb_in", dest="pdb_in", type=str, required=True, help="path to input PDB folder")
    parser.add_argument("--pdb_id",  dest="pdb_id", type=str, required=True, help="PDB file name")
    parser.add_argument("--res_num", dest="res_num", type=int, required=True, help="residue number to centre on")
    args = parser.parse_args()

    pdb_path = os.path.join(args.pdb_in, args.pdb_id + ".pdb")
    features, centers = compute_voxel_features(args.res_num, pdb_path)

    # Concatenate coordinates (centers) with feature channels
    data = np.hstack((centers, features))  # shape: (N, 10)

    # Save to CSV: first 3 columns = x, y, z; next 7 = feature values
    np.savetxt(args.pdb_id + "_feats.csv", data, delimiter=",", fmt="%.4f")


if __name__ == "__main__":
    main()