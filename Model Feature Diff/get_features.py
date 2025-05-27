from moleculekit.tools.voxeldescriptors import getVoxelDescriptors,getCenters, rotateCoordinates
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.molecule import Molecule
from Bio.PDB import PDBParser, NeighborSearch
import argparse
import os


def compute_voxel_features(mutation_site, pdb_file, verbose=False, 
        boxsize=16, voxelsize=1, rotations=None):
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
    if rotations is None:
        features, _, _ = getVoxelDescriptors(prot, center=center.flatten(), 
                boxsize=[boxsize, boxsize, boxsize], voxelsize=voxelsize, validitychecks=False)
    else:
        voxel_centers = getCenters(prot, boxsize=[boxsize, boxsize, boxsize], 
                center=center.flatten(), voxelsize=voxelsize)
        rotated_voxel_centers = rotateCoordinates(voxel_centers[0], rotations, center.flatten())
        features, _ = getVoxelDescriptors(prot, usercenters=rotated_voxel_centers, validitychecks=False)
    # return features
    nchannels = features.shape[1]
    n_voxels = int(boxsize / voxelsize)
    features = features.transpose().reshape((nchannels, n_voxels, n_voxels, n_voxels))

    return features

def main():
    parser = argparse.ArgumentParser(description="Fetch features for a given PDB file around a specified residue")
    parser.add_argument("--pdb_in", dest="pdb_in", type=str, required=True, help="path to input PDB folder")
    parser.add_argument("--pdb_id",  dest="pdb_id", type=str, required=True, help="PDB file name")
    parser.add_argument("--res_num", dest="res_num", type=int, required=True, help="residue number to centre on")
    args = parser.parse_args()

    pdb_in = args.pdb_in
    pdb_id = args.pdb_id
    input_path = os.path.join(pdb_in, pdb_id + ".pdb")
    res_num = args.res_num

    print(compute_voxel_features(res_num,input_path))


if __name__ == "__main__":
    main()