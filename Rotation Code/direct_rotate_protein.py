#!/usr/bin/env python

import argparse
import os
import numpy as np
from numpy import array, dot
from math import pi
from math import radians
from copy import deepcopy
#from Bio.PDB import *
from Bio.PDB import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB import Structure

from scipy.spatial.transform import Rotation as R


def get_pdb_structure(pdb_id, pdb_file):
    parser = PDBParser(QUIET = True)
    data = parser.get_structure(pdb_id, pdb_file)
    return data


def centre_model(model, centre_coord):
    # model_copy = deepcopy(model)        # make a deep copy
    for chain in model.get_chains():
       for residue in chain.get_residues():
           for atom in residue.get_atoms():
               newCoord = atom.coord - centre_coord
               atom.coord = newCoord


def rotate_model(model, rotation):
    for chain in model.get_chains():
        for residue in chain.get_residues():
            for atom in residue.get_atoms():
                atom_coord = np.array(atom.coord, dtype=np.float64)
                newCoord = rotation.apply(atom_coord)
                atom.coord = newCoord
    return model


def main():
    parser = argparse.ArgumentParser(description="centre and randomly orient a pdb file n times")
    # parser.add_argument("--n", dest="n", type=int, required=True, help="number of random rotations per input PDB file")
    parser.add_argument("--pdb_in", dest="pdb_in", type=str, required=True, help="path to input PDB file")
    parser.add_argument("--pdb_out", dest="pdb_out", type=str, required=True, help="path to output PDB folder location")
    parser.add_argument("--xrot_clock", dest="xrot_clock", type=float, required=True, help="clockwise rotation angle about the x-axis - 0 to 360 degrees")
    parser.add_argument("--yrot_clock", dest="yrot_clock", type=float, required=True, help="clockwise rotation angle about the y-axis - 0 to 360 degrees")
    parser.add_argument("--zrot_clock", dest="zrot_clock", type=float, required=True, help="clockwise rotation angle about the z-axis - 0 to 360 degrees")
    parser.add_argument("--res_num", dest="res_num", type=int, required=True, help="residue number to centre on")
    args = parser.parse_args()
    
    pdb_in = args.pdb_in
    pdb_id = pdb_in.split('/')[1]
    pdb_id = pdb_id.split('.')[0]
    pdb_out = args.pdb_out
    res_num = args.res_num
    structure = get_pdb_structure(pdb_id, pdb_in)
    model = list(structure.get_models())[0]
    
    residue = list(model.get_residues())[res_num-1]
    atoms = list(residue.get_atoms())
    if residue.get_resname() == 'GLY':
        central_atom = atoms[1]
    else:
        central_atom = atoms[4]
    # n_alpha = list(list(model.get_residues())[res_num-1].get_atoms())[0]
    centre_model(model, central_atom.coord)

    if not os.path.exists(pdb_out):
        os.makedirs(pdb_out)

    output_path = os.path.join(pdb_out, pdb_id)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    rotation = R.from_euler('zyx', [args.zrot_clock, args.yrot_clock, args.xrot_clock], degrees=True)

    rotated_model = rotate_model(deepcopy(model), rotation)  # specify rotation
    
    new_structure = Structure.Structure("test_dir")
    new_structure.add(rotated_model)

    io = PDBIO()
    io.set_structure(new_structure)

    output_path_new = os.path.join(output_path, pdb_id + "_dir_" + str(int(args.xrot_clock)) + "_" + str(int(args.yrot_clock)) + "_" + str(int(args.zrot_clock)) + ".pdb")

    io.save(output_path_new)

    # temp for centered model
    # new_structure = Structure.Structure("centered")
    # new_structure.add(model)

    # io = PDBIO()
    # io.set_structure(new_structure)
    # io.save("centered_cb" + pdb_id + '.pdb')
    

if __name__ == "__main__":
    main()