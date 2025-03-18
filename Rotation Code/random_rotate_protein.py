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
    parser.add_argument("--n", dest="n", type=int, required=True, help="number of random rotations per input PDB file")
    parser.add_argument("--pdb_in", dest="pdb_in", type=str, required=True, help="path to input PDB file")
    parser.add_argument("--pdb_out", dest="pdb_out", type=str, required=True, help="path to output PDB folder location")
    #parser.add_argument("--xrot_clock", dest="xrot_clock", type=float, required=True, help="clockwise rotation angle about the x-axis - 0 to 360 degrees")
    #parser.add_argument("--yrot_clock", dest="yrot_clock", type=float, required=True, help="clockwise rotation angle about the y-axis - 0 to 360 degrees")
    parser.add_argument("--res_num", dest="res_num", type=int, required=True, help="residue number to centre on")
    args = parser.parse_args()
    
    pdb_in = args.pdb_in
    pdb_id = pdb_in.split('/')[1]
    pdb_id = pdb_id.split('.')[0]
    pdb_out = args.pdb_out
    res_num = args.res_num
    structure = get_pdb_structure(pdb_id, pdb_in)
    model = list(structure.get_models())[0]
    n_alpha = list(list(model.get_residues())[res_num-1].get_atoms())[0]
    centre_model(model, n_alpha.coord)

    if not os.path.exists(pdb_out):
        os.makedirs(pdb_out)

    output_path = os.path.join(pdb_out, pdb_id)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    n = args.n  # number of rotations
    rotated_models = [rotate_model(deepcopy(model), rotation) for rotation in R.random(n)]  # specify rotation
    for i in range(len(rotated_models)):
        new_structure = Structure.Structure("test_"+ str(i))
        new_structure.add(rotated_models[i])

        io = PDBIO()
        io.set_structure(new_structure)

        name = "test_" + pdb_id + '_' + str(i+1) + '.pdb'

        output_path_new = os.path.join(output_path, name)

        io.save(output_path_new)

    # temp for centered model
    # new_structure = Structure.Structure("centered")
    # new_structure.add(model)

    # io = PDBIO()
    # io.set_structure(new_structure)
    # io.save("centered_" + pdb_id + '.pdb')
    

if __name__ == "__main__":
    main()