#!/usr/bin/env python

import argparse
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
    pdb_id = '1A0F'
    pdb_in = '1A0F.pdb'
    res_num = 1
    structure = get_pdb_structure(pdb_id, pdb_in)
    model = list(structure.get_models())[0]
    n_alpha = list(list(model.get_residues())[res_num-1].get_atoms())[0]
    centre_model(model, n_alpha.coord)

    # n = 3  # number of rotations
    # rotated_models = [rotate_model(deepcopy(model), rotation) for rotation in R.random(n)]  # specify rotation
    # for i in range(len(rotated_models)):
    #     new_structure = Structure.Structure("test_"+ str(i))
    #     new_structure.add(rotated_models[i])

    #     io = PDBIO()
    #     io.set_structure(new_structure)
    #     io.save("test_" + pdb_id + '_' + str(i+1) + '.pdb')

    # temp for centered model
    new_structure = Structure.Structure("centered")
    new_structure.add(model)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save("centered_" + pdb_id + '.pdb')
    

if __name__ == "__main__":
    main()