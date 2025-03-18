#!/usr/bin/env python

import argparse
import numpy as np
from numpy import array, dot
from math import pi
from math import radians
#from Bio.PDB import *
from Bio.PDB import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBIO import PDBIO
# SPECIFY STRUCTURE
#
# CREATE MODEL
#
# ROTATION SPECIFIED EXTENT AROUND THE X-AXIS - EXPECT 0 <= THETA <= 360

# CONVERT TO RADIANS

# CALCULATE MATRIX FOR ROTATION ABOUT X-AXIS

# CALCULATE MATRIX FOR ROTATION ABOUT Y-AXIS

# CALCULATE THE DUAL-ROTATION MATRIX

# APPLY THIS DUAL-ROTATION TO THE MODEL

#pdb_id = "1TUP"
#cif_file = "pdbs/1tup.cif"
#chain = "A"
#res_num = 156
#res_num = 190

X_90_CLOCK = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0], dtype=np.float64).T
X_180_CLOCK = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1], dtype=np.float64).T
X_270_CLOCK = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0], dtype=np.float64).T

Y_90_CLOCK = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0], dtype=np.float64).T
Y_180_CLOCK = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1], dtype=np.float64).T
Y_270_CLOCK = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0], dtype=np.float64).T

def get_cif_structure(pdb_id, cif_file):
    parser = MMCIFParser(QUIET = True)
    data = parser.get_structure(pdb_id, cif_file)
    return data

def get_pdb_structure(pdb_id, pdb_file):
    parser = PDBParser(QUIET = True)
    data = parser.get_structure(pdb_id, pdb_file)
    return data


# code example from https://www.programcreek.com/python/?CodeExample=get+rotation+matrix
#def get_rotation_matrix(axis, theta):
# 
#    Find the rotation matrix associated with counterclockwise rotation
#    about the given axis by theta radians.
#    Credit: http://stackoverflow.com/users/190597/unutbu
#
#    Args:
#        axis (list): rotation axis of the form [x, y, z]
#        theta (float): rotational angle in radians
#
#    Returns:
#        array. Rotation matrix.
#    """
#
#    axis = np.asarray(axis)
#    theta = np.asarray(theta)
#    axis = axis/math.sqrt(np.dot(axis, axis))
#    a = math.cos(theta/2.0)
#    b, c, d = -axis*math.sin(theta/2.0)
#    aa, bb, cc, dd = a*a, b*b, c*c, d*d
#    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
#    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
#                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
#                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]]) 

#def centre_model(model, centre_coord):
##    model_copy = copy.deepcopy(model)        # make a deep copy
#    for chain in model.get_chains():
#        for residue in chain.get_residues():
#            for atom in residue.get_atoms():
#                newCoord = atom.coord - centre_coord
#                atom.coord = newCoord

def rotate_model(model, ROTATE_MATRIX):
    for chain in model.get_chains():
        for residue in chain.get_residues():
            for atom in residue.get_atoms():
                atom_coord = np.array(atom.coord, dtype=np.float64)
                newCoord = np.dot(ROTATE_MATRIX, atom_coord)    # multiply coord vector by matrix A - Ax
                atom.coord = newCoord

def get_x_axis_rotate_matrix(theta):    # 0 <= theta <= 2pi
    if 0 <= theta <= np.pi/2:
        X_ROT = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)], dtype=np.float64).T
        return X_ROT
    elif np.pi/2 < theta <= np.pi:
        alpha = theta - np.pi/2   
        X_ROT = np.dot(X_90_CLOCK, np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)], dtype=np.float64).T)
        return X_ROT
    elif np.pi < theta <= 1.5*np.pi:
        alpha = theta - np.pi   
        X_ROT = np.dot(X_180_CLOCK, np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)], dtype=np.float64).T)
        return X_ROT
    elif 1.5*np.pi < theta <= 2*np.pi:
        alpha = theta - 1.5*np.pi   
        X_ROT = np.dot(X_270_CLOCK, np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)], dtype=np.float64).T)
        return X_ROT
    else:
        sys.exit("Specified rotation angle about the x-axis should be in range 0 to 360 degrees")

def get_y_axis_rotate_matrix(theta):    # 0 <= theta <= 2pi
    if 0 <= theta <= np.pi/2:
        Y_ROT = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)], dtype=np.float64).T
        return Y_ROT
    elif np.pi/2 < theta <= np.pi:
        alpha = theta - np.pi/2   
        Y_ROT = np.dot(Y_90_CLOCK, np.array([[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [np.sin(alpha), 0, np.cos(alpha)], dtype=np.float64).T)
        return Y_ROT
    elif np.pi < theta <= 1.5*np.pi:
        alpha = theta - np.pi   
        Y_ROT = np.dot(Y_180_CLOCK, np.array([[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [np.sin(alpha), 0, np.cos(alpha)], dtype=np.float64).T)
        return Y_ROT
    elif 1.5*np.pi < theta <= 2*np.pi:
        alpha = theta - 1.5*np.pi   
        Y_ROT = np.dot(Y_270_CLOCK, np.array([[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [np.sin(alpha), 0, np.cos(alpha)], dtype=np.float64).T)
        return Y_ROT
    else:
        sys.exit("Specified rotation angle about the y-axis should be in range 0 to 360 degrees")




def get_x_y_rotate_matrix(x_rot_matrix, y_rot_matrix):
    x_y_rotate_matrix = np.dot(x_rot_matrix, y_rot_matrix)
    return x_y_rotate_matrix
 
#def get_rotation_matrix_z(vector):                   # for use to rotate N->CA vector to z axis - expects vector as numpy array dtype float64
#    unit_vector = vector / np.linalg.norm(vector)   # where the norm of the vector is its length
#    unit_z = np.array([0, 0, 1], dtype=np.float64)                     # define z-axis unit vector   
#    cross = np.cross(unit_vector, unit_z)
#    unit_cross = cross / np.linalg.norm(cross)
#    B_MINUS_1 = np.column_stack((unit_cross, unit_vector, unit_z))
#    B = np.linalg.inv(B_MINUS_1)
#
#
#    theta = np.arccos(np.dot(unit_vector, unit_z))
#    print("theta:\n", theta)
#
## for the condition that theta is obtuse
#    if (theta > (np.pi/2)):
#        C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, (-2*(np.cos(np.pi - theta)))]], dtype=np.float64).T
#    else:
#        C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, (2*(np.cos(theta)))]], dtype=np.float64).T
#
#    A = np.dot(B_MINUS_1, np.dot(C, B))
#
#    rotated_vector = np.dot(A, vector)
#    return A

#def get_rotation_matrix_x(vector):          # for use to rotate CA->C to x axis - expects numpy vector of CA->C post N->CA rotation to z axis
#    xy_vector = vector[0:2]
#    unit_xy = xy_vector / np.linalg.norm(xy_vector)
#    unit_x_2d = np.array([1, 0], dtype=np.float64)
#    theta = np.arccos(np.dot(unit_xy, unit_x_2d))
#    print("theta", theta)
#    if xy_vector[1] < 0:
#        print("y is less than zero - adjusting theta")
#        theta = 2*(np.pi) - theta
#        print("theta", theta)
#    else:
#        print("y is greater than zero - no need to adjust theta")
#        print("theta", theta)
#    X = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float64).T
#    rotated_vector = np.dot(X, vector)
#    return X

#def atom_dist_test_print(model, chain, res_num):
#    centre_n = model[chain][res_num]["N"].coord
#    centre_ca = model[chain][res_num]["CA"].coord
#    centre_c = model[chain][res_num]["C"].coord
#    n_ca_vector = centre_ca - centre_n
#    ca_c_vector = centre_c - centre_ca
#    n_ca_dist = np.linalg.norm(n_ca_vector)
#    ca_c_dist = np.linalg.norm(ca_c_vector)
##    test_num_list = [110, 120, 130, 140, 150, 160]
#    test_num_list = [110, 120]
#    print("Centre residue N coord:", centre_n)
#    print("Centre residue CA coord:", centre_ca)
#    print("Centre residue C coord:", centre_c)
#    print("Centre residue N->CA vector:", n_ca_vector)
#    print("Centre residue N->CA distance:", n_ca_dist)
#    print("Centre residue CA->C vector:", ca_c_vector)
#    print("Centre residue CA->C distance:", ca_c_dist)
#    for test_num in test_num_list:
#        dist = np.linalg.norm(model[chain][test_num]["CA"].coord - centre_n)
#        print("Distance from centre residue {} N to CA of residue {}:".format(str(res_num), str(test_num)), str(dist))
#        print("Coord of CA of residue {}:".format(str(test_num)), model[chain][test_num]["CA"].coord)


def main():
    parser = argparse.ArgumentParser(description="centre and orient a pdb file")
    parser.add_argument("--pdb_in", dest="pdb_in", type=str, required=True, help="path to input PDB file")
    parser.add_argument("--pdb_out", dest="pdb_out", type=str, required=True, help="path to output PDB file")
    parser.add_argument("--xrot_clock", dest="xrot_clock", type=float, required=True, help="clockwise rotation angle about the x-axis - 0 to 360 degrees")
    parser.add_argument("--yrot_clock", dest="yrot_clock", type=float, required=True, help="clockwise rotation angle about the y-axis - 0 to 360 degrees")
#    parser.add_argument("--chain", dest="chain", type=str, required=True, help="model chain")
#    parser.add_argument("--res_num", dest="res_num", type=int, required=True, help="residue number to centre on")
    args = parser.parse_args()

#    chain = args.chain
#    res_num = args.res_num
    pdb_in = args.pdb_in  # path to pdb in file
    pdb_id = pdb_in.split("_")[0]
#    pdb_out = pdb_in.split(".")[0] + "_" + chain + "_" + str(res_num) + "_centred.pdb" # path to pdb out file 
    pdb_out = args.pdb_out
    xrot_clock_degrees = args.xrot_clock
    yrot_clock_degrees = args.yrot_clock

    xrot_clock_radians = radians(xrot_clock_degrees)
    yrot_clock_radians = radians(yrot_clock_degrees)

#    structure = get_structure_object(pdb_id, cif_file)
    structure = get_pdb_structure(pdb_id, pdb_in)

    model = list(structure.get_models())[0]
#    print("\nINITIAL MODEL")
#    atom_dist_test_print(model, chain, res_num)
#    print("\nCENTRE MODEL ON MAIN CHAIN NITROGEN")
#    centre_model(model, model[chain][res_num]["N"].coord)
#    print("centre residue n coord after centreing\n", model[chain][res_num]["N"].coord)
#    n_ca_vector = np.array(model[chain][res_num]["CA"].coord, dtype=np.float64)
#    print("n_ca_vector pre-z-rotation", n_ca_vector.dtype, n_ca_vector)
#    Z_MATRIX = get_rotation_matrix_z(n_ca_vector)
#    print("\nROTATE MODEL N->CA TO Z-AXIS")
#    rotate_model(model, Z_MATRIX)
#    centre_n = model[chain][res_num]["N"].coord
#    centre_ca = model[chain][res_num]["CA"].coord
#    print("\nROTATE MODEL CA->C TO X-AXIS")
#    ca_c_vector_post_zrotation = model[chain][res_num]["C"].coord - model[chain][res_num]["CA"].coord    
#    X_MATRIX = get_rotation_matrix_x(ca_c_vector_post_zrotation)
#    rotate_model(model, X_MATRIX)
#    atom_dist_test_print(model, chain, res_num)


    X_ROT = get_x_axis_rotate_matrix(xrot_clock_radians)
    Y_ROT = get_y_axis_rotate_matrix(yrot_clock_radians)
    X_Y_ROT = np.dot(X_ROT, Y_ROT)
    rotate_model(model, X_Y_ROT)

    
#    io = MMCIFIO()
    io = PDBIO()
    io.set_structure(structure)
#    io.save("test_" + pdb_id.lower() + "_centre_ncaz_cacx_" + chain + str(res_num)  + "_cif")
    io.save(pdb_out)

if __name__ == "__main__":
    main()
