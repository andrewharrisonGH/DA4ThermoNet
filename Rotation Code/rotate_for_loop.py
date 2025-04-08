import subprocess

for x_rot in range(0,360,5):
    for y_rot in range(0,360,5):
        #for z_rot in range(0,360,5):
        subprocess.run(["python", ".\direct_rotate_protein.py", "--pdb_in", 
                            r"C:\Users\Andrew\Documents\University\UniMelb\Masters Project\DA4ThermoNet\Training Data\PDBs", "--pdb_id", 
                            "1A23A", "--pdb_out", "pdb_out", "--xrot_clock", str(x_rot), "--yrot_clock", str(y_rot), "--zrot_clock", 
                            "0", "--res_num", "33"])

    