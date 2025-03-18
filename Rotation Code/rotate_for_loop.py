import subprocess

for x_rot in range(5,360,5):
    subprocess.run(["python", "direct_rotate_protein.py", "--pdb_in", "pdb_in/1A0F.pdb", "--pdb_out", "pdb_out", 
                    "--xrot_clock", "0", "--yrot_clock", str(x_rot), "--zrot_clock", "0", "--res_num", "8"])

    