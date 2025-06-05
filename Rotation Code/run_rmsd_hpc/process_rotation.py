import pandas as pd
import sys
import subprocess

def main():
    task_id = str(sys.argv[1])
    df = pd.read_csv('./Rotations/rotations' + task_id + '.csv')

    for idx, row in df.iterrows():
        x_rot, y_rot, z_rot = row['x_rot'], row['y_rot'], row['z_rot']

        # Ensure x, y, z are converted to strings
        subprocess.run([
            "python", "./direct_rotate_protein.py",
            "--pdb_in", "./pdb_in",
            "--pdb_id", "1A23A",
            "--pdb_out", "pdb_out",
            "--xrot_clock", str(x_rot),
            "--yrot_clock", str(y_rot),
            "--zrot_clock", str(z_rot),
            "--res_num", "33"
        ])

if __name__ == "__main__":
    main()