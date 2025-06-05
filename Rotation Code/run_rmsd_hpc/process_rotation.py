import pandas as pd
import sys
import subprocess

def main():
    task_id = str(sys.argv[1])
    df = pd.read_csv('./Rotations/rotations' + task_id + '.csv')

    for idx, row in df.iterrows():
        x, y, z = row['x'], row['y'], row['z']

        # Ensure x, y, z are converted to strings
        subprocess.run([
            "python", "./direct_rotate_protein.py",
            "--pdb_in", "./pdb_in",
            "--pdb_id", "1A23A",
            "--pdb_out", "pdb_out",
            "--xrot_clock", str(x),
            "--yrot_clock", str(y),
            "--zrot_clock", str(z),
            "--res_num", "33"
        ])

if __name__ == "__main__":
    main()