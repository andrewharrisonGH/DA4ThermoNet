import pandas as pd
import os
import sys
import subprocess
from pathlib import Path


def main():
    pdb_root = sys.argv[1]
    base_path = Path(f"./andrewh/small_test_1/{pdb_root}")
    mutations_csv = Path(f"./andrewh/small_test_1/{pdb_root}_mutants.csv")
    rotations_csv = Path("./andrewh/small_test_1/rotations.csv")
    wt_pdb = base_path / f"{pdb_root}_relaxed.pdb"
    aug_folder = base_path / "Augmented"

    # Load CSVs
    mutations_df = pd.read_csv(mutations_csv)
    rotations_df = pd.read_csv(rotations_csv)

    os.makedirs(aug_folder, exist_ok=True)

    # Track completed WT rotations
    done_wt_rotations = set()

    for _, mut_row in mutations_df.iterrows():
        pos = mut_row['pos']
        chain = mut_row['pdb_id'][-1]
        wt = mut_row['wild_type']
        mt = mut_row['mutant']
        mut_code = f"{wt}{pos}{mt}"
        mutant_pdb = base_path / f"{pdb_root}_relaxed_{mut_code}_relaxed.pdb"

        for _, rot_row in rotations_df.iterrows():
            xrot, yrot, zrot = int(rot_row['x_rot']), int(rot_row['y_rot']), int(rot_row['z_rot'])

            # Output file names
            wt_out_name = f"{pdb_root}_wt_{pos}_{xrot}_{yrot}_{zrot}.pdb"
            mt_out_name = f"{pdb_root}_mt_{mut_code}_{xrot}_{yrot}_{zrot}.pdb"
            wt_out_path = aug_folder / wt_out_name
            mt_out_path = aug_folder / mt_out_name

            # Rotate WT only if not done
            wt_key = (pos, xrot, yrot, zrot)
            if wt_key not in done_wt_rotations:
                subprocess.run([
                    "python", "./andrewh/small_test_1/direct_rotate_protein.py",
                    "--pdb_in", str(wt_pdb),
                    "--pdb_id", f"{pdb_root}{chain}",
                    "--pdb_out", str(aug_folder),
                    "--xrot_clock", str(xrot),
                    "--yrot_clock", str(yrot),
                    "--zrot_clock", str(zrot),
                    "--res_num", str(pos),
                    "--out_name", str(wt_out_path)
                ])
                done_wt_rotations.add(wt_key)

            # Rotate Mutant
            subprocess.run([
                    "python", "./andrewh/small_test_1/direct_rotate_protein.py",
                    "--pdb_in", str(mutant_pdb),
                    "--pdb_id", f"{pdb_root}{chain}",
                    "--pdb_out", str(aug_folder),
                    "--xrot_clock", str(xrot),
                    "--yrot_clock", str(yrot),
                    "--zrot_clock", str(zrot),
                    "--res_num", str(pos),
                    "--out_name", str(mt_out_path)
                ])

if __name__ == "__main__":
    main()