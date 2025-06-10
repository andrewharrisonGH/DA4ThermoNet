import pandas as pd
from pathlib import Path
import os

def split_mutation_csvs(input_csv, output_dir):
    input_dir = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    df = pd.read_csv(input_dir)
    print(f"Processing {input_dir.name}")

    if "pdb_id" not in df.columns:
        print(f"Skipping {input_dir.name} (no 'pdb_id' column)")
        return None

    for pdb_id_chain, group_df in df.groupby("pdb_id"):
        pdb_root = pdb_id_chain[:-1].upper()  # strip chain letter
        out_path = output_dir / f"{pdb_root}_mutants.csv"
        if pdb_root in ['1A23', '1A43', '1AAR']:
            group_df.to_csv(out_path, index=False)
            print(f"  -> Wrote {len(group_df)} rows to {out_path.name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split mutation CSVs into per-protein files.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV file containing mutations")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder for per-protein CSVs")
    args = parser.parse_args()

    split_mutation_csvs(args.input_csv, args.output_dir)