import os
import argparse
import pandas as pd
from Bio.PDB import PDBParser

def load_structure(pdb_id, pdb_dir):
    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.isfile(pdb_file):
        print(f"Missing PDB file: {pdb_file}")
        return None
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_file)
        return structure
    except Exception as e:
        print(f"Failed to parse {pdb_file}: {e}")
        return None

def get_residue(structure, chain_id, pos):
    for model in structure:
        chain = model[chain_id] if chain_id in model else None
        if chain:
            for res in chain:
                if res.id[1] == pos:
                    return res
    return None

def main():
    parser = argparse.ArgumentParser(description='Sanity check variants against PDB files.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file with variants')
    args = parser.parse_args()

    # Load dataframe
    df = pd.read_csv(args.csv_file)

    # Check required columns
    required_cols = {'pdb_id', 'pos', 'wild_type'} #Training Data
    #required_cols = {'pdb_id', 'mut_info'} #Testing Data
    if not required_cols.issubset(df.columns):
        print(f"CSV file must contain the columns: {required_cols}")
        return

    # Setup paths
    csv_dir = os.path.dirname(os.path.abspath(args.csv_file))
    pdb_dir = os.path.join(csv_dir, 'pdbs')

    mismatches = []

    for i, row in df.iterrows():
        pdb_id = str(row['pdb_id']).upper()[:-1]
        chain_id = str(row['pdb_id']).upper()[-1]
        pos = int(row['pos']) # Training Data
        #pos = int(str(row['mut_info']).upper()[1:-1])
        wild_type = str(row['wild_type']).upper() # Training Data
        #wild_type = str(row['mut_info']).upper()[0]
        
        structure = load_structure(pdb_id, pdb_dir)
        if not structure:
            continue

        residue = get_residue(structure, chain_id, pos)
        if not residue:
            print(f"Residue not found: {pdb_id} chain {chain_id} position {pos}")
            mismatches.append((pdb_id, chain_id, pos, 'Residue not found'))
            continue

        res_name = residue.get_resname()
        res_1letter = three_to_one(res_name)

        if res_1letter != wild_type:
            print(f"Mismatch at {pdb_id} {chain_id}{pos}: expected {wild_type}, found {res_1letter}")
            mismatches.append((pdb_id, chain_id, pos, f"Expected {wild_type}, found {res_1letter}"))

    print(f"\nSanity check complete. {len(mismatches)} issue(s) found.")
    if mismatches:
        print("Details:")
        for m in mismatches:
            print(m)

def three_to_one(resname):
    from Bio.Data.IUPACData import protein_letters_3to1
    return protein_letters_3to1.get(resname.capitalize(), '?')

if __name__ == '__main__':
    main()
