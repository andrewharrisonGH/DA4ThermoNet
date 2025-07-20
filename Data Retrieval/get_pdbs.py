import os
import argparse
import pandas as pd
import requests
import subprocess

def download_and_extract_chain(pdb_id_full, output_dir):
    pdb_code = pdb_id_full[:4].lower()
    chain_id = pdb_id_full[4].upper()
    output_filename = f"{pdb_code.upper()}.pdb"  # Ensure uppercase filename

    url = f'https://files.rcsb.org/download/{pdb_code}.pdb'
    response = requests.get(url)

    if response.status_code == 200:
        raw_pdb_path = os.path.join(output_dir, f'{pdb_code}_raw.pdb')
        output_pdb_path = os.path.join(output_dir, output_filename)

        # Save the downloaded raw PDB
        with open(raw_pdb_path, 'w') as f:
            f.write(response.text)

        # Use pdb_selchain to extract only the specified chain
        try:
            with open(output_pdb_path, 'w') as out_f:
                subprocess.run(
                    ['pdb_selchain', f'-{chain_id}', raw_pdb_path],
                    check=True,
                    stdout=out_f
                )
            print(f'Successfully wrote: {output_filename} (chain {chain_id})')
        except subprocess.CalledProcessError:
            print(f'Error: Failed to extract chain {chain_id} from {pdb_code}')
        
        os.remove(raw_pdb_path)
    else:
        print(f'Failed to download {pdb_code}.pdb (Status code: {response.status_code})')

def main():
    parser = argparse.ArgumentParser(description='Download PDBs and extract chains named in pdb_id column (e.g., 1abcA).')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the input CSV file with a "pdb_id" column.')
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_file)

    if 'pdb_id' not in df.columns:
        print('Error: CSV file must contain a column named "pdb_id" (e.g. 1abcA)')
        return
    
    unique_pdb_ids = df['pdb_id'].dropna().unique()

    # Set up output directory
    csv_dir = os.path.dirname(os.path.abspath(args.csv_file))
    pdb_dir = os.path.join(csv_dir, 'pdbs')
    os.makedirs(pdb_dir, exist_ok=True)

    for pdb_id_full in unique_pdb_ids:
        if len(pdb_id_full) != 5:
            print(f'Skipping malformed pdb_id: {pdb_id_full}')
            continue
        download_and_extract_chain(pdb_id_full, pdb_dir)
    
    print('All PDB files processed.')

if __name__ == '__main__':
    main()
