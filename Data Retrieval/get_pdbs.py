import os
import argparse
import pandas as pd
import requests

def download_pdb_file(pdb_id, output_dir):
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(output_dir, f'{pdb_id}.pdb'), 'w') as f:
            f.write(response.text)
        print(f'Successfully downloaded: {pdb_id}.pdb')
    else:
        print(f'Failed to download: {pdb_id}.pdb (Status code: {response.status_code})')

def main():
    parser = argparse.ArgumentParser(description='Download PDB files for unique pdb_ids in a CSV file.')
    parser.add_argument('--csv_file', dest="csv_file", type=str, help='Path to the CSV file')
    args = parser.parse_args()
    
    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(args.csv_file)
    
    if 'pdb_id' not in df.columns:
        print('Error: CSV file must contain a column named "pdb_id"')
        return
    
    # Extract unique PDB IDs
    unique_pdb_ids = df['pdb_id'].dropna().unique()
    
    # Create output directory
    csv_dir = os.path.dirname(os.path.abspath(args.csv_file))
    pdb_dir = os.path.join(csv_dir, 'pdbs')
    os.makedirs(pdb_dir, exist_ok=True)
    
    # Download each unique PDB file
    for pdb_id in unique_pdb_ids:
        download_pdb_file(pdb_id.upper()[:-1], pdb_dir)
    
    print('Download process completed.')

if __name__ == '__main__':
    main()
