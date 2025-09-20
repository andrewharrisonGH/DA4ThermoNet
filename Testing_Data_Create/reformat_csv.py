#!/usr/bin/env python3
import csv
import argparse

def reformat_csv(input_file: str, output_file: str) -> None:
    """
    Reads a CSV with columns: ddg, pdb_id, mut_info
    Example row: "3.2,1A23A,C23L"
    Parses mut_info into wild_type (C), position (23), mutant (L).
    Writes a new CSV with columns: pdb_id,pos,wild_type,mutant,ddg
    """
    with open(input_file, "r", newline="") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ["pdb_id", "pos", "wild_type", "mutant", "ddg"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            mut_info = row["mut_info"].strip()
            wild_type = mut_info[0]              # first character
            mutant = mut_info[-1]                # last character
            pos = mut_info[1:-1]                 # everything in between

            writer.writerow({
                "pdb_id": row["pdb_id"].strip(),
                "pos": pos,
                "wild_type": wild_type,
                "mutant": mutant,
                "ddg": row["ddg"].strip()
            })

def main():
    parser = argparse.ArgumentParser(description="Reformat ddg CSV file")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("output_csv", help="Path to output CSV file")
    args = parser.parse_args()

    reformat_csv(args.input_csv, args.output_csv)

if __name__ == "__main__":
    main()