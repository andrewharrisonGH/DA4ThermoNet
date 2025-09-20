#!/usr/bin/env python3
import argparse
import pandas as pd

def extract_ddG(input_csv, output_txt, reverse=False, repeat=1):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Check if ddG column exists
    if "ddg" not in df.columns:
        raise ValueError("The input CSV does not contain a 'ddg' column.")

    # Extract ddG column
    ddg_values = (((-1)**reverse) * df["ddg"]).repeat(repeat)

    # Save to txt file (one value per line)
    ddg_values.to_csv(output_txt, index=False, header=False)

def main():
    parser = argparse.ArgumentParser(description="Extract ddG column from CSV to TXT.")
    parser.add_argument("--input", dest="mutations", help="Input CSV file")
    parser.add_argument("--output", dest="output", help="Output TXT file")
    parser.add_argument("-r", "--reverse", dest="reverse", action='store_true', 
                        help="Output ddG as reverse mutations.")
    parser.add_argument("--repeat", dest="repeat", type=int, default=1, help="Number of times to repeat each value (default: 1)")

    args = parser.parse_args()

    extract_ddG(args.mutations, args.output, args.reverse, args.repeat)

if __name__ == "__main__":
    main()