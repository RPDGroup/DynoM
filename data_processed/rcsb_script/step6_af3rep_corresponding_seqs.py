import pandas as pd
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert seqs to tuple keys for representation lookup."
    )
    parser.add_argument("--input_pkl", type=str, required=True,
                        help="Path to input pkl file (must contain 'PDB_ID' and 'seqs')")
    parser.add_argument("--output_pkl", type=str, required=True,
                        help="Path to save output pkl file")
    return parser.parse_args()


def convert_seqs_to_key(input_pkl: str, output_pkl: str):
    if not os.path.exists(input_pkl):
        raise FileNotFoundError(f"Input file not found: {input_pkl}")
    df = pd.read_pickle(input_pkl)
    required_cols = ['PDB_ID', 'seqs']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df['seqs_key'] = df['seqs'].apply(lambda x: "_".join(x))
    df_new = df[['PDB_ID', 'seqs_key']]
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    df_new.to_pickle(output_pkl)
    print(f"Saved successfully: {output_pkl}")


def main():
    args = parse_args()
    convert_seqs_to_key(
        input_pkl=args.input_pkl,
        output_pkl=args.output_pkl
    )

if __name__ == "__main__":
    main()