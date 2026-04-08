import os
import json
import pandas as pd
from Bio import SeqIO
from collections import defaultdict, Counter
import argparse

def process_fasta_and_match(input_msa_dir, target_sequences):
    """
    Read FASTA files from the MSA directory and match target sequences.

    Parameters:
    - input_msa_dir: Path to the MSA folder corresponding to a PDB entry
    - target_sequences: List of target sequences to match

    Returns:
    - seqs_al_msa_path: Mapping between target sequences and their corresponding MSA directories
    """
    fasta_files = [os.path.join(input_msa_dir, file) for file in os.listdir(input_msa_dir) if file.endswith(".fasta")]
    if not fasta_files:
        return {}
    
    fasta_path = fasta_files[0]
    with open(fasta_path, "r") as infile:
        lines = infile.readlines()
    
    seqs_al_msa_path = {}
    for i in range(len(lines)):
        line = lines[i].strip()
        
        if line.startswith(">") and i + 1 < len(lines):
            sequence = lines[i + 1].strip()
            
            if sequence in target_sequences:
                msa_dir = os.path.join(input_msa_dir, line.split("_")[1])
                seqs_al_msa_path[sequence] = msa_dir
    
    return seqs_al_msa_path

def generate_pkl_msa_input(pkl_filepath, output_dir, precomputed_msa_dirs, pairing_db):
    df = pd.read_pickle(pkl_filepath)
    error_pdbid = []
    
    for _, row in df.iterrows():
        seq_counts = {}
        for seq in row['seqs']:
            seq_counts[seq] = seq_counts.get(seq, 0) + 1
        input_msa_dir = os.path.join(precomputed_msa_dirs, row['PDB_ID'])
        print(f"Processing MSA directory: {input_msa_dir}")
        if not os.path.exists(input_msa_dir):
            error_pdbid.append(row['PDB_ID'])
            print(f"Error: {row['PDB_ID']} - MSA directory not found: {input_msa_dir}")
            continue
        
        try:
            seqs_al_msa_path = process_fasta_and_match(input_msa_dir, list(seq_counts.keys()))
            if not seqs_al_msa_path:
                error_pdbid.append(row['PDB_ID'])
                continue
        except Exception as e:
            error_pdbid.append(row['PDB_ID'])
            print(f"Error: {row['PDB_ID']} - Sequence matching error: {e}")
            continue
        
        sequences = []
        for seq, count in seq_counts.items():
            try:
                msapath = seqs_al_msa_path[seq]
            except KeyError:
                print(f"Warning: {row['PDB_ID']} - No MSA path found for sequence.")
                continue
            
            sequence = {
                "proteinChain": {
                    "sequence": seq,
                    "count": count,
                    "msa": {
                        "precomputed_msa_dir": msapath,
                        "pairing_db": pairing_db
                    }
                }
            }
            sequences.append(sequence)
        
        json_data = {
            "sequences": sequences,
            "name": row['PDB_ID']
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        json_filename = f"{row['PDB_ID']}_msa_results.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        
        print(f"Saved MSA results for {row['PDB_ID']} to {json_filepath}")
    
    return error_pdbid

def generate_fasta_msa_input_complex(fasta_path,precomputed_msa_dirs,output_dir,pairing_db,need_process_PDBID=None):
    pdb_dict = defaultdict(list)
    for record in SeqIO.parse(fasta_path, "fasta"):
        pdb_id = record.id.split("_")[0].upper()
        if need_process_PDBID is not None and pdb_id not in need_process_PDBID:
            continue
        pdb_dict[pdb_id].append(str(record.seq))
    error_pdbid=[]
    sequences = []

    for pdb_id, seqs in pdb_dict.items():
        seq_counter = Counter(seqs)
        try:
            input_msa_dir = os.path.join(precomputed_msa_dirs, pdb_id)
            seqs_al_msa_path = process_fasta_and_match(input_msa_dir, list(seq_counter.keys()))

            if not seqs_al_msa_path:
                error_pdbid.append(pdb_id)
                continue
        except Exception as e:
            error_pdbid.append(pdb_id)
            print(f"Error: {pdb_id} - Sequence matching error: {e}")
            continue
        sequences = []
        for seq, count in seq_counter.items():
            try:
                msapath = seqs_al_msa_path[seq]
            except KeyError:
                print(f"Warning: {pdb_id} - No MSA path found for sequence.")
                continue
            
            sequence = {
                "proteinChain": {
                    "sequence": seq,
                    "count": count,
                    "msa": {
                        "precomputed_msa_dir": msapath,
                        "pairing_db": pairing_db
                    }
                }
            }
            sequences.append(sequence)

        json_data = {
            "sequences": sequences,
            "name": pdb_id
        }

        os.makedirs(output_dir, exist_ok=True)

        json_filename = f"{pdb_id}_msa_results.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        
        print(f"Saved MSA results for {pdb_id} to {json_filepath}")
    return error_pdbid

def merge_json_files(folder_path, output_file):
    """
    Read all JSON files in the specified folder and merge them into a single list.

    Parameters:
    - folder_path: Directory containing JSON files
    - output_file: Path to save the merged JSON file
    """
    combined_list = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                combined_list.append(data)
    
    with open(output_file, "w") as json_file:
        json.dump(combined_list, json_file, indent=4)
    
    print(f"Merged JSON file has been saved to {output_file}")



def parse_args():
    parser = argparse.ArgumentParser(description="Generate MSA JSON inputs and merge them.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input file path (.pkl or .fasta)")
    parser.add_argument("--one_json_output_dir", type=str, required=True,
                        help="Directory to store intermediate JSON files")
    parser.add_argument("--msa_results_dir", type=str, required=True,
                        help="Directory of precomputed MSA files")
    parser.add_argument("--merge_json_output_dir", type=str, required=True,
                        help="Directory to store merged JSON file")
    parser.add_argument("--merge_json_output_name", type=str, default="merged",
                        help="Final merged JSON file name,No need for suffix")
    parser.add_argument("--pairing_db", type=str, default="uniref100",
                        help="Pairing database (default: uniref100)")
    parser.add_argument("--need_process_PDBID_path", type=str, default=None,
                        help="Optional file containing PDB IDs to process")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.one_json_output_dir, exist_ok=True)
    os.makedirs(args.merge_json_output_dir, exist_ok=True)
    need_process_PDBID = None
    if args.need_process_PDBID_path:
        with open(args.need_process_PDBID_path, "r") as f:
            need_process_PDBID = [
                line.strip().upper() for line in f if line.strip()
            ]
    if args.input_file.endswith(".pkl"):
        error_pdbid = generate_pkl_msa_input(
            args.input_file,
            args.one_json_output_dir,
            args.msa_results_dir,
            args.pairing_db
        )
    elif args.input_file.endswith(".fasta"):
        error_pdbid = generate_fasta_msa_input_complex(
            args.input_file,
            args.msa_results_dir,
            args.one_json_output_dir,
            args.pairing_db,
            need_process_PDBID=need_process_PDBID
        )
    else:
        raise ValueError("Input file must be .pkl or .fasta")
    print("Errors encountered with PDB IDs:", error_pdbid)

    merge_json_path = os.path.join(
        args.merge_json_output_dir,
        args.merge_json_output_name + ".json"
    )

    merge_json_files(
        args.one_json_output_dir,
        merge_json_path
    )

    print(f"Merged JSON saved to: {merge_json_path}")


if __name__ == "__main__":
    main()