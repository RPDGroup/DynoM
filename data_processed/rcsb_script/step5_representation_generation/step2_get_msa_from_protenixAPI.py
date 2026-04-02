
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from Bio import SeqIO
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
PROTENIX_DIR = os.path.join(BASE_DIR, "protenix")
if PROTENIX_DIR not in sys.path:
    sys.path.append(PROTENIX_DIR)
from runner.msa_search import msa_search
import argparse

def pkl_process_row(row_dict, out_dir, log_file):
    seq_set=set(row_dict['seqs'])
    protein_seqs = sorted(seq_set)
    out_dirname = os.path.join(out_dir, str(row_dict['PDB_ID']))
    os.makedirs(out_dirname, exist_ok=True)
    print(f"Processing {row_dict['PDB_ID']} in {out_dirname}")
    with open(log_file, "a") as log:
        log.write(f"Processing {row_dict['PDB_ID']} in {out_dirname}\n")
        sys.stdout = log
        sys.stderr = log

        try:
            msa_res_subdirs = msa_search(protein_seqs, out_dirname)
            log.write(f"Completed: {row_dict['PDB_ID']}\n")
        except Exception as e:
            log.write(f"Error processing {row_dict['PDB_ID']}: {e}\n")
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
    return row_dict['PDB_ID'], msa_res_subdirs

def parse_fasta(fasta_path):
    pdb_dict = defaultdict(list)

    for record in SeqIO.parse(fasta_path, "fasta"):
        pdb_id, chain = record.id.split("_")
        pdb_id = pdb_id.upper()
        pdb_dict[pdb_id].append(str(record.seq))

    paired_count = sum(1 for v in pdb_dict.values() if len(v) > 1)
    unpaired_count = sum(1 for v in pdb_dict.values() if len(v) == 1)

    sequences = [{"PDB_ID": pdb_id, "Seqs": seqs} for pdb_id, seqs in pdb_dict.items()]

    print(f"Number of paired PDB entries: {paired_count}")
    print(f"Number of unpaired PDB entries: {unpaired_count}")

    return sequences


def fasta_process_row(row_dict, out_dir, log_file):
    """
    Process a single PDB_ID and its sequences
    """
    seq_set = set(row_dict['Seqs'])
    protein_seqs = sorted(seq_set)
    out_dirname = os.path.join(out_dir, str(row_dict['PDB_ID']))
    os.makedirs(out_dirname, exist_ok=True)

    with open(log_file, "a") as log:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = log
        sys.stderr = log

        try:
            msa_res_subdirs = msa_search(protein_seqs, out_dirname)
        except Exception as e:
            log.write(f"Error processing {row_dict['PDB_ID']}: {e}\n")
            msa_res_subdirs = None
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    return row_dict['PDB_ID'], msa_res_subdirs

def get_processed_pdb_ids(out_dir):
    processed_pdb_ids = set()

    for folder_name in os.listdir(out_dir):
        folder_path = os.path.join(out_dir, folder_name)
        if os.path.isdir(folder_path):
            processed_pdb_ids.add(folder_name.upper())  
    return processed_pdb_ids

def ensure_log_file(log_file):
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    open(log_file, "w").close()


def run_parallel_tasks(task_iterable, out_dir, log_file, max_workers, worker_func):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_func, row, out_dir, log_file): row.get("PDB_ID", "UNKNOWN")
            for row in task_iterable
        }

        print("All tasks submitted.")

        for future in futures:
            pdb_id = futures[future]
            try:
                pdb_id, msa_result = future.result()
                print(f"Completed: {pdb_id}")
            except Exception as e:
                with open(log_file, "a") as log:
                    log.write(f"Error processing {pdb_id}: {e}\n")


def process_input_file(input_path, out_dir, log_file, max_workers=10):
    """
    Automatically process input based on file type:
    - fasta → fasta_process_row
    - pkl   → pkl_process_row
    """
    ensure_log_file(log_file)

    if input_path.endswith(".fasta"):
        print("FASTA input detected")
        sequence_data = parse_fasta(input_path)
        print("Total entries to process:", len(sequence_data))

        processed_pdb_ids = get_processed_pdb_ids(out_dir)
        print("Already processed:", len(processed_pdb_ids))

        sequence_data = [
            entry for entry in sequence_data
            if entry["PDB_ID"] not in processed_pdb_ids
        ]

        print("Remaining entries to process:", len(sequence_data))

        run_parallel_tasks(
            sequence_data,
            out_dir,
            log_file,
            max_workers,
            worker_func=fasta_process_row
        )

    elif input_path.endswith(".pkl"):
        print("PKL input detected")
        df = pd.read_pickle(input_path)
        print("Total number of records:", len(df))

        task_iterable = [
            row._asdict()
            for row in df.itertuples(index=False, name="Pandas")
        ]

        run_parallel_tasks(
            task_iterable,
            out_dir,
            log_file,
            max_workers,
            worker_func=pkl_process_row
        )
    else:
        raise ValueError(f"Unsupported input type: {input_path}")


def main(args):
    process_input_file(
        input_path=args.input_path,
        out_dir=args.out_dir,
        log_file=args.log_file,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MSA files using Protenix API from FASTA or PKL inputs."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file (.fasta or .pkl) containing protein sequences"
    )
    parser.add_argument(
        "--msa_out_dir",
        type=str,
        required=True,
        help="Directory to save MSA results"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to save log file"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=20,
        help="Number of parallel workers for MSA requests"
    )
    args = parser.parse_args()
    main(args)
    
    