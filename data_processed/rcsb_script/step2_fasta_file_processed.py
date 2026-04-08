import re
import os
from Bio import SeqIO
import pandas as pd
from pathlib import Path
import sys
import math
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import defaultdict
import argparse
import pandas as pd


def extract_chains(chain_str):
    chains = []
    
    auth_matches = re.findall(r'([\w\d]+)\[auth ([\w\d]+)\]', chain_str)
    auth_map = {orig: auth for orig, auth in auth_matches}
    
    all_chains = re.findall(r'Chains? ([\w\d,\[\] auth]+)', chain_str)
    
    for match in all_chains:
        chain_list = re.split(r',\s*', match)
        for chain in chain_list:
            chain_clean = re.sub(r'\[auth ([\w\d]+)\]', '', chain).strip()
            auth_match = re.search(r'\[auth ([\w\d]+)\]', chain)

            chains.append(str(auth_map.get(chain_clean, chain_clean)))

    return list(set(chains)) if chains else []


def parse_fasta_statistics_to_complex(fasta_path, one_maxlen=650, one_minlen=20):
    sequences_arr = []
    chain_lengths_arr = []
    entities_to_chains = []

    for idx, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        chain_info = record.description.split("|")
        pdb_id = chain_info[0].split("_")[0].replace(" ", "").replace("\t", "").replace("\n", "")

        if len(chain_info[0].split("_")) > 1:
            entity_id = chain_info[0].split("_")[1].replace(" ", "").replace("\t", "").replace("\n", "")
        else:
            entity_id = idx

        if len(chain_info) >= 2:
            chain_ids = extract_chains(chain_info[1])
        else:
            chain_ids = ['U']

        sequence = str(record.seq).replace(" ", "").replace("\t", "").replace("\n", "")
        chain_length = len(sequence)

        entity_to_chain = {}
        entity_to_chain[entity_id] = chain_ids

        if len(chain_ids) > 1:
            sequences_arr.extend([sequence] * len(chain_ids))
            chain_lengths_arr.extend([chain_length] * len(chain_ids))
        else:
            sequences_arr.append(sequence)
            chain_lengths_arr.append(chain_length)

        entities_to_chains.append(entity_to_chain)

    total_aa_length = sum(chain_lengths_arr)

    if total_aa_length == 0:
        return None
    else:
        return {
            "pdb_id": pdb_id,
            "entities_to_chains": entities_to_chains,
            "sequences": sequences_arr,
            "chain_lengths": chain_lengths_arr,
            "total_aa_length": total_aa_length,
        }
    
def parse_complex_fasta(fasta_path, one_maxlen=650, one_minlen=20):
    complex_dict = defaultdict(list)
    chain_meta_dict = defaultdict(list)

    for record in SeqIO.parse(fasta_path, "fasta"):
        if "_" not in record.id:
            continue

        complex_id = record.id.rsplit("_", 1)[0]
        description_parts = record.description.split("|")

        entity_to_chain = {}
        chain_ids = []
        if len(description_parts) >= 2:
            entity_id = description_parts[0].split("_")[1] if "_" in description_parts[0] else "0"
            chain_ids = extract_chains(description_parts[1])
            if chain_ids:
                entity_to_chain[entity_id] = chain_ids

        sequence = str(record.seq).replace("X", "")
        if not (one_minlen <= len(sequence) <= one_maxlen):
            continue

        complex_dict[complex_id].append({
            "sequence": sequence,
            "length": len(sequence),
            "entity_to_chain": entity_to_chain if chain_ids else None
        })

    parsed_complexes = []
    paired_count = 0
    unpaired_count = 0

    for complex_id, seq_infos in complex_dict.items():
        if len(seq_infos) > 1:
            paired_count += 1
        else:
            unpaired_count += 1

        sequences = [x["sequence"] for x in seq_infos]
        chain_lengths = [x["length"] for x in seq_infos]
        entities_to_chains = [x["entity_to_chain"] for x in seq_infos if x["entity_to_chain"]]

        parsed_complexes.append({
            "pdb_id": complex_id,
            "sequences": sequences,
            "chain_lengths": chain_lengths,
            "total_aa_length": sum(chain_lengths),
            "entities_to_chains": entities_to_chains
        })

    print(f"Number of paired complexes: {paired_count}")
    print(f"Number of unpaired complexes: {unpaired_count}")

    return parsed_complexes

def extract_chain_ids(entities_to_chains):
    chain_ids = []
    if entities_to_chains is None:
        return chain_ids

    if isinstance(entities_to_chains, dict):
        for k, v in entities_to_chains.items():
            if isinstance(v, list):
                chain_ids.extend(v)
    elif isinstance(entities_to_chains, list):
        for item in entities_to_chains:
            if isinstance(item, dict):
                for k, v in item.items():
                    if isinstance(v, list):
                        chain_ids.extend(v)
    return list(dict.fromkeys(chain_ids))


def save_to_pkl_complex(fasta_path, data_input, one_pdb_output_pkl_dir):
    os.makedirs(one_pdb_output_pkl_dir, exist_ok=True)

    if isinstance(data_input, dict):
        data_list = [data_input]
    elif isinstance(data_input, list):
        data_list = data_input
    else:
        raise TypeError(f"Invalid data_input type: {type(data_input)}, expected dict or list[dict]")
    for data in data_list:
        pdb_id = str(data.get('pdb_id', 'unknown')).upper()
        output_pkl_path = os.path.join(one_pdb_output_pkl_dir, f"{pdb_id}.pkl")

        if not all(k in data for k in ['sequences', 'chain_lengths', 'total_aa_length', 'entities_to_chains']):
            print(f"[Warning] Missing required fields, skipping {pdb_id}")
            continue
        df = pd.DataFrame([{
            'PDB_ID': pdb_id,
            'model': 0,
            'chain_ids': extract_chain_ids(data['entities_to_chains']),
            'seqs': data['sequences'],
            'chain_lens': data['chain_lengths'],
            'total_seqLens': data['total_aa_length'],
            'entities_to_chains': data['entities_to_chains'],
        }])
        df.to_pickle(output_pkl_path)

    return output_pkl_path


def parse_fasta_statistics_to_entity(fasta_path, one_maxlen=10000, one_minlen=1):
    line_chains = []

    for idx, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        chain_info = record.description.split("|")
        pdb_id = chain_info[0].split("_")[0].replace(" ", "").replace("\t", "").replace("\n", "")

        if len(chain_info[0].split("_")) > 1:
            entity_id = chain_info[0].split("_")[1].replace(" ", "").replace("\t", "").replace("\n", "")
        else:
            entity_id = idx

        if len(chain_info) >= 2:
            chain_ids = extract_chains(chain_info[1])
        else:
            chain_ids = ['U']

        sequence = str(record.seq).replace(" ", "").replace("\t", "").replace("\n", "")
        chain_length = len(sequence)

        pdb_id_entity = str(pdb_id) + "_" + str(entity_id)

        one_line_chain_dict = {
            "PDB_ID": pdb_id_entity,
            "model": 0,
            "chain_ids": chain_ids,
            "seqs": [sequence],
            "chain_lens": [chain_length],
            "total_seqLens": chain_length,
            "entity_id": entity_id
        }

        line_chains.append(one_line_chain_dict)

    if len(line_chains) == 0:
        return None
    else:
        return line_chains
    
def save_to_pkl(fasta_path, data_list, one_pdb_output_pkl_dir):
    if not data_list:
        return None

    df = pd.DataFrame(data_list)
    one_pdb_output_pkl_path = os.path.join(one_pdb_output_pkl_dir, f"{Path(fasta_path).stem}.pkl")

    df.to_pickle(one_pdb_output_pkl_path)

    return one_pdb_output_pkl_path


def merge_pkl_files(one_pdb_output_pkl_dir, final_pkl_path, temporary_deleted=True):
    all_dataframes = []

    pkl_files = [
        os.path.join(one_pdb_output_pkl_dir, f)
        for f in os.listdir(one_pdb_output_pkl_dir)
        if f.endswith(".pkl")
    ]

    if not pkl_files:
        return print("No valid data to merge.")

    for pkl_file in pkl_files:
        try:
            df = pd.read_pickle(pkl_file)
            if isinstance(df, pd.DataFrame):
                all_dataframes.append(df)
        except Exception as e:
            print(f"Error reading {pkl_file}: {e}")

    if all_dataframes:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        merged_df.to_pickle(final_pkl_path)

        print(f"Total pkl files: {len(all_dataframes)}, final merged pkl saved at: {final_pkl_path}")

        if temporary_deleted == True:
            print("Temporary pkl files deleted.")
            for pkl_file in pkl_files:
                os.remove(pkl_file)
    else:
        print("No valid data to merge.")

def expand_chain_dicts(data_list):

    expanded = []

    for item in data_list:
        chain_ids = item.get("chain_ids", [])
        if len(chain_ids) <= 1:
            new_item = item.copy()
            new_item["PDB_ID"] = new_item["PDB_ID"][:4]+ "_" +str(new_item["model"])+ "_" + chain_ids[0]
            expanded.append(new_item)
            continue

        for cid in chain_ids:
            new_item = item.copy()
            new_item["chain_ids"] = [cid]  
            new_item["PDB_ID"] = new_item["PDB_ID"][:4] + "_" + str(new_item["model"]) + "_" + cid
            expanded.append(new_item)
    return expanded


def process_pdb_file_task(input_fasta_file, one_pdb_output_pkl_dir, mode):
    """
    Parse fasta and save as pkl, supporting three modes:
      - "complex": complex-level processing
      - "entity": entity-level splitting
      - "single": single-chain splitting, expanding homologous chains within an entity into separate entries
    Return: path to pkl if successful, otherwise None
    """
    try:
        if mode == "complex":
            result = parse_fasta_statistics_to_complex(input_fasta_file)
            save_func = save_to_pkl_complex
        elif mode == "entity":
            result = parse_fasta_statistics_to_entity(input_fasta_file)
            save_func = save_to_pkl
        elif mode == "single":
            result_entity = parse_fasta_statistics_to_entity(input_fasta_file)
            result = expand_chain_dicts(result_entity)
            save_func = save_to_pkl
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if result:
            return save_func(input_fasta_file, result, one_pdb_output_pkl_dir)
        else:
            return None
    except Exception as e:
        print(f"Processing failed: {input_fasta_file}, error: {e}")
        return None

def process_pdb_files_in_parallel(input_fasta_dir, one_pdb_output_pkl_dir, final_pkl_path, mode="complex", temporary_deleted=True, need_processed_list=None, num_workers=None):
    os.makedirs(one_pdb_output_pkl_dir, exist_ok=True)

    final_pkl_dir = os.path.dirname(final_pkl_path)
    os.makedirs(final_pkl_dir, exist_ok=True)
    failed_fasta_log = os.path.join(final_pkl_dir, "failed_fasta.log")
    if num_workers is None:
        num_workers = math.ceil(cpu_count() * 0.85)
    if os.path.isdir(input_fasta_dir):
        all_fasta_files = [f for f in os.listdir(input_fasta_dir) if f.lower().endswith((".fasta", ".fa", ".faa"))]
        print(f"Total number of fasta files: {len(all_fasta_files)}")

        if need_processed_list:
            need_processed_set = set(map(str.lower, need_processed_list))
            input_fasta_files = [
                os.path.join(input_fasta_dir, f)
                for f in all_fasta_files
                if os.path.splitext(f)[0].lower() in need_processed_set
            ]
        else:
            input_fasta_files = [os.path.join(input_fasta_dir, f) for f in all_fasta_files]
    else:
        if input_fasta_dir.endswith(".fasta"):
            print(f"Input is a merged FASTA file: {input_fasta_dir}")
            parsed_results = parse_fasta_statistics_to_entity(input_fasta_dir)
            final_pkl_dir = os.path.dirname(final_pkl_path)
            print(final_pkl_dir)

            if parsed_results is None:
                print(f"No valid sequences parsed from {input_fasta_dir}, logging to: {failed_fasta_log}")
                with open(failed_fasta_log, "a") as logf:
                    logf.write(f"{input_fasta_dir}\n")
            else:
                print(f"Successfully parsed {len(parsed_results)} sequences")
                save_to_pkl(final_pkl_path, parsed_results, os.path.dirname(final_pkl_path))
            return
        else:
            raise ValueError(f"Input path is neither a directory nor a fasta file: {input_fasta_dir}")
    print(f"Number of fasta files to process: {len(input_fasta_files)}")
    print(f"Using {num_workers} processes")
    worker_func = partial(process_pdb_file_task, one_pdb_output_pkl_dir=one_pdb_output_pkl_dir, mode=mode)
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, input_fasta_files), total=len(input_fasta_files)))
    failed_files = [f for f, r in zip(input_fasta_files, results) if r is None]
    successful_files = [r for r in results if r]
    if failed_files:
        with open(failed_fasta_log, "w") as log_file:
            log_file.write("\n".join(failed_files))
        print(f"Number of failed fasta files: {len(failed_files)}, logged to {failed_fasta_log}")
    print(f"Number of successful fasta files: {len(successful_files)}")
    if successful_files:
        merge_pkl_files(one_pdb_output_pkl_dir, final_pkl_path, temporary_deleted)
    else:
        print("No successful pkl files to merge, skipping merge step")




def main(args):
    input_fasta_dir = args.input_fasta_dir
    one_pdb_output_pkl_dir = args.one_pdb_output_pkl_dir
    final_pkl_file = args.final_pkl_file
    mode = args.mode
    temporary_deleted = args.temporary_deleted
    pdb_id_file = args.pdb_id_file
    if pdb_id_file and os.path.exists(pdb_id_file):
        with open(pdb_id_file, "r") as f:
            need_processed_list = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(need_processed_list)} PDB IDs to process.")
    else:
        need_processed_list = None
        print("No PDB ID filter provided. Processing all FASTA files.")
    process_pdb_files_in_parallel(
        input_fasta_dir,
        one_pdb_output_pkl_dir,
        final_pkl_file,
        mode,
        temporary_deleted,
        need_processed_list
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Process FASTA files and convert them into structured PKL format using parallel processing."
    )
    parser.add_argument(
        "--input_fasta_dir",
        type=str,
        required=True,
        help="Directory containing input FASTA files or a single merged FASTA file."
    )
    parser.add_argument(
        "--one_pdb_output_pkl_dir",
        type=str,
        required=True,
        help="Directory to store intermediate PKL files (one per PDB entry)."
    )
    parser.add_argument(
        "--final_pkl_file",
        type=str,
        required=True,
        help="Path to save the final merged PKL file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["complex", "entity", "single"],
        help=(
            "Processing mode:\n"
            "  - complex: keep full complex structure\n"
            "  - entity: split by biological entity\n"
            "  - single: split into single chains (default)"
        )
    )
    parser.add_argument(
        "--temporary_deleted",
        action="store_true",
        help="If set, delete intermediate PKL files after merging."
    )

    parser.add_argument(
        "--pdb_id_file",
        type=str,
        default="",
        help="Optional file containing PDB IDs to process (one per line)."
    )
    args = parser.parse_args()
    main(args)

