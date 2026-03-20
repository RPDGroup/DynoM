from Bio import PDB                          # 解析PDB结构文件
from Bio.SeqUtils import seq1               # 将三字母氨基酸转换为一字母代码
from Bio import pairwise2                   # 进行序列间的全局/局部比对
from Bio.pairwise2 import format_alignment  # 打印格式化比对结果

from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import pandas as pd
import json
import re
import pickle
import os
import argparse


'''
pdb_seqs = {
    'A': 'FGRELAAA',
    'B': 'QVQLQESGPGLVAPSQ'
}

fasta_seqs = {
    '1A2Y_3': 'KVFGRCELAAAM',
    '1A2Y_1': 'DIVLTQSPASLSA',
    '1A2Y_2': 'QVQLQESGPGLVAPSQSLSI'
}

align result：
    [{
        'chain_name':'A',
        'seq':'FGRCELAA',
        'info':{
            'pdb_chain_name':'A',
            'fasta_best_match':'1A2Y_3',
            'best_score':'258.00',
            'pdb_split_seqs':'--FGR-ELAA--'
            'fasta_seqs':'KVFGRCELAAAM'
        }
    }
    ...
    ]
'''

def extract_model0_sequences(pdb_file):
    """
    Extract the amino acid sequences (single-letter format) from model 0 of a multi-model PDB file.
    Returns a dictionary where keys are chain IDs and values are the corresponding sequences.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("my_structure", pdb_file)
    model0 = structure[0]
    chain_seq_dict = {}
    for chain in model0:
        residues = []
        for residue in chain:
            if PDB.is_aa(residue, standard=True):
                residues.append(seq1(residue.resname))
        chain_seq_dict[chain.id] = "".join(residues)

    return chain_seq_dict, structure


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
            if (len(chain_clean) > 1) and (auth_match and len(auth_match.group(1)) > 1):
                continue
            chains.append(str(auth_map.get(chain_clean, chain_clean)))

    return list(set(chains)) if chains else [] 

def fasta_to_dict(fasta_file):
    seq_dict = {}
    current_id = None
    current_seq = []

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id and current_seq:
                    seq_dict[current_id] = "".join(current_seq)
                chain_info = line[1:].split("|")
                pdbid_entityid=chain_info[0].strip()
                current_id=pdbid_entityid
                current_seq = []
            else:
                current_seq.append(line)
        if current_id and current_seq:
            seq_dict[current_id] = "".join(current_seq)
    
    return seq_dict


def align_pdb_with_fasta(pdb_query_seqs: dict, fasta_search_seqs: dict):
    aligned_results = {}

    for pdb_chain, pdb_seq in pdb_query_seqs.items():
        best_score = float('-inf')
        best_match = None
        best_alignment = None
        best_fasta_id = None

        for fasta_id, fasta_seq in fasta_search_seqs.items():
            alignments = pairwise2.align.localms(pdb_seq, fasta_seq, 2, -1, -0.5, -0.1)

            if not alignments:
                continue  

            score = alignments[0].score
            if score > best_score:
                best_score = score
                best_alignment = alignments[0]
                best_match = fasta_seq
                best_fasta_id = fasta_id

        if best_alignment:
            aligned_results[pdb_chain] = {
                "pdb_seq": best_alignment.seqA,
                "fasta_seq": best_alignment.seqB,
                "fasta_id": best_fasta_id,
                "score": best_score
            }
        else:
            aligned_results[pdb_chain] = {
                "pdb_seq": "",
                "fasta_seq": "",
                "fasta_id": None,
                "score": None
            }

    return aligned_results


def extract_core_seq_by_terminal_gaps(pdb_aln, fasta_aln):
    """
    Extract the matched region from the fasta sequence based on terminal gaps ('-') in the PDB aligned sequence.
    Internal gaps are not modified, and full residues are preserved.
    """
    assert len(pdb_aln) == len(fasta_aln), "Alignment lengths do not match"

    start = next((i for i, c in enumerate(pdb_aln) if c != '-'), None)
    end = len(pdb_aln) - next((i for i, c in enumerate(reversed(pdb_aln)) if c != '-'), None)

    core_seq_with_gap = fasta_aln[start:end]
    core_seq = core_seq_with_gap.replace('-', '')
    return core_seq


def convert_alignment_to_json_and_save(aligned_results, output_json_dir, output_pkl_dir, pdbid ):
    os.makedirs(output_json_dir, exist_ok=True)
    os.makedirs(output_pkl_dir, exist_ok=True)

    json_results = []

    chain_ids = []
    seqs = []
    chain_lens = []

    for chain_name, result in aligned_results.items():
        pdb_seq_aln = result['pdb_seq']
        fasta_seq_aln = result['fasta_seq']
        fasta_id = result['fasta_id']

        core_seq = extract_core_seq_by_terminal_gaps(pdb_seq_aln, fasta_seq_aln)

        json_results.append({
            'pdb_chain_name': chain_name,
            'seq': core_seq,
            'info': {
                'pdb_chain_name': chain_name,
                'fasta_best_match': fasta_id,
                'best_score': f"{result['score']:.2f}",
                'pdb_split_seqs': pdb_seq_aln,
                'fasta_seqs': fasta_seq_aln
            }
        })

        chain_ids.append(chain_name)
        seqs.append(core_seq)
        chain_lens.append(len(core_seq))

    total_seqLens = sum(chain_lens)

    df = pd.DataFrame([{
        'PDB_ID': pdbid,
        'model': 0,
        'chain_ids': chain_ids,
        'seqs': seqs,
        'chain_lens': chain_lens,
        'total_seqLens': total_seqLens
    }])

    # 保存 JSON 文件
    json_path = os.path.join(output_json_dir, f"{pdbid}_align_info.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=4)

    # 保存为 .pkl 文件（DataFrame）
    pkl_path = os.path.join(output_pkl_dir, f"{pdbid}.pkl")
    df.to_pickle(pkl_path)
    



def process_alignment(pdb_path, fasta_path, pdbid, json_output_dir, pkl_output_dir):
    
    try:
        pdb_chain_seq_dict,structure = extract_model0_sequences(pdb_path)
        fasta_seqs = fasta_to_dict(fasta_path)
        aligned_results = align_pdb_with_fasta(pdb_chain_seq_dict, fasta_seqs)
        pdbid = pdbid

        # 保存结果
        convert_alignment_to_json_and_save(
            aligned_results,
            output_json_dir=json_output_dir,
            output_pkl_dir=pkl_output_dir,
            pdbid=pdbid
        )
        return pdbid, "success"

    except Exception as e:
        return os.path.basename(pdb_path)[:4].upper(), "fail"




def run_multiprocess_alignment(pdb_fasta_pairs, json_output_dir, pkl_output_dir, num_processes=8, log_path="alignment_results.log"):

    args_list = [
        (pdb_path, fasta_path, pdbid, json_output_dir, pkl_output_dir)
        for pdb_path, fasta_path, pdbid in pdb_fasta_pairs
    ]

    results = []
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.starmap(process_alignment, args_list), total=len(args_list), desc="Processing Alignments"):
            results.append(result)

    with open(log_path, "w") as log_file:
        for pdbid, status in results:
            log_file.write(f"{pdbid}\t{status}\n")

    return results


def merge_pkl_files(one_pdb_output_pkl_dir, final_pkl_path,temporary_deleted=True):

    all_dataframes = [] 

    # 获取所有 .pkl 文件
    pkl_files = [os.path.join(one_pdb_output_pkl_dir, f) 
                 for f in os.listdir(one_pdb_output_pkl_dir) if f.endswith(".pkl")]
    if not pkl_files :
        return  print("No valid data to merge.")
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
        print(f"all one pkl nums:{len(all_dataframes)},Final merged pkl saved at: {final_pkl_path}")

        if temporary_deleted==True :
            print("Temporary pkl files deleted.")
            for pkl_file in pkl_files:
                os.remove(pkl_file)
    else:
        print("No valid data to merge.")


def get_pdb_ids_from_dir(directory):
    with os.scandir(directory) as entries:
        return list({
            entry.name.split(".")[0]
            for entry in entries
            if entry.is_file()
        })

def get_pdb_fasta_pairs(dir_path, pdb_id_list, fasta_dir):
    pdb_fasta_pairs = []
    for pdbid in pdb_id_list:
        pdb_path = os.path.join(
            dir_path,
            f"{pdbid}.pdb"
        )
        fasta_path = os.path.join(fasta_dir, f"{pdbid[:4].lower()}.fasta")
        pdb_fasta_pairs.append((pdb_path, fasta_path , pdbid))
    return pdb_fasta_pairs

def main(args):
    dir_path = args.pdb_dir
    fasta_dir = args.fasta_dir
    json_dir = args.json_dir
    pkl_dir = args.pkl_dir
    temporary_deleted = args.temporary_deleted
    num_processes = args.num_processes
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)
    pdb_id_list = get_pdb_ids_from_dir(dir_path)
    pdb_fasta_pairs = get_pdb_fasta_pairs(dir_path, pdb_id_list, fasta_dir)
    print(f"Total alignment tasks: {len(pdb_fasta_pairs)}")
    print("Start alignment...")
    results = run_multiprocess_alignment(
        pdb_fasta_pairs,
        json_output_dir=json_dir,
        pkl_output_dir=pkl_dir,
        num_processes=num_processes,
        log_path=os.path.join(json_dir, "alignment_log.txt")
    )
    print("Merging PKL files...")
    final_pkl_path = os.path.join(os.path.dirname(pkl_dir), args.final_pkl_name)
    merge_pkl_files(
        pkl_dir,
        final_pkl_path,
        temporary_deleted=temporary_deleted
    )
    print("Merge completed. Final PKL saved to:", final_pkl_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4-1: Align PDB-derived sequences with FASTA sequences and generate cleaned PKL data."
    )
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Directory containing processed PDB files")
    parser.add_argument("--fasta_dir", type=str, required=True,
                        help="Directory containing FASTA files")
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Directory to save alignment JSON outputs")
    parser.add_argument("--pkl_dir", type=str, required=True,
                        help="Directory to save intermediate PKL files")
    parser.add_argument("--final_pkl_name", type=str, default="step4-1_final_merged.pkl",
                        help="Name of final merged PKL file")

    parser.add_argument("--num_processes", type=int, default=32,
                        help="Number of parallel processes")
    parser.add_argument("--temporary_deleted", action="store_true",
                        help="Delete intermediate PKL files after merging")
    args = parser.parse_args()
    main(args)