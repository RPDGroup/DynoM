import os
import sys
import math
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Union, Mapping,Optional, Literal
from Bio.PDB import Select, Chain,Residue
from Bio import PDB
from tqdm import tqdm
import numpy as np
import shutil
import pprint
import copy
import mmcif_parsing  
import residue_constants
import json
import argparse

def get_valid_chain_range_and_pruned_map(chain_map):
    
    """
    Identify the first and last non-missing residue indices, and prune the chain_map.
    Return a new dictionary with keys reindexed starting from 0.

    Parameters:
        chain_map: Dict[int, ResidueAtPosition]

    Returns:
        start_idx: Original starting index
        end_idx: Original ending index
        pruned_chain_map: New dict with keys starting from 0
    """
    keys = sorted(chain_map.keys())

    start_idx = next((k for k in keys if not chain_map[k].is_missing), None)
    end_idx = next((k for k in reversed(keys) if not chain_map[k].is_missing), None)

    if start_idx is None or end_idx is None:
        raise ValueError("No valid residues found in chain_map (is_missing=False)")

    pruned_chain_map = {
        old_idx - start_idx: copy.deepcopy(chain_map[old_idx])
        for old_idx in range(start_idx, end_idx + 1)
    }
    return start_idx, end_idx, pruned_chain_map


def check_missing_atoms(residue: Residue.Residue) -> List[str]:
    """
    Check whether standard atoms are missing in a residue and remove all hydrogen atoms.

    Parameters:
        residue: Biopython Residue object

    Returns:
        List of missing atom names (empty list if none)
    """
    resname = residue.get_resname().strip().upper()
    if resname not in residue_constants.restype_3to1:
        print(f"[Skip] Unrecognized residue type: {resname} at {residue.get_id()}, treated as missing residue")
        return -1

    for atom in list(residue):
        atom_name = atom.get_name().strip()
        element = getattr(atom, "element", atom_name[0]).strip().upper()
        if element == 'H' or atom_name.startswith('H'):
            residue.detach_child(atom.id)

    expected_atoms = [name for name in residue_constants.restype_name_to_atom14_names[resname] if name]
    present_atoms = set(atom.get_name().strip() for atom in residue)

    missing_atoms = sorted([name for name in expected_atoms if name not in present_atoms])

    pos = residue.get_id()[1]
    return missing_atoms

class AtomSelect(Select):
    def accept_atom(self, atom):
        accept_criteria = not (
            (atom.get_altloc() != ' ' and atom.get_altloc() != 'A') or
            (atom.get_occupancy() < 0.5) or
            (atom.name not in residue_constants.atom_types)
        )
        return accept_criteria


def renumbered_chain_residues(
    chain: Chain.Chain,
    chain_map: Mapping[int, Any],
) -> Chain.Chain:
    """
    Remove residues not present in chain_map and renumber the remaining residues (starting from 1).
    """
    chain_copy = copy.deepcopy(chain)
    resseq_to_seq_idx_mapping = {}
    for seq_idx, res_at_pos in chain_map.items():
        if res_at_pos.is_missing:
            continue
        if res_at_pos.res_alt not in ('?', ' '):
            reflect_index = f"{res_at_pos.resseq}_{res_at_pos.res_alt}"
        else:
            reflect_index = str(res_at_pos.resseq)
        assert reflect_index not in resseq_to_seq_idx_mapping, \
            f"Duplicate assignment for {reflect_index}"
        resseq_to_seq_idx_mapping[reflect_index] = seq_idx

    filtered_chain = []
    for resi in list(chain_copy):
        if (
            resi.resname not in residue_constants.restype_3to1 or
            not all(atom in resi.child_dict for atom in ("N", "CA", "C", "O")) or
            resi.id[0] != ' '
        ):
            chain_copy.detach_child(resi.id)
            continue
        resseq = resi.id[1]
        res_alt = resi.id[2]
        if res_alt != ' ':
            key = f"{resseq}_{res_alt}"
        else:
            key = str(resseq)
        if key not in resseq_to_seq_idx_mapping:
            chain_copy.detach_child(resi.id)
            continue
        resi.xtra["original_resseq"] = resi.id[1]
        resi.xtra["original_resalt"] = resi.id[2]
        filtered_chain.append(resi)

    for temp_id, resi in enumerate(filtered_chain):
        resi.id = (resi.id[0], 10000 + temp_id, resi.id[2])

    for resi in filtered_chain:
        original_resseq = resi.xtra.get("original_resseq")
        original_res_alt = resi.xtra.get("original_resalt", ' ')
        if original_res_alt not in ('?', ' '):
            key = f"{original_resseq}_{original_res_alt}"
        else:
            key = str(original_resseq)
        seq_idx = resseq_to_seq_idx_mapping.get(key)
        if seq_idx is None:
            raise ValueError(f"Residue {key} not found in mapping.")
        resi.id = (resi.id[0], seq_idx + 1, " ")

    new_chain = Chain.Chain(chain.id)
    for residue in filtered_chain:
        new_chain.add(residue)

    return new_chain

def chain_to_npy_with_missing(
    seqres: str, # mmCIF-parsed seqres
    chain: Chain.Chain, # Biopython-parsed chain
    chain_map: Mapping[int, Any], # mmCIF-parsed chain map
) -> Dict[str, np.ndarray]:
    """
    Convert from Biopython chain to all-atom numpy array.
    """
    start_idx, end_idx, pruned_chain_map=get_valid_chain_range_and_pruned_map(chain_map)
    seqlen = len(seqres)
    assert seqlen == len(chain_map), 'seqlen != len(chain_map)'
    pruned_seqres=seqres[start_idx:end_idx+1]

    chain_map=pruned_chain_map 
    atom_coords = np.zeros((seqlen, residue_constants.atom_type_num, 3)) * np.nan # (seqlen, 37, 3)
    intermediate_residue_missing=set()
    intermediate_atom_missing=[]
    for seq_idx, res_at_pos in chain_map.items():
        if res_at_pos.is_missing: 
            intermediate_residue_missing.add(f"{seq_idx+1}")
            continue
        if res_at_pos.res_alt not in (' ', '?'):
            residue_id = (' ', res_at_pos.resseq,res_at_pos.res_alt)
        else:
            residue_id = (' ', res_at_pos.resseq,' ')
        try:
            residue = chain[residue_id]
            resname = residue.get_resname()
            atom_names = residue.child_dict.keys()
            if (
            resname not in residue_constants.restype_3to1 or
            'N' not in atom_names or
            'CA' not in atom_names or
            'C' not in atom_names or
            'O' not in atom_names
            ):
                
                intermediate_residue_missing.add(f"{seq_idx+1}")
                chain.detach_child(residue.id)
                continue
        except Exception as e:
            raise 'Failed to locate residue from chain, {e}'
        missing_atoms=check_missing_atoms(residue)
        if missing_atoms==-1:
            intermediate_residue_missing.add(f"{seq_idx+1}")
            chain.detach_child(residue.id)
        if len(missing_atoms)!=0:
            intermediate_atom_missing.append({f"{residue.resname}_{residue.id[1]}":missing_atoms})
        for atom in residue:   
            atom_coords[seq_idx, residue_constants.atom_order[atom.name]] = atom.coord
    return {
        'atom_coords': atom_coords,
        "pruned_seqres":pruned_seqres,
        "pruned_chain":chain,
        "pruned_chain_map":pruned_chain_map,
        "split_start_idx":start_idx,
        "split_end_idx":end_idx,
        "intermediate_residue_missing":sorted(intermediate_residue_missing, key=int),
        "intermediate_atom_missing":intermediate_atom_missing
    }

def process_mmcif(
    mmcif_path: Union[str, Path],
    output_pdb_dir: str,
    min_len: int ,
    max_len: int ,
    mode: str ,
) -> List[Dict[str, Any]]:
    metadata = []
    
    try:
        try:
            mmcif_object, author_chain_id_to_mmcif = mmcif_parsing.parse(mmcif_path)
        except Exception as e:
            file_name = Path(mmcif_path).stem
            error_log = Path(output_pdb_dir) / "getstruct_error.log"
            with error_log.open("a+") as file:
                file.write(f"{file_name},{str(e)},error from mmcif_parsing.parse\n")
            assert RuntimeError(f"{file_name} get mmcif_object error from mmcif_parsing")
            return []
        
        pdb_id = mmcif_object.pdb_id
        header= mmcif_object.header
        full_structure = mmcif_object.full_structure
        chain_to_seqres = mmcif_object.chain_to_seqres
        entity_to_chains = mmcif_object.entity_to_chains
        struct_mappings = mmcif_object.struct_mappings

        if len(full_structure.child_list) > 1:
            model0_chains = set(full_structure.child_list[0].child_dict.keys())
            for model_idx in range(1, len(full_structure.child_list)):
                modelx_chains = set(full_structure.child_list[model_idx].child_dict.keys())
                assert model0_chains == modelx_chains, 'Different chains across models.'
        
        for model_id, model_map in struct_mappings.items():
            for chain_id, chain_map in model_map.items():
                seqres = chain_to_seqres.get(chain_id, "")
                chain = full_structure[model_id][chain_id]
                
                chain_info = chain_to_npy_with_missing(seqres=seqres, chain=chain, chain_map=chain_map)

                atom_coords = chain_info['atom_coords']
                pruned_seqres = chain_info['pruned_seqres']
                pruned_chain = chain_info['pruned_chain']
                pruned_chain_map = chain_info['pruned_chain_map']
                split_start_idx = chain_info['split_start_idx']
                split_end_idx = chain_info['split_end_idx']
                intermediate_residue_missing = chain_info['intermediate_residue_missing']
                intermediate_atom_missing = chain_info['intermediate_atom_missing']
                num_X = pruned_seqres.count('X')
                X_res_ratio=num_X / len(pruned_seqres)
                if not (min_len <= (len(pruned_seqres) - num_X) <= max_len and (X_res_ratio) <= 0.5):
                    with (Path(output_pdb_dir) / "getstruct_error.log").open("a+") as file:
                        file.write(f"{pdb_id}_{chain_id},Chain length or effective residue not meet conditions,error from metadata\n")
                    continue
                if len(chain_id) == 3 and chain_id[0] == chain_id[1] == chain_id[2]:
                    chain_pruned.id = chain_id[0] 
                    chain_id=chain_id[0]
                chain_name = f"{pdb_id}_{model_id}_{chain_id}"
                if mode == "complex":
                    pdb_subdir = Path(output_pdb_dir) / pdb_id
                else:
                    pdb_subdir = Path(output_pdb_dir)
                pdb_subdir.mkdir(parents=True, exist_ok=True)
                pdb_path = pdb_subdir / f"{chain_name}.pdb"
                
                if pdb_path.exists():
                    os.remove(pdb_path)

                
                try:

                    chain_pruned = renumbered_chain_residues(chain=pruned_chain, chain_map=pruned_chain_map)
                    if not list(chain_pruned):
                        raise ValueError(f"Pruned chain is empty for chain: {chain_id}")
                    
                    pdbio = PDB.PDBIO()
                    pdbio.set_structure(chain_pruned)
                    pdbio.save(str(pdb_path), select=AtomSelect())
                except Exception as e:
                    with (Path(output_pdb_dir) / "getstruct_error.log").open("a+") as file:
                        file.write(f"{pdb_id}_{chain_id},{str(e)},error from metadata\n")
                    continue
                
                metadata.append({
                    'chain_name': chain_name,
                    'seqres': pruned_seqres,
                    'seqlen': len(pruned_seqres),
                    'X_res_ratio': X_res_ratio,
                    'split_start_idx': split_start_idx,
                    'split_end_idx':split_end_idx,
                    'residue_missing_num':len(intermediate_residue_missing),
                    'residue_missing':intermediate_residue_missing,
                    'atom_missing_num':len(intermediate_atom_missing),
                    'atom_missing':intermediate_atom_missing,
                })
                
        output_pdb_path = Path(output_pdb_dir) / pdb_id.upper()
        if output_pdb_path.exists() and not any(output_pdb_path.iterdir()):
            output_pdb_path.rmdir()
        
        return metadata
    except Exception as e:
        file_name = os.path.basename(mmcif_path).split(".")[0]
        with (Path(output_pdb_dir) / "getstruct_error.log").open("a+") as file:
            file.write(f"{file_name},{str(e)},error from process_mmcif\n")
        return []

def merge_two_pdb_files(pdb_files_dir, output_dir):

    chain_pdb_files = [f for f in os.listdir(pdb_files_dir) if f.endswith(".pdb")]
    pdb_name = os.path.splitext(os.path.basename(chain_pdb_files[0]))[0].split("_")[0]
    os.makedirs(output_dir, exist_ok = True)
    output_file = f"{output_dir}/{pdb_name}.pdb"
    try:
        file1=os.path.join(pdb_files_dir,chain_pdb_files[0])
        file2=os.path.join(pdb_files_dir,chain_pdb_files[1])
    except Exception as e:
        with open(os.path.join(output_dir,"getstruct_error.log"), "a+") as file:
            file.write(f"{pdb_name},{str(e)},Not have two pdb file error from merge_pdb_files \n")
        print(f"Not have two pdb file {pdb_name} in :{chain_pdb_files } error :{e}")
        return 
    try:
        with open(file1, 'r') as pdb1, open(file2, 'r') as pdb2, open(output_file, 'w') as output:
            for line in pdb1:
                if line.startswith("ATOM"):
                    output.write(line)

            for line in pdb2:
                if line.startswith("ATOM"):
                    output.write(line)

            output.write("TER\n")
         
    except Exception as e:
        with open(os.path.join(output_dir,"getstruct_error.log"), "a+") as file:
            file.write(f"{pdb_name},{str(e)},error from merge_pdb_files \n")



def is_contact(pdb_file1, pdb_file2, distance_threshold=5.0):
    """
    Determine whether two PDB structures are in contact.

    Parameters:
        pdb_file1 (str): Path to the first PDB file
        pdb_file2 (str): Path to the second PDB file
        distance_threshold (float): Distance threshold for defining contact (default: 5.0 Å)

    Returns:
        bool: True if the minimum distance between any two heavy atoms is below the threshold, otherwise False.
    """
    parser = PDB.PDBParser(QUIET=True)
    
    structure1 = parser.get_structure("PDB1", pdb_file1)
    structure2 = parser.get_structure("PDB2", pdb_file2)
    
    def get_heavy_atom_coords(structure):
        coords = []
        for atom in structure.get_atoms():
            if atom.element != "H":
                coords.append(atom.coord)
        return np.array(coords) if coords else np.empty((0, 3))

    coords1 = get_heavy_atom_coords(structure1)
    coords2 = get_heavy_atom_coords(structure2)

    if coords1.shape[0] == 0 or coords2.shape[0] == 0:
        print("Warning: At least one PDB file contains no heavy atoms!")
        return False

    dist_matrix = np.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=-1)
    min_distance = np.min(dist_matrix)

    return min_distance < distance_threshold

def convert_sets_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(elem) for elem in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

def process_pdb_file_task(input_cif_gz_file, output_pdb_dir, json_path_dir,
                          mode="single", min_len=1, max_len=10000, 
                          only_chain_num=-1, distance_threshold=5.0, 
                          max_merge_chains=None):
    file_name = os.path.basename(input_cif_gz_file).split(".")[0]
    try:
            metadata = process_mmcif(input_cif_gz_file, output_pdb_dir, min_len=min_len, max_len=max_len, mode=mode)
            if metadata is None:
                return
            else:
                pdb_id = file_name
                json_path = os.path.join(json_path_dir, f"{pdb_id}.json")
                if os.path.exists(json_path):
                    os.remove(json_path)
                metadata_serializable = convert_sets_to_lists(metadata)
                with open(json_path, "w") as f:
                    json.dump(metadata_serializable, f, indent=2)
            
            if mode == "complex":
                merge_pdb_path = os.path.join(output_pdb_dir, file_name.upper())
                pdb_files = sorted(Path(merge_pdb_path).glob("*.pdb"))
                merged_files = set()
                if  only_chain_num >0 and only_chain_num!= len(pdb_files) :
                    with open(os.path.join(output_pdb_dir, "error_output.log"), "a+") as file:
                        file.write(f"{only_chain_num } >0 and  only_chain_num:{only_chain_num} != len(pdb_files):{len(pdb_files)}\n")
                for i, pdb_file1 in enumerate(pdb_files):
                    if pdb_file1 in merged_files:
                        continue
                    for j, pdb_file2 in enumerate(pdb_files[i+1:]):
                        if pdb_file2 in merged_files:
                            continue
                        if is_contact(pdb_file1, pdb_file2, distance_threshold):
                            merge_two_pdb_files(pdb_file1, pdb_file2, merge_pdb_path)
                            merged_files.add(pdb_file1)
                            merged_files.add(pdb_file2)
                            if max_merge_chains and len(merged_files) >= max_merge_chains:
                                break
                    if max_merge_chains and len(merged_files) >= max_merge_chains:
                        break
                shutil.rmtree(merge_pdb_path, ignore_errors=True)
    except Exception as e:
        with open(os.path.join(output_pdb_dir, "error_output.log"), "a+") as file:
            file.write(f"{file_name} , {str(e)}\n")


def process_pdb_files_in_parallel_by_list(input_cif_gz_dir, need_processed, output_pdb_dir,json_path_dir,
                                        num_workers=None, mode="single", min_len=1,
                                        max_len=10000, only_chain_num=-1, distance_threshold=5.0, 
                                        max_merge_chains=None):
    os.makedirs(output_pdb_dir, exist_ok=True)
    if num_workers is None:
        num_workers = math.ceil(cpu_count() * 0.85)
    
    if need_processed:
        input_cif_gz_files = [
            os.path.join(input_cif_gz_dir, f)
            for f in os.listdir(input_cif_gz_dir)
            if f.endswith(".cif.gz") and os.path.isfile(os.path.join(input_cif_gz_dir, f)) and 
               os.path.basename(f).split(".")[0].lower() in map(str.lower, need_processed)
        ]
    else:
        input_cif_gz_files = [
            os.path.join(input_cif_gz_dir, f)
            for f in os.listdir(input_cif_gz_dir)
            if f.endswith(".cif.gz") and os.path.isfile(os.path.join(input_cif_gz_dir, f))
        ]
    
    print(f"Total files to process: {len(input_cif_gz_files)}")
    print(f"Using {num_workers} workers.")
    if input_cif_gz_files:
        print(input_cif_gz_files[0])
    
    worker_func = partial(process_pdb_file_task, output_pdb_dir=output_pdb_dir,json_path_dir=json_path_dir,mode=mode, min_len=min_len, max_len=max_len, 
                          only_chain_num=only_chain_num, distance_threshold=distance_threshold, max_merge_chains=max_merge_chains)
    
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(worker_func, input_cif_gz_files), total=len(input_cif_gz_files)))

def get_unique_pdb_ids(folder_path):
    pdb_ids = set()
    for filename in os.listdir(folder_path):
        if "_" in filename: 
            pdb_id = filename.split("_")[0]
            pdb_ids.add(pdb_id.lower())
    return pdb_ids




def main(args):
    
    input_cif_gz_dir = args.input_cif_gz_dir
    output_pdb_dir = args.output_pdb_dir
    metadata_json_path_dir = args.metadata_json_dir
    need_processed_path = args.need_processed_path
    os.makedirs(output_pdb_dir, exist_ok=True)
    os.makedirs(metadata_json_path_dir, exist_ok=True)
    entries = os.listdir(input_cif_gz_dir)

    if need_processed_path:
        with open(need_processed_path, "r") as f:
            pdb_list = [line.strip() for line in f if line.strip()]
        all_need_process = [
            pdb_id.split(".")[0].lower()
            for pdb_id in pdb_list
            if os.path.isfile(os.path.join(input_cif_gz_dir, pdb_id.lower() + ".cif.gz"))
        ]
        notexist_mmcif = [
            pdb_id.split(".")[0].lower()
            for pdb_id in pdb_list
            if not os.path.exists(os.path.join(input_cif_gz_dir, pdb_id.lower() + ".cif.gz"))
        ]
        if notexist_mmcif:
            log_path = os.path.join(output_pdb_dir, "notexist_mmcif.log")
            print("Missing mmCIF files:", notexist_mmcif)
            with open(log_path, "a+") as file:
                for i in notexist_mmcif:
                    file.write(f"{i}\n")
    else:
        print("Scanning directory for mmCIF files...")
        all_need_process = [
            entry.split(".")[0].lower()
            for entry in entries
            if os.path.isfile(os.path.join(input_cif_gz_dir, entry))
        ]
    print("Total to process:", len(all_need_process))
    processed_mmcif_to_pdbid = get_unique_pdb_ids(output_pdb_dir)
    need_processed = list(set(all_need_process) - processed_mmcif_to_pdbid)
    print("Already processed:", len(processed_mmcif_to_pdbid))
    print("Remaining:", len(need_processed))
    num_workers = math.ceil(cpu_count() * args.cpu_ratio)
    process_pdb_files_in_parallel_by_list(
        input_cif_gz_dir=input_cif_gz_dir,
        need_processed=need_processed,
        output_pdb_dir=output_pdb_dir,
        json_path_dir=metadata_json_path_dir,
        num_workers=num_workers,
        mode=args.mode,
        min_len=args.min_len,
        max_len=args.max_len,
        only_chain_num=args.only_chain_num,
        distance_threshold=args.distance_threshold,
        max_merge_chains=args.max_merge_chains
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: Convert mmCIF (.cif.gz) files to processed PDB format and metadata."
    )
    parser.add_argument("--input_cif_gz_dir", type=str, required=True,
                        help="Directory containing mmCIF (.cif.gz) files")
    parser.add_argument("--output_pdb_dir", type=str, required=True,
                        help="Directory to save processed PDB files")
    parser.add_argument("--metadata_json_dir", type=str, required=True,
                        help="Directory to save metadata JSON files")
    parser.add_argument("--need_processed_path", type=str, default="",
                        help="Optional file containing PDB IDs to process")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "complex"],
                        help="Processing mode")
    parser.add_argument("--cpu_ratio", type=float, default=0.9,
                        help="CPU usage ratio")
    parser.add_argument("--min_len", type=int, default=50,
                        help="Minimum chain length")
    parser.add_argument("--max_len", type=int, default=900,
                        help="Maximum chain length")
    parser.add_argument("--only_chain_num", type=int, default=-1,
                        help="Filter by number of chains")
    parser.add_argument("--distance_threshold", type=float, default=10000,
                        help="Distance threshold for contacts")
    parser.add_argument("--max_merge_chains", type=int, default=100,
                        help="Maximum number of chains to merge")
    args = parser.parse_args()
    main(args)
