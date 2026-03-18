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
import mmcif_parsing  # 处理 mmCIF 文件并转换为 PDB
import residue_constants
import json
def get_valid_chain_range_and_pruned_map(chain_map):
    """
    找到第一个和最后一个非缺失残基索引，并裁剪 chain_map，
    返回 key 从 0 开始连续编号的新 dict。

    参数:
        chain_map: Dict[int, ResidueAtPosition]

    返回:
        start_idx: 原始起始索引
        end_idx: 原始结束索引
        pruned_chain_map: 新的 dict，key 从 0 开始
    """
    keys = sorted(chain_map.keys())

    # 找到第一个非缺失残基的 key
    start_idx = next((k for k in keys if not chain_map[k].is_missing), None)
    # 找到最后一个非缺失残基的 key
    end_idx = next((k for k in reversed(keys) if not chain_map[k].is_missing), None)

    if start_idx is None or end_idx is None:
        raise ValueError("chain_map 中没有任何有效残基（is_missing=False）")

  # key 设为原 key - start_idx
    pruned_chain_map = {
        old_idx - start_idx: copy.deepcopy(chain_map[old_idx])
        for old_idx in range(start_idx, end_idx + 1)
    }
    return start_idx, end_idx, pruned_chain_map

def check_missing_atoms(residue: Residue.Residue) -> List[str]:
    """
    检查残基中是否缺失标准原子，并移除所有氢原子。
    
    参数：
        residue: Biopython 的 Residue 对象
    返回：
        缺失的原子名称列表（如果没有则为空列表）
    """
    resname = residue.get_resname().strip().upper()  # e.g. "ALA"
    if resname not in residue_constants.restype_3to1:
        print(f"[跳过] 未识别残基类型: {resname} at {residue.get_id()}，并加入缺失残基")
        return -1

    for atom in list(residue): 
        atom_name = atom.get_name().strip()
        element = getattr(atom, "element", atom_name[0]).strip().upper()
        #删除氢原子
        if element == 'H' or atom_name.startswith('H'):
            residue.detach_child(atom.id)

    expected_atoms = [name for name in residue_constants.restype_name_to_atom14_names[resname] if name]
    present_atoms = set(atom.get_name().strip() for atom in residue)

    missing_atoms = sorted([name for name in expected_atoms if name not in present_atoms])

    pos = residue.get_id()[1]  # 获取 residue 序号
    return missing_atoms

class AtomSelect(Select):
    # 如果返回 True，原子将被保留到新的 PDB 文件中；如果返回 False，原子将被排除。
    def accept_atom(self, atom):
        accept_criteria = not (
            # atom.is_disordered() or \
            (atom.get_altloc() != ' ' and atom.get_altloc() != 'A' ) or \
            (atom.get_occupancy() < 0.5) or \
            (atom.name not in residue_constants.atom_types)
        )
        return accept_criteria

def renumbered_chain_residues(
    chain: Chain.Chain,
    chain_map: Mapping[int, Any],
) -> Chain.Chain:
    """
    删除不在 chain_map 中的残基，并将剩余残基重编号（从 1 开始）
    """

    chain_copy = copy.deepcopy(chain)
    resseq_to_seq_idx_mapping = {}

    # 构建 resseq -> seq_idx 映射（支持插入码）
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
    # 删除无效残基，并记录原始编号信息到 xtra 字段
    filtered_chain = []
    for resi in list(chain_copy):
        if (
            resi.resname not in residue_constants.restype_3to1 or \
            not all(atom in resi.child_dict for atom in ("N", "CA", "C", "O")) or \
            resi.id[0] != ' '
        ):
            chain_copy.detach_child(resi.id)
            continue
        #删除其编号不在 resseq_to_seq_idx_mapping 中，即不在 chain_map 映射中。
        resseq = resi.id[1]
        res_alt = resi.id[2]  
        if res_alt != ' ':
            key = f"{resseq}_{res_alt}"
        else:
            key = str(resseq)
        if key not in resseq_to_seq_idx_mapping:
            chain_copy.detach_child(resi.id)
            continue
        # 保存原始编号信息
        resi.xtra["original_resseq"] = resi.id[1]
        resi.xtra["original_resalt"] = resi.id[2]

        filtered_chain.append(resi)

    # 第一步：临时使用唯一编号避免冲突
    for temp_id, resi in enumerate(filtered_chain):
        resi.id = (resi.id[0], 10000 + temp_id, resi.id[2])

    # 第二步：恢复编号
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

    # 打印输出检查
    # for residue in filtered_chain:
    #     print(f"chain_copy    Residue: {residue.resname} {residue.id}")

    # 重新组装新 Chain
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
    #更新chain_map，截取掉开头与结尾为缺失的残基，并更新对应关系
    start_idx, end_idx, pruned_chain_map=get_valid_chain_range_and_pruned_map(chain_map)
    # print("split_start_idx:",start_idx,"split_end_idx:",end_idx,'pruned_chain_map:',pruned_chain_map)
    # pprint.pprint(pruned_chain_map)
    seqlen = len(seqres)
    assert seqlen == len(chain_map), 'seqlen != len(chain_map)'
    # pprint.pprint(chain_map)
    #更新截取过的序列
    pruned_seqres=seqres[start_idx:end_idx+1]
    #更新截取过的残基
    chain_map=pruned_chain_map 
    atom_coords = np.zeros((seqlen, residue_constants.atom_type_num, 3)) * np.nan # (seqlen, 37, 3)
    intermediate_residue_missing=set()
    intermediate_atom_missing=[]
    #只对存在于chain_map的残基进行处理
    for seq_idx, res_at_pos in chain_map.items():
        # print("processing:",res_at_pos)
        if res_at_pos.is_missing: 
            # print("missing1:",res_at_pos)
            intermediate_residue_missing.add(f"{seq_idx+1}")
            continue
        if res_at_pos.res_alt not in (' ', '?'):
            residue_id = (' ', res_at_pos.resseq,res_at_pos.res_alt)
        else:
            residue_id = (' ', res_at_pos.resseq,' ')
        # print(residue_id)
        try:
            residue = chain[residue_id]
            # 判断是否为合法残基
            resname = residue.get_resname()
            atom_names = residue.child_dict.keys()
            if (
            resname not in residue_constants.restype_3to1 or
            'N' not in atom_names or
            'CA' not in atom_names or
            'C' not in atom_names or
            'O' not in atom_names
            ):
                
                # print("missing2:",res_at_pos)
                intermediate_residue_missing.add(f"{seq_idx+1}")
                chain.detach_child(residue.id)
                continue
        except Exception as e:
            raise 'Failed to locate residue from chain, {e}'
        #检查原子缺失
        missing_atoms=check_missing_atoms(residue)
        if missing_atoms==-1: #未知残基，missing步已经去除
            # print("missing3:",res_at_pos)
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
    mode: str , # 可选 "complex" 或 "single"
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
        # print("原始：")
        # pprint.pprint(struct_mappings)
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

                #该处chain也经过更新
                atom_coords = chain_info['atom_coords']
                pruned_seqres = chain_info['pruned_seqres']
                pruned_chain = chain_info['pruned_chain']
                pruned_chain_map = chain_info['pruned_chain_map']
                split_start_idx = chain_info['split_start_idx']
                split_end_idx = chain_info['split_end_idx']
                intermediate_residue_missing = chain_info['intermediate_residue_missing']
                intermediate_atom_missing = chain_info['intermediate_atom_missing']
                # print("更新map：")
                # pprint.pprint(pruned_chain_map)
                num_X = pruned_seqres.count('X')
                X_res_ratio=num_X / len(pruned_seqres)
                if not (min_len <= (len(pruned_seqres) - num_X) <= max_len and (X_res_ratio) <= 0.5):
                    with (Path(output_pdb_dir) / "getstruct_error.log").open("a+") as file:
                        file.write(f"{pdb_id}_{chain_id},Chain length or effective residue not meet conditions,error from metadata\n")
                    continue
                # 多字符链名存为pdb格式时会报错，新加：
                if len(chain_id) == 3 and chain_id[0] == chain_id[1] == chain_id[2]:#长度为3时说明是重复的问题，若为2可能是链多的问题
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
                    # Biopython chain -> PDB file
                    # print("pruned_chain:",pruned_chain)
                    # for residue in pruned_chain:
                    #     print(f"    Residue: {residue.resname} {residue.id}") 
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
    """
    将两个指定的 PDB 文件合并为一个文件，并在两条链中间加入链结束标记 TER。
    
    参数:
        file1: 第一个 PDB 文件路径
        file2: 第二个 PDB 文件路径
        output_dir: 输出的合并后的 PDB 文件夹
    """
    chain_pdb_files = [f for f in os.listdir(pdb_files_dir) if f.endswith(".pdb")]
    # 提取文件名并去除扩展名
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
            # 读取第一个 PDB 文件内容
            for line in pdb1:
                if line.startswith("ATOM"):
                    output.write(line)
            
            # 写入链结束标记
            # output.write("TER\n")
            
            # 读取第二个 PDB 文件内容
            for line in pdb2:
                if line.startswith("ATOM"):
                    output.write(line)
            
            # 再次写入链结束标记以结束最后一条链
            output.write("TER\n")
         
    except Exception as e:
        with open(os.path.join(output_dir,"getstruct_error.log"), "a+") as file:
            file.write(f"{pdb_name},{str(e)},error from merge_pdb_files \n")



def is_contact(pdb_file1, pdb_file2, distance_threshold=5.0):
    """
    判断两个 PDB 文件中的链是否“接触”。
    
    参数：
        pdb_file1 (str): 第一个 PDB 文件路径
        pdb_file2 (str): 第二个 PDB 文件路径
        distance_threshold (float): 判定接触的距离阈值，默认 5.0 Å

    返回：
        bool: 如果任意两个重原子之间的最小距离小于阈值，返回 True，否则返回 False。
    """
    parser = PDB.PDBParser(QUIET=True)
    
    # 解析 PDB 文件
    structure1 = parser.get_structure("PDB1", pdb_file1)
    structure2 = parser.get_structure("PDB2", pdb_file2)
    
    # 获取所有重原子坐标（非氢原子）
    def get_heavy_atom_coords(structure):
        coords = []
        for atom in structure.get_atoms():
            if atom.element != "H":  # 过滤氢原子
                coords.append(atom.coord)
        return np.array(coords) if coords else np.empty((0, 3))

    coords1 = get_heavy_atom_coords(structure1)
    coords2 = get_heavy_atom_coords(structure2)

    if coords1.shape[0] == 0 or coords2.shape[0] == 0:
        print("警告：至少一个 PDB 文件没有重原子！")
        return False

    # 计算最小距离
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
    # print(file_name)
    try:
            metadata = process_mmcif(input_cif_gz_file, output_pdb_dir, min_len=min_len, max_len=max_len, mode=mode)
            if metadata is None:
                return
            else:
                pdb_id = file_name
                json_path = os.path.join(json_path_dir, f"{pdb_id}.json")
                # 如果已存在同名 JSON 文件，先删除
                if os.path.exists(json_path):
                    os.remove(json_path)
                # 转换为可序列化格式并写入新 JSON 文件
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
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if "_" in filename:  # 确保文件名包含 '_'
            pdb_id = filename.split("_")[0]  # 获取 PDB ID
            pdb_ids.add(pdb_id.lower())  # 添加到集合，确保唯一
    return pdb_ids

def main():
    # 设置输入输出路径
    input_cif_gz_dir = "/opt/data/private/lyb/data_processed/temp/step3_mmcifgz_download"
    output_pdb_dir = "/opt/data/private/lyb/data_processed/temp/step4_processed_mmcifgz_pdb/pdb"
    metadata_json_path_dir='/opt/data/private/lyb/data_processed/temp/step4_processed_mmcifgz_pdb/metadata_jsonfile_output'
    entries = os.listdir(input_cif_gz_dir)
    #指定处理的PDB_id
    need_processed_path=''
    if need_processed_path :
        with open(need_processed_path, "r") as f:
            pdb_list = [line.strip() for line in f if line.strip()]
            all_need_process = [pdb_id.split(".")[0].lower() for pdb_id in pdb_list if os.path.isfile(os.path.join(input_cif_gz_dir, pdb_id.lower()+".cif.gz")) ]
            notexist_mmcif = [pdb_id.split(".")[0].lower() for pdb_id in pdb_list if not os.path.exists(os.path.join(input_cif_gz_dir, pdb_id.lower()+".cif.gz")) ]
            if len(notexist_mmcif)!=0 :
                print("下列pdb的mmcif文件不存在：",notexist_mmcif,",id已保存在:",os.path.join(output_pdb_dir, "notexist_mmcif.log"))
                with open(os.path.join(output_pdb_dir, "notexist_mmcif.log"), "a+") as file:
                    for i in notexist_mmcif:
                        file.write(f"{notexist_mmcif}\n")
    else:
        print("遍历文件夹，获取需要处理文件路径")
        all_need_process = [entry.split(".")[0].lower() for entry in entries if os.path.isfile(os.path.join(input_cif_gz_dir, entry))]
  
    print("所有需要处理：",len(all_need_process))
    # 设置需要处理的文件列表，如果为空则处理全部文件
    processed_mmcif_to_pdbid=get_unique_pdb_ids(output_pdb_dir)
    # processed_mmcif_to_pdbid=set()
    print("已处理：",len(processed_mmcif_to_pdbid))
    need_processed=[]
    need_processed=list(set(all_need_process)-processed_mmcif_to_pdbid)
    print("还需处理：",len(need_processed))

    # 配置参数
    mode = "single"  # 可以设置为 "single" 或 "complex"
    num_workers=math.ceil(cpu_count() * 0.90)
    min_len = 50 #都为为单链长度
    max_len = 900
    # 单链不需要
    only_chain_num = -1  # 如果需要处理特定链数，设置该值
    distance_threshold = 5.0  # 用于判断接触的距离阈值,
    max_merge_chains = None  # 如果需要限制合并的链数，设置该值

    # 调用处理函数
    process_pdb_files_in_parallel_by_list(
        input_cif_gz_dir=input_cif_gz_dir,
        need_processed=need_processed,
        output_pdb_dir=output_pdb_dir,
        json_path_dir=metadata_json_path_dir,
        num_workers=num_workers,
        mode=mode,
        min_len=min_len,
        max_len=max_len,
        only_chain_num=only_chain_num,
        distance_threshold=distance_threshold,
        max_merge_chains=max_merge_chains
    )


if __name__ == "__main__":
    main()
    # 差值恒定后，说明其被程序处理过，但未成功通过筛选，记录其pdb_id