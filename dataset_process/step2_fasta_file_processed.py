import re
import os
from Bio import SeqIO
import pandas as pd
from pathlib import Path
import math
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import defaultdict

def extract_chains(chain_str):
    chains = []
    
    # 先匹配 `X[auth Y]` 形式，建立原始链名到 `auth` 别名的映射（支持大小写）
    auth_matches = re.findall(r'([\w\d]+)\[auth ([\w\d]+)\]', chain_str)
    auth_map = {orig: auth for orig, auth in auth_matches}  # 生成映射，如 P -> Q
    
    # 只匹配以 "Chains" 或 "Chain" 开头的链名部分，支持大小写链名
    all_chains = re.findall(r'Chains? ([\w\d,\[\] auth]+)', chain_str)
    
    for match in all_chains:
        chain_list = re.split(r',\s*', match)  # 按逗号拆分链名
        for chain in chain_list:
            # 提取并清理链名和auth别名
            chain_clean = re.sub(r'\[auth ([\w\d]+)\]', '', chain).strip()  # 去掉 `[auth X]`，保留原始链名
            
            # 如果链名长度大于 1 或者 auth 别名的长度大于 1，则跳过
            # if (len(chain_clean) > 1) and (auth_match and len(auth_match.group(1)) > 1):
            #     continue

            # 用 auth 别名替换，否则用原值
            chains.append(str(auth_map.get(chain_clean, chain_clean)))
    # print("chains:",chains)
    return list(set(chains)) if chains else []  # 只返回有效的链名

'''
    复合物处理
'''
# 将重复链输出成复合物形式的形式
def parse_fasta_statistics_to_complex(fasta_path,one_maxlen=650,one_minlen=20):
    """
    解析单链的 FASTA 文件，提取链 ID、氨基酸序列及长度信息，并返回字典格式的数据。
    
    参数:
        fasta_path (str): FASTA 文件的路径。
        one_maxlen=650,one_minlen=20:单链最长最短长度，（初筛）
    返回:


        dict: 包含统计信息的字典，结构如下：
    """
    # 获取 PDB ID（基于文件名）

    # 初始化存储变量
    sequences_arr = []
    chain_lengths_arr = []
    entities_to_chains=[]
    # 解析 FASTA 文件

    for idx, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        chain_info = record.description.split("|")
        pdb_id = chain_info[0].split("_")[0].replace(" ", "").replace("\t", "").replace("\n", "")

        if len(chain_info[0].split("_"))>1:
            entity_id=chain_info[0].split("_")[1].replace(" ", "").replace("\t", "").replace("\n", "")
        else:
            entity_id=idx
        # 提取 Chain ID（处理可能的多个链 ID）
        if len(chain_info) >= 2:
            chain_ids =extract_chains(chain_info[1])
            # chain_ids = [chain for chain in chain_ids if len(chain) < 2]
        else:
            chain_ids=['U']
        sequence = str(record.seq).replace(" ", "").replace("\t", "").replace("\n", "")
        chain_length = len(sequence)
        # if chain_length>one_maxlen or chain_length <one_minlen :
        #     continue
        # 处理多个链 ID（用逗号分隔）
        entity_to_chain={}
        entity_to_chain[entity_id]=chain_ids
        if len(chain_ids) >1 :
            sequences_arr.extend([sequence] * len(chain_ids))
            chain_lengths_arr.extend([chain_length] * len(chain_ids))
        else:
            sequences_arr.append(sequence)
            chain_lengths_arr.append(chain_length)
        entities_to_chains.append(entity_to_chain)

    # 计算总氨基酸长度
    total_aa_length = sum(chain_lengths_arr)

    if total_aa_length== 0:
        return None
    else:
        # 返回字典
        return {
            "pdb_id": pdb_id,
            "entities_to_chains": entities_to_chains,
            "sequences": sequences_arr,
            "chain_lengths": chain_lengths_arr,
            "total_aa_length": total_aa_length,
        }
    
def parse_complex_fasta(fasta_path, one_maxlen=650,one_minlen=20):
    """
    解析包含多个复合物的 FASTA 文件。
    支持每个复合物具有两个或多个链的情况，按 ID 前缀合并。

    返回包含:
    - complex_id
    - sequences
    - chain_lengths
    - total_aa_length
    - entities_to_chains: [{entity_id: [chain_ids]}, ...]
    """
    complex_dict = defaultdict(list)

    for record in SeqIO.parse(fasta_path, "fasta"):
        if "_" not in record.id:
            continue

        complex_id = record.id.rsplit("_", 1)[0]
        description_parts = record.description.split("|")

        # 尝试提取链信息和 entity
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

    print(f"配对的复合物数量: {paired_count}")
    print(f"未配对的复合物数量: {unpaired_count}")
    return parsed_complexes


def extract_chain_ids(entities_to_chains):
    """链ID提取函数"""
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
    return list(dict.fromkeys(chain_ids))  # 去重保序


def save_to_pkl_complex(fasta_path, data_input, one_pdb_output_pkl_dir):
    """
    将 dict 或 list[dict] 格式的数据保存为单个/多个 pkl 文件
    """
    os.makedirs(one_pdb_output_pkl_dir, exist_ok=True)

    if isinstance(data_input, dict):
        data_list = [data_input]  # 单个包装成列表
    elif isinstance(data_input, list):
        data_list = data_input
    else:
        raise TypeError(f"data_input 类型错误: {type(data_input)}，应为 dict 或 list[dict]")

    for data in data_list:
        pdb_id = str(data.get('pdb_id', 'unknown')).upper()
        output_pkl_path = os.path.join(one_pdb_output_pkl_dir, f"{pdb_id}.pkl")

        # 防御性检查
        if not all(k in data for k in ['sequences', 'chain_lengths', 'total_aa_length', 'entities_to_chains']):
            print(f"[警告] 数据字段缺失，跳过 {pdb_id}")
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

        # print(f"保存 pkl 文件: {output_pkl_path}")
        df.to_pickle(output_pkl_path)
    return output_pkl_path

'''
    单链处理
'''

def parse_fasta_statistics_to_entity(fasta_path,one_maxlen=10000,one_minlen=1):
    """
    解析 FASTA 文件，提取链 ID、氨基酸序列及长度信息，并返回字典格式的数据。

    参数:
        fasta_path (str): FASTA 文件的路径。
    返回:
        dict: 包含统计信息的字典，结构如下：
    """
    # 获取 PDB ID（基于文件名）

    # 初始化存储变量
    line_chains = []
    # 解析 FASTA 文件
    for idx, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        chain_info = record.description.split("|")
        pdb_id = chain_info[0].split("_")[0].replace(" ", "").replace("\t", "").replace("\n", "")
        if len(chain_info[0].split("_"))>1:
            entity_id=chain_info[0].split("_")[1].replace(" ", "").replace("\t", "").replace("\n", "")
        else:
            entity_id=idx
        # print(entity_id)
        # 提取 Chain ID（处理可能的多个链 ID）
        if len(chain_info) >= 2:
            chain_ids =extract_chains(chain_info[1])
            # print(chain_ids)
            # chain_ids = [chain for chain in chain_ids if len(chain) < 2]
        else:
            chain_ids=['U']
        sequence = str(record.seq).replace(" ", "").replace("\t", "").replace("\n", "")
        chain_length = len(sequence)
        # if chain_length>one_maxlen or chain_length <one_minlen :
        #     continue
        # if len(chain_ids)==0 :
        #     continue
        pdb_id_entity=str(pdb_id)+"_"+str(entity_id)
        one_line_chain_dict={
            "PDB_ID": pdb_id_entity,
            "model":0,
            "chain_ids": chain_ids,
            "seqs": [sequence],
            "chain_lens": [chain_length],
            "total_seqLens": chain_length,
            "entity_id":entity_id
        }
        line_chains.append(one_line_chain_dict)
    if len(line_chains) == 0:
        return None
    else:
        return line_chains
    
def save_to_pkl(fasta_path,data_list, one_pdb_output_pkl_dir):
    """
    将字典列表转换为 Pandas DataFrame 并保存为 pkl 文件。
    
    参数:
        data_list (list): 包含字典的列表，每个字典表示一条记录。
        output_pkl_dir (str): 输出 pkl 文件的文件夹。
    """
    if not data_list:
        return None
    # 将 list of dicts 转换为 DataFrame
    df = pd.DataFrame(data_list)
    one_pdb_output_pkl_path = os.path.join(one_pdb_output_pkl_dir, f"{Path(fasta_path).stem}.pkl")
    # 保存为 pkl 文件
    df.to_pickle(one_pdb_output_pkl_path)
    return one_pdb_output_pkl_path



def merge_pkl_files(one_pdb_output_pkl_dir, final_pkl_path,temporary_deleted=True):
    """
    使用 pandas 直接读取并合并多个 pkl 文件，最终保存为一个 pkl 文件。
    """
    all_dataframes = []  # 存储 DataFrame

    # 获取所有 .pkl 文件
    pkl_files = [os.path.join(one_pdb_output_pkl_dir, f) 
                 for f in os.listdir(one_pdb_output_pkl_dir) if f.endswith(".pkl")]
    if not pkl_files :
        return  print("No valid data to merge.")
    for pkl_file in pkl_files:
        try:
            df = pd.read_pickle(pkl_file)  # 直接使用 pandas 读取
            if isinstance(df, pd.DataFrame):  
                all_dataframes.append(df)
        except Exception as e:
            print(f"Error reading {pkl_file}: {e}")

    if all_dataframes:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        merged_df.to_pickle(final_pkl_path)  # 以 pandas 格式保存 pkl
        print(f"all one pkl nums:{len(all_dataframes)},Final merged pkl saved at: {final_pkl_path}")

        # 删除小 pkl 文件
        if temporary_deleted is True :
            print("Temporary pkl files deleted.")
            for pkl_file in pkl_files:
                os.remove(pkl_file)
    else:
        print("No valid data to merge.")

def expand_chain_dicts(data_list):
    """
    展开包含多个 chain_ids 的字典，拆成多个单链字典。
    """
    expanded = []

    for item in data_list:
        chain_ids = item.get("chain_ids", [])
        # 若只有 1 个链，直接加入
        if len(chain_ids) <= 1:
            new_item = item.copy()
            # print(chain_ids[0])
            new_item["PDB_ID"] = new_item["PDB_ID"][:4]+ "_" +str(new_item["model"])+ "_" + chain_ids[0]
            expanded.append(new_item)
            continue

        # 若有多个链，逐个拆分
        for cid in chain_ids:
            new_item = item.copy()
            new_item["chain_ids"] = [cid]   # 只保留一个 chain
            new_item["PDB_ID"] = new_item["PDB_ID"][:4] + "_" + str(new_item["model"]) + "_" + cid
            expanded.append(new_item)
    return expanded


def process_pdb_file_task(input_fasta_file, one_pdb_output_pkl_dir, mode):
    """
    解析 fasta 并保存为 pkl，支持三种模式:
      - "complex" 复合物
      - "entity"  实体拆分
      - "single"  单链拆分，复制实体中的多条同源链为单链条目
    返回值: 处理成功返回 pkl 路径，失败返回 None
    """
    try:
        if mode == "complex":
            result = parse_fasta_statistics_to_complex(input_fasta_file)
            save_func = save_to_pkl_complex
        elif mode == "entity":
            result = parse_fasta_statistics_to_entity(input_fasta_file)
            # print(result)
            save_func = save_to_pkl
        elif mode == "single":
            result_entity = parse_fasta_statistics_to_entity(input_fasta_file)
            #将多条链进行进行拆链。一个实体的同源链拆解为多个条目，
            # print(result)
            result=expand_chain_dicts(result_entity)
            save_func = save_to_pkl
        else:
            raise ValueError(f"未知的模式: {mode}")
        
        if result:
            return save_func(input_fasta_file, result, one_pdb_output_pkl_dir)
        else:
            return None
    except Exception as e:
        print(f"处理失败: {input_fasta_file}, 错误: {e}")
        return None

def process_pdb_files_in_parallel(input_fasta_dir, one_pdb_output_pkl_dir, final_pkl_path, mode="complex",temporary_deleted=True, need_processed_list=None, num_workers=None):
    """
    多进程处理 fasta 文件，并合并结果，记录失败的 fasta 文件
    """
    os.makedirs(one_pdb_output_pkl_dir, exist_ok=True)
    # 生成日志文件路径，与 `final_pkl_path` 保持相同目录
    final_pkl_dir = os.path.dirname(final_pkl_path)
    os.makedirs(final_pkl_dir, exist_ok=True)  # 确保路径存在
    failed_fasta_log = os.path.join(final_pkl_dir, "failed_fasta.log")
    if num_workers is None:
        num_workers = math.ceil(cpu_count() * 0.85)  # 85% CPU 负载

    # 判断是文件夹还是单个fasta文件
    if os.path.isdir(input_fasta_dir):
        #文件夹模式（批量多个单链fasta）
        all_fasta_files = [f for f in os.listdir(input_fasta_dir) if f.lower().endswith((".fasta", ".fa", ".faa"))]
        print(f"总 fasta 文件数: {len(all_fasta_files)}")
        # 需要处理的文件列表（可选）
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
        # 情况2：单个合并fasta文件
        if input_fasta_dir.endswith(".fasta"):
            print(f"输入为汇总 FASTA 文件：{input_fasta_dir}")
            parsed_results = parse_fasta_statistics_to_entity(input_fasta_dir)
            final_pkl_dir = os.path.dirname(final_pkl_path)
            print(final_pkl_dir)
            if parsed_results is None:
                print(f"未能从 {input_fasta_dir} 解析到有效序列，记录至日志：{failed_fasta_log}")
                with open(failed_fasta_log, "a") as logf:
                    logf.write(f"{input_fasta_dir}\n")
            else:
                print(f"成功解析到 {len(parsed_results)} 条序列")
                
                save_to_pkl(final_pkl_path,parsed_results, os.path.dirname(final_pkl_path))
            return
        else:
            raise ValueError(f"输入路径既不是文件夹也不是 fasta 文件：{input_fasta_dir}")

    
    

    print(f"需处理的 fasta 文件数: {len(input_fasta_files)}")
    print(f"启用 {num_workers} 个进程")

    # 并行处理
    worker_func = partial(process_pdb_file_task, one_pdb_output_pkl_dir=one_pdb_output_pkl_dir, mode=mode)
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(worker_func, input_fasta_files), total=len(input_fasta_files)))

    # 过滤 None 结果
    failed_files = [f for f, r in zip(input_fasta_files, results) if r is None]
    successful_files = [r for r in results if r]

    # 记录失败 fasta
    if failed_files:
        with open(failed_fasta_log, "w") as log_file:
            log_file.write("\n".join(failed_files))
        print(f"失败的 fasta 数量: {len(failed_files)}，记录至 {failed_fasta_log}")

    print(f"成功的 fasta 数量: {len(successful_files)}")

    # 合并 pkl
    if successful_files:
        merge_pkl_files(one_pdb_output_pkl_dir, final_pkl_path,temporary_deleted)
    else:
        print("没有成功的 pkl 文件可合并，跳过合并步骤")


# ========== 主函数 ==========
if __name__ == "__main__":
    input_fasta_dir = "/opt/data/private/lyb/data_processed/temp/step1_fasta_download"
    one_pdb_output_pkl_dir = "/opt/data/private/lyb/data_processed/temp/step2/one_pdb_output_pkl_dir"
    final_pkl_path = "/opt/data/private/lyb/data_processed/temp/step2/final_pkl_path/all_pdb_chains_fasta_data.pkl"
    '''
      - "complex" 复合物
      - "entity"  实体拆分
      - "single"  单链拆分，复制实体中的多条同源链为单链条目
    '''
    mode = "single"  # 可选 "complex" 或 "single"或“entity”
    temporary_deleted=True  # 合并后是否删除每个pdb的临时 pkl 文件
    # 读取 PDB ID 文件（如果有）
    pdb_id_file = ""
    if os.path.exists(pdb_id_file):
        with open(pdb_id_file, "r") as f:
            need_processed_list = [line.strip() for line in f.readlines()]
        print(f"需要处理的 PDB ID 列表: {len(need_processed_list)} 个 ID")
    else:
        need_processed_list = None  # 处理整个文件夹

    process_pdb_files_in_parallel(input_fasta_dir, one_pdb_output_pkl_dir, final_pkl_path, mode, temporary_deleted,need_processed_list)
    # 测试使用
    # input_fasta_file=""
    # result_entity = parse_fasta_statistics_to_entity(input_fasta_file)
    #     #将多条链进行进行拆链。一个实体的同源链拆解为多个条目，
    #     # print(result)
    # result=expand_chain_dicts(result_entity)
    # save_func = save_to_pkl
    # save_func(input_fasta_file, result, one_pdb_output_pkl_dir)
