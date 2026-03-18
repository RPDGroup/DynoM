import os
import json
import pandas as pd
from Bio import SeqIO
from collections import defaultdict, Counter


# pkl文件
input_filepath = "/opt/data/private/lyb/test/competition_dir/final_output_final_merged.pkl"

#temp临时文件存放位置
one_json_output_dir = "/opt/data/private/lyb/test/competition_dir/temp/one_json"

#msa文件夹
precomputed_msa_dirs = "/opt/data/private/lyb/test/competition_dir/msa_dir"

#最终输出的json文件夹
merge_json_output_dir = "/opt/data/private/lyb/test/competition_dir/json"

#最终输出的json文件名
merge_json_output_name='test_3.json'


def process_fasta_and_match(input_msa_dir, target_sequences):
    """
    读取 MSA 目录中的 FASTA 文件，并匹配目标序列。
    
    Parameters:
    - input_msa_dir: PDB 对应的 MSA 文件夹路径
    - target_sequences: 需要匹配的目标序列列表
    
    Returns:
    - seqs_al_msa_path: 目标序列与其对应 MSA 目录的映射
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
        
        # 检查是否是 FASTA 标题行（以 '>' 开头）
        if line.startswith(">") and i + 1 < len(lines):
            sequence = lines[i + 1].strip()  # 获取序列
            
            if sequence in target_sequences:  # 匹配目标序列

                #指定MSA所在的存储路径
                # msa_dir=os.path.join("/home/tom/fsas/af3_repr/msa_174_4.5_1125_output/msa_result",os.path.basename(input_msa_dir),line.split("_")[1])
                #本地路径

                msa_dir = os.path.join(input_msa_dir, line.split("_")[1])  # 生成 MSA 目录路径
                seqs_al_msa_path[sequence] = msa_dir
    
    return seqs_al_msa_path

def generate_pkl_msa_input(pkl_filepath, output_dir, precomputed_msa_dirs, pairing_db):
    """
    解析 pkl 文件中的蛋白质数据，并生成 MSA 结果 JSON。
    
    Parameters:
    - pkl_filepath: 需要处理的 pkl 文件路径
    - output_dir: 结果输出目录
    - precomputed_msa_dirs: 预计算的 MSA 目录路径
    - pairing_db: 配对数据库名称（如 uniref100）
    
    Returns:
    - error_pdbid: 处理过程中出现错误的 PDB_ID 列表
    """
    df = pd.read_pickle(pkl_filepath)  # 读取 pkl 文件
    error_pdbid = []
    
    for _, row in df.iterrows():
        seq_counts = {}  # 存储去重后的序列及计数
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
        
        # 生成 JSON 数据
        json_data = {
            "sequences": sequences,
            "name": row['PDB_ID']
        }
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成 JSON 文件
        json_filename = f"{row['PDB_ID']}_msa_results.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        
        print(f"Saved MSA results for {row['PDB_ID']} to {json_filepath}")
    
    return error_pdbid

def generate_fasta_msa_input_complex(fasta_path,precomputed_msa_dirs,output_dir,pairing_db,need_process_PDBID=None):
    """解析 FASTA 文件，按 PDB ID（不含链信息）合并序列"""
    pdb_dict = defaultdict(list)
    # 解析 FASTA 文件
    for record in SeqIO.parse(fasta_path, "fasta"):
        pdb_id = record.id.split("_")[0].upper()  # 提取并大写 PDB ID
        if need_process_PDBID is not None and pdb_id not in need_process_PDBID:
            continue
        pdb_dict[pdb_id].append(str(record.seq))  # 追加序列
    # 构造输出列表
    error_pdbid=[]
    sequences = []
    print("需要构建的json文件数量：",len(pdb_dict))
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
        
        # 生成 JSON 数据
        json_data = {
            "sequences": sequences,
            "name": pdb_id
        }
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成 JSON 文件
        json_filename = f"{pdb_id}_msa_results.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        
        print(f"Saved MSA results for {pdb_id} to {json_filepath}")
    return error_pdbid

def merge_json_files(folder_path, output_file):
    """
    读取指定文件夹中的所有 JSON 文件，并合并为一个列表存储。
    
    Parameters:
    - folder_path: 包含 JSON 文件的目录路径
    - output_file: 合并后 JSON 文件的保存路径
    """
    combined_list = []
    
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):  # 筛选出 JSON 文件
            file_path = os.path.join(folder_path, file_name)  # 构建文件路径
            with open(file_path, 'r') as file:  # 读取 JSON 文件
                data = json.load(file)  # 加载 JSON 数据
                combined_list.append(data)  # 添加数据到列表
    
    # 保存合并后的 JSON 文件
    with open(output_file, "w") as json_file:
        json.dump(combined_list, json_file, indent=4)
    
    print(f"合并后的 JSON 文件已保存至 {output_file}")

if __name__ == "__main__":
    pairing_db = "uniref100"
    #以下不动
    need_process_PDBID_path=''
    # 运行 MSA 生成函数
    if input_filepath.endswith(".pkl") :
        error_pdbid = generate_pkl_msa_input(input_filepath, one_json_output_dir, precomputed_msa_dirs, pairing_db)
        print("Errors encountered with PDB IDs:", error_pdbid)
    elif input_filepath.endswith(".fasta") :
        if need_process_PDBID_path is not  None or len(need_process_PDBID_path)!= 0 :
            with open(need_process_PDBID_path, "r") as f:
                pdb_ids = [line.strip().upper() for line in f if line.strip()]
            need_process_PDBID=pdb_ids
        error_pdbid = generate_fasta_msa_input_complex(input_filepath, precomputed_msa_dirs,one_json_output_dir,  pairing_db,need_process_PDBID=need_process_PDBID)
        print("Errors encountered with PDB IDs:", error_pdbid)
    else:
        print("input pkl or fasta file")
        raise InterruptedError
    folder_path = one_json_output_dir
    merge_json_path = os.path.join(merge_json_output_dir, merge_json_output_name)
    
    # 执行合并操作
    merge_json_files(folder_path, merge_json_path)