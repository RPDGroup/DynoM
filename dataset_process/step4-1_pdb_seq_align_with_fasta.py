from Bio import PDB                          # 解析PDB结构文件
from Bio.SeqUtils import seq1               # 将三字母氨基酸转换为一字母代码
from Bio import pairwise2                   # 进行序列间的全局/局部比对
import re 

from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import json
import os


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

对齐结果：
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

#  从pdb文件中提取序列
def extract_model0_sequences(pdb_file):
    """
    从多模型 PDB 文件中，提取第 0 号 model 的氨基酸单字母序列并返回一个 dict，
    key 为链 ID，value 为对应的氨基酸单字母序列。
    """
    # 1. 初始化解析器
    parser = PDB.PDBParser(QUIET=True)
    
    # 2. 读取 PDB 文件，得到结构对象
    structure = parser.get_structure("my_structure", pdb_file)
    
    # 3. 获取第 0 号 model（Biopython 中模型计数从 0 开始）
    model0 = structure[0]

    # 4. 遍历 model0 中的链与氨基酸，并用三字母→单字母的方式收集序列
    chain_seq_dict = {}
    for chain in model0:
        residues = []
        for residue in chain:
            # 判断是不是标准氨基酸
            if PDB.is_aa(residue, standard=True):
                # residue.resname 是三字母（例如 'LYS'、'ALA' 等）
                # seq1() 用于将三字母氨基酸转换为单字母表示
                residues.append(seq1(residue.resname))
        # 将收集到的氨基酸拼接为一条序列
        chain_seq_dict[chain.id] = "".join(residues)

    return chain_seq_dict,structure


#提取fasta中链名 未用
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
            auth_match = re.search(r'\[auth ([\w\d]+)\]', chain)
            
            # 如果链名长度大于 1 或者 auth 别名的长度大于 1，则跳过
            if (len(chain_clean) > 1) and (auth_match and len(auth_match.group(1)) > 1):
                continue
            
            # 用 auth 别名替换，否则用原值
            chains.append(str(auth_map.get(chain_clean, chain_clean)))
    
    return list(set(chains)) if chains else []  # 只返回有效的链名

#  从fasta文件中提取序列
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
                # 提取 > 开头后的第一个字段作为 ID（如 1A2Y_1）
                chain_info = line[1:].split("|")
                pdbid_entityid=chain_info[0].strip()
                # 提取 Chain ID（处理可能的多个链 ID）
                # if len(chain_info) >= 2:
                    # chain_ids = extract_chains(chain_info[1])
                # chain_ids_str="".join(current_seq)
                current_id=pdbid_entityid
                current_seq = []
            else:
                current_seq.append(line)
        # 添加最后一个序列
        if current_id and current_seq:
            seq_dict[current_id] = "".join(current_seq)
    
    return seq_dict


def align_pdb_with_fasta(pdb_query_seqs: dict, fasta_search_seqs: dict):
    aligned_results = {}

    for pdb_chain, pdb_seq in pdb_query_seqs.items():
        best_score = float('-inf')
        best_alignment = None
        best_fasta_id = None

        for fasta_id, fasta_seq in fasta_search_seqs.items():
            alignments = pairwise2.align.localms(pdb_seq, fasta_seq, 2, -1, -0.5, -0.1)

            if not alignments:
                continue  # 忽略没有比对结果的情况

            score = alignments[0].score
            if score > best_score:
                best_score = score
                best_alignment = alignments[0]
                best_fasta_id = fasta_id

        # 检查是否存在有效比对
        if best_alignment:
            aligned_results[pdb_chain] = {
                "pdb_seq": best_alignment.seqA,
                "fasta_seq": best_alignment.seqB,
                "fasta_id": best_fasta_id,
                "score": best_score
            }
        else:
            # 如果该链比对失败，也记录在结果中，便于调试
            aligned_results[pdb_chain] = {
                "pdb_seq": "",
                "fasta_seq": "",
                "fasta_id": None,
                "score": None
            }

    return aligned_results


#根据 PDB 对齐序列两端的缺失（'-'）提取 fasta 中匹配区域，获取到截取后使用的序列
def extract_core_seq_by_terminal_gaps(pdb_aln, fasta_aln):
    """
    根据 PDB 对齐序列两端的缺失（'-'），提取 fasta 中匹配区域。
    中间缺失部分不处理，保留完整残基。
    """
    assert len(pdb_aln) == len(fasta_aln), "对齐长度不一致"

    # 找到 PDB 非'-' 的起始与结束索引
    start = next((i for i, c in enumerate(pdb_aln) if c != '-'), None)
    end = len(pdb_aln) - next((i for i, c in enumerate(reversed(pdb_aln)) if c != '-'), None)

    # 从 FASTA 中对应位置提取序列（保留其中的 gap）
    core_seq_with_gap = fasta_aln[start:end]
    # 去除 gap 得到最终序列
    core_seq = core_seq_with_gap.replace('-', '')
    return core_seq


#保存对齐结果
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

        # 构造 pkl 表格数据
        chain_ids.append(chain_name)
        seqs.append(core_seq)
        chain_lens.append(len(core_seq))

    total_seqLens = sum(chain_lens)

    # 构造 DataFrame，注意用 [dict] 创建单行表
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
    

##单个数据处理
# def main():
#     parser = argparse.ArgumentParser(description="Align PDB sequences to FASTA and extract core segments.")
#     parser.add_argument("--pdb", required=True, help="Path to the PDB file")
#     parser.add_argument("--fasta", required=True, help="Path to the FASTA file")
#     parser.add_argument("--json_save_dir", required=True, help="Directory to save the output JSON file")
#     parser.add_argument("--pkl_save_dir", required=True, help="Directory to save the output PKL file")
#     args = parser.parse_args()
#     pdb_path = args.pdb
#     fasta_path = args.fasta
#     json_output_dir = args.json_save_dir
#     pkl_output_dir = args.pkl_save_dir
#     # 提取PDB序列和FASTA序列
#     pdb_seqs = extract_model0_sequences(pdb_path)
#     fasta_seqs = fasta_to_dict(fasta_path)
#     # 比对
#     aligned_results = align_pdb_with_fasta(pdb_seqs, fasta_seqs)

#     # 提取文件名前四位为pdbid
#     pdbid = os.path.basename(pdb_path)[:4].upper()

#     # 保存为 JSON 和 PKL
#     convert_alignment_to_json_and_save(
#         aligned_results,
#         output_json_dir=json_output_dir,
#         output_pkl_dir=pkl_output_dir,
#         pdbid=pdbid
#     )

# if __name__ == "__main__":
#     main()


def process_alignment(pdb_path, fasta_path, json_output_dir, pkl_output_dir):
    
    try:
        # 提取PDB和FASTA序列
        pdb_chain_seq_dict,structure = extract_model0_sequences(pdb_path)
        fasta_seqs = fasta_to_dict(fasta_path)
        # 比对
        aligned_results = align_pdb_with_fasta(pdb_chain_seq_dict, fasta_seqs)
        # print("比对 成功")
        # 获取PDB ID
        pdbid = os.path.basename(pdb_path)[:4].upper()

        # 保存结果
        convert_alignment_to_json_and_save(
            aligned_results,
            output_json_dir=json_output_dir,
            output_pkl_dir=pkl_output_dir,
            pdbid=pdbid
        )
        # print(f"[成功] {pdbid} 处理完成")
        return pdbid, "success"

    except Exception as e:
        print(f"[失败] {pdb_path}: {e}")
        return os.path.basename(pdb_path)[:4].upper(), "fail"




def run_multiprocess_alignment(pdb_fasta_pairs, json_output_dir, pkl_output_dir, num_processes=8, log_path="alignment_results.log"):

    # 封装参数为多进程兼容格式
    args_list = [
        (pdb_path, fasta_path, json_output_dir, pkl_output_dir)
        for pdb_path, fasta_path in pdb_fasta_pairs
    ]

    # 进度条 + 多进程执行
    results = []
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.starmap(process_alignment, args_list), total=len(args_list), desc="Processing Alignments"):
            results.append(result)

    # 写入日志文件
    with open(log_path, "w") as log_file:
        for pdbid, status in results:
            log_file.write(f"{pdbid}\t{status}\n")

    return results


#多进程数据处理
def main():

    def get_pdb_ids_from_dir(directory):
        with os.scandir(directory) as entries:
            return list({
                entry.name[0:4].lower()
                for entry in entries
                if entry.is_file()
            })


    def get_pdb_fasta_pairs(dir_path, pdb_id_list, fasta_dir):
        pdb_fasta_pairs = []
        for pdbid in pdb_id_list:
            pdbid_upper = pdbid.upper()
            pdb_path = os.path.join(
                dir_path,
                f"{pdbid_upper}.pdb"
            )
            fasta_path = os.path.join(fasta_dir, f"{pdbid.lower()}.fasta")
            pdb_fasta_pairs.append((pdb_path, fasta_path))
        return pdb_fasta_pairs
    print("程序开始，遍历经过截取的pdb文件id")
    dir_path = "/opt/data/private/lyb/monomer_data_process/processed_testdata_align_rcsb_1-7/output_data/processed_pdb_file"
    fasta_dir = "/opt/data/private/lyb/monomer_data_process/processed_testdata_align_rcsb_1-7/middle_data/fasta_file"
    pdb_id_list = get_pdb_ids_from_dir(dir_path)  # 你已有的函数
    print(pdb_id_list[:5])
    print(f"需要对齐的截取过的pdb文件有： {len(pdb_id_list)}个")
    pdb_fasta_pairs = get_pdb_fasta_pairs(dir_path, pdb_id_list, fasta_dir)
    print(f"得到所有需要对齐的任务路径，共{len(pdb_fasta_pairs)}个")
    json_dir = "/opt/data/private/lyb/monomer_data_process/processed_testdata_align_rcsb_1-7/middle_data/align_json_output_dir"
    pkl_dir = "/opt/data/private/lyb/monomer_data_process/processed_testdata_align_rcsb_1-7/middle_data/one_pkl_output"
    # 保证输出文件夹存在（如不存在则创建）
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)
    print("开始处理：")
    results = run_multiprocess_alignment(
        pdb_fasta_pairs,
        json_output_dir=json_dir,
        pkl_output_dir=pkl_dir,
        num_processes=32,
        log_path=os.path.join(json_dir,"alignment_log.txt")
    )
    print("处理完成")

if __name__ == "__main__":
    main()