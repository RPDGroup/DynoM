import os
import pandas as pd
import requests
import tqdm

import time
import multiprocessing

LOG_FILE="failed_pdb_download.log"
def get_existing_pdb_ids(folder, extension=".cif.gz"):
    """获取指定文件夹内已存在的 PDB ID"""
    if not os.path.exists(folder):
        print(f"目录 {folder} 不存在！")
        return set()
    return {f.split('.')[0].lower() for f in os.listdir(folder) if f.endswith(extension)}

def get_pdb_ids_from_pkl(pkl_file):
    """从 PKL 文件中提取 PDB ID（长度为 4）"""
    if not os.path.exists(pkl_file):
        print(f"PKL 文件 {pkl_file} 不存在！")
        return set()
    pkldata = pd.read_pickle(pkl_file)
    # 解析 PDB_ID 并筛选长度为 4 的 ID
    pdb_ids = set()
    for pdb_entry in pkldata["PDB_ID"]:
        pdb_id = pdb_entry[:4].lower()  # 只取 `_` 前面部分
        if len(pdb_id) == 4:
            pdb_ids.add(pdb_id)

    return pdb_ids


def delete_unwanted_mmcif_files(need_delete_pdbs, folder):
    """删除指定 PDB ID 的 MMCIF 文件"""
    if not os.path.exists(folder):
        print(f"保存mmcif数据的目录 {folder} 不存在！")
        return
    
    for pdb_id in tqdm.tqdm(need_delete_pdbs, desc="Deleting unwanted files"):
        file_path = os.path.join(folder, f"{pdb_id}.cif.gz")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除: {file_path}")
        else:
            print(f"文件不存在: {file_path}")

# ========== 下载缺失的 MMCIF 文件 ==========
# 发送网络请求并处理异常
def safe_request(url, retries=0,request_interval = 0.1,max_retries = 3):
    try:
        time.sleep(request_interval)  # 限制请求速率# 请求间隔和，目的：防止请求过多无法访问api
        response = requests.get(url, timeout=20,proxies={"http": None, "https": None})
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429 and retries < max_retries:  # 最大重试次数,请求过多
            time.sleep(5)  # 稍后重试
            return safe_request(url, retries + 1)
        else:
            print(f"HTTP Error for {url}: {e}")
    except requests.RequestException as e:
        print(f"Error while requesting {url}: {e}")
        return None

def download_mmcif_file(file_id, mmcif_save_folder):
    url = f'https://files.rcsb.org/download/{file_id}.cif.gz'
    if not os.path.exists(mmcif_save_folder):
            os.makedirs(mmcif_save_folder)
    response = safe_request(url)
    if response is not None and hasattr(response, 'status_code'):
        if response.status_code == 200:
            mmcif_file_path = os.path.join(mmcif_save_folder, f'{file_id.lower() }.cif.gz')
            with open(mmcif_file_path, 'wb') as file:
                file.write(response.content)
            return True
        else:
            return False
    else:
        return False


def get_mmcif_wrapper(process_num,mmcif_ids, mmcif_data_directory, progress_queue):
    # print("ID of process num {} : {} ".format(process_num,os.getpid(),pdb_ids[0]))
    success_count = 0
    for mmcif_id in mmcif_ids:
        if  download_mmcif_file(mmcif_id, mmcif_data_directory):
            success_count += 1
        progress_queue.put(1)
    return success_count

def process_mmcif_data(filtered_result_set, mmcif_data_directory, cut_threshold=100, cpu_usage=0.75):
    """使用 75% CPU 核心数进行多进程 下载，并显示进度条"""
    num_cpu = int(multiprocessing.cpu_count() * cpu_usage)  # 75% CPU
    num_processes = min(num_cpu, (len(filtered_result_set) // cut_threshold) + 1)

    processes = []
    progress_queue = multiprocessing.Queue()
    progress_bar = tqdm.tqdm(total=len(filtered_result_set), desc="Downloading MMCIF FILES", unit="file")

    for i in range(num_processes):
        start_idx = i * cut_threshold
        end_idx = min((i + 1) * cut_threshold, len(filtered_result_set))
        pdb_ids = filtered_result_set[start_idx:end_idx]

        process = multiprocessing.Process(target=get_mmcif_wrapper, args=(i+1, pdb_ids, mmcif_data_directory, progress_queue))
        processes.append(process)

    for process in processes:
        process.start()

    # 实时更新进度条
    completed = 0
    while completed < len(filtered_result_set):
        progress_queue.get()
        completed += 1
        progress_bar.update(1)

    for process in processes:
        process.join()

    progress_bar.close()

def unsuccessful_downloaded_statistics(mmcif_folder, filtered_result_set):
    """检查未成功下载的 MMCIF 文件"""
    downloaded_files = {f.split('.')[0].lower() for f in os.listdir(mmcif_folder) if f.endswith('.cif.gz')}
    
    print("应下载 MMCIF 文件数量:", len(filtered_result_set))
    print("现有 MMCIF 文件数量:", len(downloaded_files))

    # 计算未下载成功的 PDB
    again_downloaded_files = list(set(filtered_result_set) - downloaded_files)
    print("未下载数量:", len(again_downloaded_files))
    
    return again_downloaded_files

def main():
    """主函数：多轮下载 MMCIF 文件，失败重试"""
    mmcif_save_filefolder = "/opt/data/private/lyb/data_processed/temp/step3_mmcifgz_download"
    pkl_file = "/opt/data/private/lyb/data_processed/temp/step2/final_pkl_path/all_pdb_chains_fasta_data.pkl"

    # 获取需要下载 & 需要删除的 PDB ID
    result_set = get_pdb_ids_from_pkl(pkl_file)
    existing_mmcif_files = {f.split('.')[0].lower() for f in os.listdir(mmcif_save_filefolder) if f.endswith('.cif.gz')}
    
    need_download_pdbs = list(result_set - existing_mmcif_files)
    need_delete_pdbs = list(existing_mmcif_files - result_set)

    print(f"需要下载 MMCIF 文件数: {len(need_download_pdbs)}")
    print(f"需要删除 MMCIF 文件数: {len(need_delete_pdbs)}")

    # Step 1: 删除无用的 MMCIF 文件
    delete_unwanted_mmcif_files(need_delete_pdbs, mmcif_save_filefolder)

    # Step 2: 下载缺失的 MMCIF 文件
    retry_count = 0
    max_retries = 5
    retry_threshold = 0.01  # 2% 失败率
    cpu_usage = 0.75  # 初始使用 75% CPU

    while retry_count < max_retries:
        process_mmcif_data(need_download_pdbs, mmcif_save_filefolder, cpu_usage=cpu_usage)
        # Step 3: 检查下载成功率
        failed_pdbs = unsuccessful_downloaded_statistics(mmcif_save_filefolder, need_download_pdbs)
        failed_ratio = len(failed_pdbs) / len(need_download_pdbs) if need_download_pdbs else 0
        if failed_ratio <= retry_threshold:
            print(f"下载成功率达到 {100 - failed_ratio * 100:.2f}%，无需重试！")
            break
        print(f"失败率 {failed_ratio * 100:.2f}%，重试下载...")
        retry_count += 1
        cpu_usage *= 0.9  # 每次减少 10% CPU 负载
        # 如果连续两次失败数量相同，则停止
        new_failed_pdbs = unsuccessful_downloaded_statistics(mmcif_save_filefolder, need_download_pdbs)
        if len(new_failed_pdbs) == len(failed_pdbs):
            print("两次失败数量相同，停止重试")
            break

    # Step 4: 记录最终下载失败的 PDB ID
    # 记录最终失败的 PDB ID
    if failed_pdbs:
        log_path=os.path.join(mmcif_save_filefolder,LOG_FILE)
        with open(log_path, "w") as log_file:
            log_file.write("\n".join(failed_pdbs))
        print(f"下载失败的 PDB 记录在 {log_path}")

if __name__ == "__main__":
    main()