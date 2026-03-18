import requests
import time
import os
import multiprocessing
import json
import tqdm

from urllib.parse import urlparse, parse_qs, unquote


# 设定 API 地址
SEARCH_API ="https://search.rcsb.org/rcsbsearch/v2/query"
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0'}
# 日志文件路径
LOG_FILE = "failed_pdb_download.log"

def parse_url_to_json(url, output_file="search_request.json"):
    """从 RCSB search URL 提取 request JSON，并初始化分页"""
    try:
        parsed_url = urlparse(url)
        query_dict = parse_qs(parsed_url.query)

        if "request" not in query_dict:
            raise ValueError("URL 中没有 request 参数")

        request_encoded = query_dict["request"][0]
        request_decoded = unquote(request_encoded)

        data = json.loads(request_decoded)

        #初始化分页（关键）
        if "request_options" not in data:
            data["request_options"] = {}

        data["request_options"]["paginate"] = {
            "start": 0,
            "rows": 10000   # RCSB 单次最大
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"解析成功，已保存到 {output_file}")

    except Exception as e:
        print(f"解析 URL 失败: {e}")

# 爬符合条件的PDB_name

def get_pdb_list(search_request, search_api=SEARCH_API, headers=HEADERS):
    """
    返回全量 PDB ID 列表（自动分页）
    """
    try:
        with open(search_request, "r") as f:
            request_data = json.load(f)

        all_pdb_ids = []
        start = 0
        rows = 10000

        while True:
            #更新分页参数
            request_data["request_options"]["paginate"] = {
                "start": start,
                "rows": rows
            }
            res = requests.post(search_api, headers=headers, json=request_data, verify=False)
            if res.status_code != 200:
                print(f"请求失败，状态码: {res.status_code}")
                print(res.text)
                break
            data = res.json()
            result_set = data.get("result_set", [])
            # 提取 identifier
            batch_ids = [item["identifier"] for item in result_set]
            all_pdb_ids.extend(batch_ids)
            print(f"当前批次: {len(batch_ids)}，累计: {len(all_pdb_ids)}")
            # 结束条件
            if len(result_set) < rows:
                break
            start += rows
            #防止请求过快
            time.sleep(0.1)
        return all_pdb_ids

    except Exception as e:
        print(f"获取 PDB 列表失败: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        return None
    except json.JSONDecodeError:
        print("JSON 解析失败，请检查 API 返回的内容。")
        return None

# 发送网络请求并处理异常
def safe_request(url, retries=0,request_interval = 0.1,max_retries = 2):
    try:
        time.sleep(request_interval)  # 限制请求速率# 请求间隔和，目的：防止请求过多无法访问api
        response = requests.get(url, timeout=20,proxies={"http": None, "https": None})
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429 and retries < max_retries:  # 最大重试次数,请求过多
            time.sleep(1)  # 稍后重试
            print(f"请求 {url} 失败: {e}，第{retries}请求，下一次请求")
            return safe_request(url, retries + 1)
        else:
            print(f"HTTP Error for {url}: {e}")
    except requests.RequestException as e:
        print(f"Error while requesting {url}: {e}")
        return None

def download_fasta_file(pdb_id, pdb_folder):
    pdb_url = f'https://www.rcsb.org/fasta/entry/{pdb_id}'
    if not os.path.exists(pdb_folder):
            os.makedirs(pdb_folder)
    response = safe_request(pdb_url)
    if response is not None and hasattr(response, 'status_code'):
        if response.status_code == 200:
            pdb_file_path = os.path.join(pdb_folder, f'{pdb_id}.fasta')
            with open(pdb_file_path, 'wb') as pdb_file:
                pdb_file.write(response.content)
            return True
        else:
            return False
    else:
        return False
        

def get_fasta_wrapper(process_num,pdb_ids, pdb_data_directory, progress_queue):
    # print("ID of process num {} : {} ".format(process_num,os.getpid(),pdb_ids[0]))
    success_count = 0
    for pdb_id in pdb_ids:
        if  download_fasta_file(pdb_id, pdb_data_directory):
            success_count += 1
        progress_queue.put(1)
    return success_count

def process_fasta_data(filtered_result_set, pdb_data_directory, cut_threshold=100):
    """使用 75% CPU 核心数进行多进程 FASTA 下载，并显示进度条"""
    num_cpu = int(multiprocessing.cpu_count() * 0.75)  # 75% CPU
    num_processes = min(num_cpu, (len(filtered_result_set) // cut_threshold) + 1)

    processes = []
    progress_queue = multiprocessing.Queue()
    progress_bar = tqdm.tqdm(total=len(filtered_result_set), desc="Downloading FASTA", unit="file")

    for i in range(num_processes):
        start_idx = i * cut_threshold
        end_idx = min((i + 1) * cut_threshold, len(filtered_result_set))
        pdb_ids = filtered_result_set[start_idx:end_idx]

        process = multiprocessing.Process(target=get_fasta_wrapper, args=(i+1, pdb_ids, pdb_data_directory, progress_queue))
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

def unsuccessful_downloaded_statistics(fasta_folder, filtered_result_set):
    # 使用集合存储文件名，提高查找速度
    fasta_file_count = {f.split('.')[0].lower() for f in os.listdir(fasta_folder) if f.endswith('.fasta')}
    
    print("应下载 fasta文件数量：", len(filtered_result_set))
    print("文件夹中 fasta 数量：", len(fasta_file_count))
    # 计算未下载的 PDB 文件
    again_downloaded_files = list(set(filtered_result_set) - fasta_file_count)
    print("未下载 pdb 数量：", len(again_downloaded_files),' 重新下载：')
    return again_downloaded_files

def rename_files_to_lower(folder_path):
    """
    将指定文件夹中的所有文件名转换为小写。
    
    参数:
    - folder_path (str): 目标文件夹路径

    返回:
    - None
    """
    if not os.path.exists(folder_path):
        print(f"目录不存在: {folder_path}")
        return
    
    try:
        for filename in os.listdir(folder_path):
            lower_filename = filename.lower()  # 转换为小写
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, lower_filename)

            # 避免重命名已是小写的文件
            if old_path != new_path:
                os.rename(old_path, new_path)     
        print("所有文件重命名完成！")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    url="https://www.rcsb.org/search?request=%7B%22query%22%3A%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_entry_info.resolution_combined%22%2C%22operator%22%3A%22less_or_equal%22%2C%22negation%22%3Afalse%2C%22value%22%3A5%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_accession_info.deposit_date%22%2C%22operator%22%3A%22greater_or_equal%22%2C%22negation%22%3Afalse%2C%22value%22%3A%222026-01-01%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_accession_info.deposit_date%22%2C%22operator%22%3A%22less_or_equal%22%2C%22negation%22%3Afalse%2C%22value%22%3A%222026-03-01%22%7D%7D%5D%2C%22logical_operator%22%3A%22and%22%7D%5D%2C%22label%22%3A%22text%22%7D%5D%7D%2C%22return_type%22%3A%22entry%22%2C%22request_options%22%3A%7B%22paginate%22%3A%7B%22start%22%3A0%2C%22rows%22%3A25%7D%2C%22results_content_type%22%3A%5B%22experimental%22%5D%2C%22sort%22%3A%5B%7B%22sort_by%22%3A%22score%22%2C%22direction%22%3A%22desc%22%7D%5D%2C%22scoring_strategy%22%3A%22combined%22%7D%2C%22request_info%22%3A%7B%22query_id%22%3A%229b09d8de01a914e1bf6b1cfe10d51866%22%7D%7D"
    fasta_save_folder = "/opt/data/private/lyb/data_processed/temp/step1_fasta_download"
    # Step 1: 获取 PDB 列表
    json_path = os.path.join(fasta_save_folder, "search_request.json")
    parse_url_to_json(url,output_file=json_path)
    pdb_list = get_pdb_list(json_path)
    print("符合条件的 PDB 数量：", len(pdb_list))
    # import pandas as pd
    # excel_path='/home/tom/fsas/data_16t/ab-ag_data/sabdab_summary_all_lyb1.1.tsv'
    # excel_data=pd.read_csv(excel_path, sep='\t')
    # 只保留 pdb 列每个字符串的后四位
    # pdb_list =excel_data["pdb"].astype(str).tolist()
    pdb_list = list(set([pdb.lower() for pdb in pdb_list]))
    # 输出结果
    rename_files_to_lower(fasta_save_folder)
    # Step 2: 下载 FASTA 文件
    pdb_list=unsuccessful_downloaded_statistics(fasta_save_folder, pdb_list)
    process_fasta_data(pdb_list, fasta_save_folder)

    # Step 3: 文件检查 & 重试下载
    rename_files_to_lower(fasta_save_folder)

    retry_count = 0
    max_retries = 2
    retry_threshold = 0.02  # 2% 失败率
    cpu_usage = 0.5  # 50% CPU 用于重试

    while retry_count < max_retries:
        failed_pdbs = unsuccessful_downloaded_statistics(fasta_save_folder, pdb_list)
        failed_ratio = len(failed_pdbs) / len(pdb_list)

        if failed_ratio <= retry_threshold:
            print(f"下载成功率达到 {100 - failed_ratio * 100:.2f}%，无需重试！")
            break

        print(f"失败率 {failed_ratio * 100:.2f}%，重试下载...")
        num_cpu = int(multiprocessing.cpu_count() * cpu_usage)
        process_fasta_data(failed_pdbs, fasta_save_folder, cut_threshold=num_cpu)

        retry_count += 1
        cpu_usage *= 0.9  # 降低 CPU 负载

        # 检查是否达到停止条件
        new_failed_pdbs = unsuccessful_downloaded_statistics(fasta_save_folder, pdb_list)
        if len(new_failed_pdbs) == len(failed_pdbs):
            print("两次失败数量相同，停止重试")
            break
    failed_pdbs = unsuccessful_downloaded_statistics(fasta_save_folder, pdb_list)
    # 记录最终失败的 PDB ID
    if failed_pdbs:
        log_path=os.path.join(fasta_save_folder,LOG_FILE)
        with open(log_path, "w") as log_file:
            log_file.write("\n".join(failed_pdbs))
        print(f"下载失败的 PDB 记录在 {log_path}")



# # 遍历所有 .fasta 文件删除空文件
# for file_name in os.listdir(fasta_save_folder):
#     if file_name.endswith(".fasta"):
#         file_path = os.path.join(fasta_save_folder, file_name)
    
#         # 检查文件内容是否为空或仅包含空白字符
#         with open(file_path, "r") as f:
#             content = f.read().strip()

#         if not content:  # 空文件或仅空格、换行
#             os.remove(file_path)
#             print(f"已删除空fasta文件: {file_name}")