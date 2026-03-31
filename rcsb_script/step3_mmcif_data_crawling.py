import os
import pandas as pd
import requests
import tqdm
import argparse
import time
import multiprocessing
import json

LOG_FILE="failed_pdb_download.log"
def get_existing_pdb_ids(folder, extension=".cif.gz"):
    """Get existing PDB IDs in the specified folder"""
    if not os.path.exists(folder):
        print(f"Directory {folder} does not exist!")
        return set()
    return {f.split('.')[0].lower() for f in os.listdir(folder) if f.endswith(extension)}


def get_pdb_ids_from_pkl(pkl_file):
    """Extract PDB IDs (length = 4) from a PKL file"""
    if not os.path.exists(pkl_file):
        print(f"PKL file {pkl_file} does not exist!")
        return set()

    pkldata = pd.read_pickle(pkl_file)

    pdb_ids = set()
    for pdb_entry in pkldata["PDB_ID"]:
        pdb_id = pdb_entry[:4].lower()
        if len(pdb_id) == 4:
            pdb_ids.add(pdb_id)

    return pdb_ids


def delete_unwanted_mmcif_files(need_delete_pdbs, folder):
    """Delete MMCIF files for specified PDB IDs"""
    if not os.path.exists(folder):
        print(f"Directory for mmcif data {folder} does not exist!")
        return
    
    for pdb_id in tqdm.tqdm(need_delete_pdbs, desc="Deleting unwanted files"):
        file_path = os.path.join(folder, f"{pdb_id}.cif.gz")
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")


def safe_request(url, retries=0,request_interval = 0.1,max_retries = 3):
    try:
        time.sleep(request_interval) 
        response = requests.get(url, timeout=20,proxies={"http": None, "https": None})
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429 and retries < max_retries:
            time.sleep(5)
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
    success_count = 0
    for mmcif_id in mmcif_ids:
        if  download_mmcif_file(mmcif_id, mmcif_data_directory):
            success_count += 1
        progress_queue.put(1)
    return success_count

def process_mmcif_data(filtered_result_set, mmcif_data_directory, cut_threshold=100, cpu_usage=0.75):
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

    completed = 0
    while completed < len(filtered_result_set):
        progress_queue.get()
        completed += 1
        progress_bar.update(1)

    for process in processes:
        process.join()

    progress_bar.close()

def unsuccessful_downloaded_statistics(mmcif_folder, filtered_result_set):
    """Check unsuccessfully downloaded MMCIF files"""
    downloaded_files = {f.split('.')[0].lower() for f in os.listdir(mmcif_folder) if f.endswith('.cif.gz')}
    
    print("Expected number of MMCIF files:", len(filtered_result_set))
    print("Number of existing MMCIF files:", len(downloaded_files))

    # Compute PDB IDs that failed to download
    again_downloaded_files = list(set(filtered_result_set) - downloaded_files)
    print("Number of missing files:", len(again_downloaded_files))
    
    return again_downloaded_files

def main(args):
    mmcif_save_filefolder = args.mmcif_dir
    pkl_file = args.pkl_file
    max_retries = args.max_retries
    retry_threshold = args.retry_threshold
    cpu_usage = args.cpu_usage
    os.makedirs(mmcif_save_filefolder, exist_ok=True)
    result_set = get_pdb_ids_from_pkl(pkl_file)
    existing_mmcif_files = {
        f.split('.')[0].lower()
        for f in os.listdir(mmcif_save_filefolder)
        if f.endswith('.cif.gz')
    }
    need_download_pdbs = list(result_set - existing_mmcif_files)
    need_delete_pdbs = list(existing_mmcif_files - result_set)
    print(f"PDBs to download: {len(need_download_pdbs)}")
    print(f"PDBs to delete: {len(need_delete_pdbs)}")
    delete_unwanted_mmcif_files(need_delete_pdbs, mmcif_save_filefolder)
    retry_count = 0
    while retry_count < max_retries:
        process_mmcif_data(
            need_download_pdbs,
            mmcif_save_filefolder,
            cpu_usage=cpu_usage
        )
        failed_pdbs = unsuccessful_downloaded_statistics(
            mmcif_save_filefolder,
            need_download_pdbs
        )
        failed_ratio = len(failed_pdbs) / len(need_download_pdbs) if need_download_pdbs else 0
        if failed_ratio <= retry_threshold:
            print(f"Success rate reached {100 - failed_ratio * 100:.2f}%, no retry needed.")
            break
        print(f"Failure rate {failed_ratio * 100:.2f}%, retrying...")
        retry_count += 1
        cpu_usage *= 0.9  # reduce CPU load
        new_failed_pdbs = unsuccessful_downloaded_statistics(
            mmcif_save_filefolder,
            need_download_pdbs
        )
        if len(new_failed_pdbs) == len(failed_pdbs):
            print("Failure count unchanged, stopping retries.")
            break
    if failed_pdbs:
        log_path = os.path.join(mmcif_save_filefolder, LOG_FILE)
        with open(log_path, "w") as log_file:
            log_file.write("\n".join(failed_pdbs))

        print(f"Failed PDB IDs saved to: {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3: Download mmCIF (.cif.gz) files for PDB IDs derived from Step 2 PKL data."
    )
    parser.add_argument(
        "--mmcif_dir",
        type=str,
        required=True,
        help="Directory to store downloaded mmCIF (.cif.gz) files"
    )
    parser.add_argument(
        "--pkl_file",
        type=str,
        required=True,
        help="Path to PKL file generated in Step 2 (contains PDB IDs)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts (default: 5)"
    )
    parser.add_argument(
        "--retry_threshold",
        type=float,
        default=0.01,
        help="Acceptable failure rate threshold (default: 0.01)"
    )
    parser.add_argument(
        "--cpu_usage",
        type=float,
        default=0.75,
        help="Initial CPU usage ratio for parallel download (default: 0.75)"
    )
    args = parser.parse_args()
    main(args)