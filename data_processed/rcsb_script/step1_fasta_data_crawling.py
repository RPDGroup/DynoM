
import os
import time
import json
import argparse
import multiprocessing
from urllib.parse import urlparse, parse_qs, unquote

import requests
from tqdm import tqdm
SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:95.0) Gecko/20100101 Firefox/95.0"
}
LOG_FILE = "failed_pdb_download.log"

def parse_url_to_json(url, output_file="search_request.json"):
    try:
        parsed_url = urlparse(url)
        query_dict = parse_qs(parsed_url.query)
        if "request" not in query_dict:
            raise ValueError("No request parameter found in URL")
        request_encoded = query_dict["request"][0]
        request_decoded = unquote(request_encoded)
        data = json.loads(request_decoded)
        if "request_options" not in data:
            data["request_options"] = {}
        data["request_options"]["paginate"] = {
            "start": 0,
            "rows": 10000
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Parsing successful, saved to {output_file}")
    except Exception as e:
        print(f"Failed to parse URL: {e}")


def get_pdb_list(search_request, search_api=SEARCH_API, headers=HEADERS):
    try:
        with open(search_request, "r") as f:
            request_data = json.load(f)
        all_pdb_ids = []
        start = 0
        rows = 10000
        while True:
            request_data["request_options"]["paginate"] = {
                "start": start,
                "rows": rows
            }
            res = requests.post(search_api, headers=headers, json=request_data, verify=False)
            if res.status_code != 200:
                print(f"Request failed, status code: {res.status_code}")
                print(res.text)
                break
            data = res.json()
            result_set = data.get("result_set", [])
            batch_ids = [item["identifier"] for item in result_set]
            all_pdb_ids.extend(batch_ids)
            print(f"Current batch: {len(batch_ids)}, Total: {len(all_pdb_ids)}")
            if len(result_set) < rows:
                break
            start += rows
            time.sleep(0.1)
        return all_pdb_ids

    except Exception as e:
        print(f"Failed to retrieve PDB list: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return None
    except json.JSONDecodeError:
        print("JSON parsing failed, please check API response.")
        return None

def safe_request(url, retries=0, request_interval=0.1, max_retries=2):
    try:
        time.sleep(request_interval)
        response = requests.get(url, timeout=20, proxies={"http": None, "https": None})
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429 and retries < max_retries:
            time.sleep(1)
            print(f"Request {url} failed: {e}, retry {retries}, next attempt")
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


def get_fasta_wrapper(process_num, pdb_ids, pdb_data_directory, progress_queue):
    success_count = 0
    for pdb_id in pdb_ids:
        if download_fasta_file(pdb_id, pdb_data_directory):
            success_count += 1
        progress_queue.put(1)
    return success_count

def process_fasta_data(filtered_result_set, pdb_data_directory, cut_threshold=100):
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

    completed = 0
    while completed < len(filtered_result_set):
        progress_queue.get()
        completed += 1
        progress_bar.update(1)

    for process in processes:
        process.join()

    progress_bar.close()

def unsuccessful_downloaded_statistics(fasta_folder, filtered_result_set):
    fasta_file_count = {f.split('.')[0].lower() for f in os.listdir(fasta_folder) if f.endswith('.fasta')}
    print("Expected number of fasta files:", len(filtered_result_set))
    print("Number of fasta files in folder:", len(fasta_file_count))
    again_downloaded_files = list(set(filtered_result_set) - fasta_file_count)
    print("Number of missing pdb files:", len(again_downloaded_files), "re-downloading:")
    return again_downloaded_files


def rename_files_to_lower(folder_path):
    if not os.path.exists(folder_path):
        print(f"Directory does not exist: {folder_path}")
        return
    try:
        for filename in os.listdir(folder_path):
            lower_filename = filename.lower()
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, lower_filename)

            if old_path != new_path:
                os.rename(old_path, new_path)

        print("All files have been renamed to lowercase!")

    except Exception as e:
        print(f"Error occurred: {e}")

def main(url, fasta_save_folder):
    os.makedirs(fasta_save_folder, exist_ok=True)
    json_path = os.path.join(fasta_save_folder, "search_request.json")
    parse_url_to_json(url, output_file=json_path)
    pdb_list = get_pdb_list(json_path)
    print("Total PDB entries:", len(pdb_list))
    pdb_list = list(set([pdb.lower() for pdb in pdb_list]))
    rename_files_to_lower(fasta_save_folder)
    pdb_list = unsuccessful_downloaded_statistics(fasta_save_folder, pdb_list)
    process_fasta_data(pdb_list, fasta_save_folder)
    rename_files_to_lower(fasta_save_folder)
    retry_count = 0
    max_retries = 2
    retry_threshold = 0.02  # 2% failure rate
    cpu_usage = 0.5         # initial CPU usage
    while retry_count < max_retries:
        failed_pdbs = unsuccessful_downloaded_statistics(fasta_save_folder, pdb_list)
        failed_ratio = len(failed_pdbs) / len(pdb_list) if pdb_list else 0
        if failed_ratio <= retry_threshold:
            print(f"Success rate reached {100 - failed_ratio * 100:.2f}%, no retry needed.")
            break
        print(f"Failure rate {failed_ratio * 100:.2f}%, retrying...")
        num_cpu = max(1, int(multiprocessing.cpu_count() * cpu_usage))
        process_fasta_data(failed_pdbs, fasta_save_folder, cut_threshold=num_cpu)
        retry_count += 1
        cpu_usage *= 0.9  # reduce CPU load
        new_failed_pdbs = unsuccessful_downloaded_statistics(fasta_save_folder, pdb_list)
        if len(new_failed_pdbs) == len(failed_pdbs):
            print("Failure count unchanged, stopping retries.")
            break
    failed_pdbs = unsuccessful_downloaded_statistics(fasta_save_folder, pdb_list)
    if failed_pdbs:
        log_path = os.path.join(fasta_save_folder, LOG_FILE)
        with open(log_path, "w") as log_file:
            log_file.write("\n".join(failed_pdbs))
        print(f"Failed PDB IDs saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download PDB FASTA sequences from RCSB based on search URL"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="RCSB search URL containing encoded query request"
    )
    parser.add_argument(
        "--fasta_save_folder",
        type=str,
        required=True,
        help="Directory to store downloaded FASTA files"
    )
    args = parser.parse_args()
    main(
        url=args.url,
        fasta_save_folder=args.fasta_save_folder
    )
