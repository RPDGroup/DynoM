# Inference Data Pipeline

This data pipeline is designed to generate the `.pkl` format sequence files and corresponding representations required as inputs for the DynoM model. 

> Note: Please run the execution command from the project's root directory (`DynoM/`)

## Step 1: FASTA → PKL Processing

This step is designed to convert protein FASTA files into the structured `.pkl` format, supporting various parsing modes and optional PDB ID filtering. 

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step2_fasta_file_processed \
    --input_fasta_dir <PATH_TO_YOUR_INPUT_FASTA_DIRECTORY> \
    --one_pdb_output_pkl_dir <PATH_TO_YOUR_ONE_PDB_OUTPUT_PKL_DIRECTORY> \
    --final_pkl_file <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE> \
    --mode single \

```
### Input Requirements
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--input_fasta_dir` | `Input` | The path to your input sequences (formatted to the **RCSB PDB standard**). This can be either a folder containing **individual FASTA files (one per structure)** or the path to a single **multi-FASTA file**. |
| `--one_pdb_output_pkl_dir` | `Intermediate Output` | The directory where the parsed `.pkl` files for each individual PDB/structure will be saved during processing. |
| `--final_pkl_file` | `Final Output` | The absolute path and filename for the final, aggregated `.pkl` file (e.g., `/path/to/output/merged_features.pkl`). |
| `--mode` | `Config` | **Parsing Mode:** Set to `single` when parsing a directory of individual FASTA files. *(Note: If your pipeline uses a different mode like `multi` for a multi-FASTA file, you can specify it here).* |

<br/>

## Step 2: Representation Generation
In this step, we generate protein sequence representations using the Protenix framework.

### Overview
To compute protein representations, a dedicated environment with all required dependencies must be set up. **This project integrates Protenix into a tailored pipeline for our specific needs.** *(Base reference: [Protenix v0.3.1 Release](https://github.com/bytedance/Protenix/releases/tag/v0.3.1))*

Before proceeding, please note the following built-in assets:
* **Model Weights (`model_v0.2.0.pt`):** Please download the required model weights from [this Google Drive link](https://drive.google.com/file/d/1HIm9jdhZpOO5dn6LfVpuISe0RpYl1Od0/view?usp=drive_link). Once downloaded, place the file into the `protenix/release_data/checkpoint/` directory.
* **CCD Cache (`components.v20240608.cif`)**: This required cache file will be **automatically downloaded** during the first execution. 
  > ⚠️ *Note: Please be patient, as the download may take some time depending on your network speed.*

---

### Step 2-1: MSA Generation

This step generates **Multiple Sequence Alignment (MSA)** data using the `.pkl` files produced in Step 1.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step5_representation_generation.step2_get_msa_from_protenixAPI \
    --input_file <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE_FROM_STEP1> \
    --msa_out_dir <PATH_TO_YOUR_MSA_OUTPUT_DIRECTORY> \
    --log_file <PATH_TO_YOUR_MSA_LOG_FILE> \
    --max_workers <NUMBER_OF_PARALLEL_WORKERS>
```

#### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--input_file` | `Input` | The path to your final aggregated `.pkl` file generated from Step 1. |
| `--msa_out_dir` | `Output` | The directory where the generated MSA results will be saved. |
| `--log_file` | `Logging` | The path to save the MSA generation log file. |
| `--max_workers` | `Config` | Number of parallel workers for MSA generation (e.g., `4`). |
<br/>

### Step 2-2: JSON Construction for Representation Input

This step constructs structured JSON files by pairing the precomputed MSA results with their corresponding protein sequences.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step5_representation_generation.step3_af3_input_json \
    --input_file <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE_FROM_STEP1> \
    --msa_results_dir <PATH_TO_YOUR_PRECOMPUTED_MSA_DIRECTORY> \
    --one_json_output_dir <PATH_TO_YOUR_ONE_JSON_OUTPUT_DIRECTORY> \
    --merge_json_output_dir <PATH_TO_YOUR_MERGE_JSON_OUTPUT_DIRECTORY> \
    --merge_json_output_name <MERGED_JSON_FILE_NAME_No_SUFFIX> \
    --pairing_db 'uniref100'
```

#### Parameters

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--input_file` | `Input` | **(Required)** Path to the final merged PKL file generated from Step 1. This file contains the structural or sequence feature information necessary to build the input. |
| `--msa_results_dir` | `Input` | **(Required)** Path to the directory containing the precomputed Multiple Sequence Alignment (MSA) results. |
| `--one_json_output_dir` | `Output` | **(Required)** Directory path to save the individual AlphaFold 3 input JSON files generated for each target. |
| `--merge_json_output_dir` | `Output` | **(Required)** Directory path to save the consolidated, large JSON file (typically used for batch processing or unified management). |
| `--merge_json_output_name` | `Output` | **(Required)** The base name for the merged JSON file. **Note:** Do not include the `.json` suffix. |
| `--pairing_db` | `Config` | The database used for MSA sequence pairing. The default/recommended value is `'uniref100'`. |
<br/>

### Step 2-3: Environment Setup

```bash
# 1. Create and activate a new Conda environment
conda create -n protenix_env python=3.10 -y
conda activate protenix_env

# 2. Navigate to the Protenix directory and install dependencies
cd ./data_processed/protenix
pip3 install -e .
# 3. Install additional dependencies
pip install dm-tree

```


> **Important Notes:**
> * **Modified Version:** This project uses a customized version of Protenix. Please ensure all commands are executed within the `./data_processed/protenix` directory.
> * **Terminal Execution:** Protenix does not include a Jupyter kernel. **All operations must be performed directly in the terminal.**
> * **Performance:** When processing large-scale datasets, **MSA (Multiple Sequence Alignment)** generation may be slow. Please be patient during this process.

</br>

### Step 2-4: Representation Generation

This step generates **protein representations** using a customized **Protenix inference pipeline**.
 

**1. Prepare Environment and Workspace**

Ensure you are still operating within the **`protenix_env`** Conda environment and located in the `./data_processed/protenix` directory.

**2. Configure the Inference Script**

Open `inference_demo_1.sh` in your text editor and update the following variables (leave all other parameters unchanged):
* **Input File (`input_json_path`):** Set this to the path of the JSON file generated in Step 2-2.
* **Output Directory (`dump_dir`):** Define the destination path where the generated representations will be saved.
* **GPU Allocation (`nproc_per_node`):** Update `nproc_per_node=X`, replacing `X` with the number of GPUs you intend to use.

**3. Execute the Pipeline**

Run the modified script:
```bash
bash inference_demo_1.sh
```
> **💡 Troubleshooting Note:**
> If you encounter the error `AttributeError: module 'torch.library' has no attribute 'custom_op'`, it indicates a version incompatibility between DeepSpeed and PyTorch. To fix this, ensure you are in the **`protenix_env`** and update **DeepSpeed** to version **0.15.1**:
> ```bash
> pip uninstall deepspeed -y
> pip install deepspeed==0.15.1
> ```
>

**4. Return to Root Directory**

Once the inference is complete, navigate back to the project root directory:
```bash
cd ../../
```

## Step 3: Representation Retrieval by Sequence

This step retrieves **precomputed protein representations** based on their sequences, enabling the efficient mapping and reuse of generated features.

**1. Prepare Environment**

Switch back to your main **`DynoM_env`** environment and ensure you are in the root directory: `/DynoM`:
```bash
conda deactivate
conda activate DynoM_env
```

**2. Execute the Retrieval Script**

Run the following command to map sequences to their representations:
```bash
python3 -m data_processed.rcsb_script.step6_af3rep_corresponding_seqs \
    --input_pkl <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE_FROM_STEP1> \
    --output_pkl <PATH_TO_YOUR_SEQUENCE_MAPPING_OUTPUT_PKL>
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `--input_pkl` | `Input` | The path to the final merged `.pkl` file generated in Step 1. |
| `--output_pkl` | `Output` | The destination path for the sequence mapping results. The generated `.pkl` file contains the mapping between sequences and their corresponding representations. |
<br/>


## Final Data Overview

All data required for **model inference** has now been fully prepared. The following table maps the generated components to their respective steps and the final model input arguments:

| Data Component | Source Step | Description | Model Argument Mapping |
| :--- | :--- | :--- | :--- |
| **PKL Files** | **Step 1** | Contains protein sequences, chain information, and metadata for data indexing. | `gen_dataset_test_gen_dataset` |
| **AF3 Representations** | **Step 2-4** | Includes single and pair representations (residue-level features) saved in the `dump_dir`. | `alphafold3_cfg_repr_data_root` |
| **Sequence Mapping** | **Step 3** | The final `.pkl` file establishing the map between sequences and precomputed representations. | `alphafold3_cfg_seqres_to_index_path` |

<br/>
