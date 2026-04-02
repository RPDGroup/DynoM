# RCSB PDB Dataset Training Pipeline
Please make sure to run it in the root directory of the project:DynoM/
## Step 1: Retrieve PDB IDs from RCSB Search

This step allows users to fetch PDB IDs from the RCSB database according to a specified search query and automatically download the corresponding FASTA sequences.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step1_fasta_data_crawling \
    --url "<YOUR_RCSB_SEARCH_URL>" \
    --fasta_save_folder <PATH_TO_YOUR_FASTA_OUTPUT_DIRECTORY>
```
<br/>

## Step 2: FASTA → PKL Processing

This step converts protein FASTA files into structured PKL format, supporting single or multiple FASTA files, different parsing modes, and optional PDB ID filtering.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step2_fasta_file_processed \
    --input_fasta_dir <PATH_TO_YOUR_INPUT_FASTA_DIRECTORY> \
    --one_pdb_output_pkl_dir <PATH_TO_YOUR_ONE_PDB_OUTPUT_PKL_DIRECTORY> \
    --final_pkl_file <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE> \
    --mode single 
```
<br/>

## Step 3: Download mmCIF Files

This step downloads protein structure files in mmCIF (.cif.gz) format for all PDB entries obtained from Step 2.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step3_mmcif_data_crawling \
    --mmcif_dir <PATH_TO_YOUR_MMCIF_DIRECTORY> \
    --pkl_file <PATH_TO_YOUR_INPUT_PKL_FILE> \
    --max_retries 5 \
    --retry_threshold 0.01 \
    --cpu_usage 0.75
```
<br/>

## Step 4: mmCIF → Processed PDB & Metadata

This step processes downloaded mmCIF files into structured PDBs and metadata, supporting single-chain or multi-chain processing with filtering and optional merging.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step4_pdb_parsing.structure_mmcif_processed_main \
    --input_cif_gz_dir <PATH_TO_YOUR_INPUT_CIF_GZ_DIRECTORY> \
    --output_pdb_dir <PATH_TO_YOUR_OUTPUT_PDB_DIRECTORY> \
    --metadata_json_dir <PATH_TO_YOUR_METADATA_JSON_OUTPUT_DIRECTORY> \
    --mode single
```

**Attention:** The processed data may include **large proteins** (more than **20 chains**), which can cause **processing delays** in the last few strands. If needed, it is recommended to **stop and retry multiple times**. After several attempts, the number of **successfully processed protein structures** stabilizes, ensuring that all structures except for very long proteins are processed.
<br/>

## Step 4-1: Sequence Alignment (PDB ↔ FASTA)

This optional step aligns sequences extracted from PDB structures with sequences from FASTA files, removes inconsistent terminal regions, and produces cleaned PKL data for training.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step4-1_pdb_seq_align_with_fasta \
    --pdb_dir <PATH_TO_YOUR_PROCESSED_PDB_DIRECTORY> \
    --fasta_dir <PATH_TO_YOUR_FASTA_DIRECTORY> \
    --json_dir <PATH_TO_YOUR_ALIGNMENT_JSON_OUTPUT_DIRECTORY> \
    --pkl_dir <PATH_TO_YOUR_ONE_PKL_OUTPUT_DIRECTORY> \
    --final_pkl_name <FINAL_MERGED_PKL_FILENAME> \
    --num_processes <NUMBER_OF_PROCESSES>
```
<br/>

## Step 5: Representation Generation
In this step, we generate protein sequence representations using the Protenix framework.

### Overview
To compute protein representations, a dedicated environment with all required **Protenix dependencies** must be set up. This project provides **a customized version of Protenix** tailored for our workflow. 

**model_v0.2.0.pt**: The model weights are **included** in this repository.
```text
/protenix
 └── release_data
     └── checkpoint/
         └── model_v0.2.0.pt
```
**CCD cache file**: The required CCD cache version is `components.v20240608.cif`.  
> Note: This file will be automatically downloaded during the first execution of the script.  
> Please be patient, as the download may take some time depending on your network speed.

Protenix v0.3.1 here ： [Protenix v0.3.1 Release](https://github.com/bytedance/Protenix/releases/tag/v0.3.1)
### Step 5-1: MSA Generation

This step generates **Multiple Sequence Alignment (MSA)** data using the PKL files produced in Step 4-1 as input.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step5_representation_generation.step2_get_msa_from_protenixAPI \
    --input_file <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE_FROM_STEP4-1> \
    --msa_out_dir <PATH_TO_YOUR_MSA_OUTPUT_DIRECTORY> \
    --log_file <PATH_TO_YOUR_MSA_LOG_FILE> \
    --max_workers <NUMBER_OF_PARALLEL_WORKERS>
```
<br/>

### Step 5-2: JSON Construction for Representation Input

This step constructs structured JSON files by pairing precomputed MSA results with corresponding protein sequences .

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step5_representation_generation.step3_af3_input_json \
    --input_pkl_file <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE_FROM_STEP1> \
    --msa_results_dir <PATH_TO_YOUR_PRECOMPUTED_MSA_DIRECTORY> \
    --one_json_output_dir <PATH_TO_YOUR_ONE_JSON_OUTPUT_DIRECTORY> \
    --merge_json_output_dir <PATH_TO_YOUR_MERGE_JSON_OUTPUT_DIRECTORY> \
    --merge_json_output_name <MERGED_JSON_FILE_NAME_No_SUFFIX> \
    --pairing_db 'uniref100'
```
<br/>

### Step 5-3: Environment Setup

```bash
#  Create and activate environment
conda create -n protenix_env python=3.10 -y
cd ./data_processed/protenix
pip3 install -e .
```
**Note**: This project includes a modified version of Protenix. Make sure to use this directory for execution.
<br/>

**Note:** Protenix does not include a **Jupyter kernel**; please **execute all operations in the terminal**.  When dealing with **large-scale data**, the speed of MSA may be relatively slow, so please exercise patience. 

### Step 5-4: Representation Generation

This step generates **protein representations** using a **customized Protenix inference pipeline**.

**Procedure:**  
1. **Deactivate** the current conda environment and **activate** the `protenix` environment.  
2. Navigate to the **Protenix working directory** in **protenix** dir:  
3. **Edit the inference script:**  
The script **protenix/inference_demo_1.sh** needs to be updated as follows:  
    - Set the **input JSON file path** (from Step 5-2)  
    - Set the **output directory** for representations  
    - Configure the **number of GPUs** (`nproc_per_node=X`)  
    - Keep other parameters unchanged  
4. **Run the inference script:**  
    ```bash
    bash inference_demo_1.sh
    ```
5. **Return to DynoM root directory**
    ```bash
    cd ../../
    ```
<br/>

## Step 6: Representation Retrieval by Sequence

This step retrieves **precomputed protein representations** based on sequences, enabling efficient mapping and reuse of features.

**Usage Example:**  
```bash
python3 -m data_processed.rcsb_script.step6_af3rep_corresponding_seqs \
    --input_pkl <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE_FROM_STEP4-1> \
    --output_pkl <PATH_TO_YOUR_SEQUENCE_MAPPING_OUTPUT_PKL>
```
<br/>

## Final Data Preparation for Model Training

At this stage, all data required for **model training** have been fully prepared, including the following components.

**Data Components:**

1. **PKL files**  
   Containing **protein sequences**, **chain information**, and other **metadata** for data indexing and loading. These correspond to the model input arguments `train_csv` and `val_csv`.

2. **PDB structure files**  
   Generated from standardized processing of **mmCIF files** in Step 4.

3. **AF3 representation files**  
   Including both **single** and **pair representations**, capturing **sequence features** and **residue-level interactions**, corresponding to the model input argument `complex_repr_data`.

4. **Sequence–structure mapping files**  
   Establishing the correspondence between **sequences**, **structures**, and their associated **representations**, corresponding to the model input argument `alphafold3_seqres_to_index`.
<br/>