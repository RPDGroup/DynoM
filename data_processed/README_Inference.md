# Inference Data Pipeline

Prepare the sequences for structure generation, provided either as a folder containing **individual FASTA files (one per structure)** or as **a composite FASTA file**.Ensure that the FASTA format follows the **standard used by RCSB PDB**.

## Step 1: FASTA → PKL Processing

This step converts protein FASTA files into structured PKL format, supporting single or multiple FASTA files, different parsing modes, and optional PDB ID filtering.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step2_fasta_file_processed \
    --input_fasta_dir <PATH_TO_YOUR_INPUT_FASTA_DIRECTORY> \
    --output_dir <PATH_TO_YOUR_ONE_PDB_OUTPUT_PKL_DIRECTORY> \
    --final_pkl_path <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE> \
    --mode single \
    --temporary_deleted \
    --pdb_id_file ''
```
<br/>

## Step 2: Representation Generation
In this step, we generate protein sequence representations using the Protenix framework.

### Overview
To compute protein representations, a dedicated environment with all required **Protenix dependencies** must be set up, and a **customized version of Protenix** is provided within this project. However, due to size constraints, **data files and model weights are not distributed** in the repository.  
Please download:
- **model_v0.2.0.pt** from Protenix release  
- **CCD cache file:** components.v20240608.cif 

Protenix v0.3.1 here ： [Protenix v0.3.1 Release](https://github.com/bytedance/Protenix/releases/tag/v0.3.1)
### Step 2-1: Environment Setup

```bash
conda create -n protenix_env python=3.10 -y
conda activate protenix_env
pip install protenix
cd ../protenix
```
**Note**: This project includes a modified version of Protenix. Make sure to use this directory for execution.
<br/>

### Step 2-2: MSA Generation

This step generates **Multiple Sequence Alignment (MSA)** data using the PKL files produced in Step 1 as input.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step5_representation_generation.step2_get_msa_from_protenixAPI \
    --input_path <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE_FROM_STEP1> \
    --output_dir <PATH_TO_YOUR_MSA_OUTPUT_DIRECTORY> \
    --log_file <PATH_TO_YOUR_MSA_LOG_FILE> \
    --max_workers <NUMBER_OF_PARALLEL_WORKERS>
```
<br/>

**Note:** Protenix does not include a **Jupyter kernel**; please **execute all operations in the terminal**.  When dealing with **large-scale data**, the speed of MSA may be relatively slow, so please exercise patience. 

### Step 2-3: JSON Construction for Representation Input

This step constructs structured JSON files by pairing precomputed MSA results with corresponding protein sequences .

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step5_representation_generation.step3_af3_input_json \
    --input_path <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE_FROM_STEP1> \
    --msa_dirs <PATH_TO_YOUR_PRECOMPUTED_MSA_DIRECTORY> \
    --output_dir <PATH_TO_YOUR_JSON_OUTPUT_DIRECTORY> \
    --merge_json_name <MERGED_JSON_FILE_NAME> \
    --pairing_db 'uniref100'
```
<br/>

### Step 2-4: Representation Generation

This step generates **protein representations** using a **customized Protenix inference pipeline**.

**Procedure:**  
1. **Deactivate** the current conda environment and **activate** the `protenix` environment.  
2. Navigate to the **Protenix working directory**:  
   ```bash
   cd ../data_processed/protenix
3. **Edit the inference script:**  
The script **protenix/inference_demo_1.sh** needs to be updated as follows:  
    - Set the **input JSON file path** (from Step 2-3)  
    - Set the **output directory** for representations  
    - Configure the **number of GPUs** (`nproc_per_node=X`)  
    - Keep other parameters unchanged  
4. **Run the inference script:**  
    ```bash
    bash inference_demo_1.sh
    ```
<br/>

## Step 3: Representation Retrieval by Sequence

This step retrieves **precomputed protein representations** based on sequences, enabling efficient mapping and reuse of features.

**Usage Example:**  
```bash
python3 -m data_processed.rcsb_script.step6_af3rep_corresponding_seqs \
    --input_path <PATH_TO_YOUR_FINAL_MERGED_PKL_FILE_FROM_STEP4-1> \
    --output_path <PATH_TO_YOUR_SEQUENCE_MAPPING_OUTPUT_PKL>
```
<br/>


## Final Data

All data required for **model input inference** have been fully prepared, including the following components:

**Data Components:**

1. **PKL files**
   Containing **protein sequences**, **chain information**, and other **metadata** for data indexing and loading, corresponding to the model input argument `gen_dataset_test_gen_dataset`.

2. **AF3 representation files**
   Including both **single** and **pair representations**, capturing **sequence features** and **residue-level interactions**, corresponding to the model input argument `alphafold3_cfg_repr_data_root`.

3. **Sequence–structure mapping files**
   Establishing the correspondence between **sequences** and their **representations**, corresponding to the model input argument `alphafold3_cfg_seqres_to_index_path`.

<br/>
