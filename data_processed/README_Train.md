# RCSB PDB Dataset Training Pipeline

## Step 1: Retrieve PDB IDs from RCSB Search

This step allows users to fetch PDB IDs from the RCSB database according to a specified search query and automatically download the corresponding FASTA sequences.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step1_fasta_data_crawling \\
    --url "https://www.rcsb.org/search?request=..." \\
    --fasta_save_folder data_processed/save_folder/step1_fastadownload
```
<br/>

## Step 2: FASTA → PKL Processing

This step converts protein FASTA files into structured PKL format, supporting single or multiple FASTA files, different parsing modes, and optional PDB ID filtering.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step2_fasta_file_processed \\
    --input_fasta_dir data_processed/temp/step1_fasta_download \\
    --one_pdb_output_pkl_dir data_processed/temp2/step2/one_pdb_output_pkl_dir \\
    --final_pkl_path data_processed/temp2/step2/all_pdb_chains_fasta_data.pkl \\
    --mode single \\
    --temporary_deleted \\
    --pdb_id_file ''
```
<br/>

## Step 3: Download mmCIF Files

This step downloads protein structure files in mmCIF (.cif.gz) format for all PDB entries obtained from Step 2.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step3_mmcif_data_crawling \\
    --mmcif_dir data_processed/temp2/step3 \\
    --pkl_file data_processed/temp2/step2/all_pdb_chains_fasta_data.pkl \\
    --max_retries 5 \\
    --retry_threshold 0.01 \\
    --cpu_usage 0.75
```
<br/>

## Step 4: mmCIF → Processed PDB & Metadata

This step processes downloaded mmCIF files into structured PDBs and metadata, supporting single-chain or multi-chain processing with filtering and optional merging.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step4_pdb_parsing.structure_mmcif_processed_main \\
    --input_cif_gz_dir data_processed/save_folder/step3 \\
    --output_pdb_dir data_processed/save_folder/step4/pdb \\
    --metadata_json_dir data_processed/save_folder/step4/metadata_jsonfile_output \\
    --mode single
```

**Attention:** The processed data may include **large proteins** (more than **20 chains**), which can cause **processing delays** in the last few strands. If needed, it is recommended to **stop and retry multiple times**. After several attempts, the number of **successfully processed protein structures** stabilizes, ensuring that all structures except for very long proteins are processed.
<br/>

## Step 4-1: Sequence Alignment (PDB ↔ FASTA)

This optional step aligns sequences extracted from PDB structures with sequences from FASTA files, removes inconsistent terminal regions, and produces cleaned PKL data for training.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step4-1_pdb_seq_align_with_fasta \\
    --pdb_dir data_processed/save_folder/step4/pdb \\
    --fasta_dir data_processed/save_folder/step1_fastadownload \\
    --json_dir data_processed/save_folder/step4-1_align_pdb_to_fasta/align_json_output_dir \\
    --pkl_dir data_processed/save_folder/step4-1_align_pdb_to_fasta/one_pkl_output \\
    --num_processes 32 \\
```
<br/>

## Step 5: Representation Generation
In this step, we generate protein sequence representations using the Protenix framework.

### Overview
To compute protein representations, a dedicated environment with all required **Protenix dependencies** must be set up, and a **customized version of Protenix** is provided within this project. However, due to size constraints, **data files and model weights are not distributed** in the repository.  
Please download:
- **model_v0.2.0.pt** from Protenix release  
- **CCD cache file:** components.v20240608.cif  

### Step 5-1: Environment Setup

```bash
conda create -n protenix_env python=3.10 -y
conda activate protenix_env
pip install protenix
cd ../protenix
```
**Note**: This project includes a modified version of Protenix. Make sure to use this directory for execution.
<br/>

### Step 5-2: MSA Generation

This step generates Multiple Sequence Alignment (MSA) data using the Protenix API from sequences obtained in Step 2, Step 4-1, or a composite FASTA file, for downstream representation learning.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step5_representation_generation.step2_get_msa_from_protenixAPI \\
    --input_path data_processed/save_folder/step4-1_align_pdb_to_fasta/step4-1_final_merged.pkl \\
    --out_dir data_processed/save_folder/step5_msa_output \\
    --log_file data_processed/save_folder/step5_msa_output/msa_log.log \\
    --max_workers 20
```
<br/>

**Note:** Protenix does not include a **Jupyter kernel**; please **execute all operations in the terminal**.  

When dealing with **large-scale data**, the speed of MSA may be relatively slow, so please exercise patience. If the **MSA server frequently encounters request failures**, you may consider **localizing MSA** and running it using **ColabFold in conjunction with MMseqs2**.  

For detailed instructions, please refer to the Protenix documentation:  
[ColabFold-compatible MSA](https://github.com/bytedance/Protenix/blob/main/docs/colabfoldcompatiblemsa.md)  

Furthermore, after MSA generates the **A3M file**, please proceed with **MSA Post-Processing** as outlined in Step 3 of the documentation:  
[MSA Template Pipeline](https://github.com/bytedance/Protenix/blob/main/docs/msatemplatepipeline.md)
<br/>

### Step 5-3: JSON Construction for Representation Input

This step constructs structured JSON files by pairing precomputed MSA results with corresponding protein sequences for downstream representation learning.

**Usage Example:**

```bash
python3 -m data_processed.rcsb_script.step5_representation_generation.step3_af3_input_json \\
    --input_filepath data_processed/save_folder/step4-1_align_pdb_to_fasta/step4-1_final_merged_top3.pkl \\
    --one_json_output_dir data_processed/save_folder/step5_msa_output/one_json \\
    --precomputed_msa_dirs data_processed/save_folder/step5_msa_get/msa_dir \\
    --merge_json_output_dir data_processed/save_folder/step5_msa_output/merge_json \\
    --merge_json_output_name test.json \\
    --pairing_db uniref100
```
<br/>

### Step 5-4: Representation Generation

This step generates **protein representations** using a **customized Protenix inference pipeline**.


This implementation is based on **Protenix v0.3.1**, utilizing the pretrained weights **model-v0.2.0.pt** and the **CCD cache file components.v20240608.cif**.  
The corresponding Protenix release can be accessed here: [Protenix v0.3.1 Release](https://github.com/bytedance/Protenix/releases/tag/v0.3.1)

**Procedure:**  
1. **Deactivate** the current conda environment and **activate** the `protenix` environment.  
2. Navigate to the **Protenix working directory**:  
   ```bash
   cd ../data_processed/protenix
3. **Edit the inference script:**  
The script **protenix/inference_demo_1.sh** needs to be updated as follows:  
    - Set the **input JSON file path** (from Step 5-3)  
    - Set the **output directory** for representations  
    - Configure the **number of GPUs** (`nproc_per_node=X`)  
    - Keep other parameters unchanged  
4. **Run the inference script:**  
    ```bash
    bash inference_demo_1.sh
    ```
<br/>

## Step 6: Representation Retrieval by Sequence

This step retrieves **precomputed protein representations** based on sequences, enabling efficient mapping and reuse of features.

**Usage Example:**  
```bash
python3 -m data_processed.rcsb_script.step6_af3rep_corresponding_seqs \\
    --input_pkl data_processed/temp/step4-1_align_pdb_to_fasta/step4-1_final_merged_top3.pkl \\
    --output_pkl data_processed/temp/step6/alphafold3_seqres_to_index_output.pkl
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