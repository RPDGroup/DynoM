# DynoM

## **Installation**

```bash
# clone project
git clone https://github.com/RPDGroup/DynoM.git
cd DynoM

# create conda virtual environment
conda env create -f ./environment/environment.yml

# activate virtual environment
conda activate DynoM
```

## Training

DynoM 的训练主要由预训练和微调两个阶段组成。

1. 使用 RCSB PDB dataset 对 DynoM 进行 Pretrain
2. 使用 MD dataset 对 DynoM 进行 fine-tune

### **1. Process data**

**Process RCSB PDB dataset** 

**Process  MD simulations dataset**

### 2. Pretrain **on RCSB PDB dataset**

```bash
python Pretrain.py \
    --data_train_csv_path <PATH_TO_YOUR_Train_Dataset.pkl> \
    --data_val_csv_path <PATH_TO_YOUR_EVAL_Dataset.pkl> \
    --data_train_batch_size 1 \
    --data_valid_batch_size 1 \
    --data_train_complex_pdb_data_dir <PATH_TO_YOUR_COMPLEX_PDB_DATA_DIRECTORY> \
    --data_val_complex_pdb_data_dir <PATH_TO_YOUR_COMPLEX_PDB_DATA_DIRECTORY> \
    --data_train_monomer_pdb_data_dir <PATH_TO_YOUR_MONOMER_PDB_DATA_DIRECTORY> \
    --data_val_monomer_pdb_data_dir <PATH_TO_YOUR_MONOMER_PDB_DATA_DIRECTORY> \
    --data_alphafold3_seqres_to_index_path <PATH_TO_YOUR_SEQRES.pkl> \
    --data_complex_repr_data_root <PATH_TO_YOUR_COMPLEX_DATA_ALPHAFOLD3_REPRESENTATION_DIRECTORY> \
    --data_train_csv_processor_groupby cluster70_id \
    --data_use_alphafold3_repr True \
    --train_log_path <PATH_TO_YOUR_LOG_DIRECTORY>  \
    --save_ckpt_path <PATH_TO_YOUR_SAVE_CKECKPOINT_DIRECTORY>
```

the detailed training configuration can be found

```bash
 python Pretrain.py -h
```

**model ckeckpoint**

Access pretrained models with different RCSB PDB dataset:

| **Model name** | **Dataset** | **Download** |
| --- | --- | --- |
| model C | RCSB PDB complex dataset |  |
| model M | RCSB PDB monomer dataset |  |
| **model A** | RCSB PDB all dataset |  |

### **3. Fine-tune on MD simulations dataset**

```bash
python MD_finetune.py \
    --data_target_dataset ComplexMdDataset \
    --data_train_csv_path <PATH_TO_YOUR_TRAIN_DATASET.pkl> \
    --data_val_csv_path <PATH_TO_YOUR_EVAL_DATASET.pkl> \
    --data_train_batch_size 1 \
    --data_valid_batch_size 1 \
    --data_alphafold3_seqres_to_index_path <PATH_TO_YOUR_SEQRES.pkl> \
    --data_train_complex_pdb_data_dir <PATH_TO_YOUR_COMPLEX_PDB_DATA_DIRECTORY> \
    --data_val_complex_pdb_data_dir <PATH_TO_YOUR_COMPLEX_PDB_DATA_DIRECTORY> \
    --data_monomer_repr_data_root <PATH_TO_YOUR_MONOMER_DATA_ALPHAFOLD3_REPRESENTATION_DIRECTORY> \
    --data_complex_repr_data_root <PATH_TO_YOUR_COMPLEX_DATA_ALPHAFOLD3_REPRESENTATION_DIRECTORY> \
    --train_Pretrain_ckpt_path <PATH_TO_YOUR_PRETRAIN_MONDE_CHECKPOINT_PATH> \
    --train_log_path <PATH_TO_YOUR_LOG_DIRECTORY>  \
    --save_ckpt_path <PATH_TO_YOUR_SAVE_CKECKPOINT_DIRECTORY> 
```

the detailed training configuration can be found

```bash
 python MD_finetune.py -h
```

**model ckeckpoint**

Access pretrained models with different MD **simulations** dataset:

| **Model name** | **Dataset** | **Download** |
| --- | --- | --- |
| model 1 | 1,000 MD trajectories  |  |
| model 2 | 2,000 MD trajectories  |  |
| model 3 | 3,000 MD trajectories  |  |
| DynoM | 5,502 MD trajectories  |  |

## **Inference**

input data process

```bash

```

To sample conformations using the DynoM model

```bash
python Predict.py \
    --gen_dataset_test_gen_dataset <PATH_TO_YOUR_PREDICT_DATASET.pkl> \
    --alphafold3_cfg_seqres_to_index_path <PATH_TO_YOUR_PREDICT_SEQRES.pkl> \
    --alphafold3_cfg_repr_data_root <PATH_TO_YOUR_PREDICT_DATA_ALPHAFOLD3_REPRESENTATION_DIRECTORY> \
    --gen_batch_size 10 \
    --gen_num_samples <NUM SAMPLE> \
    --model_ckpt_path  <PATH_TO_YOUR_MONDE_CHECKPOINT_PATH> \
    --output_dir <PATH_TO_YOUR_OUTPUT_DIRECTORY> 
```

the detailed training configuration can be found

```bash
 python Predict.py -h
```

## **Citation**

```markdown
@inproceedings{,
  title={xx},
  author={xx},
  booktitle={xx},
  year={xx}
}
```