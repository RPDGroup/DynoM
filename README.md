# DynoM
This is the code for "**Introducing protein interaction efficiently improved the prediction of monomer dynamic conformation**", a conditional diffusion model named **DynoM** natively designed for multi-chain inputs.

## Table of content
- [DynoM](#dynom)
  - [Table of Contents](#table-of-contents)
  - [Installation and Test](#installation)
  - [Usage](#inference)
  - [Citation](#citation)

## **Installation**

### Enviroment

```bash
# clone project
git clone https://github.com/RPDGroup/DynoM.git
cd DynoM

# create conda virtual environment
conda env create -f ./environment/env.yml

# activate virtual environment
conda activate DynoM_env
```

## **Inference**

### Data Process
Please refer to [input data process](https://github.com/RPDGroup/DynoM/blob/main/data_processed/README_Inference.md)

### Predict
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
