# DynoM
This is the code for "**Introducing protein interaction efficiently improved the prediction of monomer dynamic conformation**", a conditional diffusion model named **DynoM** natively designed for multi-chain inputs.
![image](https://github.com/RPDGroup/DynoM/blob/main/model.jpg)

## Table of content
- [DynoM](#dynom)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [inference](#inference)
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

### Data Processing
Please refer to [input data process](https://github.com/RPDGroup/DynoM/blob/main/data_processed/README_Inference.md)

### Sampling
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
If you are using our code or model, please cite the following paper:

```markdown
@article{DynoM,
  title={"Introducing protein interaction efficiently improved the prediction of monomer dynamic conformation"},
  author={Zilin Ren, Duo Liu, Qianyi Jia, Qiantong Jin, Yubin Liu, Chan Liu, Zhixiang Sui, Luyao Han, Yixiang Zhang, Xin Yang, Pingping Sun, Zhiguo Fu, Chengkun Wu, Ming Ni, Xiaochen Bo},
  journal={xx},
  year={2026},
  publisher={xx},
  doi={xx},
}
```
