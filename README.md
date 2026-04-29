# DynoM
This is the code for "**Efficient exploration of conformational ensembles via protein interaction-informed deep learning**", a conditional diffusion model named **DynoM** natively designed for multi-chain inputs.

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
To sample conformations using the DynoM model:

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

the detailed training configuration can be found:

```bash
 python Predict.py -h
```

The ```bash Predict.py ``` requires options including:

1. **gen_dataset_test_gen_dataset**: **required**, the file path to the test dataset used for prediction, a valid .pkl file path is required.
2. **alphafold3_cfg_seqres_to_index_path**: **required**, the file path to the AlphaFold3 sequence to index mapping, a valid .pkl file path is required.
3. **alphafold3_cfg_repr_data_root**: **required**, the root directory path containing the AlphaFold3 preprocessed feature representations.
4. **gen_batch_size**: **optional**, the batch size to be used during inference, default is 10.
5. **gen_num_samples**: **required**, the total number of samples you want to generate.
6. **model_ckpt_path**: **required**, the file path to the downloaded pre-trained model checkpoint.
7. **output_dir**: **required**, the directory where the inference results will be saved. If the directory does not exist, the script will typically create it automatically.

example:

```bash
python Predict.py \
    --gen_dataset_test_gen_dataset ./example/input_8QB6.pkl \
    --alphafold3_cfg_seqres_to_index_path ./example/seqres_to_index.pkl \
    --alphafold3_cfg_repr_data_root ./example/AF3_repr \
    --gen_batch_size 10 \
    --gen_num_samples 1000 \
    --model_ckpt_path  ./checkpoint/DynoM.ckpt \
    --output_dir ./output
```

## **Citation**
If you are using our code or model, please cite the following paper:

```markdown
@article{DynoM,
  title={"Efficient exploration of conformational ensembles via protein interaction-informed deep learning"},
  author={Zilin Ren, Duo Liu, Qianyi Jia, Qiantong Jin, Yubin Liu, Chan Liu, Zhixiang Sui, Luyao Han, Yixiang Zhang, Xin Yang, Pingping Sun, Zhiguo Fu, Chengkun Wu, Ming Ni, Xiaochen Bo},
  journal={xx},
  year={2026},
  publisher={xx},
  doi={xx},
}
```
