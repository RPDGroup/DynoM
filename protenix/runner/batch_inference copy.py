# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional, Union

import click
import tqdm
from Bio import SeqIO
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from rdkit import Chem
from runner.inference import InferenceRunner, download_infercence_cache, infer_predict
from runner.msa_search import contain_msa_res, msa_search, msa_search_update

from protenix.config import parse_configs
from protenix.data.json_maker import cif_to_input_json
from protenix.data.json_parser import lig_file_to_atom_info
from protenix.data.utils import pdb_to_cif
from protenix.utils.logger import get_logger

logger = get_logger(__name__)


def init_logging():
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )



def get_default_runner(seeds: Optional[list] = None) -> InferenceRunner:
    configs_base["use_deepspeed_evo_attention"] = (
        os.environ.get("USE_DEEPSPEED_EVO_ATTTENTION", False) == "true"
    )
    configs_base["model"]["N_cycle"] = 10
    configs_base["sample_diffusion"]["N_sample"] = 5
    configs_base["sample_diffusion"]["N_step"] = 200
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        fill_required_with_null=True,
    )
    if seeds is not None:
        configs.seeds = seeds
    download_infercence_cache(configs, model_version="v0.2.0")
    return InferenceRunner(configs)



def batch_inference(
    protein_msa_res: dict,
    ligand_file: str,
    out_dir: str = "./output",
    seeds: List[int] = [101],
) -> None:
    """
    ligand_file: ligand file or directory, should be in sdf format or smi with smlies list;
    protein_msa_res: the msa result for `protein`, like:
        {  "MGHHHHHHHHHHSSGH": {
                "precomputed_msa_dir": "/path/to/msa_pairing/result/msa/1",
                "pairing_db": "uniref100"
            },
            "MAEVIRSSAFWRSFPIFEEFDSE": {
                "precomputed_msa_dir": "/path/to/msa_pairing/result/msa/2",
                "pairing_db": "uniref100"
            }
        }
    out_dir: the infer outout dir, default is `./output`
    """
    # with open(json_file_path, 'r') as file:
    #     infer_jsons = json.load(file)
    # infer_jsons = generate_infer_jsons(protein_msa_res, ligand_file, seeds)
    logger.info(f"will infer with {len(infer_jsons)} jsons")
    if len(infer_jsons) == 0:
        return

    infer_errors = {}
    inference_configs["dump_dir"] = out_dir
    inference_configs["input_json_path"] = infer_jsons[0]
    runner = get_default_runner(seeds=seeds)
    configs = runner.configs
    for infer_json in tqdm.tqdm(infer_jsons):
        try:
            configs["input_json_path"] = infer_json
            if not contain_msa_res(infer_json):
                raise RuntimeError(
                    f"`{infer_json}` has no msa result for `proteinChain`, please add first."
                )
            infer_predict(runner, configs)
        except Exception as exc:
            infer_errors[infer_json] = str(exc)
    if len(infer_errors) > 0:
        logger.warning(f"run inference failed: {infer_errors}")





if __name__ == "__main__":
    init_logging()
    batch_inference(protein_msa_res, ligands_dir, out_dir=out_dir)

