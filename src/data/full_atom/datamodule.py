"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

import torch
import numpy as np
import pandas as pd
from omegaconf import ListConfig
from lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler

from .dataset import RCSBDataset, ComplexMdDataset, AtlasDataset, GenDataset, MonomerAndComplexMdDataset


class FullAtomDataModule(LightningDataModule):
    """
    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_cfg,
        **kwargs,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.cfg = data_cfg
        
        
        if self.cfg['mode'] == "train":
            if self.cfg['target_dataset'] == "RCSBDataset":
                    self.train_dataset = RCSBDataset(
                            csv_path=data_cfg["train_csv_path"],
                            mode=data_cfg["mode"],
                            se3_cfg=data_cfg['se3_cfg'],
                            monomer_pdb_data_dir=data_cfg["train_monomer_pdb_data_dir"],
                            complex_pdb_data_dir=data_cfg["train_complex_pdb_data_dir"],
                            csv_processor_cfg=data_cfg["train_csv_processor_cfg"],
                            dynamic_batching=True,
                            alphafold3_cfg= data_cfg["alphafold3_cfg"] if data_cfg["use_alphafold3_repr"] else None ,
                            dp_cfg=data_cfg["dp_cfg"] if "dp_cfg" in data_cfg else None,
                        )
                    
                    self.val_dataset = RCSBDataset(
                            csv_path=data_cfg["val_csv_path"],
                            mode=data_cfg["mode"],
                            se3_cfg=data_cfg['se3_cfg'],
                            monomer_pdb_data_dir=data_cfg["val_monomer_pdb_data_dir"],
                            complex_pdb_data_dir=data_cfg["val_complex_pdb_data_dir"],
                            csv_processor_cfg=data_cfg["valid_csv_processor_cfg"],
                            dynamic_batching=True,
                            alphafold3_cfg=data_cfg["alphafold3_cfg"] if data_cfg["use_alphafold3_repr"] else None,
                            dp_cfg=data_cfg["dp_cfg"] if "dp_cfg" in data_cfg else None,
                        )

            elif self.cfg['target_dataset'] == "ComplexMdDataset":
                    self.train_dataset = ComplexMdDataset(
                            csv_path=data_cfg["train_csv_path"],
                            mode=data_cfg["mode"],
                            se3_cfg=data_cfg['se3_cfg'],
                            monomer_pdb_data_dir=data_cfg["train_monomer_pdb_data_dir"],
                            complex_pdb_data_dir=data_cfg["train_complex_pdb_data_dir"],
                            csv_processor_cfg=data_cfg["train_csv_processor_cfg"],
                            dynamic_batching=True,
                            is_classify_sample = data_cfg["is_classify_sample"],
                            is_order_sample = data_cfg["is_order_sample"],
                            classify_num = data_cfg['classify_num'] if data_cfg["is_classify_sample"] or data_cfg['is_order_sample'] else None,
                            alphafold3_cfg=data_cfg["alphafold3_cfg"] if data_cfg["use_alphafold3_repr"] else None,
                        )

                    self.val_dataset = ComplexMdDataset(
                            csv_path=data_cfg["val_csv_path"],
                            mode=data_cfg["mode"],
                            se3_cfg=data_cfg['se3_cfg'],
                            monomer_pdb_data_dir=data_cfg["val_monomer_pdb_data_dir"],
                            complex_pdb_data_dir=data_cfg["val_complex_pdb_data_dir"],
                            csv_processor_cfg=data_cfg["valid_csv_processor_cfg"],
                            dynamic_batching=True,
                            is_classify_sample = data_cfg["is_classify_sample"],
                            is_order_sample = data_cfg["is_order_sample"],
                            classify_num = data_cfg['classify_num'] if data_cfg["is_classify_sample"] or data_cfg['is_order_sample'] else None,
                            alphafold3_cfg=data_cfg["alphafold3_cfg"] if data_cfg["use_alphafold3_repr"] else None,
                            )
            elif self.cfg['target_dataset'] ==  "AtlasDataset":
                    self.train_dataset = AtlasDataset(
                            csv_path=data_cfg["train_csv_path"],
                            mode=data_cfg["mode"],
                            se3_cfg=data_cfg['se3_cfg'],
                            monomer_pdb_data_dir=data_cfg["train_monomer_pdb_data_dir"],
                            complex_pdb_data_dir=data_cfg["train_complex_pdb_data_dir"],
                            csv_processor_cfg=data_cfg["train_csv_processor_cfg"],
                            dynamic_batching=True,
                            alphafold3_cfg=data_cfg["alphafold3_cfg"] if data_cfg["use_alphafold3_repr"] else None,
                        )

                    self.val_dataset = AtlasDataset(
                            csv_path=data_cfg["val_csv_path"],
                            mode=data_cfg["mode"],
                            se3_cfg=data_cfg['se3_cfg'],
                            monomer_pdb_data_dir=data_cfg["val_monomer_pdb_data_dir"],
                            complex_pdb_data_dir=data_cfg["val_complex_pdb_data_dir"],
                            csv_processor_cfg=data_cfg["valid_csv_processor_cfg"],
                            dynamic_batching=True,
                            alphafold3_cfg=data_cfg["alphafold3_cfg"] if data_cfg["use_alphafold3_repr"] else None,
                            )
            elif self.cfg['target_dataset'] ==  "MonomerAndComplexMdDataset":
                    self.train_dataset = MonomerAndComplexMdDataset(
                            csv_path=data_cfg["train_csv_path"],
                            mode=data_cfg["mode"],
                            se3_cfg=data_cfg['se3_cfg'],
                            monomer_pdb_data_dir=data_cfg["train_monomer_pdb_data_dir"],
                            complex_pdb_data_dir=data_cfg["train_complex_pdb_data_dir"],
                            csv_processor_cfg=data_cfg["train_csv_processor_cfg"],
                            dynamic_batching=True,
                            alphafold3_cfg=data_cfg["alphafold3_cfg"] if data_cfg["use_alphafold3_repr"] else None,
                        )

                    self.val_dataset = MonomerAndComplexMdDataset(
                            csv_path=data_cfg["val_csv_path"],
                            mode=data_cfg["mode"],
                            se3_cfg=data_cfg['se3_cfg'],
                            monomer_pdb_data_dir=data_cfg["val_monomer_pdb_data_dir"],
                            complex_pdb_data_dir=data_cfg["val_complex_pdb_data_dir"],
                            csv_processor_cfg=data_cfg["valid_csv_processor_cfg"],
                            dynamic_batching=True,
                            alphafold3_cfg=data_cfg["alphafold3_cfg"] if data_cfg["use_alphafold3_repr"] else None,
                            )
            
        elif self.cfg['mode'] == "val":        
            if data_cfg.get('gen_dataset', {}).get('test_gen_dataset') is not None:         
                self.test_gen_dataset = GenDataset(
                    csv_path=data_cfg['gen_dataset']['test_gen_dataset'],
                    num_samples=data_cfg['gen_dataset']['num_samples'],
                    alphafold3_cfg=data_cfg["alphafold3_cfg"] if data_cfg["use_alphafold3_repr"] else None,
                )
        
        self.clustering_training = data_cfg['is_clustering_training']
        
    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            print(
                f"Model fitting with train set ({len(self.train_dataset):d}) and val set ({len(self.val_dataset):d})"
            )

    def train_dataloader(self):

        if self.clustering_training:
            train_sampler = ClusterDistributedSampler(
                csv_path=self.cfg["csv_path"],
                batch_size=self.cfg['train_batch_size'],
                replace=True,
            )
            shuffle = False
        elif getattr(self.train_dataset, 'random_batch', False):
            train_sampler = None
            shuffle = True
        else:
            train_sampler = ClusterSampler(
                dataset=self.train_dataset,
                batch_size=self.cfg['train_batch_size']
            )
            shuffle = False

        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg['train_batch_size'],
            sampler=train_sampler,
            shuffle=shuffle,
            collate_fn=self.train_dataset.collate,
            drop_last=False,
            num_workers=self.cfg['train_num_workers'],
            pin_memory=self.cfg['pin_memory'],
        )

        return train_loader

    def val_dataloader(self):
        if self.clustering_training:
            val_sampler = ClusterDistributedSampler(
                csv_path=self.cfg["csv_path"],
                batch_size=self.cfg['valid_batch_size'],
                replace=True,
            )
        else:
            val_sampler = ClusterSampler(
                dataset=self.val_dataset,
                batch_size=self.cfg['valid_batch_size'],
            )
            
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg['valid_batch_size'],
            sampler=val_sampler,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
            drop_last=False,
            num_workers=self.cfg['val_num_workers'],
            pin_memory=self.cfg['pin_memory'],
        )

        if getattr(self, 'val_gen_dataset', None) is not None:
            val_loader = [val_loader]
            val_loader.append(
                torch.utils.data.DataLoader(
                    dataset=self.val_gen_dataset,
                    batch_size=self.cfg['valid_batch_size'],
                    collate_fn=self.val_gen_dataset.collate,
                    num_workers=self.cfg['val_num_workers'],
                    pin_memory=self.cfg['pin_memory'],
                )
            )

        return val_loader

    def test_dataloader(self):

        if isinstance(self.test_gen_dataset, (list, ListConfig)):
            return [
                torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=getattr(
                        dataset, "valid_batch_size", self.cfg['gen_batch_size']
                    ),
                    shuffle=False,
                    collate_fn=dataset.collate,
                    drop_last=False,
                    num_workers=self.cfg['num_workers'],
                    pin_memory=self.cfg['pin_memory'],
                )
                for dataset in self.test_gen_dataset
            ]
        else:
            return torch.utils.data.DataLoader(
                dataset=self.test_gen_dataset,
                batch_size=self.cfg['gen_dataset']['gen_batch_size'],
                shuffle=False,
                collate_fn=self.test_gen_dataset.collate,
                drop_last=False,
                num_workers=self.cfg['num_workers'],
                pin_memory=self.cfg['pin_memory'],
            )

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}


class ClusterDistributedSampler(DistributedSampler):
    def __init__(self, csv_path, batch_size):

        self.batch_size = batch_size
        if torch.distributed.is_initialized():
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            num_replicas = 8
            rank = 0
        self.rank = rank
        self.num_replicas = num_replicas
        self.epoch = 0

        self.dataframe = pd.read_pickle(csv_path)
        self.cluster_indices = list(self.dataframe.groupby("chain_name").indices.values())

        self.num_samples = len(self.dataframe) // num_replicas
        self.total_size = len(self.dataframe)

    def __iter__(self):
        batch_indices = self.reinit()
        indices = batch_indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def reinit(self):
        self.epoch += 1
        print(f"Shuffling batches... seed {41 + self.epoch}")
        np.random.seed(41 + self.epoch)
        batch_indices = []  # list of indices for each batch

        for sublist in self.cluster_indices:
            np.random.shuffle(sublist)
        np.random.shuffle(self.cluster_indices)
        sorted_indices = np.concatenate(self.cluster_indices)
        batch_indices = [
            sorted_indices[i : i + self.batch_size]
            for i in range(0, len(sorted_indices), self.batch_size)
        ]
        batch_order = np.random.permutation(len(batch_indices))  # drop last
        batch_indices = np.concatenate(
            [batch_indices[batch_id] for batch_id in batch_order]
        )
        return batch_indices


class ClusterSampler(DistributedSampler):
    def __init__(
        self, dataset: torch.utils.data.Dataset, batch_size: int, seed: int = 123
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        self.num_batches = len(self.dataset) // self.world_size
        self.total_size = self.num_batches * self.world_size
        self.iter_counter = 0
        self.seed = seed

    def __len__(self):
        return self.num_batches * self.batch_size

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.iter_counter)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices = indices[: self.total_size]

        # subsample
        indices = indices[self.rank : self.total_size : self.world_size]
        assert len(indices) == self.num_batches

        # duplicate indices for batching
        indices = np.repeat(indices, self.batch_size)

        self.iter_counter += 1

        return iter(indices.tolist())
