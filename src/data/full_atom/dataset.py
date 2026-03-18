import re
from pandas.core.groupby import DataFrameGroupBy
from typing import Optional, Literal, Dict, Any
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig
from pathlib import Path
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
import torch
import random
import os

from src.data.full_atom.feat_loader import Alphafold3ReprLoader
from src.openfold_local.data import data_transforms
from src.openfold_local.np import residue_constants as rc
from src.openfold_local.utils import rigid_utils as ru
from src.models.full_atom.diffuser.se3_diffuser import SE3Diffuser


pdb_parser = PDBParser(QUIET=True)


class RCSBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,  # path to metadata csv file
        mode: Literal["train", "val"],
        se3_cfg =None,
        monomer_pdb_data_dir: Optional[str] = None,
        complex_pdb_data_dir: Optional[str] = None,
        csv_processor_cfg: Optional[DictConfig] = None,
        alphafold3_cfg: Optional[DictConfig] = None,
        dynamic_batching: bool = True,
        is_oder_smaple: bool = False,
        **kwargs,
    ):
        self.mode = mode
        if csv_path:
            self.df = self._process_csv(
                csv_path=csv_path, csv_processor_cfg=csv_processor_cfg
            )
         
        self.monomer_pdb_data_dir = monomer_pdb_data_dir
        self.complex_pdb_data_dir = complex_pdb_data_dir
        self.diffuser = SE3Diffuser(se3_cfg=se3_cfg)
        

        # init alphafold3 repr 
        if alphafold3_cfg is not None:
            self.repr_loader = Alphafold3ReprLoader(
                complex_data_root=alphafold3_cfg["complex_repr_data_root"],
                monomer_data_root=alphafold3_cfg["monomer_repr_data_root"],
                num_recycles=alphafold3_cfg["num_recycle"],
                node_size=alphafold3_cfg["node_size"],
                edge_size=alphafold3_cfg["edge_size"],
                seqres_to_index_path = alphafold3_cfg["seqres_to_index_path"],
            )
        
        self.is_oder_smaple = is_oder_smaple
        self.dynamic_batching = dynamic_batching
        
    def _process_csv(
        self,
        csv_path: str,
        csv_processor_cfg
    ) -> pd.DataFrame:
        metadata_df = pd.read_pickle(filepath_or_buffer=csv_path)
        if csv_processor_cfg is not None:
            
            '''
                filter csv
                    length ---> release date ---> ratio ---> clustering   
                reture 
                    df: pd.DataFrame
            '''
            ## length filter
            min_seqlen = csv_processor_cfg.get("min_seqlen", 0)
            max_seqlen = csv_processor_cfg.get("max_seqlen", 1e4)
            
            filtered_index = metadata_df[
                (metadata_df['chain_lens'].apply(min) >= min_seqlen)
                & (max(metadata_df['chain_lens'].apply(max)) <= max_seqlen)
                
            ].index
            
            df = metadata_df.loc[filtered_index]

            # sequence-based clustering
            if csv_processor_cfg.get("groupby", None) is not None:
                self.group_col = group_col = csv_processor_cfg.get("groupby")
                df = df.groupby(group_col)
                self.group_keys = list(df.groups.keys())
                assert len(self.group_keys) == len(df)
            else:
                self.group_col = None
        else:
            df=metadata_df
            self.group_col = None
        return df

    @property
    def name(self):
        return "RCSBDataset"

    def __len__(self):
        return len(self.df)
    def _get_pdb_fpath(self, fname):

        pdb_fpath = Path(self.monomer_pdb_data_dir) / f"{fname}.pdb"
        if not pdb_fpath.exists():
            pdb_fpath = Path(self.complex_pdb_data_dir) / f"{fname}.pdb"
        
        return pdb_fpath
    
    def load_pdb(self,pdb_path, model, chain_id, seqlen):
        assert os.path.isfile(pdb_path), f"Cannot find pdb file at {pdb_path}."
        struct = pdb_parser.get_structure("", pdb_path)
        chain = struct[model][chain_id]

        atom_coords = (
            np.zeros((seqlen, rc.atom_type_num, 3)) * np.nan
        )  # (seqlens, 37, 3)
        for residue in chain:
            seq_idx = residue.id[1] - 1  # zero-based indexing
            for atom in residue:
                if atom.name in rc.atom_order.keys():
                    atom_coords[seq_idx, rc.atom_order[atom.name]] = atom.coord
            
        return atom_coords

    def sample_forward_diffusion(self,rigids_0, rigids_mask, seed = None):
        if seed is not None:
            random.seed(seed)
        
        t = max(0.01, random.random())

        # Sample forward diffusion
        diffused_feat_dict = self.diffuser.forward_marginal(
            rigids_0=rigids_0,
            t=t,
            diffuse_mask=rigids_mask.numpy(),
            as_tensor_7=False,
        )
        
        return t,diffused_feat_dict
    
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(self.df, pd.DataFrame):
            # sample from regular DataFrame
            row = self.df.iloc[idx]
        else:
            # sample from clusters
            assert isinstance(self.df, DataFrameGroupBy)
            sampled_group = self.df.get_group(self.group_keys[idx])
            row = sampled_group.sample(n=1).iloc[0]

        fname = row['PDB_ID']
        pdb_id = fname.split('_')[0]
        seqs = row.seqs
        
        # read pdb file
        pdb_fpath = self._get_pdb_fpath(fname)

        atom_coords = []
        for chain_id, chain_len in zip(row['chain_ids'], row['chain_lens']):
            try:
                atom_coords.append(self.load_pdb(pdb_path=pdb_fpath,model=row['model'], chain_id=chain_id, seqlen=chain_len))
            except Exception as e:
                assert 1 == 2 ,f"Error loading {pdb_id} chain {chain_id}: {e}"

        atom_coords = np.concatenate(atom_coords, axis=0)
        
        # filter seq       
        aatype = []
        chain_ids = []
        for idx, seq_item in enumerate(seqs):        
            seq_item_aatype = torch.LongTensor(
                [rc.restype_order_with_x.get(res, 20) for res in seq_item]
            )
            aatype.append(seq_item_aatype)
            chain_ids.append(torch.full((seq_item_aatype.size(0),), idx+1, dtype=torch.long))
        
        aatype = torch.cat(aatype)  
        chain_ids = torch.cat(chain_ids)  
        
        # remove center of mass
        atom_coords -= np.nanmean(atom_coords, axis=(0, 1), keepdims=True)
        all_atom_positions = torch.from_numpy(atom_coords)  # (seqlen, 37, 3)
        all_atom_mask = torch.all(
            ~torch.isnan(all_atom_positions), dim=-1
        )  # (seqlen, 37)

        all_atom_positions = torch.nan_to_num(
            all_atom_positions, 0.0
        )  # convert NaN to zero
        
        # ground truth backbone atomic coordinates
        gt_bb_coords = all_atom_positions[:, [0, 1, 2, 4], :]  # (seqlen, 4, 3)
        bb_coords_mask = all_atom_mask[:, [0, 1, 2, 4]]  # (seqlen, 4)

        # alphafold3 data transformation
        alphafold3_feat_dict = {
            "aatype": aatype.long(),
            "all_atom_positions": all_atom_positions.double(),
            "all_atom_mask": all_atom_mask.double(),
        }
        
        # alphafold3 feature processing
        alphafold3_feat_dict = data_transforms.atom37_to_frames(alphafold3_feat_dict)
        alphafold3_feat_dict = data_transforms.make_atom14_masks(alphafold3_feat_dict)
        alphafold3_feat_dict = data_transforms.make_atom14_positions(alphafold3_feat_dict)
        alphafold3_feat_dict = data_transforms.atom37_to_torsion_angles()(
            alphafold3_feat_dict
        )
        
        # ground truth rigids
        rigids_0 = ru.Rigid.from_tensor_4x4(
            alphafold3_feat_dict["rigidgroups_gt_frames"]
        )[:, 0]
        rigids_mask = alphafold3_feat_dict["rigidgroups_gt_exists"][:, 0]
        assert rigids_mask.sum() == torch.all(all_atom_mask[:, [0, 1, 2]], dim=-1).sum()
        
        # Sample forward diffusion
        seed = None
        if self.is_oder_smaple:
            seed = fname.split("_")[1] if "_" in fname and len(fname.split("_")) > 1 else None


        t, diffused_feat_dict = self.sample_forward_diffusion(rigids_0, rigids_mask, seed)
        
        # rigids_0: ru.Rigid,
        #     t: float,
        #     diffuse_mask: np.ndarray = None,
        #     as_tensor_7: bool=True,
        rigids_t = diffused_feat_dict["rigids_t"]

        for key, value in diffused_feat_dict.items():
            if isinstance(value, np.ndarray) or isinstance(value, np.float64):
                diffused_feat_dict[key] = torch.tensor(value)

        data_dict = {
            "fname": fname,
            "chain_name": row['chain_ids'],            
            "cluster_id": row[self.group_col][0] if self.group_col is not None else "NA",
            "aatype": aatype.long(),
            "chain_ids" :chain_ids,
            "rigids_0": rigids_0.to_tensor_7().float(),  # (seqlen, 7)
            "rigids_t": rigids_t.to_tensor_7().float(),  # (seqlen, 7)
            "rigids_mask": rigids_mask.float(),  # (seqlen,)
            "t": torch.tensor(t).float(),  # (,)
            "rot_score": diffused_feat_dict["rot_score"].float(),  # (seqlen, 3)
            "trans_score": diffused_feat_dict["trans_score"].float(),  # (seqlen, 3)
            "rot_score_norm": diffused_feat_dict["rot_score_scaling"].float(),  # (,)
            "trans_score_norm": diffused_feat_dict[
                "trans_score_scaling"
            ].float(),  # (,)
            "gt_torsion_angles": alphafold3_feat_dict[
                "torsion_angles_sin_cos"
            ].float(),  # (seqlen,7,2)
            "torsion_angles_mask": alphafold3_feat_dict[
                "torsion_angles_mask"
            ].float(),  # (seqlen,7)
            "rigidgroups_gt_frames": alphafold3_feat_dict[
                "rigidgroups_gt_frames"
            ].float(),
            "rigidgroups_alt_gt_frames": alphafold3_feat_dict[
                "rigidgroups_alt_gt_frames"
            ].float(),
            "rigidgroups_gt_exists": alphafold3_feat_dict[
                "rigidgroups_gt_exists"
            ].float(),
            "atom14_gt_positions": alphafold3_feat_dict["atom14_gt_positions"].float(),
            "atom14_alt_gt_positions": alphafold3_feat_dict[
                "atom14_alt_gt_positions"
            ].float(),
            "atom14_atom_is_ambiguous": alphafold3_feat_dict[
                "atom14_atom_is_ambiguous"
            ].float(),
            "atom14_gt_exists": alphafold3_feat_dict["atom14_gt_exists"].float(),
            "atom14_alt_gt_exists": alphafold3_feat_dict["atom14_alt_gt_exists"].float(),
            "atom14_atom_exists": alphafold3_feat_dict["atom14_atom_exists"].float(),
            "gt_bb_coords": gt_bb_coords.float(),  # (seqlen, 4, 3)
            "bb_coords_mask": bb_coords_mask.float(),  # (seqlen, 4)
        }

        if hasattr(self, "repr_loader"):
            pretrained_repr = self.repr_loader.load(seqres="_".join(seqs))
            data_dict["pretrained_node_repr"] = pretrained_repr.get(
                "pretrained_node_repr", None
            )
            data_dict["pretrained_edge_repr"] = pretrained_repr.get(
                "pretrained_edge_repr", None
            )
            
            assert data_dict["pretrained_node_repr"] is not None \
                    and data_dict["pretrained_edge_repr"] is not None, \
                    f"pretrained representation not found for {pdb_id}"
            
        return data_dict

    def collate(self, batch_list):
        batch = {"gt_feat": {}}
        gt_feat_name = [
            "rot_score",
            "trans_score",
            "rot_score_norm",
            "trans_score_norm",
            "gt_torsion_angles",
            "torsion_angles_mask",
            "gt_bb_coords",
            "bb_coords_mask",
            "rigids_0",
            "atom14_gt_positions",
            "atom14_alt_gt_positions",
            "atom14_atom_is_ambiguous",
            "atom14_gt_exists",
            "atom14_alt_gt_exists",
            "atom14_atom_exists",
            "rigidgroups_gt_frames",
            "rigidgroups_alt_gt_frames",
            "rigidgroups_gt_exists",
            "gt_force_0",
            "gt_energy_0",
        ]

        if "cluster_id" in batch_list[0].keys() and batch_list[0]["cluster_id"] != "NA":

            assert (
                np.array([feat_dict["cluster_id"] for feat_dict in batch_list]).std()
                == 0
            )
                
        lengths = torch.tensor(
            [feat_dict["aatype"].shape[0] for feat_dict in batch_list],
            requires_grad=False,
        )
        max_L = max(lengths)

        padding_mask = torch.arange(max_L).expand(
            len(lengths), max_L
        ) < lengths.unsqueeze(1)

        for key, val in batch_list[0].items():
            if (val is None) or (key in []):
                continue
            if key in [
                "chain_name",
                "output_name",
                "dataset_name",
                "cluster_id",
                "fname",
            ]:
                batched_val = [feat_dict[key] for feat_dict in batch_list]

            elif val.dim() == 0:
                batched_val = torch.stack([feat_dict[key] for feat_dict in batch_list])
                
            elif (val.dim() < 3) or (key not in ["pretrained_edge_repr"]):
                batched_val = pad_sequence(
                    [feat_dict[key] for feat_dict in batch_list],
                    batch_first=True,
                    padding_value=0,
                )
            else:
                assert key == "pretrained_edge_repr"
                batched_val = []
                C = batch_list[0]["pretrained_edge_repr"].shape[2]
                
                for feat_dict in batch_list:
                    edge = feat_dict["pretrained_edge_repr"]
                    L = edge.shape[0]
                    pad = torch.zeros(max_L, max_L, C)
                    pad[:L, :L, :] = edge
                    batched_val.append(pad[None, :])
                batched_val = torch.cat(batched_val, dim=0)
            if key in ["rigids_0", "rigids_t"]:
                bsz, seqlen = batched_val.shape[:2]
                batched_val = batched_val + torch.cat(
                    [~padding_mask[..., None], torch.zeros(bsz, seqlen, 6)], dim=-1
                )

            if key in gt_feat_name:
                batch["gt_feat"][key] = batched_val
            else:
                batch[key] = batched_val
            batch["padding_mask"] = padding_mask

        return batch


class ComplexMdDataset(RCSBDataset):

    def __init__(
        self,
        csv_path: str,
        mode: Literal["train", "val"],
        diffuser=None,
        monomer_pdb_data_dir: Optional[str] = None,
        complex_pdb_data_dir: Optional[str] = None,
        csv_processor_cfg: Optional[DictConfig] = None,
        repr_loader: Optional[DictConfig] = None,
        is_order_sample: bool = False,
        is_classify_sample: bool = False,
        classify_num: str = None,
        random_batch: bool = False,
        **kwargs,
    ):
        super().__init__(
            csv_path=csv_path,
            mode=mode,
            diffuser=diffuser,
            monomer_pdb_data_dir=monomer_pdb_data_dir,
            complex_pdb_data_dir=complex_pdb_data_dir,
            csv_processor_cfg=csv_processor_cfg,
            repr_loader=repr_loader,
            **kwargs,
        )

        self.is_classify_sample = is_classify_sample
        self.is_order_sample = is_order_sample
        self.random_batch = random_batch
        if self.is_classify_sample or self.is_order_sample:
            assert classify_num is not None, "if choose is_classify_sample or is_order_sample, classify_num should not be None"
            self.train_classify_num = f"{classify_num}_classify_num"

    def _process_csv(
        self,
        csv_path: str,
        csv_processor_cfg: Optional[DictConfig] = None,
    ):
        df = pd.read_pickle(csv_path)
        
        assert "chain_lens" in df.columns, "chain_lens not found in csv file"

        if csv_processor_cfg is not None:
            min_seqlen = csv_processor_cfg.get("min_seqlen", 0)
            max_seqlen = csv_processor_cfg.get("max_seqlen", 10000)
            df = df[(df["chain_lens"].apply(min) >= min_seqlen) & (df["chain_lens"].apply(max) <= max_seqlen)]

            group_col = csv_processor_cfg.get("groupby", None)
            assert group_col is None, "Atlas dataset does not need to specify 'groupby'"
            num_samples = csv_processor_cfg.get("num_samples", 1)
            if num_samples > 1:
                df = df.loc[df.index.repeat(num_samples)]

        self.group_col = None

        return df

    @property
    def name(self):
        return "ComplexMdDataset"
    
    
    def _get_pdb_fpath(self, fname):
        
        pdb_id = fname.split("_")[0]
        
        frame = np.random.randint(1,2002)
            
        pdb_fpath =  Path(f"{self.complex_pdb_data_dir}/{pdb_id}/{pdb_id.lower()}_frame{int(frame)}_renumbered.pdb")
        if (self.monomer_pdb_data_dir is not None) and  (pdb_fpath.exists()):
            pdb_fpath = Path(f"{self.monomer_pdb_data_dir}/{pdb_id}/{pdb_id.lower()}_frame{int(frame)}_renumbered.pdb")
        
        assert pdb_fpath.exists(), f"{pdb_fpath} does not exist"
        
        return pdb_fpath
    
class AtlasDataset(RCSBDataset):

    def __init__(
        self,
        csv_path: str,
        mode: Literal["train", "val"],
        diffuser=None,
        monomer_pdb_data_dir: Optional[str] = None,
        complex_pdb_data_dir: Optional[str] = None,
        csv_processor_cfg: Optional[DictConfig] = None,
        repr_loader: Optional[DictConfig] = None,
        random_batch: bool = False,
        **kwargs,
    ):
        super().__init__(
            csv_path=csv_path,
            mode=mode,
            diffuser=diffuser,
            monomer_pdb_data_dir=monomer_pdb_data_dir,
            complex_pdb_data_dir=complex_pdb_data_dir,
            csv_processor_cfg=csv_processor_cfg,
            repr_loader=repr_loader,
            **kwargs,
        )

        self.random_batch = random_batch

    def _process_csv(
        self,
        csv_path: str,
        csv_processor_cfg: Optional[DictConfig] = None,
    ):
        df = pd.read_pickle(csv_path)
        
        assert "chain_lens" in df.columns, "chain_lens not found in csv file"

        if csv_processor_cfg is not None:
            min_seqlen = csv_processor_cfg.get("min_seqlen", 0)
            max_seqlen = csv_processor_cfg.get("max_seqlen", 10000)
            df = df[(df["chain_lens"].apply(min) >= min_seqlen) & (df["chain_lens"].apply(max) <= max_seqlen)]

            group_col = csv_processor_cfg.get("groupby", None)
            assert group_col is None, "Atlas dataset does not need to specify 'groupby'"
            num_samples = csv_processor_cfg.get("num_samples", 1)
            if num_samples > 1:
                df = df.loc[df.index.repeat(num_samples)]

        self.group_col = None

        return df

    @property
    def name(self):
        return "atlas"
    
    def load_pdb(self,pdb_path, model, chain_id, seqlen):
        assert os.path.isfile(pdb_path), f"Cannot find pdb file at {pdb_path}."
        struct = pdb_parser.get_structure("", pdb_path)
        chain = struct[model][' ']

        atom_coords = (
            np.zeros((seqlen, rc.atom_type_num, 3)) * np.nan
        )  # (seqlens, 37, 3)
        for residue in chain:
            seq_idx = residue.id[1] - 1  # zero-based indexing
            for atom in residue:
                if atom.name in rc.atom_order.keys():
                    atom_coords[seq_idx, rc.atom_order[atom.name]] = atom.coord
            
        return atom_coords
    
    
    def _get_pdb_fpath(self, pdb_id):

        # Random sample a conformation
        rep = np.random.randint(3) + 1
        frame = np.random.randint(1,2002)
        pdb_fpath = Path(
            self.monomer_pdb_data_dir, f"{pdb_id.upper()}_PROD_R{rep}/{pdb_id}_prod_R{rep}_frame{frame}_renumbered.pdb"
        )

        if not pdb_fpath.exists():
            pdb_fpath = Path(
                self.complex_pdb_data_dir, f"{pdb_id.upper()}_PROD_R{rep}/{pdb_id}_prod_R{rep}_frame{frame}_renumbered.pdb"
            )
        
        return pdb_fpath

class MonomerAndComplexMdDataset(RCSBDataset):
    def __init__(
        self,
        csv_path: str,
        mode: Literal["train", "val"],
        diffuser=None,
        monomer_pdb_data_dir: Optional[str] = None,
        complex_pdb_data_dir: Optional[str] = None,
        csv_processor_cfg: Optional[DictConfig] = None,
        repr_loader: Optional[DictConfig] = None,
        is_order_sample: bool = False,
        is_classify_sample: bool = False,
        classify_num: str = None,
        random_batch: bool = False,
        **kwargs,
    ):
        super().__init__(
            csv_path=csv_path,
            mode=mode,
            diffuser=diffuser,
            monomer_pdb_data_dir=monomer_pdb_data_dir,
            complex_pdb_data_dir=complex_pdb_data_dir,
            csv_processor_cfg=csv_processor_cfg,
            repr_loader=repr_loader,
            **kwargs,
        )

        self.is_classify_sample = is_classify_sample
        self.is_order_sample = is_order_sample
        self.random_batch = random_batch
        if self.is_classify_sample or self.is_order_sample:
            assert classify_num is not None, "if choose is_classify_sample or is_order_sample, classify_num should not be None"
            self.train_classify_num = f"{classify_num}_classify_num"

    def _process_csv(
        self,
        csv_path: str,
        csv_processor_cfg: Optional[DictConfig] = None,
    ):
        df = pd.read_pickle(csv_path)
        
        assert "chain_lens" in df.columns, "chain_lens not found in csv file"

        if csv_processor_cfg is not None:
            min_seqlen = csv_processor_cfg.get("min_seqlen", 0)
            max_seqlen = csv_processor_cfg.get("max_seqlen", 10000)
            df = df[(df["chain_lens"].apply(min) >= min_seqlen) & (df["chain_lens"].apply(max) <= max_seqlen)]

            group_col = csv_processor_cfg.get("groupby", None)
            assert group_col is None, "Atlas dataset does not need to specify 'groupby'"
            num_samples = csv_processor_cfg.get("num_samples", 1)
            if num_samples > 1:
                df = df.loc[df.index.repeat(num_samples)]

        self.group_col = None

        return df

    @property
    def name(self):
        return "MonomerAndComplexMdDataset"
    
    
    def load_pdb(self,pdb_path, model, chain_id, seqlen):
        assert os.path.isfile(pdb_path), f"Cannot find pdb file at {pdb_path}."
        struct = pdb_parser.get_structure("", pdb_path)
        if "_prod_R" in pdb_path:
            chain = struct[model][' ']
        else:
            chain = struct[model][chain_id]

        atom_coords = (
            np.zeros((seqlen, rc.atom_type_num, 3)) * np.nan
        )  # (seqlens, 37, 3)
        for residue in chain:
            seq_idx = residue.id[1] - 1  # zero-based indexing
            for atom in residue:
                if atom.name in rc.atom_order.keys():
                    atom_coords[seq_idx, rc.atom_order[atom.name]] = atom.coord
            
        return atom_coords    
    
    def _get_pdb_fpath(self, fname):
        # fname 格式 {PDB_ID}_frame{int(frame)}_rep{int(rep)}_renumbered.pdb
        
        # 正则表达式
        pdb_id = fname.split("_")[0]
        frame = re.findall(r'frame(\d+)', fname)[0]
        rep = re.findall(r'rep(\d+)', fname)[0] if re.findall(r'rep(\d+)', fname) else None
        
        # complex pdb path
        pdb_fpath =  Path(f"{self.complex_pdb_data_dir}/{pdb_id}/{pdb_id.lower()}_frame{int(frame)}_renumbered.pdb")
        if not pdb_fpath.exists():
            # monomer pdb path
            chain_id = fname.split("_")[1]
            pdb_fpath = Path(self.monomer_pdb_data_dir, f"{pdb_id.upper()}_{chain_id}_PROD_R{rep}/{pdb_id}_{chain_id}_prod_R{rep}_frame{frame}_renumbered.pdb")

        return pdb_fpath
    
    
    
    
class GenDataset(torch.utils.data.Dataset):
    """Dataset for conformation generation"""

    def __init__(
        self,
        csv_path: str,
        num_samples=1000,
        alphafold3_cfg: Optional[DictConfig] = None,
        dp_cfg: Optional[DictConfig] = None,
    ):
        self.csv_path = csv_path
        self.num_samples = num_samples
        if csv_path:
            df = pd.read_pickle(csv_path)
            df = df.loc[df.index.repeat(num_samples)].reset_index(drop=True)
            df["sample_id"] = df.groupby(["PDB_ID"]).cumcount()
        self.df = df
        
        # init alphafold3 repr 
        if alphafold3_cfg is not None:
            self.repr_loader = Alphafold3ReprLoader(
                complex_data_root=alphafold3_cfg["complex_repr_data_root"],
                monomer_data_root=alphafold3_cfg["monomer_repr_data_root"],
                num_recycles=alphafold3_cfg["num_recycle"],
                node_size=alphafold3_cfg["node_size"],
                edge_size=alphafold3_cfg["edge_size"],
                seqres_to_index_path = alphafold3_cfg["seqres_to_index_path"]
            )
        else:
            self.repr_loader = None
        

        self.dynamic_batching = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        seqs = row.seqs

        data_dict = {
            "fname": row.PDB_ID,
            "output_name": f"{row.PDB_ID}_sample{row.sample_id}.pdb",
            "chain_name": row.chain_ids,
        }

        # pretrained representations
        if self.repr_loader is not None:
            pretrained_repr = self.repr_loader.load("_".join(seqs))
            # pretrained_repr = self.repr_loader.load(pdb_id=row.PDB_ID)
            data_dict["pretrained_node_repr"] = pretrained_repr.get(
                "pretrained_node_repr", None
            )
            data_dict["pretrained_edge_repr"] = pretrained_repr.get(
                "pretrained_edge_repr", None
            )
        
        # filter seq       
        aatype = []
        chain_ids = []
        for idx, seq_item in enumerate(seqs):        
            seq_item_aatype = torch.LongTensor(
                [rc.restype_order_with_x.get(res, 20) for res in seq_item]
            )
            aatype.append(seq_item_aatype)
            chain_ids.append(torch.full((seq_item_aatype.size(0),), idx+1, dtype=torch.long))
        
        aatype = torch.cat(aatype)
        chain_ids = torch.cat(chain_ids)  
        data_dict["aatype"] = aatype.long()
        data_dict["chain_ids"] = chain_ids.long()
        data_dict['atom14_atom_exists'] = data_transforms.make_atom14_masks(data_dict)['atom14_atom_exists']

        return data_dict

    def collate(self, batch_list):
        return RCSBDataset.collate(self, batch_list)
