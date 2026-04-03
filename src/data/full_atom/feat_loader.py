import torch
import numpy as np
import pandas as pd
from pathlib import PosixPath, Path
from typing import Dict, Union

PATH_TYPE = Union[str, PosixPath]


class Alphafold3ReprLoader(object):
    def __init__(
        self,
        monomer_data_root: str,
        complex_data_root: str,
        seqres_to_index_path: str,
        num_recycles: int = 3,
        node_size: int = 384,
        edge_size: int = 128,
    ):
        self.num_recycles = num_recycles
        self.node_size = node_size
        self.edge_size = edge_size

        self.seqres_to_index = pd.read_pickle(seqres_to_index_path)
        self.seqres_to_index = self.seqres_to_index.set_index("seqs_key")
        self.monomer_data_root  = monomer_data_root
        self.complex_data_root  = complex_data_root

    def load_repr(self, repr_type: str, pdb_id: str) -> torch.Tensor:
        repr_path = f"{self.complex_data_root}/{pdb_id}_{repr_type}_repr_recycle{self.num_recycles}.npy"
        if not Path(repr_path).exists():
            repr_path = f"{self.monomer_data_root}/{pdb_id}_{repr_type}_repr_recycle{self.num_recycles}.npy"

        assert Path(repr_path).exists(), f"{repr_path} repr_file not found!"
        
        try:
            if Path(repr_path).exists() and ".pt" in repr_path:
                return torch.load(repr_path).float()
            elif Path(repr_path).exists() and ".npy" in repr_path:
                return torch.from_numpy(np.load(repr_path, mmap_mode="r")).float()
        except Exception as e:
            raise FileNotFoundError(
                f"ERROR: {e}, {pdb_id}: {repr_type} not found: {str(repr_path)}"
            )

    def load(self, seqres: str ) -> Dict[str, torch.Tensor]:
        """Load node and/or edge representations from pretrained model
        Returns:
            {
                node_repr: Tensor[seqlen, repr_dim], float
                edge_repr: Tensor[seqlen, seqlen, repr_dim], float
            }
        """

        pdb_id = self.seqres_to_index.at[seqres, "PDB_ID"]

        if isinstance(pdb_id, (list, pd.Series)):
            pdb_id = pdb_id[0]
        else:
            pdb_id = pdb_id

        repr_dict = {}

        # -------------------- Node repr --------------------
        if self.node_size > 0:
            repr_dict["pretrained_node_repr"] = self.load_repr("single", pdb_id)
            # print(repr_dict['pretrained_node_repr'].shape)

        # -------------------- Edge repr --------------------
        if self.edge_size > 0:
            repr_dict["pretrained_edge_repr"] = self.load_repr("pair", pdb_id)

        return repr_dict
