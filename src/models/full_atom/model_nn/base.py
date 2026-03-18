import torch
from torch import nn


from src.openfold_local.np import residue_constants as rc
from src.openfold_local.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from .embedder import Embedder
from .structure_module import StructureModule

def compute_res_idx(aatype: torch.Tensor, chain_ids: torch.Tensor) -> torch.Tensor:
    """
    根据 `chain_ids` 和 `aatype` 批量生成每条链的残基编码。
    参数:
    - aatype: [batch_size, sequence_length]，残基种类数组
    - chain_ids: [batch_size, sequence_length]，残基所属链的指示数组
    返回:
    - res_idx: [batch_size, sequence_length]，每条链的残基编号
    """
    batch_size, sequence_length = aatype.size()
    res_idx = torch.zeros_like(chain_ids)  # 初始化与 chain_ids 大小相同的张量
    
    for batch_idx in range(batch_size):
        # 获取当前 batch 的数据
        seq_mask = chain_ids[batch_idx]
        for chain_id in seq_mask.unique():  # 遍历当前样本的每条链
            chain_mask = seq_mask == chain_id  # 当前链的掩码
            res_idx[batch_idx, chain_mask] = torch.arange(chain_mask.sum(), device=aatype.device)
    
    return res_idx

class FoldNet(nn.Module):
    def __init__(
        self,
        embedder_cfg,
        structure_module_cfg,
        confidence_head_cfg,
    ):
        super().__init__()

        self.embedder = Embedder(time_emb_size=embedder_cfg['time_emb_size'],
                                 scale_t=embedder_cfg['scale_t'],
                                 res_idx_emb_size=embedder_cfg['res_idx_emb_size'],
                                 r_max = embedder_cfg['r_max'],
                                 num_rbf=embedder_cfg['num_rbf'],
                                 rbf_min=embedder_cfg['rbf_min'],
                                 rbf_max=embedder_cfg['rbf_max'],
                                 pretrained_node_repr_size=embedder_cfg['pretrained_node_repr_size'],
                                 pretrained_edge_repr_size=embedder_cfg['pretrained_edge_repr_size'],
                                 node_emb_size=embedder_cfg['node_emb_size'],
                                 edge_emb_size=embedder_cfg['edge_emb_size'],
                                 use_af3_relative_pos_encoding = embedder_cfg['use_af3_relative_pos_encoding'],
                                 )
        
        self.structure_module = StructureModule(num_ipa_blocks=structure_module_cfg['num_ipa_blocks'],
                                                c_s=structure_module_cfg['c_s'],
                                                c_z=structure_module_cfg['c_z'],
                                                c_hidden=structure_module_cfg['c_hidden'],
                                                c_skip=structure_module_cfg['c_skip'],
                                                no_heads=structure_module_cfg['no_heads'],
                                                no_qk_points=structure_module_cfg['no_qk_points'],
                                                no_v_points=structure_module_cfg['no_v_points'],
                                                seq_tfmr_num_heads=structure_module_cfg['seq_tfmr_num_heads'],
                                                seq_tfmr_num_layers=structure_module_cfg['seq_tfmr_num_layers'],
                                                )

    def forward(
        self,
        aatype,
        chain_ids,
        padding_mask,
        t,
        atom14_atom_exists,
        rigids_t,  # unscaled
        rigids_mask,
        res_idx=None,
        pretrained_node_repr=None,
        pretrained_edge_repr=None,
        **kwargs,
    ):
            
        if res_idx is None:
            res_idx=compute_res_idx(aatype=aatype, chain_ids=chain_ids)

        node_feat, edge_feat = self.embedder(
            padding_mask=padding_mask,
            t=t,
            res_idx=res_idx,
            chain_ids=chain_ids,
            rigids_t=rigids_t,
            pretrained_node_repr=pretrained_node_repr,
            pretrained_edge_repr=pretrained_edge_repr,
        )

        model_out = self.structure_module(
            rigids_t=rigids_t,  # (B, L, 7) # unscaled
            node_feat=node_feat,  # (B, L, node_emb_size)
            edge_feat=edge_feat,  # (B, L, L, edge_emb_size)
            node_mask=rigids_mask,  # (B, L)
            padding_mask=padding_mask,  # (B, L)
        )
        
        all_frames_to_global = self.torsion_angles_to_frames(
            model_out["pred_rigids_0"],
            model_out["pred_torsions"],
            aatype,
        )

        model_out["pred_atom14"] = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            torch.fmod(aatype, 20),
        )
        
        model_out["pred_sidechain_frames"] = all_frames_to_global.to_tensor_4x4()

        return model_out

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    rc.restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    rc.restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    rc.restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    rc.restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
        self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
