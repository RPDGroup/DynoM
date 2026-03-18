"""
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""
import math
import torch
from torch import nn
from torch.nn.functional import one_hot
from torch.nn import Linear

from src.openfold_local.utils import rigid_utils as ru

def sinusoidal_embedding(
        pos: torch.Tensor,
        emb_size: int,
        max_pos: int = 10000,
) -> torch.Tensor:
    assert -max_pos <= pos.min().item() <= max_pos
    assert emb_size % 2 == 0, "Please use an even embedding size."
    half_emb_size = emb_size // 2
    idx = torch.arange(half_emb_size, dtype=torch.float32, device=pos.device)
    exponent = -1 * idx * math.log(max_pos) / (half_emb_size - 1)
    emb = pos[..., None] * torch.exp(exponent)  # (..., half_emb_size)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (..., emb_size)
    assert emb.size() == pos.size() + torch.Size([emb_size]), "Embedding size mismatch."
    return emb


class GaussianSmearing(nn.Module):
    def __init__(
            self,
            num_rbf: int,
            rbf_min: float,
            rbf_max: float,
    ):
        super().__init__()
        offset = torch.linspace(rbf_min, rbf_max, num_rbf)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    @property
    def device(self):
        return self.offset.device

    def forward(
            self,
            dist: torch.Tensor,
    ) -> torch.Tensor:
        diff = dist[..., None] - self.offset
        return torch.exp(self.coeff * torch.pow(diff, 2)).float()


class Embedder(nn.Module):
    def __init__(
            self,
            time_emb_size: int,
            scale_t: float,  # upscale node_t for sinusoidal embedding
            res_idx_emb_size: int,
            r_max: int,
            num_rbf: int,
            rbf_min: float,
            rbf_max: float,
            pretrained_node_repr_size: int,
            pretrained_edge_repr_size: int,
            node_emb_size: int,
            edge_emb_size: int,
            use_dp_repr: bool = False,
            dp_repr_size: int = 0,
            use_af3_relative_pos_encoding: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.time_emb_func = lambda x: sinusoidal_embedding(
            pos=x, emb_size=time_emb_size
        )
        self.res_idx_emb_func = lambda x: sinusoidal_embedding(
            pos=x, emb_size=res_idx_emb_size
        )

        self.gaussian_smearing = GaussianSmearing(
            num_rbf=num_rbf, rbf_min=rbf_min, rbf_max=rbf_max
        )

        self.scale_t = scale_t
        self.r_max = r_max
        self.use_af3_relative_pos_encoding = use_af3_relative_pos_encoding
        
        if pretrained_node_repr_size > 0:
            self.pretrained_node_repr_layernorm = nn.LayerNorm(
                pretrained_node_repr_size
            )
        if pretrained_edge_repr_size > 0:
            self.pretrained_edge_repr_layernorm = nn.LayerNorm(
                pretrained_edge_repr_size
            )
        if self.use_af3_relative_pos_encoding:
            self.linear_layer = Linear(2 * self.r_max + 2, res_idx_emb_size,bias = False)
        # node_inp_size = time_emb_size + res_idx_emb_size + pretrained_node_repr_size + 1  # chain_id_size = 1
        node_inp_size = time_emb_size + res_idx_emb_size + pretrained_node_repr_size
        self.node_mlp = nn.Sequential(
            nn.Linear(node_inp_size, node_emb_size),
            nn.ReLU(),
            nn.Linear(node_emb_size, node_emb_size),
            nn.ReLU(),
            nn.Linear(node_emb_size, node_emb_size),
            nn.LayerNorm(node_emb_size),
        )

        edge_inp_size = (
                time_emb_size + res_idx_emb_size + num_rbf + pretrained_edge_repr_size
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_inp_size, edge_emb_size),
            nn.ReLU(),
            nn.Linear(edge_emb_size, edge_emb_size),
            nn.ReLU(),
            nn.Linear(edge_emb_size, edge_emb_size),
            nn.LayerNorm(edge_emb_size),
        )


    @property
    def device(self):
        return self.gaussian_smearing.device

    def forward(
            self,
            res_idx,
            chain_ids,
            padding_mask,
            t,
            rigids_t,
            pretrained_node_repr=None,
            pretrained_edge_repr=None,
            dp_repr = None,
            **kwargs,
    ):

        B, L = res_idx.size()[:2]

        """ Node features. """

        # time embedding
        node_t = t * self.scale_t  # upscale time for sinusoidal embedding
        node_time_emb = torch.tile(
            self.time_emb_func(node_t)[:, None, :], (1, L, 1)  # (B, L, time_emb_size)
        )

        # residue index embedding
        res_idx_emb = self.res_idx_emb_func(res_idx)  # (B, L, res_idx_emb_size)
        res_idx_emb = res_idx_emb * padding_mask.unsqueeze(-1).type_as(res_idx_emb)
        # chain_ids_emb = chain_ids.unsqueeze(dim=2)

        # concat node features
        # node_feat = torch.cat([node_time_emb, res_idx_emb, chain_ids_emb], dim=-1)
        node_feat = torch.cat([node_time_emb, res_idx_emb], dim=-1)

        if (pretrained_node_repr is not None) and (
                self.pretrained_node_repr_layernorm is not None
        ):
            node_repr = self.pretrained_node_repr_layernorm(pretrained_node_repr)
            node_feat = torch.cat([node_feat, node_repr], dim=-1)
        node_feat = self.node_mlp(node_feat)

        """ Edge features. """
        # time embedding
        edge_time_emb = torch.tile(
            self.time_emb_func(node_t)[:, None, None, :],
            (1, L, L, 1),  # (B, L, L, time_emb_size)
        )
        
        if self.use_af3_relative_pos_encoding:
            # relative residue embedding
            chain_index = chain_ids[:, :, None] == chain_ids[:, None, :]
            rel_pos = (res_idx[:, :, None] - res_idx[:, None, :])
            d_residue = torch.clip(
                rel_pos + self.r_max,
                0,
                2 * self.r_max,
            )
            d_residue = torch.where(
                chain_index, d_residue, torch.zeros_like(d_residue) + 2 * self.r_max + 1
            )
            rel_pos = one_hot(d_residue, 2 * self.r_max + 2)
            rel_res_idx_emb = self.linear_layer(rel_pos.float())
        else:
            pair_mask = chain_ids[:, :, None] == chain_ids[:, None, :]

            # relative residue index embedding
            rel_res_idx_emb = self.res_idx_emb_func(
                (res_idx[:, :, None] - res_idx[:, None, :]) * pair_mask
            )  # (B, L, L, res_idx_emb_size)
            
            
        # edge length embedding
        if not isinstance(rigids_t, ru.Rigid):
            trans_t = rigids_t[..., 4:]
        else:
            trans_t = rigids_t.get_trans()  # (B, L, 3)
        padding_2d = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
        trans_t = trans_t * 0.1  # (B, L, 3) * padding
        edge_len = (
                torch.norm(
                    trans_t[:, :, None, :] - trans_t[:, None, :, :], dim=-1, keepdim=False
                )
                * padding_2d
        )
        edge_len_rbf = self.gaussian_smearing(edge_len)  # (B, L, L, num_rbf)

        # concat edge features
        edge_feat = torch.cat([edge_time_emb, rel_res_idx_emb, edge_len_rbf], dim=-1)

        if (pretrained_edge_repr is not None) and (
                self.pretrained_edge_repr_layernorm is not None
        ):
            edge_repr = self.pretrained_edge_repr_layernorm(pretrained_edge_repr)
            edge_feat = torch.cat([edge_feat, edge_repr], dim=-1)

        edge_feat = self.edge_mlp(edge_feat)  # (B, L, L, edge_emb_size)

        """dp_repr -> update edge and node"""
        if dp_repr is not None:
            node_scale = self.dp_node_scale(dp_repr)  # (B, node_emb_size)
            node_shift = self.dp_node_shift(dp_repr)
            node_feat = node_feat * node_scale[:, None, :] + node_shift[:, None, :]

            edge_scale = self.dp_edge_scale(dp_repr)  # (B, edge_emb_size)
            edge_shift = self.dp_edge_shift(dp_repr)
            edge_feat = edge_feat * edge_scale[:, None, None, :] + edge_shift[:, None, None, :]


        return node_feat, edge_feat

