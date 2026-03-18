from typing import Optional,Dict
from src.openfold_local.np import residue_constants
import torch

def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    asym_id: Optional[torch.Tensor] = None,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    fp_type = atom14_pred_positions.dtype
    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None]
        < residue_index[..., None, :, None, None]
    )

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(2), num_classes=14
    )
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(0), num_classes=14
    )
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (residue_index[..., :, None] + 1) == residue_index[..., None, :]

    if asym_id is not None:
        neighbour_mask = neighbour_mask & (asym_id[..., :, None] == asym_id[..., None, :])

    neighbour_mask = neighbour_mask[..., None, None]

    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = residue_constants.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(
        *((1,) * len(residue_index.shape[:-1])), 1
    ).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, dim=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    per_atom_num_clash = torch.sum(clash_mask, dim=(-4, -2)) + torch.sum(clash_mask, dim=(-3, -1))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, dim=(-4, -2)),
        torch.amax(clash_mask, dim=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
        "per_atom_num_clash": per_atom_num_clash  # shape (N, 14)
    }


def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]

def get_atom_radius(
    atom14_pred_positions: torch.Tensor,
    aatype: torch.Tensor,
    atom14_atom_exists: torch.Tensor,) -> torch.Tensor:

    atomtype_radius = [
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]

    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)

    residx_atom14_to_atom37 = get_rc_tensor(
        residue_constants.RESTYPE_ATOM14_TO_ATOM37, aatype
    )
    atom14_atom_radius = (
        atom14_atom_exists
        * atomtype_radius[residx_atom14_to_atom37.long()]
    )
    
    return atom14_atom_radius


if __name__ == "__main__":
    # test code
    atom14_pred_positions = torch.load("/home/tom/fsas/Dynamic_Proteins/confdiff_complex/tmp/clash/9FZD_pred_atom14.pth")
    aatype = torch.load("/home/tom/fsas/Dynamic_Proteins/confdiff_complex/tmp/clash/9FZD_aatype.pth")
    asym_id = torch.load("/home/tom/fsas/Dynamic_Proteins/confdiff_complex/tmp/clash/9FZD_chain_ids.pth")
    gt_feat = torch.load("/home/tom/fsas/Dynamic_Proteins/confdiff_complex/tmp/clash/9FZD_gt_feat.pth")
    residue_index = torch.load("/home/tom/fsas/Dynamic_Proteins/confdiff_complex/tmp/clash/9FZD_res_idx.pth")
    atom14_atom_exists = gt_feat['atom14_atom_exists'].to(torch.float32)
    atom14_atom_radius = get_atom_radius(atom14_pred_positions=atom14_pred_positions,
                                        aatype=aatype, 
                                        atom14_atom_exists=atom14_atom_exists)
    # atom14_atom_radius

    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        asym_id=asym_id,
    )
    print(between_residue_clashes)
