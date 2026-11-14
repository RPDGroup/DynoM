"""Compare MD and predicted C-alpha RMSF profiles for one sample."""

import mdtraj as md
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from analysis_common import load_trajectories, parse_args


def calculate_rmsf_profiles(reference, predicted, output_dir):
    """Superpose both ensembles on MD frame zero and compare common C-alpha atoms.

    Keep the existing reference-based md.rmsf call. The first MD frame now
    supplies the reference structure instead of a hard-coded external PDB.
    All input frames participate; distances and RMSF values are in nanometers.
    """
    backbone = reference.topology.select("backbone")
    ca_indices = reference.topology.select("name CA and protein")
    if len(backbone) == 0 or len(ca_indices) == 0:
        raise ValueError("No common backbone/C-alpha atoms remain for RMSF analysis.")
    anchor = reference[0]
    md_aligned = reference.superpose(anchor, frame=0, atom_indices=backbone,
                                     ref_atom_indices=backbone)
    pred_aligned = predicted.superpose(anchor, frame=0, atom_indices=backbone,
                                      ref_atom_indices=backbone)
    md_rmsf = md.rmsf(md_aligned, anchor, atom_indices=ca_indices, parallel=False)
    pred_rmsf = md.rmsf(pred_aligned, anchor, atom_indices=ca_indices, parallel=False)
    labels = []
    for index in ca_indices:
        residue = reference.topology.atom(index).residue
        labels.append(f"{residue.name}_{residue.chain.index}_{residue.resSeq}")

    # Use identical residue labels in both profiles; no cross-sample CSV merging
    # or label-cleaning notebook is needed for the Pearson comparison.
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("md", md_rmsf), ("pred", pred_rmsf)):
        pd.DataFrame({
            "Residue_Index": np.arange(len(labels)),
            "Residue_Label": labels,
            "RMSF(nm)": values,
        }).to_csv(output_dir / f"{name}_rmsf.csv", index=False, float_format="%.6g")
    pd.DataFrame({
        "Residue_Index": np.arange(len(labels)), "Residue_Label": labels,
        "MD_RMSF(nm)": md_rmsf, "Pred_RMSF(nm)": pred_rmsf,
    }).to_csv(output_dir / "rmsf_comparison.csv", index=False, float_format="%.6g")
    correlation = float("nan")
    if len(labels) >= 2 and np.std(md_rmsf) > 0 and np.std(pred_rmsf) > 0:
        correlation = float(pearsonr(md_rmsf, pred_rmsf)[0])
    return {
        "rmsf_pearson": correlation, "n_common_ca": len(labels),
        "n_md_frames": reference.n_frames, "n_pred_frames": predicted.n_frames,
    }


def main(argv=None):
    args = parse_args(__doc__, argv, require_lagtimes=False)
    reference, predicted = load_trajectories(args.md_traj, args.md_top, args.pred_traj)
    result = calculate_rmsf_profiles(reference, predicted, args.output_dir)
    result.update(sample=args.pred_traj.stem, md_traj=str(args.md_traj),
                  md_top=str(args.md_top) if args.md_top else "", pred_traj=str(args.pred_traj))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(args.output_csv, index=False)
    print(f"RMSF analysis complete: {args.output_csv}")


if __name__ == "__main__":
    main()
