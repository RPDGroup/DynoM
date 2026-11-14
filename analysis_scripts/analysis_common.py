"""Shared terminal arguments and trajectory alignment for one sample."""

import argparse
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def parse_args(description, argv=None, require_lagtimes=True, chain_mapping=False):
    """Accept one MD trajectory and one predicted ensemble, without batch lists."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--md_traj", required=True, type=Path,
                        help="MD trajectory file for one sample.")
    parser.add_argument("--md_top", type=Path,
                        help="Optional MD topology; required for formats such as XTC or DCD.")
    parser.add_argument("--pred_traj", required=True, type=Path,
                        help="Predicted ensemble with embedded topology, such as a multi-model PDB.")
    parser.add_argument("--lagtimes", nargs="+", type=positive_int, required=require_lagtimes,
                        help="Positive TICA lag times in MD frames, separated by spaces.")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory for this sample's analysis artifacts.")
    parser.add_argument("--output_csv", required=True, type=Path,
                        help="Summary CSV path; replaced by this invocation's sample results.")
    if chain_mapping:
        parser.add_argument("--chain_map", default="",
                            help="Optional MD:prediction chain mapping, for example A:X,B:Y.")
    args = parser.parse_args(argv)
    for key in ("md_traj", "md_top", "pred_traj", "output_dir", "output_csv"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    for path in (args.md_traj, args.md_top, args.pred_traj):
        if path is not None and not path.is_file():
            parser.error(f"Input must be a file: {path}")
    if args.output_csv in {args.md_traj, args.md_top, args.pred_traj}:
        parser.error("--output_csv must differ from the input files")
    if args.lagtimes is not None:
        args.lagtimes = list(dict.fromkeys(args.lagtimes))
    return args


def map_common_atoms(reference, predicted):
    """Match named atoms at aligned protein-residue positions in trajectory order.

    Retain the release's BLOSUM62 alignment and internal gap penalties. Both
    trajectories are sliced with the resulting correspondence, so their atom
    and residue dimensions agree even when one sequence has missing residues.
    """
    from Bio.Align import PairwiseAligner
    from Bio.SeqUtils import seq1

    ref_seq = "".join(reference.topology.to_fasta())
    pred_seq = "".join(predicted.topology.to_fasta())
    if not ref_seq or not pred_seq:
        raise ValueError("No protein sequence could be extracted from the trajectories.")
    aligner = PairwiseAligner(scoring="blastp")
    aligner.open_internal_insertion_score = -14
    aligner.extend_internal_insertion_score = -2
    aligner.open_internal_deletion_score = -14
    aligner.extend_internal_deletion_score = -2
    aligned_ref, aligned_pred = aligner.align(ref_seq, pred_seq)[0]
    ref_residues = list(reference.topology.residues)
    pred_residues = list(predicted.topology.residues)
    ref_indices, pred_indices = [], []
    ref_pos = pred_pos = 0
    for ref_char, pred_char in zip(aligned_ref, aligned_pred):
        if ref_char != "-" and pred_char != "-":
            ref_res = ref_residues[ref_pos]
            pred_res = pred_residues[pred_pos]
            if seq1(ref_res.name) == ref_char and seq1(pred_res.name) == pred_char:
                ref_atoms = {atom.name: atom.index for atom in ref_res.atoms}
                pred_atoms = {atom.name: atom.index for atom in pred_res.atoms}
                for name in sorted(ref_atoms.keys() & pred_atoms.keys()):
                    ref_indices.append(ref_atoms[name])
                    pred_indices.append(pred_atoms[name])
        if ref_char != "-":
            ref_pos += 1
        if pred_char != "-":
            pred_pos += 1
    if not ref_indices:
        raise ValueError("No common protein atoms were found after sequence alignment.")
    return ref_indices, pred_indices


def load_trajectories(md_traj, md_top, pred_traj, match_atoms=True):
    """Load all frames and map protein atoms; no implicit stride or path inference."""
    import mdtraj as md

    options = {"top": str(md_top)} if md_top is not None else {}
    try:
        reference = md.load(str(md_traj), **options)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load MD trajectory {md_traj}; supply --md_top if needed: {exc}") from exc
    try:
        predicted = md.load(str(pred_traj))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load {pred_traj}; --pred_traj must contain its own topology: {exc}") from exc
    if reference.n_frames < 1 or predicted.n_frames < 1:
        raise ValueError("Both trajectories must contain at least one frame.")
    if not match_atoms:
        # Complex analysis resolves chain correspondence before atom selection.
        return reference, predicted
    # Remove solvent/ligands before sequence alignment so residue indices follow
    # the protein sequences rather than the full simulation topology.
    reference = reference.atom_slice(reference.topology.select("protein"))
    predicted = predicted.atom_slice(predicted.topology.select("protein"))
    ref_indices, pred_indices = map_common_atoms(reference, predicted)
    reference = reference.atom_slice(ref_indices)
    predicted = predicted.atom_slice(pred_indices)
    predicted.topology = reference.topology
    return reference, predicted
