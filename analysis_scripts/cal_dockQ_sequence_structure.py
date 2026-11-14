"""Evaluate DockQ for one predicted PDB ensemble and one native complex."""

import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio import pairwise2
from biopandas.pdb import PandasPdb
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
from structure_common import parse_structure_args, prediction_frames, structure_parser


# Merge each specified receptor/ligand group into one DockQ partner.
def merge_chains(model, chains_to_merge):
    target_id = chains_to_merge[0]

    for chain_id in chains_to_merge[1:]:
        target = model[target_id]
        source = model[chain_id]

        for res in list(source):
            res.id = (chain_id, res.id[1], res.id[2])
            target.add(res)

        if hasattr(target, "sequence") and hasattr(source, "sequence"):
            target.sequence += source.sequence

        model.detach_child(chain_id)

    model[target_id].id = "".join(chains_to_merge)
    return model


# Extract protein sequences grouped by PDB chain ID.
def get_sequence(ppdb) -> pd.DataFrame:
    sequence = ppdb.amino3to1()
    sequence_df = (
        sequence.groupby(["chain_id"]).agg({"residue_name": list}).reset_index()
    )
    sequence_df["sequence"] = sequence_df["residue_name"].apply(lambda x: "".join(x))
    sequence_df.drop(columns=["residue_name"], inplace=True)

    return sequence_df



# ===============================
# Map predicted chains one-to-one to native chains by sequence and rigid fit.
# ===============================
def _read_chain_ca(pdb_path, sequence_df):
    """Read first-model CA coordinates in sequence order."""
    model = next(PDBParser(QUIET=True).get_structure("structure", str(pdb_path)).get_models())
    by_id = {chain.id: chain for chain in model}
    result = {}
    for row in sequence_df.itertuples():
        if row.chain_id not in by_id:
            raise ValueError(f"Chain {row.chain_id!r} is absent from first model of {pdb_path}")
        sequence, coordinates = [], []
        for residue in by_id[row.chain_id]:
            if residue.id[0] != " ":
                continue
            amino_acid = seq1(residue.resname, custom_map={"UNK": "X"})
            if len(amino_acid) != 1 or amino_acid == "X":
                continue
            sequence.append(amino_acid)
            coordinates.append(residue["CA"].coord if "CA" in residue else None)
        if "".join(sequence) != row.sequence:
            raise ValueError(
                f"Sequence/coordinate residue mismatch for chain {row.chain_id!r} in {pdb_path}"
            )
        result[row.chain_id] = coordinates
    return result


def annotate_pred_df(pred_df, ref_df, pred_pdb, ref_pdb):
    """Sequence-optimal one-to-one mapping; first-model complex RMSD breaks ties."""
    pred_df, ref_df = pred_df.copy(), ref_df.copy()
    pred_rows = list(pred_df.itertuples())
    ref_rows = list(ref_df.itertuples())
    if len(pred_rows) < len(ref_rows):
        raise ValueError("One-to-one mapping requires at least as many predicted as reference chains")

    scores = np.array([
        [pairwise2.align.globalxx(r.sequence, p.sequence, score_only=True)
         for p in pred_rows] for r in ref_rows
    ])
    rows, cols = linear_sum_assignment(scores, maximize=True)
    best_sequence_score = scores[rows, cols].sum()
    ref_ca = _read_chain_ca(ref_pdb, ref_df)
    pred_ca = _read_chain_ca(pred_pdb, pred_df)
    paired_coords = {}

    def get_pair(i, j):
        if (i, j) not in paired_coords:
            r, p = ref_rows[i], pred_rows[j]
            aln = pairwise2.align.globalxx(r.sequence, p.sequence, one_alignment_only=True)[0]
            ri = pi = 0
            x, y = [], []
            for aa, bb in zip(aln.seqA, aln.seqB):
                if aa != "-" and bb != "-":
                    a, b = ref_ca[r.chain_id][ri], pred_ca[p.chain_id][pi]
                    if a is not None and b is not None:
                        x.append(a)
                        y.append(b)
                ri += aa != "-"
                pi += bb != "-"
            paired_coords[i, j] = np.asarray(x), np.asarray(y)
        return paired_coords[i, j]

    best_rmsd, best_assignment, candidate_count = np.inf, None, 0

    def visit(chosen, remaining, score):
        nonlocal best_rmsd, best_assignment, candidate_count
        i = len(chosen)
        if i == len(ref_rows):
            if not np.isclose(score, best_sequence_score, rtol=0, atol=1e-8):
                return
            candidate_count += 1
            if candidate_count > 100000:
                raise ValueError("Too many equally scoring chain mappings")
            pairs = [get_pair(k, j) for k, j in enumerate(chosen)]
            if any(len(x) == 0 for x, _ in pairs):
                return
            x = np.concatenate([p[0] for p in pairs]).astype(float)
            y = np.concatenate([p[1] for p in pairs]).astype(float)
            if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
                return
            x -= x.mean(axis=0)
            y -= y.mean(axis=0)
            u, _, vt = np.linalg.svd(x.T @ y)
            correction = np.eye(3)
            correction[-1, -1] = np.linalg.det(u @ vt)
            rmsd = float(np.sqrt(np.mean(np.sum((x @ (u @ correction @ vt) - y) ** 2, axis=1))))
            if rmsd < best_rmsd:
                best_rmsd, best_assignment = rmsd, list(chosen)
            return
        for j in remaining:
            rest = [k for k in remaining if k != j]
            upper = score + scores[i, j]
            if i + 1 < len(ref_rows):
                sub = scores[i + 1:, rest]
                sr, sc = linear_sum_assignment(sub, maximize=True)
                upper += sub[sr, sc].sum()
            if np.isclose(upper, best_sequence_score, rtol=0, atol=1e-8):
                visit(chosen + [j], rest, score + scores[i, j])

    visit([], list(range(len(pred_rows))), 0.0)
    if best_assignment is None:
        raise ValueError("No sequence-optimal mapping has enough matched CA atoms")
    pred_to_ref = {
        pred_rows[j].chain_id: ref_rows[i].chain_id
        for i, j in enumerate(best_assignment)
    }
    print(f"Chain mapping (predicted -> reference): {pred_to_ref}; "
          f"first-model CA RMSD: {best_rmsd:.3f} Å; "
          f"sequence-tied candidates: {candidate_count}", flush=True)
    pred_df["ref_chain_id"] = pred_df["chain_id"].map(pred_to_ref)
    return pred_df


def validate_native_groups(ref_sequence_df, receptor, ligand):
    specified = receptor + ligand
    if len(set(specified)) != len(specified):
        raise ValueError("Receptor and ligand chain groups must be disjoint and contain no duplicates.")
    if set(specified) != set(ref_sequence_df["chain_id"]):
        raise ValueError("Specify every reference protein chain in exactly one receptor/ligand group.")


def build_chain_idx_mapping(pred_pdb, ref_pdb, native_receptor_chains, native_ligand_chains):
    """Infer native:model chain groups once for all models of a sample."""
    pred_pdb_df = PandasPdb().read_pdb(str(pred_pdb))
    ref_pdb_df = PandasPdb().read_pdb(str(ref_pdb))
    pred_sequence_df = get_sequence(pred_pdb_df)
    ref_sequence_df = get_sequence(ref_pdb_df)
    if len(ref_sequence_df) < 2 or len(pred_sequence_df) < 2:
        raise ValueError("DockQ requires at least two protein chains in both inputs.")
    validate_native_groups(ref_sequence_df, native_receptor_chains, native_ligand_chains)

    # Some predictions fuse multiple native chains (for example, an scFv
    # joined by a linker). In that case, assign one predicted chain to each
    # user-defined partner by summing its sequence matches to that partner's
    # native chains.
    if len(pred_sequence_df) < len(ref_sequence_df):
        available_pred_chains = set(pred_sequence_df["chain_id"])
        chain_mapping = {}
        for native_chains in (native_receptor_chains, native_ligand_chains):
            native_sequences = [
                ref_sequence_df.loc[
                    ref_sequence_df["chain_id"] == chain, "sequence"
                ].iloc[0]
                for chain in native_chains
            ]
            best_pred_chain = max(
                sorted(available_pred_chains),
                key=lambda pred_chain: sum(
                    pairwise2.align.globalxx(
                        pred_sequence_df.loc[
                            pred_sequence_df["chain_id"] == pred_chain,
                            "sequence",
                        ].iloc[0],
                        native_sequence,
                        score_only=True,
                    )
                    for native_sequence in native_sequences
                ),
            )
            chain_mapping["".join(native_chains)] = best_pred_chain
            available_pred_chains.remove(best_pred_chain)
        return chain_mapping

    pred_sequence_df = annotate_pred_df(
        pred_sequence_df, ref_sequence_df, pred_pdb, ref_pdb
    )

    def get_pred_chains(native_chains):
        pred_chains = []
        for native_chain in native_chains:
            matches = pred_sequence_df[
                pred_sequence_df["ref_chain_id"] == native_chain
            ]
            if matches.empty:
                raise ValueError(
                    f"No predicted chain matches native chain {native_chain} "
                    f"in {pred_pdb}"
                )
            pred_chains.append(matches.iloc[0]["chain_id"])
        return pred_chains

    pred_receptor_chains = get_pred_chains(native_receptor_chains)
    pred_ligand_chains = get_pred_chains(native_ligand_chains)
    return {
        "".join(native_receptor_chains): "".join(pred_receptor_chains),
        "".join(native_ligand_chains): "".join(pred_ligand_chains),
    }


def score_frame(pred_pdb, ref_pdb, chain_mapping):
    """Score the specified interface using the sample's first-frame chain map."""
    model = load_PDB(str(pred_pdb))
    native = load_PDB(str(ref_pdb))
    for native_chains, model_chains in chain_mapping.items():
        model = merge_chains(model, list(model_chains))
        native = merge_chains(native, list(native_chains))
    interfaces = run_on_all_native_interfaces(
        model_structure=model, native_structure=native, chain_map=chain_mapping)[0]
    expected = set(chain_mapping)
    key = next((key for key in interfaces
                if (isinstance(key, tuple) and set(key) == expected)
                or key == "".join(chain_mapping)), None)
    if key is None:
        raise KeyError(f"DockQ did not return the requested interface {chain_mapping}.")
    return dict(interfaces[key])


def main(argv=None):
    parser = structure_parser(__doc__)
    parser.add_argument("--receptor_chains", required=True, nargs="+",
                        help="Reference receptor chain IDs, for example H L.")
    parser.add_argument("--ligand_chains", required=True, nargs="+",
                        help="Reference ligand chain IDs, for example A.")
    parser.add_argument("--chain_map", default="",
                        help="Optional native:prediction partner groups, for example HL:XY,A:Z.")
    args = parse_structure_args(parser, argv)
    rows = []
    with prediction_frames(args) as (_, frames):
        if args.chain_map:
            ref_sequence = get_sequence(PandasPdb().read_pdb(str(args.ref_pdb)))
            validate_native_groups(ref_sequence, args.receptor_chains, args.ligand_chains)
            items = [item.strip().split(":") for item in args.chain_map.split(",")]
            if any(len(item) != 2 or not all(item) for item in items):
                raise ValueError("--chain_map must use native:prediction groups, for example HL:XY,A:Z.")
            mapping = dict(items)
            expected = {"".join(args.receptor_chains), "".join(args.ligand_chains)}
            if len(items) != 2 or set(mapping) != expected:
                raise ValueError("--chain_map must specify the receptor and ligand groups exactly once.")
            selected = "".join(mapping.values())
            pred_chains = set(get_sequence(PandasPdb().read_pdb(str(frames[0][1])))["chain_id"])
            if len(selected) != len(set(selected)) or not set(selected).issubset(pred_chains):
                raise ValueError("Predicted chain groups must contain distinct chains present in the input.")
        else:
            mapping = build_chain_idx_mapping(frames[0][1], args.ref_pdb,
                                              args.receptor_chains, args.ligand_chains)
        (args.output_dir / "chain_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
        for index, frame in frames:
            result = score_frame(frame, args.ref_pdb, mapping)
            result.update(pdb_name=args.pred_traj.stem, frame=index,
                          chain_mapping=json.dumps(mapping, sort_keys=True))
            rows.append(result)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"DockQ scores saved to {args.output_csv}")


if __name__ == "__main__":
    main()
