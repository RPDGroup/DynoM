#!/usr/bin/env python3
"""Compare one complex MD trajectory with one predicted ensemble using JS-PWD/Rg.

Histogram ranges pool the MD reference and the supplied predicted ensemble.
Chain maps use reference:prediction identifiers, e.g. A:X,B:Y (zero-based
indices use a # prefix, e.g. #0:#1). Identical chains with tied optimal
assignments are paired by order of appearance.
Coordinates must already represent whole complexes, not PBC-wrapped chains.
"""

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.SeqUtils import seq1
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from analysis_common import load_trajectories, parse_args

PSEUDO_C = 1e-6
N_BINS = 50
PWD_OFFSET = 3


# Match protein chains and sequence positions before comparing C-alpha features.
def protein_chains(top):
    chains = []
    for chain in top.chains:
        residues = [r for r in chain.residues if r.is_protein]
        if residues:
            chains.append(
                (
                    chain,
                    residues,
                    "".join(seq1(r.name, custom_map={"UNK": "X"}) for r in residues),
                )
            )
    if not chains:
        raise ValueError("No protein chains found in topology")
    return chains


def chain_label(chain):
    return chain.chain_id if chain.chain_id else f"#{chain.index}"


def assign_chains(rc, pc, mapping, aligner):
    """Globally match chain sequences; resolve identical chains in input order."""
    scores = np.array([[aligner.score(r[2], p[2]) for p in pc] for r in rc])
    if mapping:

        def resolve(token, chains):
            found = [
                i
                for i, (c, _, _) in enumerate(chains)
                if token == f"#{c.index}" or (c.chain_id and token == c.chain_id)
            ]
            if len(found) != 1:
                raise ValueError(f"Chain {token!r} is missing or ambiguous; use #index")
            return found[0]

        assignment = {}
        for item in mapping.split(","):
            parts = item.strip().split(":")
            if len(parts) != 2:
                raise ValueError("chain_map format must be A:X,B:Y")
            i, j = resolve(parts[0].strip(), rc), resolve(parts[1].strip(), pc)
            if i in assignment or j in assignment.values():
                raise ValueError("Each chain must occur once in chain_map")
            assignment[i] = j
        if len(assignment) != len(rc):
            raise ValueError("chain_map must specify every protein chain")
    else:
        rows, cols = linear_sum_assignment(scores, maximize=True)
        best = scores[rows, cols].sum()
        optimal = dict(zip(rows, cols))
        inverse = {j: i for i, j in optimal.items()}
        # Identical chains can be assigned in order; other tied alignments
        # need an explicit mapping rather than a guess based on chain order.
        if len(rc) > 1:
            for i, j in optimal.items():
                altered = scores.copy()
                altered[i, j] = -np.inf
                ar, ac = linear_sum_assignment(altered, maximize=True)
                if np.isclose(altered[ar, ac].sum(), best, rtol=0, atol=1e-8):
                    for row, col in zip(ar, ac):
                        old_col = optimal[row]
                        if (
                            col != old_col
                            and rc[row][2] != rc[inverse[col]][2]
                            and pc[col][2] != pc[old_col][2]
                        ):
                            raise ValueError(
                                "Ambiguous alignment of non-identical chains; specify --chain_map"
                            )
        # Lexicographically smallest optimal assignment: for each reference
        # chain, choose the first prediction chain that preserves the optimum.
        assignment, remaining, remaining_score = {}, list(range(len(pc))), best
        for i in range(len(rc)):
            for j in remaining:
                rest = [col for col in remaining if col != j]
                if rest:
                    sub = scores[i + 1 :, rest]
                    sr, sc = linear_sum_assignment(sub, maximize=True)
                    candidate = scores[i, j] + sub[sr, sc].sum()
                else:
                    candidate = scores[i, j]
                if np.isclose(candidate, remaining_score, rtol=0, atol=1e-8):
                    assignment[i] = j
                    remaining.remove(j)
                    remaining_score -= scores[i, j]
                    break
            else:
                raise RuntimeError("Could not resolve optimal chain assignment")
    return assignment


def align_chains(ref, pred, mapping):
    rc, pc = protein_chains(ref.topology), protein_chains(pred.topology)
    if len(rc) != len(pc):
        raise ValueError(
            f"Protein chain counts differ: reference {len(rc)}, prediction {len(pc)}"
        )
    aligner = PairwiseAligner(scoring="blastp")
    aligner.open_gap_score = -14
    aligner.extend_gap_score = -2
    assignment = assign_chains(rc, pc, mapping, aligner)
    ref_indices, pred_indices, chain_indices, positions, report = [], [], [], [], []
    for i, (rchain, rr, rs) in enumerate(rc):
        pchain, pr, ps = pc[assignment[i]]
        alignment = aligner.align(rs, ps)[0]
        ri, pi = alignment.indices
        paired = [(int(a), int(b)) for a, b in zip(ri, pi) if a >= 0 and b >= 0]
        matches = sum(rs[a] == ps[b] for a, b in paired)
        if not paired or matches / max(len(rs), len(ps)) < 0.5:
            raise ValueError(
                f"Low sequence identity/coverage for {chain_label(rchain)}:{chain_label(pchain)}"
            )
        count = 0
        for a, b in paired:
            ra = [atom.index for atom in rr[a].atoms if atom.name == "CA"]
            pa = [atom.index for atom in pr[b].atoms if atom.name == "CA"]
            if len(ra) == len(pa) == 1:
                ref_indices.append(ra[0])
                pred_indices.append(pa[0])
                chain_indices.append(i)
                positions.append(a)
                count += 1
        if not count:
            raise ValueError(f"No matched CA atoms for chain {chain_label(rchain)}")
        report.append(
            f"{chain_label(rchain)}:{chain_label(pchain)}({count}/{len(rr)},{len(pr)})"
        )
    return (
        np.array(ref_indices),
        np.array(pred_indices),
        np.array(chain_indices),
        np.array(positions),
        ";".join(report),
    )


# Retain the original joint-range JS-Rg and chunked JS-PWD calculations.


def radius_of_gyration(coords, masses=None, frame_chunk_size=256):
    """(..., N, 3) -> (...)"""
    if coords.ndim == 3 and len(coords) > frame_chunk_size:
        result = np.empty(len(coords), dtype=np.float64)
        for start in range(0, len(coords), frame_chunk_size):
            stop = start + frame_chunk_size
            result[start:stop] = radius_of_gyration(
                coords[start:stop], masses=masses, frame_chunk_size=frame_chunk_size
            )
        return result

    if masses is None:
        masses = np.ones(coords.shape[-2])

    weights = masses / masses.sum()
    centered = coords - coords.mean(-2, keepdims=True)
    squared_dists = (centered**2).sum(-1)
    return (squared_dists * weights).sum(-1) ** 0.5


def js_rg(ca_coords_dict, ref_key="target", n_bins=N_BINS, weights=None):
    ca_rg = {k: radius_of_gyration(v) for k, v in ca_coords_dict.items()}

    if weights is None:
        weights = {}
    weights.update(
        {k: np.ones(len(v)) for k, v in ca_coords_dict.items() if k not in weights}
    )

    # Use the joint range of MD and the supplied prediction.
    d_min = min(v.min() for v in ca_rg.values())
    d_max = max(v.max() for v in ca_rg.values())
    ca_rg_binned = {
        k: np.histogram(v, bins=n_bins, weights=weights[k], range=(d_min, d_max))[0]
        + PSEUDO_C
        for k, v in ca_rg.items()
    }
    results = {
        k: distance.jensenshannon(v, ca_rg_binned[ref_key], axis=0).mean()
        for k, v in ca_rg_binned.items()
        if k != ref_key
    }

    results[ref_key] = 0.0
    return {k: np.around(v, decimals=4) for k, v in results.items()}


def js_pwd(
    ca_coords_dict,
    ref_key="target",
    n_bins=N_BINS,
    pwd_offset=PWD_OFFSET,
    weights=None,
    frame_chunk_size=1024,
    pair_chunk_size=64,
    chains=None,
    positions=None,
):
    """Compute all-frame pair statistics in chunks without a frames x L x L array."""
    ref = ca_coords_dict[ref_key]
    weights = {} if weights is None else weights
    totals = {key: 0.0 for key in ca_coords_dict if key != ref_key}
    n_pairs = 0

    def blocks(coords, row, columns):
        for start in range(0, len(coords), frame_chunk_size):
            block = coords[start : start + frame_chunk_size]
            delta = block[:, row, None, :] - block[:, columns, :]
            values = np.sqrt(np.sum(delta**2, axis=-1))
            yield start, values

    def histograms(coords, key, row, columns, edges):
        n_columns = len(columns)
        hist = np.zeros((n_columns, n_bins), dtype=np.float64)
        low, high = edges[0], edges[-1]
        column_ids = np.arange(n_columns)[None, :]
        for start, values in blocks(coords, row, columns):
            values = values.astype(edges.dtype, copy=False)
            valid = (values >= low) & (values <= high)
            # Apply equal-width binning and boundary corrections as in numpy.histogram.
            indices = ((values - low) / (high - low)) * n_bins
            indices = np.clip(indices, 0, n_bins - 1).astype(np.intp)
            indices -= values < edges[indices, column_ids]
            indices += (indices != n_bins - 1) & (
                values >= edges[indices + 1, column_ids]
            )
            flat = indices + column_ids * n_bins
            frame_weights = (
                np.ones(len(values))
                if key not in weights
                else np.asarray(weights[key])[start : start + len(values)]
            )
            hist += np.bincount(
                flat[valid],
                weights=np.broadcast_to(frame_weights[:, None], values.shape)[valid],
                minlength=n_columns * n_bins,
            ).reshape(n_columns, n_bins)
        return hist.T + PSEUDO_C

    # Follow np.triu_indices order without allocating a full pair-index array.
    for row in range(ref.shape[1]):
        for first in range(row + 1, ref.shape[1], pair_chunk_size):
            columns = np.arange(first, min(first + pair_chunk_size, ref.shape[1]))
            # Filter within-chain neighbors by sequence offset; retain all inter-chain pairs.
            columns = columns[
                (chains[columns] != chains[row])
                | (positions[columns] - positions[row] >= pwd_offset)
            ]
            if not len(columns):
                continue
            lower = np.full(len(columns), np.inf, dtype=ref.dtype)
            upper = np.full(len(columns), -np.inf, dtype=ref.dtype)
            # Use the MD/prediction joint range for every C-alpha pair.
            for coords in ca_coords_dict.values():
                for _, values in blocks(coords, row, columns):
                    lower = np.minimum(lower, values.min(axis=0))
                    upper = np.maximum(upper, values.max(axis=0))
            edges = np.stack(
                [
                    np.histogram_bin_edges(
                        np.empty(0, dtype=ref.dtype), bins=n_bins, range=(lo, hi)
                    )
                    for lo, hi in zip(lower, upper)
                ],
                axis=1,
            )
            ref_hist = histograms(ref, ref_key, row, columns, edges)
            for key, coords in ca_coords_dict.items():
                if key == ref_key:
                    continue
                pred_hist = histograms(coords, key, row, columns, edges)
                totals[key] += distance.jensenshannon(pred_hist, ref_hist, axis=0).sum()
            n_pairs += len(columns)
    if n_pairs == 0:
        raise ValueError("No residue pairs for the selected pwd_offset")
    results = {
        key: np.around(total / n_pairs, decimals=4) for key, total in totals.items()
    }
    results[ref_key] = 0.0
    return results


def main(argv=None):
    args = parse_args(__doc__, argv, require_lagtimes=False, chain_mapping=True)
    reference, predicted = load_trajectories(args.md_traj, args.md_top, args.pred_traj,
                                            match_atoms=False)
    ri, pi, chains, positions, mapping = align_chains(reference, predicted, args.chain_map)
    coords = {"MD": reference.xyz[:, ri, :], "Pred": predicted.xyz[:, pi, :]}
    if any(not np.isfinite(values).all() for values in coords.values()):
        raise ValueError("Non-finite coordinates in the matched complex.")
    pwd = js_pwd(coords, ref_key="MD", chains=chains, positions=positions)["Pred"]
    rg = js_rg(coords, ref_key="MD")["Pred"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "matched_ca_indices.npz", md_indices=ri, pred_indices=pi,
             chain_indices=chains, sequence_positions=positions)
    (args.output_dir / "chain_mapping.txt").write_text(mapping + "\n")
    row = dict(name=args.pred_traj.stem, js_PwD=pwd, js_Rg=rg,
               n_matched_ca=len(ri), n_ref_frames=reference.n_frames,
               n_pred_frames=predicted.n_frames, chain_mapping=mapping)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output_csv, index=False)
    print(f"Complex JS scores saved to {args.output_csv}")


if __name__ == "__main__":
    main()
