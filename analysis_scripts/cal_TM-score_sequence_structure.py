"""Calculate per-frame TM-score after sequence/structure-based chain matching."""

import math
from itertools import permutations
import numpy as np
import pandas as pd
from Bio import pairwise2
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from structure_common import (
    calculate_tmscore, executable_path, parse_structure_args,
    prediction_frames, structure_parser,
)

def _chains(path):
    model = next(PDBParser(QUIET=True).get_structure('structure', str(path)).get_models())
    chains = {}
    for chain in model:
        residues = []
        for residue in chain:
            if residue.id[0] != ' ' or 'CA' not in residue:
                continue
            aa = seq1(residue.resname, custom_map={'UNK': 'X'})
            if len(aa) == 1 and aa != 'X':
                residues.append((aa, residue['CA'].coord.astype(float)))
        if residues:
            chains[chain.id] = residues
    return chains


def _pairs(a, b, min_identity, min_coverage):
    sa = ''.join(x[0] for x in a)
    sb = ''.join(x[0] for x in b)
    alignment = pairwise2.align.globalms(sa, sb, 2, -1, -10, -0.5, one_alignment_only=True)[0]
    i = j = matches = 0
    coords_a, coords_b = [], []
    for x, y in zip(alignment.seqA, alignment.seqB):
        if x != '-' and y != '-':
            coords_a.append(a[i][1]); coords_b.append(b[j][1])
            matches += x == y
        i += x != '-'
        j += y != '-'
    aligned = len(coords_a)
    if not aligned or matches / aligned < min_identity or aligned / max(len(sa), len(sb)) < min_coverage:
        return None
    return np.array(coords_a), np.array(coords_b)


def find_chain_mapping(pred_pdb, ref_pdb, min_identity=0.8, min_coverage=0.8, max_candidates=100000):
    """Return best one-to-one mapping and ranked (RMSD, mapping) candidates.

    Scores use all aligned C-alpha atoms after one common least-squares rigid
    fit, so relative chain positions contribute to the ranking.
    """
    pred, ref = _chains(pred_pdb), _chains(ref_pdb)
    if not pred or len(pred) > len(ref):
        raise ValueError('Protein chains missing or more predicted chains than reference chains')
    keys, targets = list(pred), list(ref)
    count = math.perm(len(targets), len(keys))
    if count > max_candidates:
        raise ValueError(f'{count} chain permutations exceed limit {max_candidates}; restrict candidate chains')
    pairs = {(p, r): _pairs(pred[p], ref[r], min_identity, min_coverage)
             for p in keys for r in targets}
    results = []
    for perm in permutations(targets, len(keys)):
        selected = [pairs[p, r] for p, r in zip(keys, perm)]
        if any(pair is None for pair in selected):
            continue
        x = np.concatenate([pair[0] for pair in selected])
        y = np.concatenate([pair[1] for pair in selected])
        if len(x) < 3:
            continue
        x -= x.mean(axis=0); y -= y.mean(axis=0)
        u, _, vt = np.linalg.svd(x.T @ y)
        correction = np.eye(3); correction[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        rmsd = float(np.sqrt(np.mean(np.sum((x @ rotation - y) ** 2, axis=1))))
        results.append((rmsd, dict(zip(keys, perm))))
    if not results:
        raise ValueError('No one-to-one chain mapping satisfies sequence identity and coverage thresholds')
    results.sort(key=lambda x: x[0])
    return results[0][1], results


def score_frame(pred_pdb, ref_pdb, workspace, executable, log_path):
    """Relabel prediction chains using the original one-to-one rigid-fit ranking."""
    mapping, candidates = find_chain_mapping(pred_pdb, ref_pdb)
    prediction = PandasPdb().read_pdb(str(pred_pdb))
    reference = PandasPdb().read_pdb(str(ref_pdb))
    atoms = prediction.df["ATOM"].copy()
    atoms["chain_id"] = atoms["chain_id"].map(mapping)
    native_order = reference.df["ATOM"]["chain_id"].unique()
    atoms = pd.concat([atoms[atoms["chain_id"] == chain] for chain in native_order]).copy()
    atoms["atom_number"] = range(1, len(atoms) + 1)
    atoms["line_idx"] = range(1, len(atoms) + 1)
    prediction.df["OTHERS"] = prediction.df["OTHERS"][
        prediction.df["OTHERS"]["record_name"] != "TER"]
    prediction.df["ATOM"] = atoms
    filtered = workspace / f"{pred_pdb.stem}_mapped.pdb"
    prediction.to_pdb(str(filtered))
    return {
        "tm_score": calculate_tmscore(filtered, ref_pdb, executable, log_path),
        "chain_mapping": ", ".join(f"{key}->{value}" for key, value in mapping.items()),
        "mapping_rmsd": candidates[0][0],
    }


def main(argv=None):
    parser = structure_parser(__doc__)
    parser.add_argument("--tmscore_bin", default="TMscore", help="TMscore executable or path.")
    args = parse_structure_args(parser, argv)
    executable = executable_path(args.tmscore_bin)
    rows = []
    with prediction_frames(args) as (workspace, frames):
        for index, frame in frames:
            result = score_frame(frame, args.ref_pdb, workspace, executable,
                                 args.output_dir / f"frame_{index:06d}_tmscore.log")
            result.update(pdb_name=args.pred_traj.stem, frame=index)
            rows.append(result)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"TM-scores saved to {args.output_csv}")


if __name__ == "__main__":
    main()


