"""Compute PwD, Rg, and TICA distribution comparisons for one sample.

A single invocation loads one MD trajectory and one predicted ensemble, compares
their distributions, and evaluates TICA at the requested lag times.
"""

import random

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyemma
from msmbuilder.decomposition import tICA
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import RobustScaler

from analysis_common import load_trajectories, parse_args


PSEUDO_C = 1e-6


def get_backbone_torsions_feature(traj):
    """Encode backbone torsions as sine/cosine pairs without periodic imaging."""
    featurizer = pyemma.coordinates.featurizer(traj.topology)
    featurizer.add_backbone_torsions(cossin=True, periodic=False)
    return featurizer.transform(traj)


def distance_matrix_ca(coords):
    """Return C-alpha distances with shape (frames, residues, residues)."""
    delta = coords[..., None, :, :] - coords[..., None, :]
    return np.sqrt(np.sum(delta ** 2, axis=-1))


def pairwise_distance_ca(coords, k=1):
    """Extract the upper triangle with sequence-index offset k."""
    distances = distance_matrix_ca(coords)
    rows, columns = np.triu_indices(distances.shape[-1], k=k)
    return distances[..., rows, columns]


def radius_of_gyration(coords, masses=None):
    """Return per-frame Rg; the analysis uses equal weights for C-alpha atoms."""
    if masses is None:
        masses = np.ones(coords.shape[-2])
    weights = masses / masses.sum()
    centered = coords - coords.mean(-2, keepdims=True)
    return (((centered ** 2).sum(-1)) * weights).sum(-1) ** 0.5


def histogram_js(values, ref_key="MD", n_bins=50):
    """Average Jensen-Shannon distances using reference-defined histogram ranges.

    Retain the release's 50 bins, pseudocount, and exclusion of observations
    outside the MD range. The reported quantity is the Jensen-Shannon distance
    returned by scipy.spatial.distance.jensenshannon, not its square.
    """
    reference = np.asarray(values[ref_key])
    if reference.ndim == 1:
        reference = reference[:, None]
    lower, upper = reference.min(axis=0), reference.max(axis=0)
    histograms = {}
    for key, value in values.items():
        value = np.asarray(value)
        if value.ndim == 1:
            value = value[:, None]
        histograms[key] = np.stack([
            np.histogram(value[:, index], bins=n_bins,
                         range=(lower[index], upper[index]))[0] + PSEUDO_C
            for index in range(reference.shape[1])
        ], axis=1)
    results = {
        key: np.around(distance.jensenshannon(histogram, histograms[ref_key], axis=0).mean(), 4)
        for key, histogram in histograms.items() if key != ref_key
    }
    results[ref_key] = 0.0
    return results


def js_rg(ca_coords, ref_key="MD", n_bins=50):
    return histogram_js({key: radius_of_gyration(value) for key, value in ca_coords.items()},
                        ref_key, n_bins)


def js_pwd(ca_coords, ref_key="MD", n_bins=50, pwd_offset=3):
    return histogram_js({key: pairwise_distance_ca(value, k=pwd_offset)
                         for key, value in ca_coords.items()}, ref_key, n_bins)


def js_RMWD_tica(md_feature, pred_feature, dim, lagtime, n_bins=50):
    """Fit TICA on MD, transform both ensembles, and compare the first two components."""
    model = tICA(n_components=dim, lag_time=lagtime).fit([md_feature])
    projections = {
        "MD": model.transform([md_feature])[0],
        "Pred": model.transform([pred_feature])[0],
    }
    if any(value.shape[1] < 2 for value in projections.values()):
        raise ValueError(f"TICA did not produce two components at lag {lagtime}.")
    jsd = histogram_js(projections, "MD", n_bins)["Pred"]
    wd_x = wasserstein_distance(projections["MD"][:, 0], projections["Pred"][:, 0])
    wd_y = wasserstein_distance(projections["MD"][:, 1], projections["Pred"][:, 1])
    rmwd = np.around(np.sqrt((wd_x ** 2 + wd_y ** 2) / 2.0), 4)
    return jsd, rmwd, projections, model


def get_features(reference, predicted, output_dir):
    """Extract corresponding C-alpha and backbone-torsion features for one pair."""
    backbone = reference.topology.select("backbone")
    if len(backbone) == 0:
        raise ValueError("No common backbone atoms remain after alignment.")
    reference = reference.atom_slice(backbone)
    predicted = predicted.atom_slice(backbone)
    ca_indices = reference.topology.select("name CA")
    if len(ca_indices) < 4:
        raise ValueError("At least four common C-alpha atoms are needed for PwD offset 3.")
    ca_md = reference.xyz[:, ca_indices, :]
    ca_pred = predicted.xyz[:, ca_indices, :]
    torsions_md = get_backbone_torsions_feature(reference)
    torsions_pred = get_backbone_torsions_feature(predicted)
    if torsions_md.shape[1] != torsions_pred.shape[1] or torsions_md.shape[1] < 2:
        raise ValueError("At least two matching torsion features are required for 2D TICA.")
    # Preserve the release's independent RobustScaler fits. Changing to a shared
    # fit would change the existing analysis convention and is a separate choice.
    scaled_md = RobustScaler().fit_transform(torsions_md)
    scaled_pred = RobustScaler().fit_transform(torsions_pred)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Recompute for the current inputs; never reuse an unrelated sample's cache.
    np.savez(output_dir / "all_features.npz", ca_coords_md=ca_md, ca_coords_pred=ca_pred,
             dihedral_md=torsions_md, dihedral_pred=torsions_pred,
             scaled_dihedral_md=scaled_md, scaled_dihedral_pred=scaled_pred)
    return ca_md, ca_pred, scaled_md, scaled_pred


def main(argv=None):
    args = parse_args(__doc__, argv)
    random.seed(42)
    np.random.seed(42)
    reference, predicted = load_trajectories(args.md_traj, args.md_top, args.pred_traj)
    for lagtime in args.lagtimes:
        if lagtime >= reference.n_frames:
            raise ValueError(f"Lag {lagtime} must be smaller than the {reference.n_frames} MD frames.")
    ca_md, ca_pred, scaled_md, scaled_pred = get_features(reference, predicted, args.output_dir)
    coordinates = {"MD": ca_md, "Pred": ca_pred}
    pwd = js_pwd(coordinates)["Pred"]
    rg = js_rg(coordinates)["Pred"]
    rows = []
    for lagtime in args.lagtimes:
        print(f"Analyzing {args.pred_traj.stem}: lagtime={lagtime} frames", flush=True)
        jsd, rmwd, projections, model = js_RMWD_tica(scaled_md, scaled_pred, dim=2, lagtime=lagtime)
        figure, axes = plt.subplots()
        for name in ("MD", "Pred"):
            axes.scatter(projections[name][:, 0], projections[name][:, 1], alpha=0.5, label=name)
        axes.set_xlabel("TIC 1")
        axes.set_ylabel("TIC 2")
        axes.set_title(f"Lag: {lagtime} frames; JS: {jsd}; RMWD: {rmwd}")
        axes.legend()
        figure.savefig(args.output_dir / f"lagtime_{lagtime}_tica.png")
        plt.close(figure)
        np.savez(args.output_dir / f"lagtime_{lagtime}_tica.npz", **projections)
        eigenvalues = np.asarray(model.eigenvalues_, dtype=float)
        valid = np.isfinite(eigenvalues) & (eigenvalues > 0) & (eigenvalues < 1)
        timescales = np.full(eigenvalues.shape, np.nan)
        timescales[valid] = -lagtime / np.log(eigenvalues[valid])
        pd.DataFrame({
            "pdb_name": args.pred_traj.stem,
            "component": np.arange(1, len(eigenvalues) + 1),
            "lag_frames": lagtime, "eigenvalue": eigenvalues,
            "valid_positive_slow_mode": valid, "timescale_frames": timescales,
            "n_ref_frames": len(scaled_md), "n_lagged_pairs": len(scaled_md) - lagtime,
        }).to_csv(args.output_dir / f"lagtime_{lagtime}_timescales.csv",
                  index=False, float_format="%.12g")
        rows.append({
            "pdb_name": args.pred_traj.stem, "md_traj": str(args.md_traj),
            "md_top": str(args.md_top) if args.md_top else "", "pred_traj": str(args.pred_traj),
            "js_PwD": pwd, "js_Rg": rg, "lagtime": lagtime,
            "js-TICA": jsd, "rmwd-TICA": rmwd,
        })
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Multiple rows represent lag times for this one sample, never other samples.
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"TICA analysis complete: {args.output_csv}")


if __name__ == "__main__":
    main()
