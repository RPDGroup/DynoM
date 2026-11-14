"""Single-sample PDB inputs and external-tool helpers for structure scoring."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import os
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory


def structure_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ref_pdb", required=True, type=Path,
                        help="One reference PDB structure containing a single model.")
    parser.add_argument("--pred_traj", required=True, type=Path,
                        help="One sample's predicted PDB, optionally containing multiple models.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Detailed results directory.")
    parser.add_argument("--output_csv", required=True, type=Path, help="Per-frame score CSV.")
    return parser


def parse_structure_args(parser, argv=None):
    args = parser.parse_args(argv)
    for name in ("ref_pdb", "pred_traj", "output_dir", "output_csv"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for path in (args.ref_pdb, args.pred_traj):
        if not path.is_file():
            parser.error(f"Input must be a file: {path}")
    if args.output_csv in {args.ref_pdb, args.pred_traj}:
        parser.error("--output_csv must differ from the input files")
    return args


def executable_path(value):
    path = shutil.which(os.path.expanduser(str(value)))
    if path is None:
        raise FileNotFoundError(f"Executable not found: {value}. Supply its path with the corresponding *_bin argument.")
    return path


@contextmanager
def prediction_frames(args):
    """Export each model of one PDB sample into an isolated temporary directory."""
    from Bio.PDB import PDBIO, PDBParser

    parser = PDBParser(QUIET=True)
    native = parser.get_structure("reference", str(args.ref_pdb))
    if len(list(native.get_models())) != 1:
        raise ValueError("--ref_pdb must contain exactly one reference model.")
    prediction = parser.get_structure("prediction", str(args.pred_traj))
    models = list(prediction.get_models())
    if not models:
        raise ValueError("No models found in --pred_traj.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="structure_eval_", dir=args.output_dir) as directory:
        workspace = Path(directory)
        frames = []
        for index, model in enumerate(models):
            frame = workspace / f"frame_{index:06d}.pdb"
            writer = PDBIO()
            writer.set_structure(model)
            writer.save(str(frame))
            frames.append((index, frame))
        yield workspace, frames


def calculate_tmscore(pred_pdb, ref_pdb, executable, log_path):
    """Keep the original TMscore options and last reported normalization score."""
    result = subprocess.run([
        executable, "-c", "-ter", "1", "-seq", str(pred_pdb), str(ref_pdb),
    ], check=False, capture_output=True, text=True)
    log_path.write_text(result.stdout + result.stderr)
    result.check_returncode()
    scores = re.findall(r"TM-score\s*=\s*([0-9]*\.[0-9]+)", result.stdout)
    if not scores:
        raise ValueError(f"No TM-score found in tool output. See {log_path}")
    return float(scores[-1])
