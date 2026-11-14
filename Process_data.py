"""Prepare all DynoM inference inputs from RCSB-style protein FASTA files.

Run with --help for examples and environment options. Metadata preparation uses
the current Python environment; MSA search and feature extraction use Protenix.
"""

import argparse
from collections import OrderedDict
from itertools import groupby
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PROTENIX = ROOT / "data_processed" / "protenix"
RECYCLES = 10  # Must match the feature filenames requested by Predict.py.


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="FASTA -> metadata -> MSA -> Protenix representations -> DynoM inputs.",
        epilog="Example: python Process_data.py --input-fasta ./fasta --output-dir ./prepared",
    )
    parser.add_argument("--input-fasta", required=True, type=Path,
                        help="RCSB-style FASTA file or directory (.fasta, .fa, .faa).")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New or empty directory for this processing run.")
    parser.add_argument("--mode", choices=["complex", "single"], default="complex",
                        help="Keep all chains together (default), or prepare each chain separately.")
    env = parser.add_mutually_exclusive_group()
    env.add_argument("--protenix-env", default="protenix_env",
                     help="Conda environment for MSA and representation generation (default: protenix_env).")
    env.add_argument("--protenix-python",
                     help="Protenix Python executable; use this instead of conda run.")
    parser.add_argument("--gpus", type=positive_int, default=1,
                        help="Number of local GPU processes for Protenix (default: 1).")
    parser.add_argument("--msa-workers", type=positive_int, default=4,
                        help="Parallel MSA requests (default: 4).")
    parser.add_argument("--msa-dir", type=Path,
                        help="Reuse MSA results arranged as <target>/<index>/{pairing,non_pairing}.a3m.")
    parser.add_argument("--repr-dir", type=Path,
                        help="Reuse existing cycle-10 representations; skip MSA and Protenix execution.")
    parser.add_argument("--protenix-checkpoint", type=Path,
                        help="Optional model_v0.2.0.pt path; otherwise use the bundled runner's default.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the processing plan without running steps or writing files.")
    args = parser.parse_args(argv)
    for name in ("input_fasta", "output_dir", "msa_dir", "repr_dir", "protenix_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    return args


def prepare_metadata(input_path, mode):
    """Expand FASTA entities into ordered chains and create DynoM metadata.

    A FASTA record can describe several identical chains. The same expansion is
    used for both file and directory inputs, including merged multi-target files.
    """
    import pandas as pd
    from Bio import SeqIO

    paths = ([input_path] if input_path.is_file() else
             sorted(p for p in input_path.iterdir() if p.suffix.lower() in {".fasta", ".fa", ".faa"}))
    targets = OrderedDict()
    for path in paths:
        for record in SeqIO.parse(str(path), "fasta"):
            fields = record.description.split("|")
            if len(fields) < 2 or "_" not in fields[0]:
                raise ValueError(f"Expected an RCSB header such as '>9R71_1|Chains A, B|...': {record.description}")
            target = fields[0].strip().rsplit("_", 1)[0].upper()
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", target) or target in {".", ".."}:
                raise ValueError(f"Unsupported target name: {target}")
            chain_field = re.sub(r"^Chains?\s+", "", fields[1].strip())
            chains = []
            for value in chain_field.split(","):
                match = re.fullmatch(r"\s*(\w+)(?:\s*\[auth\s+(\w+)\])?\s*", value)
                if not match:
                    raise ValueError(f"Cannot parse chain identifier in: {record.description}")
                chains.append(match.group(2) or match.group(1))
            sequence = str(record.seq).upper()
            if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWYX"):
                raise ValueError(f"Expected a protein sequence (20 standard residues or X): {record.id}")
            entry = targets.setdefault(target, {"chain_ids": [], "seqs": []})
            for chain in chains:
                if chain in entry["chain_ids"]:
                    raise ValueError(f"Duplicate chain {chain} for {target}; provide each entity once.")
                entry["chain_ids"].append(chain)
                entry["seqs"].append(sequence)

    rows = []
    for target, entry in targets.items():
        groups = [(target, entry["chain_ids"], entry["seqs"])]
        if mode == "single":
            groups = [(f"{target}_0_{chain}", [chain], [seq])
                      for chain, seq in zip(entry["chain_ids"], entry["seqs"])]
        for name, chains, sequences in groups:
            lengths = [len(seq) for seq in sequences]
            rows.append(dict(PDB_ID=name, model=0, chain_ids=chains, seqs=sequences,
                             chain_lens=lengths, total_seqLens=sum(lengths)))
    if not rows:
        raise ValueError(f"No protein FASTA records found in {input_path}")
    return pd.DataFrame(rows)


def build_input_json(metadata, msa_dir, output_path):
    """Match MSA queries to sequences and preserve residue order in Protenix JSON."""
    from Bio import SeqIO

    jobs = []
    for row in metadata.itertuples(index=False):
        lookup = {}
        target_dir = msa_dir / row.PDB_ID
        for query_file in sorted(target_dir.glob("*/non_pairing.a3m")):
            query = next(SeqIO.parse(str(query_file), "fasta"), None)
            pairing = query_file.with_name("pairing.a3m")
            pair_query = next(SeqIO.parse(str(pairing), "fasta"), None) if pairing.is_file() else None
            if query and pair_query and str(query.seq) == str(pair_query.seq):
                lookup[str(query.seq)] = str(query_file.parent.resolve())
        missing = set(row.seqs) - set(lookup)
        if missing:
            raise ValueError(f"Incomplete or mismatched MSA for {row.PDB_ID} in {target_dir}. "
                             "Check logs/msa.log and the MSA search output before retrying.")
        sequences = []
        # Merge only adjacent identical chains. Global counting would reorder
        # an A/B/A sequence layout to A/A/B and misalign the feature residue axes.
        for sequence, copies in groupby(row.seqs):
            sequences.append({"proteinChain": {
                "sequence": sequence, "count": sum(1 for _ in copies),
                "msa": {"precomputed_msa_dir": lookup[sequence], "pairing_db": "uniref100"},
            }})
        jobs.append({"name": row.PDB_ID, "sequences": sequences})
    output_path.write_text(json.dumps(jobs, indent=2) + "\n")


def validate_representations(metadata, repr_dir):
    """Check every expected representation without loading full pair arrays into RAM."""
    import numpy as np

    for row in metadata.itertuples(index=False):
        length = sum(row.chain_lens)
        for kind, shape in (("single", (length, 384)), ("pair", (length, length, 128))):
            path = repr_dir / f"{row.PDB_ID}_{kind}_repr_recycle{RECYCLES}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing representation: {path}. Check logs/representations.log and AF3_repr/ERR.")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
                raise ValueError(f"Invalid representation {path}: expected floating-point {shape}, "
                                 f"found {array.dtype} {array.shape}.")


def protenix_python(args):
    """Select the Protenix environment without changing the user's active shell."""
    if args.protenix_python:
        executable = os.path.expanduser(args.protenix_python)
        return [shutil.which(executable) or str(Path(executable).resolve())]
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda") or "conda"
    return [conda, "run", "--no-capture-output", "-n", args.protenix_env, "python"]


def run_step(label, command, cwd, log_path, dry_run=False):
    print(f"{label}\n  {shlex.join(str(x) for x in command)}\n  Log: {log_path}", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(PROTENIX), str(ROOT), env.get("PYTHONPATH", "")])
    env["LAYERNORM_TYPE"] = "False"
    with log_path.open("w") as log:
        result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} failed (exit {result.returncode}). See {log_path}")


def main(argv=None):
    args = parse_args(argv)
    if not args.input_fasta.exists():
        raise FileNotFoundError(args.input_fasta)
    for path in (args.msa_dir, args.repr_dir):
        if path is not None and not path.is_dir():
            raise NotADirectoryError(path)
    if args.protenix_checkpoint is not None and not args.protenix_checkpoint.is_file():
        raise FileNotFoundError(args.protenix_checkpoint)
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("--output-dir must be new or empty. Reuse previous MSA/representations with --msa-dir/--repr-dir.")

    output = args.output_dir
    metadata_path = output / "dataset.pkl"
    mapping_path = output / "seqres_to_index.pkl"
    json_path = output / "protenix_input.json"
    msa_dir = args.msa_dir or output / "msa"
    repr_dir = args.repr_dir or output / "AF3_repr"
    logs = output / "logs"
    python = protenix_python(args)

    # Stage 1: Build model metadata before requesting alignments or allocating a GPU.
    print(f"[1/5] Parse FASTA ({args.mode}) -> {metadata_path}", flush=True)
    if not args.dry_run:
        metadata = prepare_metadata(args.input_fasta, args.mode)
        logs.mkdir(parents=True)
        metadata.to_pickle(metadata_path)
        print(f"Prepared {len(metadata)} targets.", flush=True)

    if args.repr_dir is None:
        # Stage 2: Use the existing Protenix MSA client in its own environment.
        if args.msa_dir is None:
            if not args.dry_run:
                msa_dir.mkdir()
            run_step("[2/5] Search MSAs", python + [
                "-m", "data_processed.rcsb_script.step5_representation_generation.step2_get_msa_from_protenixAPI",
                "--input_file", metadata_path, "--msa_out_dir", msa_dir,
                "--log_file", logs / "msa.log", "--max_workers", str(args.msa_workers),
            ], ROOT, logs / "msa_search.log", args.dry_run)
        else:
            print(f"[2/5] Reuse MSAs from {msa_dir}", flush=True)

        # Stage 3: Require an MSA for every sequence before building encoder inputs.
        print(f"[3/5] Match MSAs and build {json_path}", flush=True)
        if not args.dry_run:
            build_input_json(metadata, msa_dir, json_path)

        # Stage 4: Match the release's encoder-only configuration and recycle count.
        # Native attention avoids requiring a local CUTLASS kernel installation.
        command = python + [
            "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}",
            "runner/inference.py", "--seeds", "101", "--dump_dir", repr_dir,
            "--input_json_path", json_path, "--model.N_cycle", str(RECYCLES),
            "--sample_diffusion.N_sample", "5", "--sample_diffusion.N_step", "200",
            "--only_encoder", "True", "--use_deepspeed_evo_attention", "False",
        ]
        if args.protenix_checkpoint:
            command += ["--load_checkpoint_path", args.protenix_checkpoint]
        run_step("[4/5] Generate representations", command, PROTENIX,
                 logs / "representations.log", args.dry_run)
    else:
        print(f"[2-4/5] Reuse representations from {repr_dir}", flush=True)

    # Stage 5: Publish the sequence lookup and prediction command only after all
    # target feature files exist and match DynoM's expected residue dimensions.
    print(f"[5/5] Validate representations and write {mapping_path}", flush=True)
    if args.dry_run:
        return
    validate_representations(metadata, repr_dir)
    mapping = metadata[["PDB_ID", "seqs"]].copy()
    mapping["seqs_key"] = mapping.pop("seqs").apply(lambda seqs: "_".join(seqs))
    # The runtime loader expects one scalar PDB_ID for each ordered sequence key.
    # Identical targets can share the first matching representation.
    mapping.drop_duplicates("seqs_key", keep="first").to_pickle(mapping_path)
    paths = {
        "gen_dataset_test_gen_dataset": str(metadata_path),
        "alphafold3_cfg_seqres_to_index_path": str(mapping_path),
        "alphafold3_cfg_repr_data_root": str(repr_dir),
    }
    (output / "data_paths.json").write_text(json.dumps(paths, indent=2) + "\n")
    predict = [sys.executable, str(ROOT / "Predict.py")]
    for key, path in paths.items():
        predict += ["--" + key, path]
    predict += ["--gen_batch_size", "10", "--gen_num_samples", "1000",
                "--model_ckpt_path", str(ROOT / "checkpoint" / "DynoM.ckpt"),
                "--output_dir", str(output / "samples")]
    command = shlex.join(predict)
    (output / "predict.sh").write_text("#!/usr/bin/env bash\nset -e\n" + command + "\n")
    print(f"Preparation complete. Run inference with:\n  bash {shlex.quote(str(output / 'predict.sh'))}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
