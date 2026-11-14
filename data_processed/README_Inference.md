# Inference data preparation

See the [project README](../README.md) for DynoM installation and sampling.
All commands below assume the repository root as the working directory unless
an absolute script path is used.

Run `Process_data.py` once to prepare all three inputs required by `Predict.py`:

```text
RCSB FASTA -> dataset.pkl -> MSA search -> Protenix input JSON
                                      -> single/pair representations
                                      -> seqres_to_index.pkl
```

The script handles intermediate paths, environment selection, logs, and final
feature checks. You do not need to edit `inference_demo_1.sh` or manually switch
environments between processing steps.

## Contents

- [One-time setup](#one-time-setup)
- [Prepare inputs with one command](#prepare-inputs-with-one-command)
- [Environment and compute options](#environment-and-compute-options)
- [Outputs and prediction](#outputs-and-prediction)
- [Reuse existing results](#reuse-existing-results)
- [Troubleshooting](#troubleshooting)
- [CPU tests](#cpu-tests)

## One-time setup

Follow the [Installation tutorial](../README.md#installation) to create
`DynoM_env` and [the Protenix environment](../README.md#protenix-environment).
Then activate `DynoM_env` before running the commands below.

`Process_data.py` calls `protenix_env` automatically. MSA search and first-use
model/cache downloads require network access; representation generation uses
the Protenix encoder checkpoint `model_v0.2.0.pt`.

## Prepare inputs with one command

From the DynoM repository, with `DynoM_env` activated:

```bash
python Process_data.py \
    --input-fasta /path/to/input.fasta \
    --output-dir /path/to/prepared_data
```

`--input-fasta` accepts either a FASTA file (including multiple targets) or a
directory of `.fasta`, `.fa`, and `.faa` files. `--output-dir` must be new or empty.
The script can also be invoked by its absolute path from another directory.

Input headers use the RCSB format, for example:

```fasta
>9R71_1|Chains A, B|Adenylate kinase
MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKD
```

This record describes two identical chains. A `Chain X [auth A]` annotation uses
the author chain identifier `A`. Each target/entity should appear once. Sequences
must contain standard protein residue letters or `X`; gaps and stop symbols are
not accepted.

- `--mode complex` (default) keeps all chains of a target together and preserves
  their order. Identical-chain copies are expanded in the metadata.
- `--mode single` makes an independent target for every chain, named
  `<PDB_ID>_0_<chain>`. This changes the names of the corresponding MSA folders
  and feature files.

The included ADK example can be prepared with:

```bash
python Process_data.py \
    --input-fasta example/Test_Task/ADK/01_Data_Process/Step1_Convert_Fasta_Into_Pkl/Input/9R71.fasta \
    --output-dir ./prepared_9R71
```

## Environment and compute options

```bash
python Process_data.py \
    --input-fasta ./fasta \
    --output-dir ./prepared_data \
    --protenix-env protenix_env \
    --msa-workers 4 \
    --gpus 1
```

The wrapper uses `conda run` to invoke Protenix while keeping the main environment
active. Alternatively, pass `--protenix-python /path/to/protenix_env/bin/python`
instead of `--protenix-env`. Set `CUDA_VISIBLE_DEVICES` before running the command
to select GPUs. `--gpus` is the number of local worker processes.

Use `--dry-run` to print the plan and external commands without writing files or
running MSA/model computation. `python Process_data.py --help` lists all options.

| Argument | Default | Description |
| --- | --- | --- |
| `--input-fasta` | Required | RCSB-style FASTA file or directory |
| `--output-dir` | Required | New or empty run directory |
| `--mode` | `complex` | `complex` or `single` |
| `--protenix-env` | `protenix_env` | Conda environment used for MSA and encoding |
| `--protenix-python` | Unset | Python executable to use instead of `--protenix-env` |
| `--gpus` | `1` | Positive number of local Protenix worker processes |
| `--msa-workers` | `4` | Positive number of parallel MSA requests |
| `--msa-dir` | Unset | Existing MSA results; skips MSA search |
| `--repr-dir` | Unset | Existing representations; skips both MSA and encoding |
| `--protenix-checkpoint` | Runner default | Optional encoder checkpoint file |
| `--dry-run` | Disabled | Print the plan only |

`--protenix-env` and `--protenix-python` are mutually exclusive. When
`--repr-dir` is supplied, the MSA options are not used. Argument paths are
resolved relative to the directory from which the script is invoked.

## Outputs and prediction

| Output | Purpose / `Predict.py` argument |
| --- | --- |
| `dataset.pkl` | `--gen_dataset_test_gen_dataset` |
| `seqres_to_index.pkl` | `--alphafold3_cfg_seqres_to_index_path` |
| `AF3_repr/` | `--alphafold3_cfg_repr_data_root` |
| `msa/` | MSA results indexed by target and unique sequence |
| `protenix_input.json` | Encoder inputs with absolute MSA paths |
| `logs/` | MSA and representation-generation logs |
| `data_paths.json` | Final input paths, written after validation succeeds |
| `predict.sh` | Ready-to-run DynoM prediction command |

The default generated layout is:

```text
prepared_data/
├── dataset.pkl
├── msa/
├── protenix_input.json
├── AF3_repr/
│   ├── <target>_single_repr_recycle10.npy
│   └── <target>_pair_repr_recycle10.npy
├── seqres_to_index.pkl
├── logs/
│   ├── msa.log
│   ├── msa_search.log
│   └── representations.log
├── data_paths.json
└── predict.sh
```

Reusing an external MSA or representation directory omits the corresponding
generation steps and local outputs. With `--repr-dir`, no MSA or encoder-input
JSON is generated; `data_paths.json` points to the supplied feature directory.

Features use 10 recycling iterations, matching the current `Predict.py`
configuration. For a target with `L` residues, the wrapper checks for floating-point
arrays of shapes `(L, 384)` and `(L, L, 128)`. Missing targets or mismatched shapes
stop processing. Sequence lookup keys follow the same chain order as the metadata.
Targets with identical ordered sequences share one lookup entry.

After preparation succeeds, run:

```bash
bash /path/to/prepared_data/predict.sh
```

The generated command uses the Python executable that launched `Process_data.py`,
the repository's `checkpoint/DynoM.ckpt`, batch size 10, and 1,000 samples per
target. Edit these prediction options in `predict.sh` if needed. Data preparation
does not start DynoM sampling automatically.

## Reuse existing results

Reuse compatible MSA results to skip network search:

```bash
python Process_data.py \
    --input-fasta ./input.fasta \
    --output-dir ./prepared_with_existing_msa \
    --msa-dir /path/to/existing/msa
```

Each target must have `<target>/<index>/pairing.a3m` and `non_pairing.a3m` files.
The first sequence in both files must match the intended query. The wrapper
matches directories by query sequence rather than assuming their numeric order.

To reuse representations and skip both MSA and encoder execution:

```bash
python Process_data.py \
    --input-fasta ./input.fasta \
    --output-dir ./prepared_with_existing_features \
    --repr-dir /path/to/existing/AF3_repr
```

Use representations generated for the **same sequences, target names, chain
order, and chain multiplicity**. Shape checks alone cannot establish sequence
identity. External representations are referenced in place, without copying.

On failure, inspect the reported log. Retry with a new output directory, optionally
reusing completed MSA results with `--msa-dir`. Existing output directories are
not automatically resumed or overwritten. Completion files are created only when
all expected representation files pass validation.

## Troubleshooting

| Message or symptom | Action |
| --- | --- |
| Output directory is not empty | Choose a new output directory; use `--msa-dir` or `--repr-dir` to reuse results |
| RCSB header or duplicate-chain error | Use one record per entity with explicit chain IDs; do not repeat the same chain within a target |
| Conda environment or executable not found | Complete the Protenix setup or specify its Python with `--protenix-python` |
| MSA step fails or a query has no matching MSA | Inspect `logs/msa_search.log` and `logs/msa.log`; confirm both A3M files contain the expected query |
| Protenix generation fails | Inspect `logs/representations.log` and any per-target reports under `AF3_repr/ERR/` |
| Representation file is missing or has the wrong shape | Confirm target names, chain multiplicity, residue order, and cycle-10 filenames |

The individual scripts in `rcsb_script/` remain available for manual workflows.
The unified entry point only prepares inference data; the training pipeline is
documented separately in [README_Train.md](README_Train.md).

## CPU tests

With pytest installed in an environment containing NumPy, pandas, and Biopython:

```bash
python -m pytest -q tests/test_process_data.py
```

These tests exercise parsing, chain ordering, MSA matching, feature validation,
environment routing, and failure handling. They do not run the remote MSA service
or the GPU encoder.
