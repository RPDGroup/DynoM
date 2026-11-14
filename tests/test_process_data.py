"""CPU tests for inference preparation; no network calls or model weights needed."""

import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "Process_data.py"
spec = importlib.util.spec_from_file_location("process_data", SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def write_fasta(path):
    path.write_text(
        ">1ABC_1|Chains B [auth X], A|First entity\nACD\nEF\n"
        ">1ABC_2|Chain C|Second entity\nGHIK\n"
        ">2DEF_1|Chain Z|Another target\nLMNP\n"
    )
    return path


def write_msas(metadata, msa_dir):
    for row in metadata.itertuples(index=False):
        for index, sequence in enumerate(sorted(set(row.seqs))):
            folder = msa_dir / row.PDB_ID / str(index)
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("pairing.a3m", "non_pairing.a3m"):
                (folder / name).write_text(f">query\n{sequence}\n")


def write_representations(metadata, repr_dir):
    repr_dir.mkdir(parents=True, exist_ok=True)
    for row in metadata.itertuples(index=False):
        length = sum(row.chain_lens)
        np.save(repr_dir / f"{row.PDB_ID}_single_repr_recycle10.npy",
                np.zeros((length, 384), dtype=np.float32))
        np.save(repr_dir / f"{row.PDB_ID}_pair_repr_recycle10.npy",
                np.zeros((length, length, 128), dtype=np.float32))


def test_file_directory_and_single_modes_preserve_chains(tmp_path):
    source = tmp_path / "fasta"
    source.mkdir()
    fasta = write_fasta(source / "merged.fa")
    complex_data = pipeline.prepare_metadata(fasta, "complex")
    pd.testing.assert_frame_equal(complex_data, pipeline.prepare_metadata(source, "complex"))
    assert complex_data.iloc[0].chain_ids == ["X", "A", "C"]
    assert complex_data.iloc[0].seqs == ["ACDEF", "ACDEF", "GHIK"]
    assert complex_data.iloc[0].chain_lens == [5, 5, 4]
    assert list(complex_data.PDB_ID) == ["1ABC", "2DEF"]
    single = pipeline.prepare_metadata(fasta, "single")
    assert list(single.PDB_ID) == ["1ABC_0_X", "1ABC_0_A", "1ABC_0_C", "2DEF_0_Z"]
    assert all(len(row.seqs) == len(row.chain_ids) == 1 for row in single.itertuples())


@pytest.mark.parametrize("contents, message", [
    ("", "No protein FASTA"),
    (">unknown\nACDE\n", "Expected an RCSB header"),
    (">1ABC_1|Chain A|\nACD*\n", "Expected a protein sequence"),
    (">1ABC_1|Chain A|\nACDE\n>1ABC_2|Chain A|\nFGHI\n", "Duplicate chain"),
])
def test_invalid_fasta_is_reported(tmp_path, contents, message):
    path = tmp_path / "input.fasta"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        pipeline.prepare_metadata(path, "complex")


def test_json_matches_msa_by_sequence_without_reordering(tmp_path):
    data = pd.DataFrame([dict(PDB_ID="1ABC", seqs=["ACDE", "FGHI", "ACDE"])])
    msa = tmp_path / "msa"
    write_msas(data, msa)
    output = tmp_path / "input.json"
    pipeline.build_input_json(data, msa, output)
    job = json.loads(output.read_text())[0]
    expanded = [seq["proteinChain"]["sequence"] for seq in job["sequences"]
                for _ in range(seq["proteinChain"]["count"])]
    assert expanded == data.iloc[0].seqs
    assert all(Path(seq["proteinChain"]["msa"]["precomputed_msa_dir"]).is_absolute()
               for seq in job["sequences"])
    (msa / "1ABC" / "1" / "pairing.a3m").unlink()
    with pytest.raises(ValueError, match="Incomplete or mismatched MSA"):
        pipeline.build_input_json(data, msa, output)


def test_representation_shape_and_missing_files(tmp_path):
    metadata = pipeline.prepare_metadata(write_fasta(tmp_path / "input.fa"), "complex")
    repr_dir = tmp_path / "features"
    write_representations(metadata, repr_dir)
    pipeline.validate_representations(metadata, repr_dir)
    broken = repr_dir / "1ABC_pair_repr_recycle10.npy"
    np.save(broken, np.zeros((1, 1, 128), dtype=np.float32))
    with pytest.raises(ValueError, match="Invalid representation"):
        pipeline.validate_representations(metadata, repr_dir)
    broken.unlink()
    with pytest.raises(FileNotFoundError, match="Missing representation"):
        pipeline.validate_representations(metadata, repr_dir)


def test_cached_pipeline_cli_handles_spaces_and_unique_sequence_keys(tmp_path):
    fasta = tmp_path / "input with spaces.fasta"
    fasta.write_text(
        ">1ABC_1|Chain A|One\nACDEF\n"
        ">2DEF_1|Chain B|Same sequence\nACDEF\n"
    )
    data = pipeline.prepare_metadata(fasta, "complex")
    repr_dir = tmp_path / "feature files"
    write_representations(data, repr_dir)
    output = tmp_path / "prepared data"
    result = subprocess.run([sys.executable, str(SCRIPT), "--input-fasta", str(fasta),
                             "--output-dir", str(output), "--repr-dir", str(repr_dir)],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    paths = json.loads((output / "data_paths.json").read_text())
    mapping = pd.read_pickle(paths["alphafold3_cfg_seqres_to_index_path"])
    assert mapping.set_index("seqs_key").at["ACDEF", "PDB_ID"] == "1ABC"
    assert len(pd.read_pickle(paths["gen_dataset_test_gen_dataset"])) == 2
    command = shlex.split((output / "predict.sh").read_text().splitlines()[-1])
    assert command[command.index("--alphafold3_cfg_repr_data_root") + 1] == str(repr_dir)
    assert "Preparation complete" in result.stdout


def test_full_orchestration_uses_selected_environments_and_outputs(tmp_path, monkeypatch):
    fasta = write_fasta(tmp_path / "input.fasta")
    output = tmp_path / "prepared"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        data = pd.read_pickle(output / "dataset.pkl")
        if "--msa_out_dir" in command:
            write_msas(data, Path(command[command.index("--msa_out_dir") + 1]))
        else:
            assert json.loads((output / "protenix_input.json").read_text())
            write_representations(data, Path(command[command.index("--dump_dir") + 1]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline.main(["--input-fasta", str(fasta), "--output-dir", str(output),
                   "--protenix-python", "/custom env/bin/python", "--gpus", "2"])
    assert len(calls) == 2
    assert all(call[0][0] == "/custom env/bin/python" for call in calls)
    assert "--nproc_per_node=2" in calls[1][0]
    assert calls[1][1]["cwd"] == pipeline.PROTENIX
    assert str(pipeline.PROTENIX) in calls[0][1]["env"]["PYTHONPATH"]
    assert (output / "predict.sh").is_file()


def test_reused_msa_skips_network_stage(tmp_path, monkeypatch):
    fasta = write_fasta(tmp_path / "input.fasta")
    metadata = pipeline.prepare_metadata(fasta, "complex")
    msa = tmp_path / "existing_msa"
    write_msas(metadata, msa)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        write_representations(metadata, Path(command[command.index("--dump_dir") + 1]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline.main(["--input-fasta", str(fasta), "--output-dir", str(tmp_path / "out"),
                   "--msa-dir", str(msa)])
    assert len(calls) == 1
    assert "torch.distributed.run" in calls[0]
    assert ["run", "--no-capture-output", "-n", "protenix_env", "python"] == calls[0][1:6]


@pytest.mark.parametrize("returncode", [0, 2])
def test_failed_or_incomplete_msa_cannot_publish_success(tmp_path, monkeypatch, returncode):
    fasta = write_fasta(tmp_path / "input.fasta")
    output = tmp_path / "out"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    with pytest.raises((RuntimeError, ValueError)):
        pipeline.main(["--input-fasta", str(fasta), "--output-dir", str(output)])
    assert len(calls) == 1
    assert not (output / "data_paths.json").exists()
    assert not (output / "predict.sh").exists()


def test_dry_run_writes_nothing_and_preserves_existing_output(tmp_path):
    fasta = write_fasta(tmp_path / "input.fasta")
    output = tmp_path / "out"
    pipeline.main(["--input-fasta", str(fasta), "--output-dir", str(output), "--dry-run"])
    assert not output.exists()
    output.mkdir()
    marker = output / "user_file"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="new or empty"):
        pipeline.main(["--input-fasta", str(fasta), "--output-dir", str(output)])
    assert marker.read_text() == "keep"
