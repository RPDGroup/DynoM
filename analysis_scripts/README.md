# Single-sample analysis

Run from the DynoM directory. Each command analyzes one sample; a predicted PDB
may contain multiple models. Replace the example paths with your own files.

## PWD, Rg, and TICA

```bash
python analysis_scripts/calculate_PWD_RG_TIC.py \
    --md_traj /path/to/md.xtc --md_top /path/to/topology.pdb \
    --pred_traj /path/to/predicted.pdb --lagtimes 10 20 50 100 \
    --output_dir ./results/tica --output_csv ./results/tica.csv
```

## RMSF and Pearson correlation

```bash
python analysis_scripts/cal_RMSF.py \
    --md_traj /path/to/md.xtc --md_top /path/to/topology.pdb \
    --pred_traj /path/to/predicted.pdb \
    --output_dir ./results/rmsf --output_csv ./results/rmsf.csv
```

## Complex PWD and Rg

```bash
python analysis_scripts/cal_complex_js_pwd_rg_total_range.py \
    --md_traj /path/to/md.xtc --md_top /path/to/topology.pdb \
    --pred_traj /path/to/predicted.pdb \
    --output_dir ./results/complex_js --output_csv ./results/complex_js.csv
```

This compares one MD trajectory and one prediction using their joint histogram
range. To set MD:prediction chain pairs manually, add `--chain_map A:X,B:Y`.
Complexes should be whole, without chains split across periodic boundaries.

## RMSD, lDDT, GDT-TS, and TM-score

```bash
python analysis_scripts/cal_RMSD_lddt_GDT_TMscore.py \
    --ref_pdb /path/to/reference.pdb --pred_traj /path/to/predicted.pdb \
    --ost_bin /path/to/ost --tmscore_bin /path/to/TMscore \
    --output_dir ./results/structure --output_csv ./results/structure.csv
```

## TM-score with chain matching

```bash
python analysis_scripts/cal_TM-score_sequence_structure.py \
    --ref_pdb /path/to/reference.pdb --pred_traj /path/to/predicted.pdb \
    --tmscore_bin /path/to/TMscore \
    --output_dir ./results/tm --output_csv ./results/tm.csv
```

## DockQ

Specify receptor and ligand chain IDs from the **reference PDB**:

```bash
python analysis_scripts/cal_dockQ_sequence_structure.py \
    --ref_pdb /path/to/reference.pdb --pred_traj /path/to/predicted.pdb \
    --receptor_chains H L --ligand_chains A \
    --output_dir ./results/dockq --output_csv ./results/dockq.csv
```

Chain matching is inferred from the first predicted model. For a manual
native:prediction partner mapping, add `--chain_map HL:XY,A:Z`.

## Inputs and results

- `--md_top` is optional when the MD file contains topology. Predicted trajectory
  files must contain topology; multi-model PDB is supported.
- TICA lag times are positive integers in MD frames, smaller than the MD frame
  count. RMSF and complex JS do not need lag times.
- Structure-scoring scripts require one reference PDB model and a predicted PDB
  ensemble. Their CSV contains one row per predicted model (`frame` starts at 0).
- `--output_csv` stores scores; `--output_dir` stores detailed results. Use separate
  directories for each sample and analysis. Matching output filenames are overwritten.
- Keep `analysis_common.py` and `structure_common.py` alongside the scripts.

Dependencies: MDTraj, Biopython, NumPy, pandas, and SciPy. TICA also uses PyEMMA,
MSMBuilder, scikit-learn, and Matplotlib. Chain-matched TM-score and DockQ use
BioPandas; DockQ also needs the DockQ package. Install OpenStructure and TMscore
separately for structure scoring. Their executable paths can be supplied as above;
the defaults are `ost` and `TMscore` on `PATH`.
