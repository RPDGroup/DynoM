"""Score one predicted PDB ensemble against one reference structure."""

import json
import subprocess

import pandas as pd

from structure_common import (
    calculate_tmscore, executable_path, parse_structure_args,
    prediction_frames, structure_parser,
)


def main(argv=None):
    parser = structure_parser(__doc__)
    parser.add_argument("--ost_bin", default="ost", help="OpenStructure executable or path.")
    parser.add_argument("--tmscore_bin", default="TMscore", help="TMscore executable or path.")
    args = parse_structure_args(parser, argv)
    ost = executable_path(args.ost_bin)
    tmscore = executable_path(args.tmscore_bin)
    rows = []
    with prediction_frames(args) as (workspace, frames):
        for index, frame in frames:
            json_path = workspace / f"frame_{index:06d}.json"
            # Keep the release's backbone-lDDT and rigid-score options.
            result = subprocess.run([
                ost, "compare-structures", "-m", str(frame), "-r", str(args.ref_pdb),
                "-o", str(json_path), "--fault-tolerant", "--min-pep-length", "4",
                "--min-nuc-length", "4", "--bb-lddt", "--rigid-scores",
            ], check=False, capture_output=True, text=True)
            (args.output_dir / f"frame_{index:06d}_ost.log").write_text(result.stdout + result.stderr)
            result.check_returncode()
            data = json.loads(json_path.read_text())
            (args.output_dir / f"frame_{index:06d}_ost.json").write_text(json.dumps(data, indent=2) + "\n")
            rows.append({
                "pdb_name": args.pred_traj.stem, "frame": index,
                "bb_lddt": data.get("bb_lddt"), "gdt_ts": data.get("oligo_gdtts"),
                "rmsd": data.get("rmsd"),
                "tm_score": calculate_tmscore(
                    frame, args.ref_pdb, tmscore,
                    args.output_dir / f"frame_{index:06d}_tmscore.log"),
            })
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Structure scores saved to {args.output_csv}")


if __name__ == "__main__":
    main()
