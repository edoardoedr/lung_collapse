"""Fill a new output_dir from an old one: the model setup and the runs of the current code only.

    python scripts/init_output_dir.py --configs configs/*.json                # dry run: prints what it would copy
    python scripts/init_output_dir.py --configs configs/*.json --apply

The new folder is each config's output_dir; the old one is --old (a template with {patient},
{dataset} = the dataset folder of output_dir, e.g. HD_dataset, and {root} = its parent). Copied:

  geometry   lung_inflated_mesh.vtp, lung_collapsed_mesh.vtp
  hilum      hilum_anchor.mrk.json, hilum/
  checks     checks/check_summary.json
  fem_setup  fem/setup/ (problem.npz, setup_summary.json, reference.vtp, volume.vtu, target.vtp; not netgen/)
  runs       runs/<name>/ with a summary.json (new layout), except --skip patterns; sequence/ left out

The old folder is not changed. Nothing in the new one is overwritten (an existing file is
reported and kept). The setup keeps its volume mesh, so the copied runs stay valid there.
"""

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eurelaps_lung.config import load_config                              # noqa: E402

FILES = ["lung_inflated_mesh.vtp", "lung_collapsed_mesh.vtp", "hilum_anchor.mrk.json", "checks/check_summary.json",
         "fem/setup/problem.npz", "fem/setup/setup_summary.json", "fem/setup/reference.vtp", "fem/setup/volume.vtu",
         "fem/setup/target.vtp"]
DIRS = ["hilum"]


def plan(old, new, skip):
    """[(source, destination)] of the files to copy."""
    pairs = [(old / f, new / f) for f in FILES if (old / f).is_file()]
    for d in DIRS:
        if (old / d).is_dir():
            pairs += [(p, new / p.relative_to(old)) for p in sorted((old / d).rglob("*")) if p.is_file()]
    runs = old / "runs"
    if runs.is_dir():
        for r in sorted(runs.iterdir()):
            if not (r / "summary.json").is_file() or any(fnmatch.fnmatch(r.name, s) for s in skip):
                continue
            pairs += [(p, new / p.relative_to(old)) for p in sorted(r.rglob("*"))
                      if p.is_file() and "sequence" not in p.relative_to(r).parts]
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--old", default="{root}/{dataset}/FEM_fit_{suffix}/{patient}_noreg_hilum_wall",
                    help="old output_dir template ({root}, {dataset}, {suffix} = HD / CZ, {patient})")
    ap.add_argument("--skip", nargs="*", default=["v2_*"], help="run names (glob) not to copy")
    ap.add_argument("--apply", action="store_true", help="copy (default: only print)")
    args = ap.parse_args()

    for c in args.configs:
        cfg = load_config(c, [])
        new = Path(cfg.output_dir)
        dataset = new.parent.parent                                  # .../Dataset_Lungs/HD_dataset
        old = Path(args.old.format(root=dataset.parent, dataset=dataset.name, suffix=dataset.name.split("_")[0],
                                   patient=cfg.patient))
        if not old.is_dir():
            print("%-11s old folder %s missing: skipped" % (cfg.patient, old))
            continue
        if old.resolve() == new.resolve():
            print("%-11s output_dir is the old folder: skipped" % cfg.patient)
            continue
        pairs = plan(old, new, args.skip)
        runs = sorted({d.relative_to(new).parts[1] for _, d in pairs if d.relative_to(new).parts[0] == "runs"})
        setup = (old / "fem/setup/problem.npz").is_file()
        print("%-11s %s -> %s: %d files, setup %s, runs %s" % (cfg.patient, old, new, len(pairs),
                                                             "yes" if setup else "MISSING", " ".join(runs) or "-"))
        if not args.apply:
            continue
        kept = 0
        for s, d in pairs:
            if d.exists():
                kept += 1
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
        if kept:
            print("            %d files already there, kept" % kept)
    if not args.apply:
        print("dry run: add --apply to copy")


if __name__ == "__main__":
    main()
