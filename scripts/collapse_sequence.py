"""Collapse sequence of a run for 3D Slicer / ParaView (pipeline/collapse/sequence.py), after the
collapse or fit step of the config:

    python scripts/collapse_sequence.py --config configs/karl04.json
    python scripts/collapse_sequence.py --config configs/karl04.json --frames 40 --volume
    python scripts/collapse_sequence.py --config configs/karl04.json --mode linear   # no FEM library needed
    python scripts/collapse_sequence.py --config configs/karl04.json --set run_name=gE1000

Writes output_dir/runs/<run_name>/sequence/ (self-contained: copy it to the machine with Slicer).
The collapse step does the same with collapse.sequence_frames > 0.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.collapse.sequence import make_sequence              # noqa: E402
from pipeline.config import load_config                           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="patient config (uses its fem_setup and run outputs)")
    ap.add_argument("--frames", type=int, default=30, help="load steps after the initial frame")
    ap.add_argument("--mode", default="fem", choices=["fem", "linear"],
                    help="fem = re-solve along the load ramp (physical); linear = straight morph")
    ap.add_argument("--volume", action="store_true", help="also write volume frames (fem mode)")
    ap.add_argument("--fps", type=float, default=8.0, help="playback speed in Slicer")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", default=[], help="config overrides as in main.py")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    make_sequence(load_config(args.config, args.set), args.frames, args.mode, args.volume, args.fps)


if __name__ == "__main__":
    main()
