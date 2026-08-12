#!/usr/bin/env python3
"""Generate the default benchmark campaign inputs.

Run from the repository root:

    python3 benchmarks/generate_instances.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PBF = REPO_ROOT / "packages/instance-generation/regions/sao-paulo.osm.pbf"


def make_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate TPP benchmark instances from an OSM PBF extract.")
	parser.add_argument("--campaign", default="sao-paulo", help="Campaign name under benchmarks/campaigns/.")
	parser.add_argument("--pbf", type=Path, default=DEFAULT_PBF, help="Input .osm.pbf file.")
	parser.add_argument("--instances", default="100", help="Instances per generated input file.")
	parser.add_argument("--sample-size", default="40", help="Number of matrix entries to sample.")
	parser.add_argument("--seed", default="42", help="Random seed for reproducible sampling.")
	parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to benchmarks/tpp.py generate-matrix.")
	return parser


def main() -> int:
	args = make_parser().parse_args()
	pbf = args.pbf.resolve()
	if not pbf.exists():
		raise SystemExit(f"Missing input PBF: {pbf}")

	command = [
		sys.executable,
		"benchmarks/tpp.py",
		"generate-matrix",
		args.campaign,
		str(pbf),
		"--instances",
		str(args.instances),
		"--sample-size",
		str(args.sample_size),
		"--seed",
		str(args.seed),
		*args.extra,
	]
	print("+", " ".join(command), flush=True)
	return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
	raise SystemExit(main())
