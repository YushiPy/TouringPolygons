#!/usr/bin/env python3
"""Run the unordered C++ solver on a tracked binary suite, with optional external comparison."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from benchmark_cases import read_encoded_cases
from unordered_runner import run_unordered_solver
from unordered_validation import validate_path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--suite', type=Path, default=ROOT / 'benchmarks/suites/algorithm-dev-v1.bin')
	parser.add_argument('--solver', type=Path, default=ROOT / '.build/unordered/tpp')
	parser.add_argument('--seconds', type=float, default=5)
	parser.add_argument('--max-calls', type=int, default=10000000)
	parser.add_argument('--case', type=int, action='append')
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args()
	args.output.parent.mkdir(parents=True, exist_ok=True)
	with args.output.open('w') as file:
		for case in read_encoded_cases(args.suite):
			if args.case is not None and case.case_index not in args.case:
				continue
			s = struct.unpack_from('<dddd', case.data)
			row = {'case': case.case_index, 'sha256': case.digest, 'polygons': case.polygon_count}
			try:
				row.update(run_unordered_solver(args.solver, s[:2], s[2:], case.polygons, args.max_calls, args.seconds))
				try:
					row['validation'] = validate_path(s[:2], s[2:], case.polygons, row['path'], 1e-7)
					row['max_polygon_distance'] = row['validation']['max_polygon_distance']
					row['valid'] = row['validation']['valid']
				except ImportError:
					row['validation'] = None
					row['valid'] = None
			except RuntimeError as error:
				row['error'] = str(error)
			file.write(json.dumps(row) + '\n')
			file.flush()
			print(json.dumps({k: v for k, v in row.items() if k not in ('path', 'order', 'sha256')}), flush=True)


if __name__ == '__main__':
	main()
