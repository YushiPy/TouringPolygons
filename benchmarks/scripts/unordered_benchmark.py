#!/usr/bin/env python3
"""Run the unordered C++ solver on a tracked binary suite, with optional external comparison."""
from __future__ import annotations

import argparse
import json
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
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
	parser.add_argument('--workers', type=int, default=1)
	parser.add_argument('--solver-argument', action='append', default=[])
	parser.add_argument('--case', type=int, action='append')
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--resume', action='store_true')
	args = parser.parse_args()
	if args.workers < 1:
		parser.error('--workers must be positive')
	args.output.parent.mkdir(parents=True, exist_ok=True)
	cases = [case for case in read_encoded_cases(args.suite)
		if args.case is None or case.case_index in args.case]
	completed = set()
	if args.resume and args.output.exists():
		for line in args.output.read_text().splitlines():
			try:
				completed.add(json.loads(line)['case'])
			except (json.JSONDecodeError, KeyError):
				continue
	cases = [case for case in cases if case.case_index not in completed]

	def solve(case):
		s = struct.unpack_from('<dddd', case.data)
		row = {'case': case.case_index, 'sha256': case.digest, 'polygons': case.polygon_count}
		try:
			row.update(run_unordered_solver(
				args.solver, s[:2], s[2:], case.polygons, args.max_calls, args.seconds,
				arguments=args.solver_argument,
			))
			try:
				row['validation'] = validate_path(s[:2], s[2:], case.polygons, row['path'], 1e-7)
				row['max_polygon_distance'] = row['validation']['max_polygon_distance']
				row['valid'] = row['validation']['valid']
			except ImportError:
				row['validation'] = None
				row['valid'] = None
		except RuntimeError as error:
			row['error'] = str(error)
		return row

	mode = 'a' if args.resume else 'w'
	with args.output.open(mode) as file, ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(cases)))) as executor:
		futures = [executor.submit(solve, case) for case in cases]
		for future in as_completed(futures):
			row = future.result()
			file.write(json.dumps(row) + '\n')
			file.flush()
			print(json.dumps({k: v for k, v in row.items() if k not in ('path', 'order', 'sha256')}), flush=True)


if __name__ == '__main__':
	main()
