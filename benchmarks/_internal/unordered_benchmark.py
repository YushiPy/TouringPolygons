#!/usr/bin/env python3
"""Run the unordered C++ solver on a tracked binary suite, with optional external comparison."""
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from benchmark_cases import EncodedCase, read_encoded_cases
from unordered_runner import run_unordered_solver
from unordered_validation import validate_path

ROOT = Path(__file__).resolve().parents[2]


def read_initial_paths(path: Path, cases: list[EncodedCase]) -> dict[int, list[list[float]]]:
	selected = {case.case_index: case for case in cases}
	paths: dict[int, list[list[float]]] = {}
	with path.open(newline='') as file:
		reader = csv.DictReader(file, delimiter=';')
		if not reader.fieldnames or not {'case_index', 'sha256', 'path'} <= set(reader.fieldnames):
			raise ValueError('Initial paths CSV needs case_index, sha256, and path columns.')
		for row in reader:
			index = int(row['case_index'])
			if index not in selected:
				continue
			if index in paths:
				raise ValueError(f'Duplicate initial path for case {index}.')
			if row['sha256'] != selected[index].digest:
				raise ValueError(f'Initial path hash mismatch for case {index}.')
			points = json.loads(row['path'])
			if not isinstance(points, list) or len(points) < 2 or any(
				not isinstance(point, list) or len(point) != 2
				or any(isinstance(value, bool) or not isinstance(value, (float, int))
					or not math.isfinite(value) for value in point)
				for point in points
			):
				raise ValueError(f'Invalid initial path for case {index}.')
			paths[index] = points
	missing = selected.keys() - paths.keys()
	if missing:
		raise ValueError(f'Missing initial paths for cases: {sorted(missing)[:10]}')
	return paths


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--suite', type=Path, default=ROOT / 'benchmarks/suites/algorithm-dev-v1.bin')
	parser.add_argument('--solver', type=Path, default=ROOT / '.build/unordered/tpp')
	parser.add_argument('--seconds', type=float, default=5)
	parser.add_argument('--max-calls', type=int, default=10000000)
	parser.add_argument('--workers', type=int, default=1)
	parser.add_argument('--solver-argument', action='append', default=[])
	parser.add_argument('--initial-paths', type=Path,
		help='Semicolon-delimited CSV with case_index, sha256, and path columns; replaces the solver initial heuristic.')
	parser.add_argument('--case', type=int, action='append')
	parser.add_argument('--case-list', type=Path,
		help='Text file with one case index per line; combines with repeated --case.')
	parser.add_argument('--output', type=Path, required=True)
	parser.add_argument('--resume', action='store_true')
	args = parser.parse_args(argv)
	if args.workers < 1:
		parser.error('--workers must be positive')
	args.output.parent.mkdir(parents=True, exist_ok=True)
	requested_cases = set(args.case or [])
	if args.case_list:
		requested_cases.update(int(line) for line in args.case_list.read_text().splitlines() if line.strip())
	cases = [case for case in read_encoded_cases(args.suite)
		if not requested_cases or case.case_index in requested_cases]
	initial_paths = read_initial_paths(args.initial_paths, cases) if args.initial_paths else {}
	initial_path_source = str(args.initial_paths.resolve()) if args.initial_paths else None
	completed = set()
	if args.resume and args.output.exists():
		for line in args.output.read_text().splitlines():
			try:
				previous = json.loads(line)
				if previous.get('initial_path_source') != initial_path_source:
					raise ValueError('Cannot resume with a different initial-path source.')
				completed.add(previous['case'])
			except (json.JSONDecodeError, KeyError):
				continue
	cases = [case for case in cases if case.case_index not in completed]

	def solve(case):
		s = struct.unpack_from('<dddd', case.data)
		row = {'case': case.case_index, 'sha256': case.digest, 'polygons': case.polygon_count}
		if initial_path_source:
			row['initial_path_source'] = initial_path_source
		try:
			row.update(run_unordered_solver(
				args.solver, s[:2], s[2:], case.polygons, args.max_calls, args.seconds,
				arguments=args.solver_argument,
				initial_path=initial_paths.get(case.case_index),
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
	return 0


if __name__ == '__main__':
	main()
