#!/usr/bin/env python3
"""Audit unordered JSONL runs against their suite and write independent summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
from collections import Counter
from pathlib import Path

from benchmark_cases import EncodedCase, read_encoded_cases
from unordered_validation import validate_path


def audit(source: Path, cases: dict[int, EncodedCase], selected: set[int]) -> tuple[dict, list[dict]]:
	rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
	indices = [row['case'] for row in rows]
	if len(indices) != len(set(indices)) or set(indices) != selected:
		raise ValueError(f'{source}: duplicate, missing or unexpected cases')
	records = []
	for row in rows:
		case = cases[row['case']]
		if row.get('error') or row['sha256'] != case.digest:
			raise ValueError(f'{source}: case {case.case_index}: error or wrong input hash')
		start_x, start_y, target_x, target_y = struct.unpack_from('<dddd', case.data)
		start, target = (start_x, start_y), (target_x, target_y)
		lb, ub = row['lower_bound'], row['upper_bound']
		tolerance = 1e-7 + 1e-9 * abs(ub)
		if not all(math.isfinite(v) for v in (lb, ub, row['seconds'])):
			raise ValueError(f'{source}: nonfinite bounds or time')
		if lb > ub or lb < math.dist(start, target) - tolerance:
			raise ValueError(f'{source}: inconsistent bounds')
		closed = ub - lb <= tolerance
		if row['exact'] != closed or (row['termination'] == 'optimal') != closed:
			raise ValueError(f'{source}: inconsistent numerical certification')
		if row['termination'] not in ('optimal', 'call_limit', 'time_limit', 'numerical_limit'):
			raise ValueError(f'{source}: unknown termination')
		if sorted(row['order']) != list(range(case.polygon_count)):
			raise ValueError(f'{source}: order is not a permutation')
		if not all(math.isfinite(v) for point in row['path'] for v in point):
			raise ValueError(f'{source}: nonfinite path')
		validation = validate_path(start, target, case.polygons, row['path'], 1e-7)
		length_error = abs(validation['recomputed_length'] - ub)
		if not validation['valid'] or length_error > tolerance:
			raise ValueError(f'{source}: invalid geometry or inconsistent length in case {case.case_index}')
		profile = row['profile']
		if row['seconds'] <= 0 or any(not math.isfinite(value) or value < 0
			for key, value in profile.items() if key.endswith('_seconds')):
			raise ValueError(f'{source}: invalid profiling time')
		if row['fallback_calls'] != row['fallback_geometric_path_invalid_calls'] + row['fallback_certificate_gap_calls']:
			raise ValueError(f'{source}: inconsistent fallback counters')
		records.append({
			'run': str(source), 'case': case.case_index, 'sha256': case.digest,
			'polygons': case.polygon_count, 'lower_bound': lb, 'upper_bound': ub,
			'absolute_gap': ub - lb, 'relative_gap': (ub - lb) / max(abs(ub), 1e-30),
			'gap_tolerance': tolerance, 'exact': closed, 'termination': row['termination'],
			'valid': validation['valid'], 'max_polygon_distance': validation['max_polygon_distance'],
			'start_distance': validation['start_distance'], 'target_distance': validation['target_distance'],
			'length_error': length_error, 'seconds': row['seconds'], 'calls': row['calls'],
			'fallback_calls': row['fallback_calls'],
			'fallback_path_calls': row['fallback_geometric_path_invalid_calls'],
			'fallback_gap_calls': row['fallback_certificate_gap_calls'],
			'extended_precision_calls': row['extended_precision_calls'],
			'repaired_geometric_path_calls': row['repaired_geometric_path_calls'],
			'order': json.dumps(row['order']),
			**{key: value for key, value in profile.items() if key.endswith('_seconds')},
		})
	total_seconds = sum(row['seconds'] for row in records)
	certified = sum(row['exact'] for row in records)
	return {
		'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
		'cases': len(records), 'valid_paths': sum(row['valid'] for row in records),
		'exact_certified': certified,
		# Kept as an audit compatibility alias for older comparison scripts.
		'numerically_certified': certified,
		'termination_counts': dict(Counter(row['termination'] for row in records)),
		'solver_seconds': total_seconds,
		'mean_relative_gap': statistics.mean(row['relative_gap'] for row in records),
		'max_relative_gap': max(row['relative_gap'] for row in records),
		'max_polygon_distance': max(row['max_polygon_distance'] for row in records),
		'max_length_error': max(row['length_error'] for row in records),
		'max_solver_seconds': max(row['seconds'] for row in records),
		'calls': sum(row['calls'] for row in records),
		'fallback_calls': sum(row['fallback_calls'] for row in records),
		'fallback_time_fraction': sum(row['convex_fallback_seconds'] for row in records) / total_seconds,
		'oracle_time_fraction': sum(row['convex_oracle_seconds'] for row in records) / total_seconds,
		'validation_tolerance': 1e-7,
		'certification': 'Exact certification for completed searches; geometry independently checked.',
	}, records


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('runs', type=Path, nargs='+')
	parser.add_argument('--suite', type=Path, required=True)
	parser.add_argument('--case', type=int, action='append')
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args()
	if len({source.resolve() for source in args.runs}) != len(args.runs):
		raise ValueError('Repeated input file is not an independent repetition')
	cases = {case.case_index: case for case in read_encoded_cases(args.suite)}
	selected = set(args.case) if args.case is not None else set(cases)
	if not selected or not selected <= cases.keys():
		raise ValueError('Unknown or empty case selection')
	summaries, records = [], []
	for source in args.runs:
		summary, audited = audit(source, cases, selected)
		summaries.append(summary)
		records.extend(audited)
	args.output.mkdir(parents=True, exist_ok=False)
	with (args.output / 'cases.csv').open('w', newline='') as file:
		writer = csv.DictWriter(file, fieldnames=list(records[0]))
		writer.writeheader()
		writer.writerows(records)
	result = {'suite_sha256': hashlib.sha256(args.suite.read_bytes()).hexdigest(), 'runs': summaries}
	if len(args.runs) > 1:
		repetitions = []
		for index in sorted(selected):
			group = [row for row in records if row['case'] == index]
			repetitions.append({
				'case': index, 'repetitions': len(group),
				'termination_counts': dict(Counter(row['termination'] for row in group)),
				'seconds_min': min(row['seconds'] for row in group),
				'seconds_median': statistics.median(row['seconds'] for row in group),
				'seconds_max': max(row['seconds'] for row in group),
				'calls_min': min(row['calls'] for row in group),
				'calls_max': max(row['calls'] for row in group),
				'lower_bound_min': min(row['lower_bound'] for row in group),
				'lower_bound_max': max(row['lower_bound'] for row in group),
				'upper_bound_min': min(row['upper_bound'] for row in group),
				'upper_bound_max': max(row['upper_bound'] for row in group),
				'identical_orders': len({row['order'] for row in group}) == 1,
				**{key + '_fraction_median': statistics.median(row[key] / row['seconds'] for row in group)
					for key in ('convex_oracle_seconds', 'convex_fallback_seconds', 'convex_fallback_extended_precision_seconds')},
			})
		result['repetitions_by_case'] = repetitions
	(args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
	print(json.dumps(result, indent=2))


if __name__ == '__main__':
	main()
