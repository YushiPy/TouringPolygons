#!/usr/bin/env python3
"""Summarize a matched unordered-TPP JSONL / external path-mode CSV run."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('ours', type=Path)
	parser.add_argument('external', type=Path)
	parser.add_argument('--output', type=Path, required=True)
	args = parser.parse_args()
	ours = [json.loads(line) for line in args.ours.read_text().splitlines()]
	with args.external.open() as file:
		external = {int(row['case_index']): row for row in csv.DictReader(file)}
	rows = []
	for a in ours:
		b = external[a['case']]
		if a['sha256'] != b['sha256'] or b['mode'] != 'path':
			raise ValueError(f"Unmatched instance or problem variant: {a['case']}")
		if 'error' in a:
			raise ValueError(f"Solver error: {a['case']}: {a['error']}")
		ub, lb, seconds = (float(b[key]) for key in ('upper_bound', 'lower_bound', 'solve_seconds'))
		common = a['exact'] and b['is_optimal'] == 'True'
		if common and abs(a['upper_bound'] - ub) > 1e-6 * max(1, ub):
			raise ValueError(f"Objective disagreement: {a['case']}")
		profile = a.get('profile', {})
		external_raw_valid = None if not b.get('raw_valid') else b['raw_valid'] == 'True'
		external_snapped_valid = None if not b.get('snapped_valid') else b['snapped_valid'] == 'True'
		row = {
			'case': a['case'], 'polygons': a['polygons'],
			'ours_declared_optimal': a['exact'], 'external_declared_optimal': b['is_optimal'] == 'True',
			'ours_verified_feasible': a.get('valid'), 'external_raw_verified_feasible': external_raw_valid,
			'external_endpoint_repaired_feasible': external_snapped_valid,
			'raw_results_comparable': a.get('valid') is True and external_raw_valid,
			'ours_seconds': a['seconds'], 'external_seconds': seconds,
			'external_process_seconds': float(b['process_seconds']) if b.get('process_seconds') else None,
			'ours_lower': a['lower_bound'], 'ours_upper': a['upper_bound'],
			'external_lower': lb, 'external_upper': ub,
			'external_recomputed_length': float(b['recomputed_length']) if b.get('recomputed_length') else None,
			'external_snapped_length': float(b['snapped_recomputed_length']) if b.get('snapped_recomputed_length') else None,
			'ours_gap': (a['upper_bound'] - a['lower_bound']) / a['upper_bound'] if a['upper_bound'] else 0,
			'external_gap': (ub - lb) / ub if ub else 0,
			'common_speedup': seconds / a['seconds'] if common else None,
		}
		for key, value in profile.items():
			if key != 'timing_semantics':
				row[f'ours_{key}'] = value
		for key in ('calls', 'nodes', 'fallback_calls', 'fallback_geometric_path_invalid_calls',
			'fallback_certificate_gap_calls', 'extended_precision_calls',
			'repaired_geometric_path_calls', 'insertion_branches', 'decomposition_branches', 'peak_queue'):
			row[f'ours_{key}'] = a.get(key)
		rows.append(row)
	common = [r['common_speedup'] for r in rows if r['common_speedup'] is not None]
	validated_common = [r['common_speedup'] for r in rows if r['common_speedup'] is not None and r['raw_results_comparable']]
	summary = {
		'instances': len(rows), 'ours_declared_optimal': sum(r['ours_declared_optimal'] for r in rows),
		'external_declared_optimal': sum(r['external_declared_optimal'] for r in rows),
		'ours_valid': sum(a.get('valid') is True for a in ours),
		'external_raw_validation_available': sum(r['external_raw_verified_feasible'] is not None for r in rows),
		'external_raw_verified_feasible': sum(r['external_raw_verified_feasible'] is True for r in rows),
		'external_endpoint_repaired_feasible': sum(r['external_endpoint_repaired_feasible'] is True for r in rows),
		'external_endpoint_check_failures': sum(b['is_valid_trajectory'] != 'True' for b in external.values()),
		'common_exact': len(common), 'median_speedup_common': statistics.median(common) if common else None,
		'ours_faster_common': sum(s > 1 for s in common),
		'endpoint_validated_common': len(validated_common),
		'median_speedup_endpoint_validated_common': statistics.median(validated_common) if validated_common else None,
		'ours_mean_relative_gap': statistics.mean(r['ours_gap'] for r in rows),
		'external_mean_relative_gap': statistics.mean(r['external_gap'] for r in rows),
		'ours_total_seconds': sum(r['ours_seconds'] for r in rows),
		'external_total_seconds': sum(r['external_seconds'] for r in rows),
		'ours_total_fallback_calls': sum(r['ours_fallback_calls'] for r in rows)
			if all(r['ours_fallback_calls'] is not None for r in rows) else None,
		'ours_total_repaired_geometric_path_calls': sum(r['ours_repaired_geometric_path_calls'] for r in rows)
			if all(r['ours_repaired_geometric_path_calls'] is not None for r in rows) else None,
	}
	args.output.mkdir(parents=True, exist_ok=True)
	(args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
	with (args.output / 'comparison.csv').open('w') as file:
		writer = csv.DictWriter(file, fieldnames=rows[0].keys())
		writer.writeheader()
		writer.writerows(rows)
	print(json.dumps(summary, indent=2))


if __name__ == '__main__':
	main()
