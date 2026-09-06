"""Campaign runner for endpoint TPP with free visit order."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from benchmark_cases import read_encoded_cases
from unordered_runner import run_unordered_solver
from unordered_validation import validate_path

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / '.build/unordered/tpp'
EXTERNAL_PYTHON = ROOT / 'tspn-comparison/solver/.venv/bin/python'
EXTERNAL_RUNNER = ROOT / 'tspn-comparison/benchmarks/run_comparison.py'


def ensure_binary(no_build: bool = False) -> Path:
	if no_build:
		if not BINARY.exists():
			raise FileNotFoundError('Build the free-order solver before using --no-build.')
		return BINARY
	sources = [p for package in ('common-geometry', 'convex-tpp', 'nonconvex-tpp', 'optimal-convex-partition')
		for p in (ROOT / 'packages' / package / 'cpp').rglob('*') if p.suffix in ('.cpp', '.h', '.txt')]
	if BINARY.exists() and all(p.stat().st_mtime_ns <= BINARY.stat().st_mtime_ns for p in sources):
		return BINARY
	subprocess.run(['cmake', '-S', str(ROOT / 'packages/nonconvex-tpp/cpp'), '-B', str(BINARY.parent), '-DTARGET=main-unordered'], check=True)
	subprocess.run(['cmake', '--build', str(BINARY.parent), '--target', 'tpp', '-j', '8'], check=True)
	return BINARY


def atomic_json(path: Path, data: dict) -> None:
	temporary = path.with_suffix('.tmp')
	temporary.write_text(json.dumps(data, allow_nan=False))
	temporary.replace(path)


def finite(value: str) -> float | None:
	try:
		number = float(value)
		return number if math.isfinite(number) else None
	except (TypeError, ValueError):
		return None


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('campaign', type=Path)
	parser.add_argument('--solver', choices=('unordered', 'tspn'), action='append')
	parser.add_argument('--max-instances', type=int, default=5000)
	parser.add_argument('--max-calls', type=int, default=1000000)
	parser.add_argument('--max-seconds', type=float, default=30)
	parser.add_argument('--threads', type=int, choices=(1,), default=1)
	parser.add_argument('--no-build', action='store_true')
	parser.add_argument('--force', action='store_true')
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args(argv)
	if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.max_calls < 0 or args.max_instances < 1:
		parser.error('Expected positive seconds/instance cap and nonnegative calls.')
	campaign = args.campaign if args.campaign.is_absolute() or args.campaign.parent != Path('.') else ROOT / 'benchmarks/campaigns' / args.campaign
	metadata = json.loads((campaign / 'campaign.json').read_text())
	cases = [case for record in metadata['inputs'] for case in read_encoded_cases(campaign / record['file'])][:args.max_instances]
	if not cases:
		parser.error('Campaign has no cases.')
	solvers = list(dict.fromkeys(args.solver or ['unordered']))
	if 'tspn' in solvers and args.max_seconds != int(args.max_seconds):
		parser.error('The external runner requires an integer time limit in seconds.')
	config = {'visit_order': 'free', 'solvers': solvers, 'threads': 1, 'max_calls': args.max_calls,
		'max_seconds': args.max_seconds, 'hashes': [c.digest for c in cases],
		'ours_optimality': {'absolute_gap': 1e-7, 'relative_gap': 1e-9},
		'external_optimality_eps': 1e-9, 'solver_feasibility_tolerance': 1e-8,
		'independent_validation_tolerance': 1e-7}
	key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
	results = campaign / 'results/free-order'
	if args.dry_run:
		print(json.dumps(config, indent=2)); return 0
	if not args.force and results.exists():
		for prior in results.glob('*/report.json'):
			old = json.loads(prior.read_text())
			if old.get('key') == key and old.get('status') == 'completed':
				prior.touch()
				print(f'Reusing {prior}', flush=True); return 0
	if 'unordered' in solvers:
		ensure_binary(args.no_build)
	if 'tspn' in solvers and not (EXTERNAL_PYTHON.exists() and EXTERNAL_RUNNER.exists()):
		raise FileNotFoundError('External TSPN checkout/environment is unavailable.')
	run = results / (datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6])
	run.mkdir(parents=True)
	report = {'key': key, 'config': config, 'visit_order': 'free', 'title': metadata.get('name', campaign.name),
		'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'rows': [],
		'notes': ['Fixed endpoints; free visit order. One worker.',
			'Our tolerance is 1e-7 + 1e-9 × UB. The external API only exposes a relative ratio, so eps=1e-9 is conservative rather than algebraically identical.',
			'Both solvers use feasibility tolerance 1e-8 where their APIs permit it; independent validation uses 1e-7.',
			'External raw and endpoint-snapped trajectories are reported separately; snapping never changes the declared solver result.',
			'External calls are SOCP calls; our calls invoke a certified convex oracle. They are not identical units.']}
	atomic_json(run / 'report.json', report)
	try:
		for solver in solvers:
			print(f'## {solver}', flush=True)
			if solver == 'unordered':
				for i, case in enumerate(cases):
					sx, sy, tx, ty = struct.unpack_from('<dddd', case.data)
					row = {'case': i, 'sha256': case.digest, 'solver': solver, 'polygons': len(case.polygons),
						'geometry': {'start': [sx, sy], 'target': [tx, ty], 'polygons': case.polygons}}
					try:
						row.update(run_unordered_solver(BINARY, (sx, sy), (tx, ty), case.polygons, args.max_calls, args.max_seconds))
						row['visit_order'] = 'free'
						row['length'] = row['upper_bound']
						row['validation'] = validate_path((sx, sy), (tx, ty), case.polygons, row['path'], 1e-7)
						row['valid'] = row['validation']['valid']
					except (RuntimeError, subprocess.TimeoutExpired) as error:
						row['error'] = str(error)
					report['rows'].append(row)
					atomic_json(run / 'report.json', report)
					print(f'cases | [free] {i + 1}/{len(cases)}', flush=True)
			else:
				suite = run / 'input.bin'
				suite.write_bytes(b''.join(c.data for c in cases))
				subprocess.run([str(EXTERNAL_PYTHON), str(EXTERNAL_RUNNER), '--suite', str(suite), '--mode', 'path',
					'--threads', '1', '--time-limit', str(int(args.max_seconds)), '--eps', '0.000000001',
					'--feasibility-tolerance', '0.00000001', '--validation-tolerance', '0.0000001',
					'--output', str(run / 'external')], check=True)
				csv_path = next((run / 'external').glob('*/*-tspn-path.csv'))
				with csv_path.open() as file:
					for external in csv.DictReader(file):
						i = int(external['case_index'])
						if external['sha256'] != cases[i].digest:
							raise ValueError('External instance hash mismatch.')
						report['rows'].append({'case': i, 'sha256': cases[i].digest, 'solver': solver,
							'polygons': len(cases[i].polygons), 'upper_bound': finite(external['upper_bound']),
							'lower_bound': finite(external['lower_bound']), 'seconds': finite(external['solve_seconds']),
							'calls': finite(external['soc_num_calls']), 'exact': external['is_optimal'] == 'True',
							'path': json.loads(external['trajectory_json']) if external.get('trajectory_json') else None,
							'endpoint_valid': external['is_valid_trajectory'] == 'True',
							'valid': external.get('raw_valid') == 'True',
							'endpoint_repaired_valid': external.get('snapped_valid') == 'True',
							'validation': {'start_distance': finite(external.get('start_distance')),
								'target_distance': finite(external.get('target_distance')),
								'max_polygon_distance': finite(external.get('max_polygon_distance')),
								'recomputed_length': finite(external.get('recomputed_length'))},
							'termination': external['status'], 'error': external['error'] or None})
				print(f'cases | [free] {len(cases)}/{len(cases)}', flush=True)
		report['status'] = 'failed' if any(r.get('error') for r in report['rows']) else 'completed'
	except Exception as error:
		report['status'] = 'failed'; report['error'] = str(error)
		raise
	finally:
		atomic_json(run / 'report.json', report)
	print(f'Report: {run / "report.json"}', flush=True)
	return int(report['status'] != 'completed')


if __name__ == '__main__':
	raise SystemExit(main())
