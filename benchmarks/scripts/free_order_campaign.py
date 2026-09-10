"""Campaign runner for endpoint TPP with free visit order."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import struct
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
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
		print('Build: free-order solver is up to date.', flush=True)
		return BINARY
	print('Build: configuring free-order solver...', flush=True)
	subprocess.run(['cmake', '-S', str(ROOT / 'packages/nonconvex-tpp/cpp'), '-B', str(BINARY.parent), '-DTARGET=main-unordered'], check=True)
	print('Build: compiling free-order solver...', flush=True)
	subprocess.run(['cmake', '--build', str(BINARY.parent), '--target', 'tpp', '-j', '8'], check=True)
	print('Build: complete.', flush=True)
	return BINARY


def atomic_json(path: Path, data: dict) -> None:
	temporary = path.with_suffix('.tmp')
	temporary.write_text(json.dumps(data, allow_nan=False))
	temporary.replace(path)


def case_geometry(case) -> dict:
	sx, sy, tx, ty = struct.unpack_from('<dddd', case.data)
	return {'start': [sx, sy], 'target': [tx, ty], 'polygons': case.polygons}


def finite(value: str) -> float | None:
	try:
		number = float(value)
		return number if math.isfinite(number) else None
	except (TypeError, ValueError):
		return None


def run_external(command: list[str], time_limit: float) -> None:
	process = subprocess.Popen(command, start_new_session=True)
	try:
		returncode = process.wait(timeout=time_limit + max(15.0, time_limit * 0.25))
	except subprocess.TimeoutExpired:
		os.killpg(process.pid, signal.SIGTERM)
		try:
			process.wait(timeout=2)
		except subprocess.TimeoutExpired:
			os.killpg(process.pid, signal.SIGKILL)
			process.wait()
		raise
	if returncode:
		raise subprocess.CalledProcessError(returncode, command)


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('campaign', type=Path)
	parser.add_argument('--solver', choices=('unordered', 'tspn'), action='append')
	parser.add_argument('--max-instances', type=int, default=5000)
	parser.add_argument('--max-calls', type=int, default=1000000)
	parser.add_argument('--max-seconds', type=float, default=30)
	parser.add_argument('--threads', type=int, default=1)
	parser.add_argument('--no-build', action='store_true')
	parser.add_argument('--force', action='store_true')
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args(argv)
	if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.max_calls < 0 or args.max_instances < 1 or args.threads < 1:
		parser.error('Expected positive seconds/instance cap and nonnegative calls.')
	campaign = args.campaign if args.campaign.is_absolute() or args.campaign.parent != Path('.') else ROOT / 'benchmarks/campaigns' / args.campaign
	metadata = json.loads((campaign / 'campaign.json').read_text())
	cases = [case for record in metadata['inputs'] for case in read_encoded_cases(campaign / record['file'])][:args.max_instances]
	if not cases:
		parser.error('Campaign has no cases.')
	solvers = list(dict.fromkeys(args.solver or ['unordered']))
	if 'tspn' in solvers and args.max_seconds != int(args.max_seconds):
		parser.error('The external runner requires an integer time limit in seconds.')
	config = {'visit_order': 'free', 'solvers': solvers, 'threads': args.threads, 'max_calls': args.max_calls,
		'max_seconds': args.max_seconds, 'hashes': [c.digest for c in cases],
		'ours_optimality': {'absolute_gap': 1e-7, 'relative_gap': 1e-9},
		'external_optimality_eps': 1e-9, 'solver_feasibility_tolerance': 1e-8,
		'independent_validation_tolerance': 1e-7}
	key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
	results = campaign / 'results/free-order'
	if args.dry_run:
		print(json.dumps(config, indent=2))
		return 0
	resume_path = None
	if not args.force and results.exists():
		for prior in sorted(results.glob('*/report.json'), key=lambda path: path.stat().st_mtime_ns, reverse=True):
			try:
				old = json.loads(prior.read_text())
			except (OSError, json.JSONDecodeError):
				continue
			if old.get('key') == key and old.get('status') == 'completed':
				prior.touch()
				print(f'Reusing {prior}', flush=True)
				return 0
			if old.get('key') == key and resume_path is None:
				resume_path = prior
	if 'unordered' in solvers:
		ensure_binary(args.no_build)
	if 'tspn' in solvers and not (EXTERNAL_PYTHON.exists() and EXTERNAL_RUNNER.exists()):
		raise FileNotFoundError('External TSPN checkout/environment is unavailable.')
	if resume_path:
		run = resume_path.parent
		report = json.loads(resume_path.read_text())
		report['status'] = 'running'
		report.pop('error', None)
		report['resumed_at'] = datetime.now(UTC).isoformat()
		print(f'Resuming checkpoint: {resume_path}', flush=True)
	else:
		run = results / (datetime.now(UTC).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6])
		run.mkdir(parents=True)
		report = {'schema_version': 2, 'key': key, 'config': config, 'visit_order': 'free', 'title': metadata.get('name', campaign.name),
			'created_at': datetime.now(UTC).isoformat(), 'status': 'running', 'rows': [],
			'notes': [f'Fixed endpoints; free visit order. {args.threads} campaign worker(s); each solver process is single-threaded.',
			'Our tolerance is 1e-7 + 1e-9 × UB. The external API only exposes a relative ratio, so eps=1e-9 is conservative rather than algebraically identical.',
			'Both solvers use feasibility tolerance 1e-8 where their APIs permit it; independent validation uses 1e-7.',
			'External raw and endpoint-snapped trajectories are reported separately; snapping never changes the declared solver result.',
			'External calls are SOCP calls; our calls invoke a certified convex oracle. They are not identical units.']}
	geometry_catalog = {case.digest: case_geometry(case) for case in cases}
	geometry_dir = run / 'geometry'
	geometry_dir.mkdir(exist_ok=True)
	for digest, geometry in geometry_catalog.items():
		path = geometry_dir / f'{digest}.json'
		if not path.exists():
			atomic_json(path, geometry)
	atomic_json(run / 'geometry.json', {'schema_version': 2, 'hashes': list(geometry_catalog)})
	for row in report['rows']:
		if row.get('geometry'):
			row['geometry_sha256'] = row.get('sha256')
			row.pop('geometry', None)

	def successful_pairs() -> set[tuple[str, int]]:
		return {(row.get('solver'), int(row.get('case', -1))) for row in report['rows'] if not row.get('error')}

	def save_checkpoint() -> None:
		report['checkpoint'] = {
			'completed_pairs': len(successful_pairs()),
			'total_pairs': len(cases) * len(solvers),
			'updated_at': datetime.now(UTC).isoformat(),
		}
		atomic_json(run / 'report.json', report)

	save_checkpoint()
	try:
		for solver in solvers:
			print(f'## {solver}', flush=True)
			if solver == 'unordered':
				def solve_case(i: int) -> dict:
					case = cases[i]
					geometry = geometry_catalog[case.digest]
					sx, sy = geometry['start']
					tx, ty = geometry['target']
					row = {'case': i, 'sha256': case.digest, 'solver': solver, 'polygons': len(case.polygons),
						'geometry_sha256': case.digest}
					try:
						row.update(run_unordered_solver(BINARY, (sx, sy), (tx, ty), case.polygons, args.max_calls, args.max_seconds))
						row['visit_order'] = 'free'
						row['length'] = row['upper_bound']
						row['validation'] = validate_path((sx, sy), (tx, ty), case.polygons, row['path'], 1e-7)
						row['valid'] = row['validation']['valid']
					except (RuntimeError, subprocess.TimeoutExpired) as error:
						row['error'] = str(error)
					return row

				pending = [i for i in range(len(cases)) if (solver, i) not in successful_pairs()]
				with ThreadPoolExecutor(max_workers=min(args.threads, max(1, len(pending)))) as executor:
					futures = {}
					for i in pending:
						print(f'instance | [free] {i + 1}/{len(cases)} queued', flush=True)
						futures[executor.submit(solve_case, i)] = i
					for completed, future in enumerate(as_completed(futures), 1):
						row = future.result()
						report['rows'] = [item for item in report['rows'] if not (item.get('solver') == solver and item.get('case') == row['case'])]
						report['rows'].append(row)
						report['rows'].sort(key=lambda item: (item['case'], solvers.index(item['solver'])))
						save_checkpoint()
						print(f'cases | [free] {completed}/{len(cases)} | case {row["case"] + 1} complete', flush=True)
			else:
				if all((solver, i) in successful_pairs() for i in range(len(cases))):
					print(f'cases | [free] {len(cases)}/{len(cases)} | resumed', flush=True)
					continue
				suite = run / 'input.bin'
				suite.write_bytes(b''.join(c.data for c in cases))
				external_command = [str(EXTERNAL_PYTHON), str(EXTERNAL_RUNNER), '--suite', str(suite), '--mode', 'path',
					'--threads', '1', '--time-limit', str(int(args.max_seconds)), '--eps', '0.000000001',
					'--feasibility-tolerance', '0.00000001', '--validation-tolerance', '0.0000001',
					'--output', str(run / 'external')]
				run_external(external_command, args.max_seconds)
				csv_path = next((run / 'external').glob('*/*-tspn-path.csv'))
				with csv_path.open() as file:
					for external in csv.DictReader(file):
						i = int(external['case_index'])
						if external['sha256'] != cases[i].digest:
							raise ValueError('External instance hash mismatch.')
						report['rows'] = [item for item in report['rows'] if not (item.get('solver') == solver and item.get('case') == i)]
						report['rows'].append({'case': i, 'sha256': cases[i].digest, 'geometry_sha256': cases[i].digest, 'solver': solver,
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
		complete = len(successful_pairs()) == len(cases) * len(solvers)
		report['status'] = 'completed' if complete else 'failed'
	except Exception as error:
		report['status'] = 'failed'
		report['error'] = str(error)
		raise
	finally:
		save_checkpoint()
	print(f'Report: {run / "report.json"}', flush=True)
	return int(report['status'] != 'completed')


if __name__ == '__main__':
	raise SystemExit(main())
