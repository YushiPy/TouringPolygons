"""Campaign runner for endpoint TPP with free visit order."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
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
EXTERNAL_PYTHON = ROOT / 'third_party/tspn-socg/.venv/bin/python'
EXTERNAL_RUNNER = ROOT / 'benchmarks/_internal/tspn_run_comparison.py'
EXTERNAL_SOURCE = ROOT / 'third_party/tspn-socg'


def _build_cache_matches_checkout(build_dir: Path) -> bool:
	cache = build_dir / 'CMakeCache.txt'
	if not cache.exists():
		return True
	values = {}
	try:
		for line in cache.read_text().splitlines():
			for key in ('CMAKE_HOME_DIRECTORY:INTERNAL=', 'CMAKE_CACHEFILE_DIR:INTERNAL='):
				if line.startswith(key):
					values[key.split(':', 1)[0]] = line[len(key):]
	except OSError:
		return False
	source_dir = (ROOT / 'packages/nonconvex-tpp/cpp').resolve()
	return values.get('CMAKE_HOME_DIRECTORY') == str(source_dir) and values.get('CMAKE_CACHEFILE_DIR') == str(build_dir.resolve())


def ensure_binary(no_build: bool = False) -> Path:
	if no_build:
		if not BINARY.exists():
			raise FileNotFoundError('Build the free-order solver before using --no-build.')
		return BINARY
	sources = [p for package in ('common-geometry', 'convex-tpp', 'nonconvex-tpp', 'optimal-convex-partition')
		for p in (ROOT / 'packages' / package / 'cpp').rglob('*') if p.suffix in ('.cpp', '.h', '.txt')]
	cache_matches = _build_cache_matches_checkout(BINARY.parent)
	if BINARY.exists() and cache_matches and all(p.stat().st_mtime_ns <= BINARY.stat().st_mtime_ns for p in sources):
		print('Build: free-order solver is up to date.', flush=True)
		return BINARY
	if not cache_matches:
		print('Build: discarding a relocated CMake build cache...', flush=True)
		shutil.rmtree(BINARY.parent)
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


def optional_bool(value: str | None) -> bool | None:
	if value == 'True':
		return True
	if value == 'False':
		return False
	return None


def write_comparison_summary(path: Path, report: dict, expected_cases: int) -> None:
	rows = report.get('rows', [])
	by_solver = {
		name: {int(row['case']): row for row in rows if row.get('solver') == name and not row.get('error')}
		for name in ('unordered', 'tspn')
	}
	config = report.get('config', {})
	lines = [
		'# Free-order multithreaded solver comparison', '',
		f"- Instances in suite: {expected_cases}",
		f"- Instance workers: {config.get('instance_workers', 'unknown')} (instances processed concurrently)",
		f"- Threads per instance: {config.get('threads_per_instance', 'unknown')}",
		f"- Per-instance time cap: {config.get('max_seconds', 'unknown')} s",
		f"- Our gap: absolute {config.get('ours_optimality', {}).get('absolute_gap', 'unknown')} + relative {config.get('ours_optimality', {}).get('relative_gap', 'unknown')} × |UB|",
		f"- Fekete tolerance: UB/LB ≤ 1 + {config.get('external_optimality_eps', 'unknown')}",
		f"- Fekete source revision: `{config.get('external_source_revision', 'unknown')}`; binding SHA-256: `{config.get('external_binding_sha256', 'unknown')}`",
		f"- Initial strategies: {', '.join(name for name, enabled in config.get('initial_strategies', {}).items() if enabled) or 'none'}", '',
		'Fekete uses its relative-ratio termination test; our solver closes an additive absolute-plus-relative gap. Their thresholds are close, not identical.',
		'The runtime ratios below compare the two solver configurations. They do not isolate the speedup from threading, because no single-thread control run is included.', '',
		'| Solver | Recorded cases | Closed requested gap | Independent valid paths | Median solve time | Total solve time |',
		'|---|---:|---:|---:|---:|---:|',
	]
	for solver, label in (('unordered', 'Our solver'), ('tspn', 'Fekete')):
		group = list(by_solver[solver].values())
		times = [float(row['seconds']) for row in group if row.get('seconds') is not None]
		closed = sum(row.get('exact') is True for row in group)
		valid = sum(row.get('valid') is True for row in group)
		valid_known = sum(row.get('valid') is not None for row in group)
		lines.append(f"| {label} | {len(group)} | {closed}/{len(group)} | {valid}/{valid_known} known | "
			f"{statistics.median(times):.3f}s | {sum(times):.3f}s |" if times else
			f"| {label} | {len(group)} | {closed}/{len(group)} | {valid}/{valid_known} known | n/a | n/a |")
	paired = [(by_solver['unordered'][i], by_solver['tspn'][i]) for i in sorted(set(by_solver['unordered']) & set(by_solver['tspn']))]
	ratios = [float(ours['seconds']) / float(fekete['seconds']) for ours, fekete in paired
		if ours.get('seconds') and fekete.get('seconds') and float(ours['seconds']) > 0 and float(fekete['seconds']) > 0]
	if ratios:
		lines.extend([
			'', f"Paired runtime data: {len(ratios)} instances.",
			f"Median our/Fekete runtime ratio: {statistics.median(ratios):.3f}× (below 1 means our solver was faster).",
			f"Our solver faster: {sum(ratio < 1 for ratio in ratios)}; Fekete faster: {sum(ratio > 1 for ratio in ratios)}; equal: {sum(ratio == 1 for ratio in ratios)}.",
		])
	if any(row.get('valid') is None for group in by_solver.values() for row in group.values()):
		lines.extend(['', 'Independent geometric validation was unavailable for some rows because Shapely was not installed in that solver environment. Those rows are marked unknown, not valid.'])
	reference_path = ROOT / 'benchmarks/results-saved/german-comparison/fekete.csv'
	if reference_path.exists() and 'tspn' in config.get('solvers', []):
		with reference_path.open(newline='') as file:
			reference_rows = {int(row['case_index']): row for row in csv.DictReader(file)}
		hashes = config.get('hashes', [])
		compatible_reference = len(hashes) == expected_cases and all(
			i in reference_rows and reference_rows[i]['sha256'] == digest
			and int(reference_rows[i]['threads']) == 1
			and float(reference_rows[i]['eps']) == float(config.get('external_optimality_eps', -1))
			and int(reference_rows[i]['time_limit_seconds']) == int(config.get('max_seconds', -1))
			for i, digest in enumerate(hashes)
		)
		if compatible_reference:
			thread_pairs = [(float(reference_rows[i]['solve_seconds']), float(row['seconds']))
				for i, row in by_solver['tspn'].items() if row.get('seconds') is not None
				and float(reference_rows[i]['solve_seconds']) > 0 and float(row['seconds']) > 0]
			if thread_pairs:
				speedups = [single / multi for single, multi in thread_pairs]
				thread_count = config.get('threads_per_instance', 'unknown')
				lines.extend([
					'', f"Fekete thread-count control against the saved 1-thread run: {len(thread_pairs)} paired cases, same hashes, eps, and per-instance cap.",
					f"Median 1-thread/{thread_count}-thread runtime ratio: {statistics.median(speedups):.3f}× (above 1 means the multithreaded run was faster).",
					f"Multithreaded run faster: {sum(value > 1 for value in speedups)}; 1 thread faster: {sum(value < 1 for value in speedups)}; equal: {sum(value == 1 for value in speedups)}.",
					'This is a historical paired comparison; machine load and software environment may differ between campaigns.',
				])
	our_rows = list(by_solver['unordered'].values())
	parallel_cases = sum(int(row.get('parallel_oracle_calls', 0) or 0) > 0 for row in our_rows)
	parallel_calls = sum(int(row.get('parallel_oracle_calls', 0) or 0) for row in our_rows)
	parallel_batches = sum(int(row.get('parallel_oracle_batches', 0) or 0) for row in our_rows)
	if our_rows:
		lines.extend(['', f"Our solver launched parallel oracle batches on {parallel_cases}/{len(our_rows)} completed instances "
			f"({parallel_calls} calls in {parallel_batches} batches)."])
	lines.extend(['', f"Campaign status: {report.get('status', 'unknown')}.", ''])
	path.write_text('\n'.join(lines))


def run_external(command: list[str], timeout_seconds: float) -> None:
	process = subprocess.Popen(command, start_new_session=True)
	try:
		returncode = process.wait(timeout=timeout_seconds)
	except KeyboardInterrupt:
		try:
			os.killpg(process.pid, signal.SIGINT)
			raise
		finally:
			try:
				process.wait(timeout=15)
			except subprocess.TimeoutExpired:
				os.killpg(process.pid, signal.SIGTERM)
				process.wait()
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
	parser.add_argument('--threads-per-instance', type=int, default=1,
		help='Solver threads used inside one instance.')
	parser.add_argument('--workers', type=int, default=1,
		help='Different instances processed concurrently; use 1 for sequential cases.')
	parser.add_argument('--absolute-gap', type=float, default=1e-7)
	parser.add_argument('--relative-gap', type=float, default=1e-9)
	parser.add_argument('--eps', type=float, default=1e-9,
		help='Fekete relative ratio tolerance (UB/LB <= 1 + eps).')
	parser.add_argument('--sampled-perimeter-initial', action='store_true')
	parser.add_argument('--convex-initial-refinement', action='store_true')
	parser.add_argument('--bidirectional-initial', action='store_true')
	parser.add_argument('--feasibility-tolerance', type=float, default=1e-8)
	parser.add_argument('--validation-tolerance', type=float, default=1e-7)
	parser.add_argument('--external-python', type=Path, default=EXTERNAL_PYTHON,
		help='Python environment for Fekete; defaults to the submodule .venv.')
	parser.add_argument('--external-build', type=Path, default=EXTERNAL_SOURCE,
		help='Fekete source/build tree containing the compiled Python binding.')
	parser.add_argument('--no-build', action='store_true')
	parser.add_argument('--force', action='store_true')
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args(argv)
	if (not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.max_calls < 0
		or args.max_instances < 1 or args.threads_per_instance < 1 or args.workers < 1
		or not math.isfinite(args.absolute_gap) or args.absolute_gap < 0
		or not math.isfinite(args.relative_gap) or args.relative_gap < 0
		or not math.isfinite(args.eps) or args.eps <= 0
		or not math.isfinite(args.feasibility_tolerance) or args.feasibility_tolerance <= 0
		or not math.isfinite(args.validation_tolerance) or args.validation_tolerance <= 0):
		parser.error('Expected positive time, thread, worker, and tolerance values; gaps may be zero.')
	campaign = args.campaign if args.campaign.is_absolute() or args.campaign.parent != Path('.') else ROOT / 'benchmarks/campaigns' / args.campaign
	metadata = json.loads((campaign / 'campaign.json').read_text())
	cases = [case for record in metadata['inputs'] for case in read_encoded_cases(campaign / record['file'])][:args.max_instances]
	if not cases:
		parser.error('Campaign has no cases.')
	solvers = list(dict.fromkeys(args.solver or ['unordered']))
	if 'tspn' in solvers and args.max_seconds != int(args.max_seconds):
		parser.error('The external runner requires an integer time limit in seconds.')
	initial_strategies = {
		'sampled_perimeter': args.sampled_perimeter_initial,
		'convex_order_refinement': args.convex_initial_refinement,
		'bidirectional': args.bidirectional_initial,
	}
	external_python = args.external_python.resolve()
	external_build = args.external_build.resolve()
	external_bindings = sorted((external_build / 'python/tspn_bnb2/core').glob('_tspn_bindings*.so'))
	external_binding_sha256 = None
	if external_bindings:
		external_binding_sha256 = hashlib.sha256(external_bindings[0].read_bytes()).hexdigest()
	try:
		external_revision = subprocess.run(['git', '-C', str(EXTERNAL_SOURCE), 'rev-parse', 'HEAD'],
			check=True, capture_output=True, text=True).stdout.strip()
	except subprocess.CalledProcessError:
		external_revision = 'unknown'
	config = {'visit_order': 'free', 'solvers': solvers,
		'threads_per_instance': args.threads_per_instance, 'instance_workers': args.workers,
		'max_calls': args.max_calls, 'max_seconds': args.max_seconds, 'hashes': [c.digest for c in cases],
		'ours_optimality': {'absolute_gap': args.absolute_gap, 'relative_gap': args.relative_gap},
		'external_optimality_eps': args.eps, 'initial_strategies': initial_strategies,
		'external_source_revision': external_revision,
		'external_binding_sha256': external_binding_sha256,
		'external_build_path': str(external_build),
		'perimeter_work_budget': os.environ.get('TPP_APPROX_WORK_BUDGET', '1000000'),
		'perimeter_budget_mode': os.environ.get('TPP_APPROX_BUDGET_MODE', 'fixed'),
		'solver_feasibility_tolerance': args.feasibility_tolerance,
		'independent_validation_tolerance': args.validation_tolerance}
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
	if 'tspn' in solvers and not (external_python.exists() and EXTERNAL_RUNNER.exists()
		and (external_build / 'python/tspn_bnb2/core').exists()):
		raise FileNotFoundError('External TSPN Python or built binding is unavailable; use --external-python and --external-build.')
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
			'notes': [f'Fixed endpoints; free visit order. {args.workers} concurrent instance worker(s); {args.threads_per_instance} solver thread(s) per instance.',
			f'Our target gap is {args.absolute_gap:g} + {args.relative_gap:g} × |UB|. Fekete accepts UB/LB <= 1 + {args.eps:g}; these gap formulas are close but not algebraically identical.',
			'Fekete eps is implemented as a ratio test. Our free-order solver uses an additive absolute-plus-relative gap.',
			f'Both solvers use feasibility tolerance {args.feasibility_tolerance:g} where their APIs permit it; independent validation uses {args.validation_tolerance:g}.',
			'External raw and endpoint-snapped trajectories are reported separately; snapping never changes the declared solver result.',
			'Fekete calls use its per-instance child-evaluation threading; our calls use per-instance sibling-oracle threading. Oracle-call counters are not equivalent units.',
			f"Fekete source revision: {external_revision}; executed binding SHA-256: {external_binding_sha256 or 'unavailable'}.",
			'Only completed rows with matching campaign configuration and input hashes are reused on resume.']}
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
		return {(row.get('solver'), int(row.get('case', -1))) for row in report['rows']
			if not row.get('error') and row.get('status') in (None, 'optimal', 'limit')}

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
						arguments = ['--threads', str(args.threads_per_instance),
							'--absolute-gap', str(args.absolute_gap), '--relative-gap', str(args.relative_gap)]
						if args.sampled_perimeter_initial:
							arguments.append('--sampled-perimeter-initial')
						if args.convex_initial_refinement:
							arguments.append('--convex-initial-refinement')
						if args.bidirectional_initial:
							arguments.append('--bidirectional-initial')
						row.update(run_unordered_solver(BINARY, (sx, sy), (tx, ty), case.polygons,
							args.max_calls, args.max_seconds, arguments=arguments))
						row['visit_order'] = 'free'
						row['length'] = row['upper_bound']
						try:
							row['validation'] = validate_path((sx, sy), (tx, ty), case.polygons,
								row['path'], args.validation_tolerance)
						except ModuleNotFoundError as error:
							if error.name != 'shapely':
								raise
							row['validation'] = {'valid': None, 'reason': 'shapely_unavailable'}
						row['valid'] = row['validation']['valid']
					except (RuntimeError, subprocess.TimeoutExpired) as error:
						row['error'] = str(error)
					return row

				pending = [i for i in range(len(cases)) if (solver, i) not in successful_pairs()]
				with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as executor:
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
				external_root = run / 'external'
				external_root.mkdir(exist_ok=True)
				prior_csvs = sorted(external_root.glob('*/*-tspn-path.csv'),
					key=lambda path: path.stat().st_mtime_ns, reverse=True)
				resume_csv = prior_csvs[0] if prior_csvs else None
				completed_external = set()
				if resume_csv:
					with resume_csv.open(newline='') as file:
						completed_external = {int(row['case_index']) for row in csv.DictReader(file)
							if row.get('status') in {'optimal', 'limit'} and not row.get('error')}
				pending_external = len(cases) - len(completed_external)
				external_command = [str(external_python), str(EXTERNAL_RUNNER), '--suite', str(suite), '--mode', 'path',
					'--tspn-repo', str(external_build),
					'--threads', str(args.threads_per_instance), '--workers', str(args.workers),
					'--time-limit', str(int(args.max_seconds)), '--eps', str(args.eps),
					'--feasibility-tolerance', str(args.feasibility_tolerance),
					'--validation-tolerance', str(args.validation_tolerance)]
				if resume_csv:
					external_command.extend(['--resume', str(resume_csv)])
				else:
					external_command.extend(['--output', str(external_root)])
				run_external(external_command,
					max(120.0, pending_external * (args.max_seconds + 125.0) + 60.0))
				if resume_csv:
					csv_path = resume_csv
				else:
					csv_path = max(external_root.glob('*/*-tspn-path.csv'), key=lambda path: path.stat().st_mtime_ns)
				with csv_path.open() as file:
					for external in csv.DictReader(file):
						i = int(external['case_index'])
						if external['sha256'] != cases[i].digest:
							raise ValueError('External instance hash mismatch.')
						report['rows'] = [item for item in report['rows'] if not (item.get('solver') == solver and item.get('case') == i)]
						report['rows'].append({'case': i, 'sha256': cases[i].digest, 'geometry_sha256': cases[i].digest, 'solver': solver,
							'status': external['status'],
							'polygons': len(cases[i].polygons), 'upper_bound': finite(external['upper_bound']),
							'lower_bound': finite(external['lower_bound']), 'seconds': finite(external['solve_seconds']),
							'threads_per_instance': int(external['threads']),
							'calls': finite(external['soc_num_calls']), 'exact': external['is_optimal'] == 'True',
							'path': json.loads(external['trajectory_json']) if external.get('trajectory_json') else None,
							'endpoint_valid': external['is_valid_trajectory'] == 'True',
							'valid': optional_bool(external.get('raw_valid')),
							'endpoint_repaired_valid': optional_bool(external.get('snapped_valid')),
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
		write_comparison_summary(run / 'comparison.md', report, len(cases))
	print(f'Report: {run / "report.json"}', flush=True)
	print(f'Comparison: {run / "comparison.md"}', flush=True)
	return int(report['status'] != 'completed')


if __name__ == '__main__':
	raise SystemExit(main())
