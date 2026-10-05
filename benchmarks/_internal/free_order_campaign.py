"""Campaign runner for endpoint TPP with free visit order."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import signal
import statistics
import struct
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import native_build
import workspace
from benchmark_cases import read_encoded_cases
from live_progress import LiveStatus
from unordered_runner import interrupt_running_solvers, run_unordered_solver, terminate_running_solvers
from unordered_validation import validate_path

ROOT = Path(__file__).resolve().parents[2]
BINARY = native_build.tool_path('tpp-unordered')
EXTERNAL_PYTHON = ROOT / 'third_party/tspn-socg/.venv/bin/python'
EXTERNAL_RUNNER = ROOT / 'benchmarks/_internal/tspn_run_comparison.py'
EXTERNAL_SOURCE = ROOT / 'third_party/tspn-socg'
DEFAULT_EXTERNAL_EPS = 1e-3
DEFAULT_OUR_RELATIVE_GAP = DEFAULT_EXTERNAL_EPS / (1 + DEFAULT_EXTERNAL_EPS)
SOLVER_DISPLAY_NAMES = {'unordered': 'tpp-ours', 'tspn': 'tpp-fekete'}
SHUTDOWN_BEFORE_START = 'shutdown requested before solver start'
SOLVER_ALIASES = {
	'tpp-ours': 'unordered', 'tpp-fekete': 'tspn',
	'unordered': 'unordered', 'tspn': 'tspn',
}


def parse_solver_name(value: str) -> str:
	try:
		return SOLVER_ALIASES[value]
	except KeyError as error:
		raise argparse.ArgumentTypeError('choose tpp-ours or tpp-fekete') from error
def ensure_binary(no_build: bool = False) -> Path:
	return native_build.ensure_tool('tpp-unordered', no_build=no_build)


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


def optional_bool(value: str | bool | None) -> bool | None:
	if value is True or value == 'True':
		return True
	if value is False or value == 'False':
		return False
	return None


def _resume_compatible_config(previous: dict, current: dict) -> bool:
	"""Reuse matching per-solver rows when workers or solver selection changes."""
	ignored = {
		'campaign_runner_sha256', 'external_runner_sha256', 'queue_policy',
		'instance_workers', 'solvers',
	}
	current_solvers = set(current.get('solvers', []))
	if 'unordered' not in current_solvers:
		ignored.update({
			'max_calls', 'ours_optimality', 'unordered_binary_sha256',
			'initial_strategies', 'perimeter_work_budget', 'perimeter_budget_mode',
		})
	if 'tspn' not in current_solvers:
		ignored.update({
			'external_optimality_eps', 'external_source_revision',
			'external_binding_sha256', 'external_build_path',
		})
	previous_identity = {key: value for key, value in previous.items() if key not in ignored}
	current_identity = {key: value for key, value in current.items() if key not in ignored}
	return previous_identity == current_identity


def _find_compatible_report(results: Path, key: str, config: dict) -> tuple[Path, dict] | None:
	"""Return the newest report that can be resumed with the requested configuration."""
	for prior in sorted(results.glob('*/report.json'),
		key=lambda path: path.stat().st_mtime_ns, reverse=True):
		try:
			old = json.loads(prior.read_text())
		except (OSError, json.JSONDecodeError):
			continue
		if old.get('key') == key or _resume_compatible_config(old.get('config', {}), config):
			return prior, old
	return None


def _sort_report_rows(rows: list[dict]) -> None:
	"""Sort mixed-solver reports independently of the solver selection for this run."""
	solver_order = {'unordered': 0, 'tspn': 1}
	rows.sort(key=lambda row: (
		int(row.get('case', -1)), solver_order.get(row.get('solver'), len(solver_order)),
		row.get('solver', ''),
	))


def _job_was_not_started(result: dict) -> bool:
	"""Identify shutdown sentinels that must not become checkpoint rows."""
	if result.get('not_started'):
		return True
	return result.get('row', {}).get('error') == SHUTDOWN_BEFORE_START


def _pending_solver_jobs(case_count: int, solvers: list[str], completed: set[tuple[str, int]]) -> list[tuple[str, int]]:
	"""Order pending work by solver, then by case index, for the shared FIFO pool."""
	return [
		(solver, index)
		for solver in ('unordered', 'tspn') if solver in solvers
		for index in range(case_count) if (solver, index) not in completed
	]


def _read_complete_csv_rows(path: Path) -> list[dict[str, str]]:
	"""Read only newline-terminated CSV records while another process may append."""
	if not path.exists():
		return []
	data = path.read_bytes()
	last_newline = data.rfind(b'\n')
	if last_newline < 0:
		return []
	return list(csv.DictReader(io.StringIO(data[:last_newline + 1].decode('utf-8'))))


def _tspn_report_row(external: dict[str, str], cases: list, solver: str = 'tspn') -> dict:
	index = int(external['case_index'])
	if not 0 <= index < len(cases):
		raise ValueError(f'External case index outside suite: {index}')
	if external['sha256'] != cases[index].digest:
		raise ValueError(f'External instance hash mismatch for case {index}.')
	row = {
		'case': index, 'sha256': cases[index].digest, 'geometry_sha256': cases[index].digest,
		'solver': solver, 'status': external['status'], 'polygons': len(cases[index].polygons),
		'upper_bound': finite(external.get('upper_bound')), 'lower_bound': finite(external.get('lower_bound')),
		'seconds': finite(external.get('solve_seconds')),
		'threads_per_instance': int(external.get('threads') or 0),
		'calls': finite(external.get('soc_num_calls')), 'exact': optional_bool(external.get('is_optimal')) is True,
		'path': json.loads(external['trajectory_json']) if external.get('trajectory_json') else None,
		'endpoint_valid': optional_bool(external.get('is_valid_trajectory')) is True,
		'valid': optional_bool(external.get('raw_valid')),
		'endpoint_repaired_valid': optional_bool(external.get('snapped_valid')),
		'validation': {'start_distance': finite(external.get('start_distance')),
			'target_distance': finite(external.get('target_distance')),
			'max_polygon_distance': finite(external.get('max_polygon_distance')),
			'recomputed_length': finite(external.get('recomputed_length'))},
		'termination': external.get('status'), 'error': external.get('error') or None,
	}
	# Earlier workers could lose the virtual environment and export trajectories
	# without Shapely validation. Recheck those stored paths outside solver timing.
	if row['valid'] is None and row['path']:
		geometry = case_geometry(cases[index])
		tolerance = float(external['validation_tolerance'])
		try:
			validation = validate_path(geometry['start'], geometry['target'],
				cases[index].polygons, row['path'], tolerance)
			snapped = json.loads(external['snapped_trajectory_json']) if external.get('snapped_trajectory_json') else None
			snapped_validation = validate_path(geometry['start'], geometry['target'],
				cases[index].polygons, snapped, tolerance) if snapped else None
		except ModuleNotFoundError as error:
			if error.name != 'shapely':
				raise
		else:
			row['valid'] = validation['valid']
			row['validation'] = {**validation, 'source': 'stored_trajectory_revalidation',
				'tolerance': tolerance}
			if snapped_validation is not None:
				row['endpoint_repaired_valid'] = snapped_validation['valid']
	return row


def write_comparison_summary(path: Path, report: dict, expected_cases: int) -> None:
	rows = report.get('rows', [])
	by_solver = {
		name: {int(row['case']): row for row in rows if row.get('solver') == name and not row.get('error')}
		for name in ('unordered', 'tspn')
	}
	config = report.get('config', {})
	selected_solvers = config.get('solvers', ['unordered', 'tspn'])
	time_limit = config.get('max_seconds', 'unknown')
	time_limit_text = 'unlimited' if time_limit == -1 else f'{time_limit} s'
	ours_tolerance = config.get('ours_optimality', {})
	eps = float(config.get('external_optimality_eps', -1))
	relative_gap = float(ours_tolerance.get('relative_gap', math.nan))
	tolerance_matched = float(ours_tolerance.get('absolute_gap', math.nan)) == 0 and eps > 0 and math.isclose(
		relative_gap, eps / (1 + eps), rel_tol=1e-12, abs_tol=1e-15,
	)
	lines = [
		'# Free-order TPP campaign results', '',
		f"- Instances in suite: {expected_cases}",
		f"- Instance workers: {config.get('instance_workers', 'unknown')} (instances processed concurrently)",
		f"- Threads per instance: {config.get('threads_per_instance', 'unknown')}",
		f"- Pending-job order: {config.get('queue_policy', 'solver-major FIFO shared worker pool')}",
		f'- Per-instance time cap: {time_limit_text}',
		f"- Solvers selected: {', '.join(SOLVER_DISPLAY_NAMES[solver] for solver in selected_solvers)}",
		f"- Initial strategies: {', '.join(name for name, enabled in config.get('initial_strategies', {}).items() if enabled) or 'none'}",
	]
	if 'unordered' in selected_solvers:
		lines.append(f"- tpp-ours gap: absolute {ours_tolerance.get('absolute_gap', 'unknown')} + relative {ours_tolerance.get('relative_gap', 'unknown')} × |UB|")
	if 'tspn' in selected_solvers:
		lines.extend([
			f"- tpp-fekete tolerance: UB/LB ≤ 1 + {config.get('external_optimality_eps', 'unknown')}",
			f"- tpp-fekete source revision: `{config.get('external_source_revision', 'unknown')}`; binding SHA-256: `{config.get('external_binding_sha256', 'unknown')}`",
		])
	if 'unordered' in selected_solvers and 'tspn' in selected_solvers:
		tolerance_note = (
			'With zero absolute gap, the tpp-ours relative-gap threshold is algebraically equivalent to the tpp-fekete UB/LB ratio test.'
			if tolerance_matched else
			'The two solvers use different stopping thresholds; see the recorded gap parameters.'
		)
		lines.extend([
			tolerance_note,
			'The runtime ratios compare the recorded solver configurations; they do not control for system load or isolate threading effects.',
		])
	lines.extend([
		'',
		'| Solver | Recorded cases | Errors | Closed requested gap | Independent valid paths | Median solve time | Total solve time |',
		'|---|---:|---:|---:|---:|---:|---:|',
	])
	for solver, label in (('unordered', 'tpp-ours'), ('tspn', 'tpp-fekete')):
		if solver not in selected_solvers:
			continue
		group = list(by_solver[solver].values())
		recorded = sum(row.get('solver') == solver for row in rows)
		errors = sum(row.get('solver') == solver and bool(row.get('error')) for row in rows)
		times = [float(row['seconds']) for row in group if row.get('seconds') is not None]
		closed = sum(row.get('exact') is True for row in group)
		valid = sum(row.get('valid') is True for row in group)
		valid_known = sum(row.get('valid') is not None for row in group)
		lines.append(f"| {label} | {recorded} | {errors} | {closed}/{recorded} | {valid}/{valid_known} known | "
			f"{statistics.median(times):.3f}s | {sum(times):.3f}s |" if times else
			f"| {label} | {recorded} | {errors} | {closed}/{recorded} | {valid}/{valid_known} known | n/a | n/a |")
	paired = []
	ratios = []
	if 'unordered' in selected_solvers and 'tspn' in selected_solvers:
		paired = [(by_solver['unordered'][i], by_solver['tspn'][i]) for i in sorted(set(by_solver['unordered']) & set(by_solver['tspn']))]
		ratios = [float(ours['seconds']) / float(fekete['seconds']) for ours, fekete in paired
			if ours.get('exact') is True and fekete.get('exact') is True
			and ours.get('valid') is True and fekete.get('valid') is True
			and ours.get('seconds') and fekete.get('seconds') and float(ours['seconds']) > 0 and float(fekete['seconds']) > 0]
	if ratios:
		lines.extend([
			'', f"Paired runtime data with both gaps closed and independently valid raw paths: {len(ratios)} instances.",
			f"Median tpp-ours/tpp-fekete runtime ratio: {statistics.median(ratios):.3f}× (below 1 means tpp-ours was faster).",
			f"tpp-ours faster: {sum(ratio < 1 for ratio in ratios)}; tpp-fekete faster: {sum(ratio > 1 for ratio in ratios)}; equal: {sum(ratio == 1 for ratio in ratios)}.",
		])
	if 'unordered' in selected_solvers and 'tspn' in selected_solvers:
		lines.extend(['', 'Runtime ratios exclude errors, unfinished gaps, invalid paths, and paths without independent validation. '
			'A subset of paired cases does not establish the overall corpus speedup.'])
	if any(row.get('valid') is None for solver in selected_solvers for row in by_solver[solver].values()):
		lines.extend(['', 'Independent geometric validation is missing for some rows. Those rows are marked unknown, not valid.'])
	reference_path = ROOT / 'benchmarks/results-saved/fekete-comparison/fekete.csv'
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
			'', f"tpp-fekete thread-count control against the saved 1-thread run: {len(thread_pairs)} paired cases, same hashes, eps, and per-instance cap.",
					f"Median 1-thread/{thread_count}-thread runtime ratio: {statistics.median(speedups):.3f}× (above 1 means the multithreaded run was faster).",
					f"Multithreaded run faster: {sum(value > 1 for value in speedups)}; 1 thread faster: {sum(value < 1 for value in speedups)}; equal: {sum(value == 1 for value in speedups)}.",
					'This is a historical paired comparison; machine load and software environment may differ between campaigns.',
				])
	our_rows = list(by_solver['unordered'].values()) if 'unordered' in selected_solvers else []
	parallel_cases = sum(int(row.get('parallel_oracle_calls', 0) or 0) > 0 for row in our_rows)
	parallel_calls = sum(int(row.get('parallel_oracle_calls', 0) or 0) for row in our_rows)
	parallel_batches = sum(int(row.get('parallel_oracle_batches', 0) or 0) for row in our_rows)
	if our_rows:
		lines.extend(['', f"tpp-ours launched parallel oracle batches on {parallel_cases}/{len(our_rows)} completed instances "
			f"({parallel_calls} calls in {parallel_batches} batches)."])
	lines.extend(['', f"Campaign status: {report.get('status', 'unknown')}.", ''])
	attempts = report.get('attempts', [])
	if attempts:
		latest = attempts[-1]
		finished = latest.get('finished_at') or 'still running'
		elapsed = latest.get('elapsed_wall_seconds')
		elapsed_text = f'{elapsed:.1f}s wall time' if isinstance(elapsed, (int, float)) else 'elapsed time unavailable'
		lines.extend([f"Latest attempt: {latest.get('started_at', 'unknown')} to {finished} ({elapsed_text}; {latest.get('status', 'unknown')}).", ''])
	path.write_text('\n'.join(lines))


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('campaign', type=Path)
	parser.add_argument('--solver', type=parse_solver_name, action='append',
		metavar='{tpp-ours,tpp-fekete}',
		help='Select tpp-ours and/or tpp-fekete; may be repeated (default: tpp-ours).')
	parser.add_argument('--max-instances', type=int, default=5000)
	parser.add_argument('--max-calls', type=int, default=1000000)
	parser.add_argument('--max-seconds', type=float, default=30,
		help='Maximum seconds per instance; -1 means unlimited.')
	parser.add_argument('--threads-per-instance', type=int, default=1,
		help='Solver threads used inside one instance.')
	parser.add_argument('--workers', type=int, default=1,
		help='Concurrent cases across the selected solver queue(s).')
	parser.add_argument('--absolute-gap', type=float, default=0.0)
	parser.add_argument('--relative-gap', type=float, default=DEFAULT_OUR_RELATIVE_GAP)
	parser.add_argument('--eps', type=float, default=DEFAULT_EXTERNAL_EPS,
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
	parser.add_argument('--progress-interval', type=float, default=60.0,
		help='Seconds between status lines (bounds, calls, queue) of each running tpp-ours instance, also kept in '
			'results/free-order/RUN/live.json for `tpp.py live`; 0 disables.')
	parser.add_argument('--no-build', action='store_true')
	parser.add_argument('--force', action='store_true')
	parser.add_argument('--dry-run', action='store_true')
	args = parser.parse_args(argv)
	if ((args.max_seconds != -1 and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0)) or args.max_calls < 0
		or args.max_instances < 1 or args.threads_per_instance < 1 or args.workers < 1
		or not math.isfinite(args.progress_interval) or args.progress_interval < 0
		or not math.isfinite(args.absolute_gap) or args.absolute_gap < 0
		or not math.isfinite(args.relative_gap) or args.relative_gap < 0
		or not math.isfinite(args.eps) or args.eps <= 0
		or not math.isfinite(args.feasibility_tolerance) or args.feasibility_tolerance <= 0
		or not math.isfinite(args.validation_tolerance) or args.validation_tolerance <= 0):
		parser.error('Expected positive time (or -1 for unlimited), thread, worker, and tolerance values; gaps may be zero.')
	campaign = workspace.campaign_path(args.campaign)
	metadata = json.loads((campaign / 'campaign.json').read_text())
	cases = [case for record in metadata['inputs'] for case in read_encoded_cases(campaign / record['file'])][:args.max_instances]
	if not cases:
		parser.error('Campaign has no cases.')
	requested_solvers = set(args.solver or ['unordered'])
	solvers = [solver for solver in ('unordered', 'tspn') if solver in requested_solvers]
	if 'tspn' in solvers and args.max_seconds != int(args.max_seconds):
		parser.error('The external runner requires an integer time limit in seconds.')
	initial_strategies = {
		'sampled_perimeter': args.sampled_perimeter_initial,
		'convex_order_refinement': args.convex_initial_refinement,
		'bidirectional': args.bidirectional_initial,
	}
	# Preserve the .venv entry point: resolve() bypasses its site-packages and
	# freezes a versioned Homebrew target that an upgrade can remove mid-campaign.
	external_python = args.external_python.absolute()
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
		'queue_policy': 'solver-major FIFO shared worker pool: ' + ' then '.join(
			SOLVER_DISPLAY_NAMES[solver] for solver in solvers),
		'max_calls': args.max_calls, 'max_seconds': args.max_seconds, 'hashes': [c.digest for c in cases],
		'ours_optimality': {'absolute_gap': args.absolute_gap, 'relative_gap': args.relative_gap},
		'unordered_binary_sha256': None,
		'campaign_runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
		'external_runner_sha256': hashlib.sha256(EXTERNAL_RUNNER.read_bytes()).hexdigest(),
		'external_optimality_eps': args.eps, 'initial_strategies': initial_strategies,
		'external_source_revision': external_revision,
		'external_binding_sha256': external_binding_sha256,
		'external_build_path': str(external_build),
		'perimeter_work_budget': os.environ.get('TPP_APPROX_WORK_BUDGET', '1000000'),
		'perimeter_budget_mode': os.environ.get('TPP_APPROX_BUDGET_MODE', 'fixed'),
		'solver_feasibility_tolerance': args.feasibility_tolerance,
		'independent_validation_tolerance': args.validation_tolerance}
	if args.dry_run:
		print(json.dumps(config, indent=2))
		return 0
	if 'unordered' in solvers:
		ensure_binary(args.no_build)
		config['unordered_binary_sha256'] = hashlib.sha256(BINARY.read_bytes()).hexdigest()
	key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
	results = campaign / 'results/free-order'
	resume_path = None
	if not args.force and results.exists():
		match = _find_compatible_report(results, key, config)
		if match is not None:
			prior, old = match
			old_rows = old.get('rows', [])
			old_completed = {
				(row.get('solver'), int(row.get('case', -1))) for row in old_rows
				if row.get('solver') in solvers and not row.get('error')
				and row.get('status') in (None, 'optimal', 'limit')
			}
			all_requested_cases_complete = all(
				(solver, index) in old_completed
				for solver in solvers for index in range(len(cases))
			)
			if old.get('status') == 'completed' and all_requested_cases_complete:
				prior.touch()
				print(f'Reusing {prior}', flush=True)
				write_comparison_summary(prior.with_name('comparison.md'), old, len(cases))
				return 0
			if old.get('key') != key or old.get('config') != config:
				previous_config = old.get('config', {})
				old.setdefault('configuration_history', []).append({
					'migrated_at': datetime.now(UTC).isoformat(),
					'previous_instance_workers': previous_config.get('instance_workers'),
					'instance_workers': config.get('instance_workers'),
					'previous_campaign_runner_sha256': previous_config.get('campaign_runner_sha256'),
					'previous_external_runner_sha256': previous_config.get('external_runner_sha256'),
				})
				old['key'] = key
				old['config'] = config
				old.setdefault('notes', []).append(
					'Resumed with the shared FIFO solver-case queue; prior completed rows were kept.')
				atomic_json(prior, old)
			resume_path = prior
	if 'tspn' in solvers and not (external_python.exists() and EXTERNAL_RUNNER.exists()
		and (external_build / 'python/tspn_bnb2/core').exists()):
		raise FileNotFoundError('tpp-fekete Python or built binding is unavailable; use --external-python and --external-build.')
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
		notes = [f'Fixed endpoints; free visit order. {args.workers} shared worker(s); {args.threads_per_instance} solver thread(s) per case.',
			f"Pending cases are queued for {' then '.join(SOLVER_DISPLAY_NAMES[solver] for solver in solvers)} in one FIFO worker pool."]
		if 'unordered' in solvers:
			notes.append(f'tpp-ours target gap is {args.absolute_gap:g} + {args.relative_gap:g} × |UB|.')
		if 'tspn' in solvers:
			notes.append(f'tpp-fekete accepts UB/LB <= 1 + {args.eps:g}.')
		if 'unordered' in solvers and 'tspn' in solvers:
			notes.append(
				('With zero absolute gap, the configured relative gap is algebraically equivalent to the tpp-fekete UB/LB ratio test.'
				if args.absolute_gap == 0 and math.isclose(args.relative_gap, args.eps / (1 + args.eps), rel_tol=1e-12, abs_tol=1e-15)
				else 'The two solvers use different stopping thresholds; both configured criteria are recorded explicitly.')
			)
		notes.append(
			f'Selected solver(s) use feasibility tolerance {args.feasibility_tolerance:g} where their APIs permit it; independent validation uses {args.validation_tolerance:g}.'
		)
		if 'tspn' in solvers:
			notes.extend([
				'External raw and endpoint-snapped trajectories are reported separately; snapping never changes the declared solver result.',
				'tpp-fekete uses per-instance child-evaluation threads; tpp-ours uses per-instance sibling-oracle threads. Oracle-call counters are not equivalent units.',
			f"tpp-fekete source revision: {external_revision}; executed binding SHA-256: {external_binding_sha256 or 'unavailable'}.",
			])
		notes.append('Only completed rows with matching campaign configuration and input hashes are reused on resume.')
		report = {'schema_version': 2, 'key': key, 'config': config, 'visit_order': 'free', 'title': metadata.get('name', campaign.name),
		'created_at': datetime.now(UTC).isoformat(), 'status': 'running', 'rows': [],
		'notes': notes}
	report.setdefault('attempts', [])
	attempt = {'started_at': datetime.now(UTC).isoformat(), 'finished_at': None,
		'elapsed_wall_seconds': None, 'status': 'running',
		'instance_workers': args.workers, 'threads_per_instance': args.threads_per_instance,
		'machine': workspace.machine_state(), 'git': workspace.git_state(), 'origin': workspace.origin()}
	report['attempts'].append(attempt)
	# Geometry is not stored per run: rows reference an instance by its hash and the
	# campaign's input .bin is the single copy (the dashboard resolves hashes there).
	geometry_catalog = {case.digest: case_geometry(case) for case in cases}
	for row in report['rows']:
		if row.get('geometry'):
			row['geometry_sha256'] = row.get('sha256')
			row.pop('geometry', None)

	def successful_pairs() -> set[tuple[str, int]]:
		return {(row.get('solver'), int(row.get('case', -1))) for row in report['rows']
			if row.get('solver') in solvers and not row.get('error')
			and row.get('status') in (None, 'optimal', 'limit')}

	def save_checkpoint() -> None:
		report['checkpoint'] = {
			'completed_pairs': len(successful_pairs()),
			'total_pairs': len(cases) * len(solvers),
			'updated_at': datetime.now(UTC).isoformat(),
		}
		atomic_json(run / 'report.json', report)

	save_checkpoint()
	interrupted = False
	external_runner = None
	external_cases = []
	external_args = None
	external_csv_path = None
	external_log_file = None
	external_log_lock = threading.Lock()
	shutdown_requested = threading.Event()
	if 'tspn' in solvers:
		import tspn_run_comparison as external_runner
		suite = run / 'input.bin'
		suite.write_bytes(b''.join(case.data for case in cases))
		external_cases = external_runner.read_cases(suite)
		external_root = run / 'external'
		external_root.mkdir(exist_ok=True)
		prior_csvs = sorted(external_root.glob('*/*-tspn-path.csv'),
			key=lambda path: path.stat().st_mtime_ns, reverse=True)
		if prior_csvs:
			external_csv_path = prior_csvs[0]
		else:
			external_output = external_root / datetime.now(UTC).strftime('%Y%m%d-%H%M%S')
			external_output.mkdir(parents=True, exist_ok=True)
			external_csv_path = external_output / f'{suite.stem}-tspn-path.csv'
			with external_csv_path.open('w', newline='') as file:
				csv.DictWriter(file, fieldnames=external_runner.RESULT_FIELDS).writeheader()
		external_log_file = (external_csv_path.parent / 'solver.log').open('a')
		external_args = argparse.Namespace(
			suite=suite, tspn_repo=external_build, mode='path', time_limit=int(args.max_seconds),
			threads=args.threads_per_instance, eps=args.eps,
			feasibility_tolerance=args.feasibility_tolerance,
			validation_tolerance=args.validation_tolerance, oracle_backend='socp',
			oracle_tolerance=1e-7, worker_python=external_python,
		)
		for external in _read_complete_csv_rows(external_csv_path):
			row = _tspn_report_row(external, cases)
			report['rows'] = [item for item in report['rows']
				if not (item.get('solver') == 'tspn' and item.get('case') == row['case'])]
			report['rows'].append(row)
			if prior_csvs:
				_sort_report_rows(report['rows'])
				save_checkpoint()

	def upsert_report_row(row: dict) -> None:
		report['rows'] = [item for item in report['rows']
			if not (item.get('solver') == row.get('solver') and item.get('case') == row.get('case'))]
		report['rows'].append(row)
		_sort_report_rows(report['rows'])

	def save_external_csv_row(row: dict) -> None:
		if external_csv_path is None or external_runner is None:
			return
		with external_csv_path.open('a', newline='') as file:
			csv.DictWriter(file, fieldnames=external_runner.RESULT_FIELDS).writerow(row)
			file.flush()

	def compact_external_csv() -> None:
		if external_csv_path is None or external_runner is None:
			return
		latest: dict[int, dict[str, str]] = {}
		for row in _read_complete_csv_rows(external_csv_path):
			if row.get('case_index', '').isdigit():
				latest[int(row['case_index'])] = row
		temporary = external_csv_path.with_suffix(external_csv_path.suffix + '.tmp')
		with temporary.open('w', newline='') as file:
			writer = csv.DictWriter(file, fieldnames=external_runner.RESULT_FIELDS)
			writer.writeheader()
			for index in sorted(latest):
				writer.writerow(latest[index])
		temporary.replace(external_csv_path)
		external_runner.write_summary(external_csv_path, external_csv_path.with_suffix('.md'), external_args)

	live = LiveStatus(run / 'live.json' if args.progress_interval > 0 else None)

	def solve_unordered_case(index: int) -> dict:
		case = cases[index]
		geometry = geometry_catalog[case.digest]
		sx, sy = geometry['start']
		tx, ty = geometry['target']
		row = {'case': index, 'sha256': case.digest, 'solver': 'unordered',
			'polygons': len(case.polygons), 'geometry_sha256': case.digest}
		try:
			arguments = ['--threads', str(args.threads_per_instance),
				'--absolute-gap', str(args.absolute_gap), '--relative-gap', str(args.relative_gap)]
			if args.sampled_perimeter_initial:
				arguments.append('--sampled-perimeter-initial')
			if args.convex_initial_refinement:
				arguments.append('--convex-initial-refinement')
			if args.bidirectional_initial:
				arguments.append('--bidirectional-initial')
			solver_time_limit = math.inf if args.max_seconds == -1 else args.max_seconds
			key = f'unordered-{index}'
			report = live.reporter(key, f'free case {index + 1}/{len(cases)} tpp-ours', max_seconds=solver_time_limit,
				max_calls=args.max_calls, target_gap=args.relative_gap)
			try:
				row.update(run_unordered_solver(BINARY, (sx, sy), (tx, ty), case.polygons,
					args.max_calls, solver_time_limit, arguments=arguments,
					progress=report, progress_interval=args.progress_interval))
			finally:
				live.finish(key)
			if row.get('error') == SHUTDOWN_BEFORE_START:
				return None
			if row.get('termination') == 'interrupted':
				row['status'] = 'interrupted'
			row['visit_order'] = 'free'
			row['length'] = row.get('upper_bound')
			if not row.get('path'):
				row['validation'] = {'valid': None, 'reason': 'no_incumbent_path_before_shutdown'}
				row['valid'] = None
				return row
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

	def solve_tspn_case(index: int) -> dict:
		payload = external_runner.run_case(external_args, index, external_log_file, external_log_lock)
		if payload.get('error') == SHUTDOWN_BEFORE_START:
			return {'solver': 'tspn', 'case': index, 'not_started': True}
		external_row = external_runner.result_row(external_args, index, external_cases[index], {}, payload)
		return {'solver': 'tspn', 'case': index, 'row': _tspn_report_row(external_row, cases),
			'external_row': external_row}

	def dispatch_job(job: tuple[str, int]) -> dict:
		solver, index = job
		if shutdown_requested.is_set():
			return {'solver': solver, 'case': index, 'not_started': True}
		try:
			if solver == 'unordered':
				row = solve_unordered_case(index)
				if row is None:
					return {'solver': solver, 'case': index, 'not_started': True}
				return {'solver': solver, 'case': index, 'row': row}
			return solve_tspn_case(index)
		except Exception as error:
			if solver == 'unordered':
				return {'solver': solver, 'case': index, 'row': {
					'case': index, 'sha256': cases[index].digest, 'solver': solver,
					'polygons': len(cases[index].polygons), 'status': 'error', 'error': str(error)}}
			payload = {'status': 'error', 'error': str(error), 'solve_seconds': 0.0,
				'is_optimal': False, 'is_valid_trajectory': False}
			external_row = external_runner.result_row(external_args, index, external_cases[index], {}, payload)
		return {'solver': solver, 'case': index,
			'row': _tspn_report_row(external_row, cases), 'external_row': external_row}

	def dispatch_job_interrupted(job: tuple[str, int], error: Exception) -> dict:
		solver, index = job
		message = str(error) or 'solver stopped during shutdown'
		if solver == 'unordered':
			return {'solver': solver, 'case': index, 'row': {
				'case': index, 'sha256': cases[index].digest, 'solver': solver,
				'polygons': len(cases[index].polygons), 'status': 'interrupted',
				'termination': 'interrupted', 'error': message}}
		payload = {'status': 'interrupted', 'error': message, 'solve_seconds': 0.0,
			'is_optimal': False, 'is_valid_trajectory': False}
		external_row = external_runner.result_row(external_args, index, external_cases[index], {}, payload)
		return {'solver': solver, 'case': index,
			'row': _tspn_report_row(external_row, cases), 'external_row': external_row}

	def commit_job_result(result: dict) -> None:
		if result['solver'] == 'tspn':
			save_external_csv_row(result['external_row'])
		upsert_report_row(result['row'])
		save_checkpoint()

	jobs = _pending_solver_jobs(len(cases), solvers, successful_pairs())
	unordered_pending = sum(solver == 'unordered' for solver, _ in jobs)
	tspn_pending = sum(solver == 'tspn' for solver, _ in jobs)
	print(f'Queue: tpp-ours={unordered_pending}, tpp-fekete={tspn_pending}, workers={args.workers}, '
		f'threads/job={args.threads_per_instance}', flush=True)
	for solver in solvers:
		label = SOLVER_DISPLAY_NAMES[solver]
		print(f'## {label}: {sum(item[0] == solver for item in jobs)} pending job(s)', flush=True)

	try:
		if jobs:
			executor = ThreadPoolExecutor(max_workers=min(args.workers, len(jobs)))
			futures = {}
			written_jobs: set[tuple[str, int]] = set()
			for job in jobs:
				futures[executor.submit(dispatch_job, job)] = job
			try:
				for completed, future in enumerate(as_completed(futures), 1):
					result = future.result()
					commit_job_result(result)
					job = (result['solver'], result['case'])
					written_jobs.add(job)
					job_status = result['row'].get('status') or result['row'].get('termination') or 'finished'
					label = SOLVER_DISPLAY_NAMES[job[0]]
					print(f'jobs | [free] {completed}/{len(jobs)} | {label} case {job[1] + 1} finished ({job_status})', flush=True)
			except KeyboardInterrupt:
				previous_sigint_handler = signal.getsignal(signal.SIGINT)
				forced_shutdown = False

				def force_stop_on_second_interrupt(signum, frame) -> None:
					nonlocal forced_shutdown
					if forced_shutdown:
						return
					forced_shutdown = True
					print('Second Ctrl+C: force-stopping active solver processes; saving completed checkpoints.',
						file=sys.stderr, flush=True)
					terminate_running_solvers()
					if external_runner is not None:
						external_runner.stop_active_processes()

				signal.signal(signal.SIGINT, force_stop_on_second_interrupt)
				print('Shutting down: asking active solvers to save incumbent paths and bounds...',
					file=sys.stderr, flush=True)
				shutdown_requested.set()
				interrupt_running_solvers()
				if external_runner is not None:
					external_runner.stop_active_processes()
				for future in futures:
					future.cancel()
				print('Waiting for active solver calls to finish. Press Ctrl+C again to force-stop them.',
					file=sys.stderr, flush=True)
				try:
					executor.shutdown(wait=True, cancel_futures=True)
					for future, job in futures.items():
						if future.cancelled() or job in written_jobs:
							continue
						try:
							result = future.result()
						except Exception as error:
							result = dispatch_job_interrupted(job, error)
						if _job_was_not_started(result):
							continue
						commit_job_result(result)
						written_jobs.add(job)
						row = result['row']
						print(f'partial | [free] {SOLVER_DISPLAY_NAMES[job[0]]} case {job[1] + 1}: '
							f'{row.get("termination", row.get("status", "interrupted"))}; '
							f'UB={row.get("upper_bound", "n/a")} LB={row.get("lower_bound", "n/a")}', flush=True)
				finally:
					signal.signal(signal.SIGINT, previous_sigint_handler)
				interrupted = True
			else:
				executor.shutdown(wait=True)
		if not interrupted:
			complete = len(successful_pairs()) == len(cases) * len(solvers)
			report['status'] = 'completed' if complete else 'failed'
		else:
			report['status'] = 'interrupted'
	except KeyboardInterrupt:
		print('Shutting down: recording completed and partial case results...', file=sys.stderr, flush=True)
		interrupt_running_solvers()
		if external_runner is not None:
			external_runner.stop_active_processes()
		report['status'] = 'interrupted'
	except Exception as error:
		report['status'] = 'failed'
		report['error'] = str(error)
		raise
	finally:
		if external_csv_path is not None:
			try:
				compact_external_csv()
			except Exception as error:
				report.setdefault('notes', []).append(f'Could not compact the Fekete CSV checkpoint: {error}')
		if external_log_file is not None:
			external_log_file.close()
		finished_at = datetime.now(UTC)
		attempt['finished_at'] = finished_at.isoformat()
		attempt['elapsed_wall_seconds'] = (finished_at - datetime.fromisoformat(attempt['started_at'])).total_seconds()
		attempt['status'] = report['status'] if report['status'] != 'running' else 'interrupted'
		save_checkpoint()
		write_comparison_summary(run / 'comparison.md', report, len(cases))
	print(f'Report: {run / "report.json"}', flush=True)
	print(f'Comparison: {run / "comparison.md"}', flush=True)
	return 130 if report['status'] == 'interrupted' else int(report['status'] != 'completed')


if __name__ == '__main__':
	raise SystemExit(main())
