import contextlib
import csv
import io
import json
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


INTERNAL = Path(__file__).resolve().parents[1] / '_internal'
sys.path.insert(0, str(INTERNAL))

from free_order_campaign import (
	_find_compatible_report, _job_was_not_started, _pending_solver_jobs,
	_resume_compatible_config, _sort_report_rows,
)
import free_order_campaign
import tspn_run_comparison


class FreeOrderQueueTests(unittest.TestCase):
	def test_pending_jobs_are_solver_major_and_skip_completed_pairs(self):
		completed = {('tpp-ours', 0), ('tpp-fekete', 0), ('tpp-fekete', 2)}
		self.assertEqual(
			_pending_solver_jobs(4, ['tpp-fekete', 'tpp-ours'], completed),
			[('tpp-ours', 1), ('tpp-ours', 2), ('tpp-ours', 3), ('tpp-fekete', 1), ('tpp-fekete', 3)],
		)

	def test_resume_allows_worker_and_runner_changes_but_not_solver_changes(self):
		previous = {
			'solvers': ['tpp-ours', 'tpp-fekete'], 'instance_workers': 1,
			'threads_per_instance': 12, 'max_calls': 100_000_000,
			'campaign_runner_sha256': 'old-runner', 'external_runner_sha256': 'old-external',
			'queue_policy': 'solver phases',
		}
		current = {
			**previous, 'instance_workers': 2, 'campaign_runner_sha256': 'new-runner',
			'external_runner_sha256': 'new-external', 'queue_policy': 'shared FIFO',
		}
		self.assertTrue(_resume_compatible_config(previous, current))
		current['threads_per_instance'] = 8
		self.assertFalse(_resume_compatible_config(previous, current))

	def test_resume_selects_newest_compatible_report_not_oldest_report(self):
		with tempfile.TemporaryDirectory() as temporary:
			results = Path(temporary)
			newer = results / 'newer' / 'report.json'
			older = results / 'older' / 'report.json'
			newer.parent.mkdir()
			older.parent.mkdir()
			current = {
				'solvers': ['tpp-ours'], 'instance_workers': 8, 'threads_per_instance': 1,
				'max_calls': 100, 'unordered_binary_sha256': 'same-binary',
				'campaign_runner_sha256': 'current', 'queue_policy': 'current',
			}
			compatible = {
				**current, 'instance_workers': 2,
				'campaign_runner_sha256': 'previous', 'queue_policy': 'previous',
			}
			incompatible = {**compatible, 'threads_per_instance': 8}
			newer.write_text(json.dumps({'key': 'old-key', 'config': compatible, 'rows': []}))
			older.write_text(json.dumps({'key': 'older-key', 'config': incompatible, 'rows': []}))
			os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
			os.utime(older, ns=(1_000_000_000, 1_000_000_000))

			match = _find_compatible_report(results, 'current-key', current)

		self.assertIsNotNone(match)
		self.assertEqual(match[0], newer)

	def test_report_rows_sort_when_only_one_solver_is_selected(self):
		rows = [
			{'case': 1, 'solver': 'tpp-fekete'},
			{'case': 0, 'solver': 'tpp-fekete'},
			{'case': 0, 'solver': 'tpp-ours'},
		]

		_sort_report_rows(rows)

		self.assertEqual(
			[(row['case'], row['solver']) for row in rows],
			[(0, 'tpp-ours'), (0, 'tpp-fekete'), (1, 'tpp-fekete')],
		)

	def test_shutdown_skips_jobs_that_never_started_a_solver(self):
		self.assertTrue(_job_was_not_started({'not_started': True}))
		self.assertTrue(_job_was_not_started({
			'row': {'error': 'shutdown requested before solver start'},
		}))
		self.assertFalse(_job_was_not_started({
			'row': {'status': 'interrupted', 'upper_bound': 123.0, 'lower_bound': 100.0},
		}))

	def test_fekete_starts_from_the_shared_queue_while_our_case_is_still_running(self):
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			campaign = root / 'campaign'
			campaign.mkdir()
			suite_path = campaign / 'tiny.bin'
			suite_path.write_bytes(_two_case_suite())
			(campaign / 'campaign.json').write_text(json.dumps({
				'name': 'two-case queue test', 'inputs': [{'file': 'tiny.bin'}],
			}))
			binary = root / 'tpp'
			binary.write_text('stub solver binary')
			build = root / 'fekete-build'
			bindings = build / 'python/tspn_bnb2/core'
			bindings.mkdir(parents=True)
			(bindings / '_tspn_bindings_test.so').write_bytes(b'stub binding')
			venv_python = root / '.venv/bin/python'
			venv_python.parent.mkdir(parents=True)
			venv_python.symlink_to(sys.executable)
			slow_case_started = threading.Event()
			release_slow_case = threading.Event()
			fekete_overlapped = threading.Event()
			active_our_cases = 0
			active_lock = threading.Lock()

			def fake_our_solver(_binary, _start, _target, polygons, _max_calls, _seconds, arguments=(), **_progress):
				nonlocal active_our_cases
				index = int(polygons[0][0][0] // 10)
				with active_lock:
					active_our_cases += 1
				try:
					if index == 1:
						slow_case_started.set()
						if not release_slow_case.wait(5):
							raise TimeoutError('Fekete did not enter the queue')
					else:
						time.sleep(0.05)
					return {
						'status': 'optimal', 'termination': 'gap', 'exact': True,
						'upper_bound': 10.0, 'lower_bound': 10.0,
						'path': [[0.0, 0.0], [10.0, 0.0]], 'seconds': 0.05,
					}
				finally:
					with active_lock:
						active_our_cases -= 1

			def fake_fekete_solver(_args, index, _log_file, _log_lock, **_options):
				self.assertEqual(_args.worker_python, venv_python)
				self.assertTrue(slow_case_started.is_set())
				with active_lock:
					if active_our_cases:
						fekete_overlapped.set()
					release_slow_case.set()
				return {
					'status': 'optimal', 'is_optimal': True, 'is_valid_trajectory': True,
					'lower_bound': 10.0, 'upper_bound': 10.0, 'absolute_gap': 0.0,
					'relative_gap': 0.0, 'solve_seconds': 0.05,
					'trajectory': [[0.0, 0.0], [10.0, 0.0]],
					'snapped_trajectory': [[0.0, 0.0], [10.0, 0.0]],
					'validation': {'valid': True}, 'snapped_validation': {'valid': True},
					'statistics': {'num_iterations': 1, 'soc_num_calls': 1},
				}

			with patch.object(free_order_campaign, 'BINARY', binary), \
				patch.object(free_order_campaign, 'ensure_binary'), \
				patch.object(free_order_campaign, 'run_unordered_solver', side_effect=fake_our_solver), \
				patch.object(free_order_campaign, 'validate_path', return_value={'valid': True}), \
				patch.object(tspn_run_comparison, 'run_case', side_effect=fake_fekete_solver):
				status = free_order_campaign.main([
					str(campaign), '--solver', 'tpp-ours', '--solver', 'tpp-fekete',
					'--threads-per-instance', '1', '--workers', '2', '--max-seconds', '-1',
					'--max-calls', '100', '--external-python', str(venv_python),
					'--external-build', str(build),
				])

			self.assertEqual(status, 0)
			self.assertTrue(fekete_overlapped.is_set())

	def test_resume_retries_only_external_errors_and_keeps_time_limited_results(self):
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			campaign = root / 'campaign'
			campaign.mkdir()
			(campaign / 'tiny.bin').write_bytes(_two_case_suite())
			(campaign / 'campaign.json').write_text(json.dumps({
				'inputs': [{'file': 'tiny.bin'}],
			}))
			binary = root / 'tpp'
			binary.write_bytes(b'frozen test binary')
			build = root / 'fekete-build'
			bindings = build / 'python/tspn_bnb2/core'
			bindings.mkdir(parents=True)
			(bindings / '_tspn_bindings_test.so').write_bytes(b'frozen test binding')
			options = [str(campaign), '--solver', 'tpp-ours', '--solver', 'tpp-fekete',
				'--max-seconds', '3600', '--workers', '2', '--no-build',
				'--external-python', sys.executable, '--external-build', str(build)]
			ours = {'exact': True, 'termination': 'optimal', 'seconds': 1,
				'upper_bound': 10., 'lower_bound': 10., 'path': [[0., 0.], [10., 0.]]}
			partial = {'status': 'limit', 'is_optimal': False, 'solve_seconds': 3601.,
				'upper_bound': 11., 'lower_bound': 10., 'trajectory': [[0., 0.], [10., 0.]],
				'validation': {'valid': True}, 'statistics': {}}
			failure = {'status': 'error', 'error': 'missing interpreter', 'solve_seconds': 0.}
			with patch.object(free_order_campaign, 'BINARY', binary), \
				patch.object(free_order_campaign, 'ensure_binary'), \
				patch.object(free_order_campaign, 'run_unordered_solver', return_value=ours) as native, \
				patch.object(free_order_campaign, 'validate_path', return_value={'valid': True}), \
				patch.object(tspn_run_comparison, 'run_case',
					side_effect=lambda _args, index, *_rest, **_options: partial if index == 0 else failure) as external, \
				contextlib.redirect_stdout(io.StringIO()):
				self.assertEqual(free_order_campaign.main(options), 2)  # solver error, not a campaign failure
				native.reset_mock()
				external.reset_mock()
				external.side_effect = None
				external.return_value = {**partial, 'status': 'optimal', 'is_optimal': True,
					'upper_bound': 10., 'solve_seconds': 2.}
				self.assertEqual(free_order_campaign.main(options), 0)
				native.assert_not_called()
				external.assert_called_once()
				self.assertEqual(external.call_args.args[1], 1)
				external.reset_mock()
				self.assertEqual(free_order_campaign.main(options), 0)
				external.assert_not_called()
			report = json.loads(next((campaign / 'results').glob('*/report.json')).read_text())
			self.assertEqual(len(report['rows']), 4)
			preserved = next(row for row in report['rows'] if row['solver'] == 'tpp-fekete' and row['case'] == 0)
			self.assertEqual(preserved['status'], 'limit')
			self.assertEqual(preserved['seconds'], 3601.)
			self.assertEqual(report['status'], 'completed')
			self.assertNotIn('solver_errors', report)

	def test_revalidate_stored_external_path_without_changing_solver_result(self):
		with tempfile.TemporaryDirectory() as temporary:
			suite = Path(temporary) / 'tiny.bin'
			suite.write_bytes(_two_case_suite())
			cases = free_order_campaign.read_encoded_cases(suite)
			# Endpoint drift is above the independent tolerance, while the line
			# still visits the polygon. Snapping is only a diagnostic candidate.
			external = {'case_index': '0', 'sha256': cases[0].digest, 'status': 'optimal',
				'is_optimal': 'True', 'upper_bound': '10', 'lower_bound': '10', 'solve_seconds': '2',
				'validation_tolerance': '1e-7',
				'trajectory_json': json.dumps([[0., 1e-5], [0., 1.5], [10., 0.]]),
				'snapped_trajectory_json': json.dumps([[0., 0.], [0., 1.5], [10., 0.]])}
			row = free_order_campaign._tspn_report_row(external, cases)
			self.assertFalse(row['valid'])
			self.assertTrue(row['endpoint_repaired_valid'])
			self.assertEqual(row['seconds'], 2.)
			self.assertTrue(row['exact'])
			self.assertEqual(row['path'][0], [0., 1e-5])
			external['raw_valid'] = 'False'
			with patch.object(free_order_campaign, 'validate_path') as validate:
				self.assertFalse(free_order_campaign._tspn_report_row(external, cases)['valid'])
				validate.assert_not_called()

	def test_summary_excludes_unfinished_invalid_unknown_and_error_pairs(self):
		rows = []
		for index in range(5):
			base = {'case': index, 'exact': True, 'valid': True, 'seconds': 1.}
			rows.append({**base, 'solver': 'tpp-ours'})
			external = {**base, 'solver': 'tpp-fekete', 'seconds': 2.}
			if index == 1:
				external['valid'] = False
			elif index == 2:
				external['exact'] = False
			elif index == 3:
				external['valid'] = None
			elif index == 4:
				external['error'] = 'missing interpreter'
			rows.append(external)
		with tempfile.TemporaryDirectory() as temporary:
			path = Path(temporary) / 'comparison.md'
			free_order_campaign.write_comparison_summary(path, {
				'rows': rows, 'config': {'solvers': ['tpp-ours', 'tpp-fekete']}, 'status': 'failed',
			}, 5)
			text = path.read_text()
			self.assertIn('independently valid raw paths: 1 instances.', text)
			self.assertIn('| tpp-fekete | 5 | 1 | 3/5 |', text)


class ExternalWorkerRuntimeTests(unittest.TestCase):
	def test_worker_uses_venv_symlink_even_when_its_target_changes(self):
		with tempfile.TemporaryDirectory() as temporary:
			root = Path(temporary)
			link = root / '.venv/bin/python'
			link.parent.mkdir(parents=True)
			args = SimpleNamespace(worker_python=link, tspn_repo=root, suite=root / 'tiny.bin',
				mode='path', time_limit=3600, threads=1, eps=.001, feasibility_tolerance=.001,
				validation_tolerance=1e-7, oracle_backend='socp', oracle_tolerance=1e-7)
			for version in ('python-old', 'python-new'):
				target = root / version
				target.write_text('test interpreter placeholder')
				if link.is_symlink():
					old_target = link.resolve()
					link.unlink()
					old_target.unlink()
				link.symlink_to(target)
				def fake_spawn(command, **_kwargs):
					self.assertEqual(command[0], str(link))
					result_path = Path(command[command.index('--worker-result') + 1])
					result_path.write_text(json.dumps({'status': 'optimal', 'upper_bound': 10.}))
					process = Mock(returncode=0, stdout=io.StringIO(''), stderr=io.StringIO(''))
					process.wait.return_value = 0
					return process
				with patch.object(tspn_run_comparison.subprocess, 'Popen', side_effect=fake_spawn):
					result = tspn_run_comparison.run_case(args, 0, io.StringIO())
				self.assertEqual(result['status'], 'optimal')

class SingleReportFileTests(unittest.TestCase):
	"""A run is one report.json: Fekete telemetry inside, nothing else to keep in sync."""

	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		root = Path(self.directory.name)
		self.campaign = root / 'campaign'
		self.campaign.mkdir()
		(self.campaign / 'tiny.bin').write_bytes(_two_case_suite())
		(self.campaign / 'campaign.json').write_text(json.dumps({'inputs': [{'file': 'tiny.bin'}]}))
		self.binary = root / 'tpp'
		self.binary.write_bytes(b'stub')
		self.build = root / 'fekete-build'
		(self.build / 'python/tspn_bnb2/core').mkdir(parents=True)
		(self.build / 'python/tspn_bnb2/core/_tspn_bindings_test.so').write_bytes(b'stub')
		self.fekete_calls = []

	def ours(self, *_args, **_options):
		return {'status': 'optimal', 'exact': True, 'termination': 'optimal', 'seconds': 0.1,
			'upper_bound': 10.0, 'lower_bound': 10.0, 'path': [[0.0, 0.0], [10.0, 0.0]]}

	def fekete(self, args, index, _log_file, _log_lock, **_options):
		self.fekete_calls.append((args.suite, index))
		return {'status': 'optimal', 'is_optimal': True, 'is_valid_trajectory': True,
			'lower_bound': 10.0, 'upper_bound': 10.0, 'absolute_gap': 0.0, 'relative_gap': 0.0,
			'solve_seconds': 0.2, 'trajectory': [[0.0, 0.0], [10.0, 0.0]],
			'snapped_trajectory': [[0.0, 0.0], [10.0, 0.0]],
			'validation': {'valid': True}, 'snapped_validation': {'valid': True},
			'statistics': {'num_iterations': 7, 'num_branches': 3, 'soc_num_calls': 42, 'soc_total_seconds': 0.5}}

	def run_main(self, *extra):
		with patch.object(free_order_campaign, 'BINARY', self.binary), \
			patch.object(free_order_campaign, 'ensure_binary'), \
			patch.object(free_order_campaign, 'run_unordered_solver', side_effect=self.ours), \
			patch.object(free_order_campaign, 'validate_path', return_value={'valid': True}), \
			patch.object(tspn_run_comparison, 'run_case', side_effect=self.fekete):
			return free_order_campaign.main([str(self.campaign), '--solver', 'tpp-ours', '--solver', 'tpp-fekete',
				'--max-seconds', '60', '--no-build', '--external-python', sys.executable,
				'--external-build', str(self.build), *extra])

	def run_directory(self):
		return next((self.campaign / 'results').iterdir())

	def test_a_finished_run_leaves_only_the_report(self):
		self.assertEqual(self.run_main(), 0)
		self.assertEqual(sorted(path.name for path in self.run_directory().iterdir()), ['report.json'])

	def test_fekete_rows_carry_their_telemetry_without_duplicating_the_path(self):
		self.run_main()
		report = json.loads((self.run_directory() / 'report.json').read_text())
		row = next(item for item in report['rows'] if item['solver'] == 'tpp-fekete')
		self.assertEqual(row['calls'], 42)
		external = row['external']
		self.assertEqual((external['num_iterations'], external['num_branches'], external['soc_total_seconds']), (7, 3, 0.5))
		self.assertEqual(external['snapped_trajectory'], [[0.0, 0.0], [10.0, 0.0]])
		self.assertNotIn('trajectory_json', external)
		self.assertNotIn('case_index', external)
		json.dumps(report, allow_nan=False)

	def test_the_fekete_runner_reads_the_campaign_input_directly(self):
		self.run_main()
		self.assertEqual({suite.resolve() for suite, _ in self.fekete_calls}, {(self.campaign / 'tiny.bin').resolve()})

	def test_several_input_files_use_a_temporary_suite_that_is_removed(self):
		(self.campaign / 'more.bin').write_bytes(_two_case_suite())
		(self.campaign / 'campaign.json').write_text(json.dumps({'inputs': [{'file': 'tiny.bin'}, {'file': 'more.bin'}]}))
		self.run_main()
		suites = {suite for suite, _ in self.fekete_calls}
		self.assertEqual(len(suites), 1)
		self.assertNotIn((self.campaign / 'tiny.bin').resolve(), {suite.resolve() for suite in suites})
		self.assertFalse(next(iter(suites)).exists())
		self.assertEqual(sorted(path.name for path in self.run_directory().iterdir()), ['report.json'])

	def test_resuming_a_legacy_run_imports_its_csv_instead_of_solving_again(self):
		self.run_main()
		run = self.run_directory()
		report = json.loads((run / 'report.json').read_text())
		legacy_rows = [item for item in report['rows'] if item['solver'] == 'tpp-fekete']
		report['rows'] = [item for item in report['rows'] if item['solver'] != 'tpp-fekete']
		report['status'] = 'interrupted'
		(run / 'report.json').write_text(json.dumps(report))
		suite = self.campaign / 'tiny.bin'
		cases = tspn_run_comparison.read_cases(suite)
		namespace = SimpleNamespace(mode='path', time_limit=60, threads=1, eps=0.001, oracle_backend='socp',
			feasibility_tolerance=1e-8, validation_tolerance=1e-7)
		(run / 'external/20260926-000000').mkdir(parents=True)
		csv_path = run / 'external/20260926-000000/tiny-tspn-path.csv'
		with csv_path.open('w', newline='') as file:
			writer = csv.DictWriter(file, fieldnames=tspn_run_comparison.RESULT_FIELDS)
			writer.writeheader()
			for index in range(2):
				writer.writerow(tspn_run_comparison.result_row(namespace, index, cases[index], {}, self.fekete(
					SimpleNamespace(suite=suite), index, None, None)))
		self.fekete_calls.clear()
		self.assertEqual(self.run_main(), 0)
		self.assertEqual(self.fekete_calls, [])
		resumed = json.loads((run / 'report.json').read_text())
		restored = sorted((item for item in resumed['rows'] if item['solver'] == 'tpp-fekete'), key=lambda item: item['case'])
		self.assertEqual([item['case'] for item in restored], [0, 1])
		self.assertEqual(restored[0]['external']['num_iterations'], 7)
		self.assertEqual(restored[0]['upper_bound'], legacy_rows[0]['upper_bound'])
		self.assertTrue(any('legacy CSV' in note for note in resumed['notes']))

	def test_the_summary_is_computed_on_demand_from_the_report(self):
		self.run_main()
		output = io.StringIO()
		with contextlib.redirect_stdout(output):
			self.assertEqual(free_order_campaign.report_main([str(self.run_directory())]), 0)
		self.assertIn('Free-order TPP campaign results', output.getvalue())
		self.assertIn('Instances in suite: 2', output.getvalue())


class FinalStatusTests(unittest.TestCase):
	solvers = ['tpp-ours', 'tpp-fekete']

	def rows(self, *failed):
		return [
			{'solver': solver, 'case': case, **({'error': 'boom\nGurobi status 3'} if (solver, case) in failed else {})}
			for solver in self.solvers for case in range(3)
		]

	def status(self, rows):
		ok = {(r['solver'], r['case']) for r in rows if not r.get('error')}
		return free_order_campaign._final_status(rows, ok, 3, self.solvers)

	def test_clean_run_is_completed(self):
		self.assertEqual(self.status(self.rows()), ('completed', {}))

	def test_a_solver_error_names_the_solver_and_is_not_a_campaign_failure(self):
		status, errors = self.status(self.rows(('tpp-fekete', 1)))
		self.assertEqual(status, 'completed_with_errors')
		self.assertEqual(list(errors), ['tpp-fekete'])
		self.assertEqual(errors['tpp-fekete']['cases'], [1])

	def test_cases_that_never_ran_fail_the_campaign(self):
		rows = [row for row in self.rows() if row['case'] != 2]
		self.assertEqual(self.status(rows)[0], 'failed')

	def test_summary_reports_which_solver_failed(self):
		report = {
			'config': {'solvers': self.solvers}, 'rows': [], 'status': 'completed_with_errors',
			'solver_errors': {'tpp-fekete': {'cases': [42], 'first_error': 'invalid model status: 3.'}},
		}
		text = free_order_campaign.render_comparison_summary(report, 3)
		self.assertIn('Campaign status: completed with errors.', text)
		self.assertIn('tpp-fekete failed on 1 case(s) (case 43, numbered as in `tpp.py live`): invalid model status: 3.', text)


class SolverNamesTests(unittest.TestCase):
	def test_older_reports_are_renamed_on_read(self):
		report = {
			'config': {'solvers': ['unordered', 'tspn']},
			'rows': [{'solver': 'unordered'}, {'solver': 'tspn'}, {'solver': 'tpp-ours'}],
		}
		free_order_campaign.normalize_report_solvers(report)
		self.assertEqual(report['config']['solvers'], ['tpp-ours', 'tpp-fekete'])
		self.assertEqual(
			[row['solver'] for row in report['rows']], ['tpp-ours', 'tpp-fekete', 'tpp-ours']
		)

	def test_solver_option_still_accepts_the_old_names(self):
		self.assertEqual(free_order_campaign.parse_solver_name('unordered'), 'tpp-ours')
		self.assertEqual(free_order_campaign.parse_solver_name('tspn'), 'tpp-fekete')


def _two_case_suite() -> bytes:
	encoded = bytearray()
	for index in range(2):
		encoded.extend(struct.pack('<ddddQ', 0.0, 0.0, 10.0, 0.0, 1))
		encoded.extend(struct.pack('<Q', 4))
		for x, y in ((index * 10.0, 1.0), (index * 10.0 + 1.0, 1.0),
			(index * 10.0 + 1.0, 2.0), (index * 10.0, 2.0)):
			encoded.extend(struct.pack('<dd', x, y))
		encoded.extend(struct.pack('<Q', 0))
	return bytes(encoded)


if __name__ == '__main__':
	unittest.main()
