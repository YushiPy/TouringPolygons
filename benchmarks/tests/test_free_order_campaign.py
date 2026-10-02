import json
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch


INTERNAL = Path(__file__).resolve().parents[1] / '_internal'
sys.path.insert(0, str(INTERNAL))

from free_order_campaign import (
	_find_compatible_report, _pending_solver_jobs, _resume_compatible_config, _sort_report_rows,
)
import free_order_campaign
import tspn_run_comparison


class FreeOrderQueueTests(unittest.TestCase):
	def test_pending_jobs_are_solver_major_and_skip_completed_pairs(self):
		completed = {('unordered', 0), ('tspn', 0), ('tspn', 2)}
		self.assertEqual(
			_pending_solver_jobs(4, ['tspn', 'unordered'], completed),
			[('unordered', 1), ('unordered', 2), ('unordered', 3), ('tspn', 1), ('tspn', 3)],
		)

	def test_resume_allows_worker_and_runner_changes_but_not_solver_changes(self):
		previous = {
			'solvers': ['unordered', 'tspn'], 'instance_workers': 1,
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
				'solvers': ['unordered'], 'instance_workers': 8, 'threads_per_instance': 1,
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
			{'case': 1, 'solver': 'tspn'},
			{'case': 0, 'solver': 'tspn'},
			{'case': 0, 'solver': 'unordered'},
		]

		_sort_report_rows(rows)

		self.assertEqual(
			[(row['case'], row['solver']) for row in rows],
			[(0, 'unordered'), (0, 'tspn'), (1, 'tspn')],
		)

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
			slow_case_started = threading.Event()
			release_slow_case = threading.Event()
			fekete_overlapped = threading.Event()
			active_our_cases = 0
			active_lock = threading.Lock()

			def fake_our_solver(_binary, _start, _target, polygons, _max_calls, _seconds, arguments=()):
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

			def fake_fekete_solver(_args, index, _log_file, _log_lock):
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
					str(campaign), '--solver', 'unordered', '--solver', 'tspn',
					'--threads-per-instance', '1', '--workers', '2', '--max-seconds', '-1',
					'--max-calls', '100', '--external-python', sys.executable,
					'--external-build', str(build),
				])

			self.assertEqual(status, 0)
			self.assertTrue(fekete_overlapped.is_set())

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
