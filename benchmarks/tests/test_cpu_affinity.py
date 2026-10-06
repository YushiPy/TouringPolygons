import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '_internal'))

import cpu_affinity
from cpu_affinity import CorePool, detect_cores, parse_cpu_list

CPUS = range(24)


def make_sysfs(root: Path) -> Path:
	"""An i9-12900K: 8 P cores with two hyperthreads (cpus 0-15) and 8 E cores (16-23)."""
	(root / 'cpu_core').mkdir(parents=True)
	(root / 'cpu_core' / 'cpus').write_text('0-15\n')
	(root / 'cpu_atom').mkdir()
	(root / 'cpu_atom' / 'cpus').write_text('16-23\n')
	for cpu in CPUS:
		topology = root / 'system' / 'cpu' / f'cpu{cpu}' / 'topology'
		topology.mkdir(parents=True)
		siblings = f'{cpu - cpu % 2}-{cpu - cpu % 2 + 1}' if cpu < 16 else str(cpu)
		topology.joinpath('thread_siblings_list').write_text(siblings + '\n')
	return root


def write_stat(path: Path, busy: dict[int, int], total: int) -> None:
	"""Cumulative jiffies: `busy[cpu]` user time out of `total`, the rest idle."""
	lines = ['cpu  0 0 0 0 0 0 0 0 0 0']
	for cpu in CPUS:
		used = busy.get(cpu, 0)
		lines.append(f'cpu{cpu} {used} 0 0 {total - used} 0 0 0 0 0 0')
	path.write_text('\n'.join(lines) + '\n')


class CpuAffinityTests(unittest.TestCase):
	def setUp(self):
		self.directory = tempfile.TemporaryDirectory()
		root = Path(self.directory.name)
		self.sysfs = make_sysfs(root / 'sys')
		self.stat = root / 'stat'
		write_stat(self.stat, {}, 1000)
		self.pinned = []
		self.messages = []

	def tearDown(self):
		self.directory.cleanup()

	def pool(self, workers, threads=1, mode='p-cores', busy_after=None):
		"""Build a pool; `busy_after` is the load seen during its initial half-second sample."""
		def sample(_seconds):
			write_stat(self.stat, busy_after or {}, 2000)
		with patch.object(cpu_affinity.os, 'sched_setaffinity', create=True), \
				patch.object(cpu_affinity.time, 'sleep', side_effect=sample):
			return CorePool(workers, threads, mode, sysfs=self.sysfs, proc_stat=self.stat, allowed=set(CPUS),
				pin=lambda pid, cpus: self.pinned.append((pid, cpus)),
				log=lambda message, **_: self.messages.append(message))

	def test_parse_cpu_list(self):
		self.assertEqual(parse_cpu_list('0-2,5,7-8\n'), {0, 1, 2, 5, 7, 8})

	def test_detects_physical_performance_cores_first(self):
		cores = detect_cores(self.sysfs, set(CPUS))
		self.assertEqual([core.cpus for core in cores[:8]], [(i, i + 1) for i in range(0, 16, 2)])
		self.assertTrue(all(core.performance for core in cores[:8]))
		self.assertEqual([core.cpus for core in cores[8:]], [(i,) for i in range(16, 24)])
		self.assertFalse(any(core.performance for core in cores[8:]))

	def test_without_hybrid_listing_every_core_is_performance(self):
		(self.sysfs / 'cpu_core' / 'cpus').unlink()
		self.assertTrue(all(core.performance for core in detect_cores(self.sysfs, {0, 1, 2, 3})))

	def test_workers_get_distinct_performance_cores_and_are_pinned(self):
		pool = self.pool(4)
		self.assertEqual(pool.warnings, [])
		leases = [pool.acquire() for _ in range(4)]
		self.assertEqual(len({lease.cpus for lease in leases}), 4)
		self.assertTrue(all(core.performance for lease in leases for core in lease.cores))
		pool.pin(123, leases[0])
		self.assertEqual(self.pinned, [(123, set(leases[0].cpus))])
		self.assertTrue(all(lease.note == '' for lease in leases))

	def test_more_workers_than_performance_cores_warns_and_uses_efficiency_cores(self):
		pool = self.pool(10)
		self.assertIn('Only 8 performance core(s) for 10 worker(s)', pool.warnings[0])
		self.assertIn('--workers 8', pool.warnings[0])
		leases = [pool.acquire() for _ in range(10)]
		self.assertEqual([lease.note for lease in leases[8:]], ['efficiency core'] * 2)
		self.assertTrue(any('no idle performance core' in message for message in self.messages))

	def test_threads_per_worker_take_one_physical_core_each(self):
		pool = self.pool(4, threads=2)
		self.assertEqual(pool.warnings, [])
		lease = pool.acquire()
		self.assertEqual(len(lease.cores), 2)
		self.assertEqual(len(lease.cpus), 4)

	def test_busy_core_is_announced_avoided_and_its_load_recorded(self):
		pool = self.pool(2, busy_after={0: 2000, 1: 1500})  # another user on the first P core
		self.assertTrue(any('already busy' in warning for warning in pool.warnings))
		lease = pool.acquire()
		self.assertNotIn(0, lease.cpus)
		write_stat(self.stat, {lease.cpus[0]: 1000}, 3000)
		placement = pool.release(lease)
		self.assertTrue(placement['performance'])
		self.assertEqual(placement['other_load'], 0.0)  # only our own solver ran there

	def test_release_reports_load_from_other_users(self):
		pool = self.pool(1)
		lease = pool.acquire()
		first, second = lease.cpus
		write_stat(self.stat, {first: 1000, second: 500}, 3000)  # our thread plus half a sibling
		self.assertEqual(pool.release(lease)['other_load'], 0.5)

	def test_released_core_is_reused(self):
		pool = self.pool(1)
		first = pool.acquire()
		pool.release(first)
		self.assertEqual(pool.acquire().cpus, first.cpus)

	def test_off_mode_does_not_pin(self):
		pool = self.pool(4, mode='off')
		self.assertIsNone(pool.acquire())
		self.assertIsNone(pool.release(None))
		self.assertFalse(pool.describe()['enabled'])


if __name__ == '__main__':
	unittest.main()
