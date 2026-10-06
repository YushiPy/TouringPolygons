"""Pin benchmark solver processes to whole performance cores (Linux only).

On a hybrid Intel CPU such as the i9-12900K, Linux lists the performance
(P) logical CPUs in /sys/devices/cpu_core/cpus and the efficiency (E) ones in
/sys/devices/cpu_atom/cpus; without that split every core counts as P. Each
solver thread gets one physical core, pinned to all of its hyperthreads so the
scheduler can still avoid a sibling busy with another user's work. Among free
cores the least loaded P cores are chosen first; E cores and cores already
holding one of our solvers are used only when the P cores run out, and that is
announced. The pool cannot reserve cores against other users: it measures
their load on our cores during each case and reports it in the row.

Run ``python3 benchmarks/_internal/cpu_affinity.py [WORKERS [THREADS]]`` on the
benchmark machine to see the detected cores and the plan.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

MODES = ('auto', 'p-cores', 'off')
# A core whose other-user load stays above this fraction is reported as shared.
BUSY_FRACTION = 0.10


def parse_cpu_list(text: str) -> set[int]:
	"""Parse the kernel's CPU list format, e.g. "0-7,16,18-19"."""
	cpus: set[int] = set()
	for part in text.strip().split(','):
		if not part:
			continue
		first, _, last = part.partition('-')
		cpus.update(range(int(first), int(last or first) + 1))
	return cpus


@dataclass(frozen=True)
class Core:
	cpus: tuple[int, ...]  # hyperthreads of one physical core
	performance: bool


def detect_cores(sysfs: Path = Path('/sys/devices'), allowed: set[int] | None = None) -> list[Core]:
	"""Physical cores usable by this process, P cores first."""
	if allowed is None:
		allowed = set(os.sched_getaffinity(0))
	p_list = sysfs / 'cpu_core' / 'cpus'
	performance = parse_cpu_list(p_list.read_text()) if p_list.exists() else set(allowed)
	cores: dict[tuple[int, ...], Core] = {}
	for cpu in sorted(allowed):
		siblings = sysfs / 'system' / 'cpu' / f'cpu{cpu}' / 'topology' / 'thread_siblings_list'
		group = parse_cpu_list(siblings.read_text()) if siblings.exists() else {cpu}
		key = tuple(sorted(group & allowed))
		cores.setdefault(key, Core(key, bool(set(key) & performance)))
	return sorted(cores.values(), key=lambda core: (not core.performance, core.cpus))


def read_cpu_times(proc_stat: Path = Path('/proc/stat')) -> dict[int, tuple[int, int]]:
	"""(busy, total) jiffies per logical CPU."""
	times = {}
	for line in proc_stat.read_text().splitlines():
		name, *fields = line.split()
		if not name.startswith('cpu') or name == 'cpu':
			continue
		values = [int(value) for value in fields]
		idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
		total = sum(values[:8])  # guest time is already inside user/nice
		times[int(name[3:])] = (total - idle, total)
	return times


def busy_fraction(cpus: tuple[int, ...], before: dict, after: dict) -> float:
	"""Busy logical-CPU equivalents of a core between two /proc/stat samples."""
	load = 0.0
	for cpu in cpus:
		if cpu in before and cpu in after:
			busy = after[cpu][0] - before[cpu][0]
			total = after[cpu][1] - before[cpu][1]
			load += busy / total if total > 0 else 0.0
	return load


@dataclass
class Lease:
	cores: tuple[Core, ...]
	before: dict
	note: str = ''

	@property
	def cpus(self) -> tuple[int, ...]:
		return tuple(cpu for core in self.cores for cpu in core.cpus)


class CorePool:
	"""Hands out whole cores to concurrent solver processes."""

	def __init__(self, workers: int, threads_per_worker: int = 1, mode: str = 'auto', *,
			sysfs: Path = Path('/sys/devices'), proc_stat: Path = Path('/proc/stat'),
			allowed: set[int] | None = None, pin=None, log=print) -> None:
		if mode not in MODES:
			raise ValueError(f'cpu affinity mode must be one of {", ".join(MODES)}')
		self.threads = max(1, threads_per_worker)
		self.proc_stat = proc_stat
		self.log = log
		self.lock = threading.Lock()
		self.holders: dict[Core, int] = {}
		supported = hasattr(os, 'sched_setaffinity') and proc_stat.exists()
		self.enabled = mode != 'off' and supported
		self.pin_process = pin or (os.sched_setaffinity if hasattr(os, 'sched_setaffinity') else None)
		self.cores: list[Core] = detect_cores(sysfs, allowed) if self.enabled else []
		# Other users' load per core, refreshed whenever at least a second has passed.
		self.load: dict[Core, float] = {}
		self.sample: tuple[float, dict] | None = None
		self.warnings: list[str] = []
		if mode == 'p-cores' and not supported:
			self.warnings.append('CPU pinning needs Linux (sched_setaffinity and /proc/stat); solvers run unpinned.')
		if self.enabled:
			needed = workers * self.threads
			p_cores = sum(core.performance for core in self.cores)
			if p_cores < needed:
				self.warnings.append(
					f'Only {p_cores} performance core(s) for {workers} worker(s) x {self.threads} thread(s): '
					f'{needed - p_cores} solver thread(s) will run on efficiency cores or share a core. '
					f'Use --workers {max(1, p_cores // self.threads)} to keep every solver on its own P core.')
			before = read_cpu_times(proc_stat)
			time.sleep(0.5)
			self._refresh_load(before, read_cpu_times(proc_stat))
			busy = [core for core in self.cores if core.performance and self.load[core] > BUSY_FRACTION]
			if busy:
				self.warnings.append(
					f'{len(busy)} of {p_cores} performance core(s) are already busy with other work '
					f'(cpus {"; ".join(",".join(map(str, core.cpus)) for core in busy)}); '
					'the least loaded cores are used first and each row records the other load it saw.')
		elif mode == 'auto' and not supported:
			self.log('CPU pinning unavailable here (needs Linux); solvers run unpinned.', flush=True)
		for warning in self.warnings:
			self.log(f'WARNING: {warning}', flush=True)

	def _refresh_load(self, before: dict, after: dict) -> None:
		self.sample = (time.monotonic(), after)
		for core in self.cores:
			own = self.holders.get(core, 0)
			self.load[core] = max(0.0, busy_fraction(core.cpus, before, after) - own)

	def describe(self) -> dict:
		return {'enabled': self.enabled, 'threads_per_worker': self.threads,
			'performance_cores': [list(core.cpus) for core in self.cores if core.performance],
			'efficiency_cores': [list(core.cpus) for core in self.cores if not core.performance],
			'warnings': self.warnings}

	def acquire(self) -> Lease | None:
		if not self.enabled:
			return None
		with self.lock:
			times = read_cpu_times(self.proc_stat)
			if self.sample is None or time.monotonic() - self.sample[0] >= 1.0:
				if self.sample is not None:
					self._refresh_load(self.sample[1], times)
				else:
					self.sample = (time.monotonic(), times)
			# Free cores before ours, P before E, then the least other load.
			ranked = sorted(self.cores, key=lambda core: (
				self.holders.get(core, 0), not core.performance, self.load.get(core, 0.0) > BUSY_FRACTION,
				self.load.get(core, 0.0)))
			chosen = tuple(ranked[:self.threads])
			for core in chosen:
				self.holders[core] = self.holders.get(core, 0) + 1
			notes = []
			if any(not core.performance for core in chosen):
				notes.append('efficiency core')
			if any(self.holders[core] > 1 for core in chosen):
				notes.append('core shared with another worker')
			if any(self.load.get(core, 0.0) > BUSY_FRACTION for core in chosen):
				notes.append('core busy with other work')
			lease = Lease(chosen, times, ', '.join(notes))
		if lease.note:
			self.log(f'WARNING: no idle performance core free; a solver runs on cpus {lease.cpus} ({lease.note}).', flush=True)
		return lease

	def pin(self, pid: int, lease: Lease | None) -> None:
		if lease is not None and self.pin_process is not None:
			self.pin_process(pid, set(lease.cpus))

	def release(self, lease: Lease | None) -> dict | None:
		"""Free the cores and summarize them for the result row."""
		if lease is None:
			return None
		after = read_cpu_times(self.proc_stat)
		with self.lock:
			for core in lease.cores:
				self.holders[core] -= 1
				if not self.holders[core]:
					del self.holders[core]
		# Our solver keeps about one logical CPU per thread busy; the rest is someone else's.
		other = max(0.0, busy_fraction(lease.cpus, lease.before, after) - self.threads)
		summary = {'cpus': list(lease.cpus), 'performance': all(core.performance for core in lease.cores),
			'other_load': round(other, 3)}
		if lease.note:
			summary['note'] = lease.note
		return summary


def main(argv: list[str]) -> int:
	workers = int(argv[0]) if argv else 8
	threads = int(argv[1]) if len(argv) > 1 else 1
	pool = CorePool(workers, threads, 'p-cores')
	for key, value in pool.describe().items():
		print(f'{key}: {value}')
	return 0


if __name__ == '__main__':
	raise SystemExit(main(sys.argv[1:]))
