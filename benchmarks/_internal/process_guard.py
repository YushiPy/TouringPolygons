"""Stops a solver process that grows past a memory limit, so it can report instead of being killed.

The operating system's out-of-memory killer (or a hurried SIGKILL) leaves nothing behind;
asking the solver to stop with SIGINT/SIGTERM first lets it hand back its incumbent and
bounds. Only the standard library is used: the resident size comes from ``/proc`` on Linux
and from ``ps`` elsewhere.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable

POLL_SECONDS = 5.0


def rss_bytes(pid: int) -> int | None:
	"""Resident memory of ``pid`` in bytes, or None when it cannot be read (process gone)."""
	try:
		with open(f"/proc/{pid}/statm") as handle:
			return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
	except (OSError, ValueError, IndexError):
		pass
	try:
		output = subprocess.run(
			["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5
		).stdout.strip()
		return int(output) * 1024 if output else None
	except (OSError, ValueError, subprocess.SubprocessError):
		return None


def descendants(pid: int) -> list[int]:
	"""Every process below ``pid``, whatever its process group or session (solver workers start their own)."""
	try:
		table = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
	except (OSError, subprocess.SubprocessError):
		return []
	children: dict[int, list[int]] = {}
	for line in table.splitlines():
		try:
			child, parent = (int(value) for value in line.split())
		except ValueError:
			continue
		children.setdefault(parent, []).append(child)
	found, pending = [], [pid]
	while pending:
		for child in children.get(pending.pop(), []):
			found.append(child)
			pending.append(child)
	return found


def alive(pid: int) -> bool:
	"""True while ``pid`` runs; an exited process waiting to be reaped (zombie) does not count."""
	try:
		os.kill(pid, 0)
	except ProcessLookupError:
		return False
	except PermissionError:
		return True
	try:
		state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
	except (OSError, subprocess.SubprocessError):
		return True
	return bool(state.strip()) and not state.strip().startswith("Z")


def _send(pid: int, number: int) -> None:
	try:
		os.kill(pid, number)
	except OSError:
		pass


def kill_tree(pid: int, grace: float = 8.0) -> list[int]:
	"""Stop ``pid`` and everything below it: SIGINT to every descendant, a grace period for them to
	report and exit, SIGKILL for any left, and only then SIGKILL for ``pid`` and its process group.

	Solver workers run in their own session, so a group signal alone never reaches them. Returns
	the descendants that needed SIGKILL.
	"""
	below = descendants(pid)
	for child in below:
		_send(child, signal.SIGINT)
	deadline = time.monotonic() + grace
	while time.monotonic() < deadline and any(alive(child) for child in below):
		time.sleep(0.2)
	stubborn = [child for child in descendants(pid) + below if alive(child)]
	for child in dict.fromkeys(stubborn):
		_send(child, signal.SIGKILL)
	try:
		os.killpg(pid, signal.SIGKILL)
	except OSError:
		pass
	_send(pid, signal.SIGKILL)
	return list(dict.fromkeys(stubborn))


class MemoryGuard:
	"""Polls one process; above ``limit_bytes`` it sends ``stop_signal`` to the process group once."""

	def __init__(
		self,
		pid: int,
		limit_bytes: int,
		stop_signal: int = signal.SIGINT,
		on_limit: Callable[[int], None] | None = None,
		poll_seconds: float = POLL_SECONDS,
	) -> None:
		self.pid, self.limit_bytes, self.stop_signal = pid, limit_bytes, stop_signal
		self.on_limit, self.poll_seconds = on_limit, poll_seconds
		self.peak = 0
		self.triggered = False
		self._done = threading.Event()
		self._thread = threading.Thread(target=self._watch, daemon=True)
		self._thread.start()

	def _watch(self) -> None:
		while not self._done.wait(self.poll_seconds):
			used = rss_bytes(self.pid)
			if used is None:
				return
			self.peak = max(self.peak, used)
			if used > self.limit_bytes:
				self.triggered = True
				try:
					if on_limit := self.on_limit:
						on_limit(used)
				except Exception:
					pass
				try:
					os.killpg(self.pid, self.stop_signal)
				except (ProcessLookupError, PermissionError, OSError):
					try:
						os.kill(self.pid, self.stop_signal)
					except OSError:
						pass
				return

	def close(self) -> None:
		self._done.set()
