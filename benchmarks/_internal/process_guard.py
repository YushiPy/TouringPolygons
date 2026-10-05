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
