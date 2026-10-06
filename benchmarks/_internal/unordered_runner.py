"""Shared process runner for the free-order C++ solver."""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Callable, Sequence

from process_guard import MemoryGuard

# The solver reads max_calls as an unsigned 64-bit number; -1 ("no limit") is sent as its largest value.
UNLIMITED_CALLS = 2**64 - 1

Point = Sequence[float]
Polygon = Sequence[Point]

_ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()
_ACTIVE_PROCESSES_LOCK = threading.Lock()
_SHUTDOWN_REQUESTED = False


class SolverKilled(RuntimeError):
	"""The solver process died from a signal (SIGKILL from the user or the out-of-memory killer...)."""

	def __init__(self, message: str, signal_number: int) -> None:
		super().__init__(message)
		self.signal_number = signal_number


def interrupt_running_solvers() -> None:
	"""Request cooperative shutdown from every active native solver process."""
	global _SHUTDOWN_REQUESTED
	with _ACTIVE_PROCESSES_LOCK:
		_SHUTDOWN_REQUESTED = True
		processes = list(_ACTIVE_PROCESSES)
	for process in processes:
		if process.poll() is not None:
			continue
		try:
			if os.name == 'posix':
				os.killpg(process.pid, signal.SIGINT)
			else:
				process.send_signal(signal.SIGINT)
		except ProcessLookupError:
			pass


def terminate_running_solvers() -> None:
	"""Force-stop active native solver processes after a second interrupt."""
	with _ACTIVE_PROCESSES_LOCK:
		processes = list(_ACTIVE_PROCESSES)
	for process in processes:
		if process.poll() is not None:
			continue
		try:
			if os.name == 'posix':
				os.killpg(process.pid, signal.SIGTERM)
			else:
				process.terminate()
		except ProcessLookupError:
			pass


def encode_instance(
	start: Point,
	target: Point,
	polygons: Sequence[Polygon],
	max_calls: int,
	max_seconds: float,
	initial_path: Sequence[Point] | None = None,
) -> str:
	seconds_text = "1e308" if math.isinf(max_seconds) else str(max_seconds)
	calls = UNLIMITED_CALLS if max_calls < 0 else max_calls
	lines = [' '.join(list(map(str, (*start, *target, len(polygons), calls))) + [seconds_text])]
	lines.extend(f'{len(polygon)} ' + ' '.join(str(coordinate) for vertex in polygon for coordinate in vertex)
		for polygon in polygons)
	if initial_path is not None:
		lines.append(f'{len(initial_path)} ' + ' '.join(str(coordinate) for point in initial_path for coordinate in point))
	return '\n'.join(lines) + '\n'


class _StreamedProcess:
	"""Feed a solver its input and read both its streams, handing progress lines to a callback.

	The solver reports on stderr as one JSON object per line starting with
	{"progress"; any other stderr text is kept for error messages. A failing
	callback never affects the run.
	"""

	def __init__(self, process: subprocess.Popen, text: str, on_progress: Callable[[dict], None]) -> None:
		self.process = process
		self.stdout: list[str] = []
		self.stderr: list[str] = []
		self.threads = [
			threading.Thread(target=self._write, args=(text,), daemon=True),
			threading.Thread(target=lambda: self.stdout.append(process.stdout.read()), daemon=True),
			threading.Thread(target=self._read_stderr, args=(on_progress,), daemon=True),
		]
		for thread in self.threads:
			thread.start()

	def _write(self, text: str) -> None:
		try:
			self.process.stdin.write(text)
			self.process.stdin.close()
		except (BrokenPipeError, OSError, ValueError):
			pass

	def _read_stderr(self, on_progress: Callable[[dict], None]) -> None:
		for line in self.process.stderr:
			if line.startswith('{"progress"'):
				try:
					on_progress(json.loads(line))
				except Exception:
					pass
			else:
				self.stderr.append(line)

	def wait(self, timeout: float | None) -> None:
		self.process.wait(timeout=timeout)

	def finish(self) -> tuple[str, str]:
		self.process.wait()
		for thread in self.threads:
			thread.join()
		for stream in (self.process.stdout, self.process.stderr):
			stream.close()
		return ''.join(self.stdout), ''.join(self.stderr)


def run_unordered_solver(
	solver: Path,
	start: Point,
	target: Point,
	polygons: Sequence[Polygon],
	max_calls: int,
	max_seconds: float,
	arguments: Sequence[str] = (),
	initial_path: Sequence[Point] | None = None,
	process_timeout: float | None = None,
	progress: Callable[[dict], None] | None = None,
	progress_interval: float | None = None,
	on_start: Callable[[int], None] | None = None,
	max_memory_bytes: int | None = None,
	on_memory_limit: Callable[[int], None] | None = None,
) -> dict:
	"""Run the solver; with ``progress`` and a positive ``progress_interval`` (seconds), call
	``progress`` with each periodic status the solver prints while it searches.

	``on_start`` receives the solver's pid. ``max_memory_bytes`` asks the solver to stop (SIGINT, so it
	returns its incumbent and bounds) when its resident memory passes the limit; ``on_memory_limit``
	is told how much it used. A solver killed by a signal raises ``SolverKilled``."""
	streaming = progress is not None and bool(progress_interval) and progress_interval > 0
	command = [str(solver.resolve()), *arguments, *(['--initial-path'] if initial_path is not None else [])]
	if streaming:
		command += ['--progress-interval', str(progress_interval)]
	if process_timeout is None and math.isfinite(max_seconds):
		process_timeout = max(30, max_seconds + 30)
	with _ACTIVE_PROCESSES_LOCK:
		if _SHUTDOWN_REQUESTED:
			return {"termination": "interrupted", "status": "interrupted", "error": "shutdown requested before solver start"}
		process = subprocess.Popen(
			command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			text=True, start_new_session=(os.name == 'posix'),
		)
		_ACTIVE_PROCESSES.add(process)
	if on_start is not None:
		try:
			on_start(process.pid)
		except Exception:
			pass
	guard = (
		MemoryGuard(process.pid, max_memory_bytes, signal.SIGINT, on_memory_limit)
		if max_memory_bytes
		else None
	)
	text = encode_instance(start, target, polygons, max_calls, max_seconds, initial_path)
	streams = _StreamedProcess(process, text, progress) if streaming else None

	def drain() -> tuple[str, str]:
		return streams.finish() if streams else process.communicate()

	try:
		try:
			if streams:
				streams.wait(process_timeout)
				stdout, stderr = streams.finish()
			else:
				stdout, stderr = process.communicate(text, timeout=process_timeout)
		except KeyboardInterrupt:
			interrupt_running_solvers()
			drain()
			raise
		except subprocess.TimeoutExpired:
			if os.name == 'posix':
				os.killpg(process.pid, signal.SIGKILL)
			else:
				process.kill()
			drain()
			raise
	finally:
		if guard is not None:
			guard.close()
		with _ACTIVE_PROCESSES_LOCK:
			_ACTIVE_PROCESSES.discard(process)
	if process.returncode and process.returncode < 0:
		name = signal.Signals(-process.returncode).name
		raise SolverKilled(f'solver killed by {name}' + (f': {stderr.strip()}' if stderr.strip() else ''), -process.returncode)
	if process.returncode:
		raise RuntimeError(stderr.strip() or 'Free-order solver failed.')
	return json.loads(stdout)
