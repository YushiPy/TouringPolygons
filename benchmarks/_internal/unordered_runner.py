"""Shared process runner for the free-order C++ solver."""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Sequence

Point = Sequence[float]
Polygon = Sequence[Point]

_ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()
_ACTIVE_PROCESSES_LOCK = threading.Lock()
_SHUTDOWN_REQUESTED = False


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
	lines = [' '.join(list(map(str, (*start, *target, len(polygons), max_calls))) + [seconds_text])]
	lines.extend(f'{len(polygon)} ' + ' '.join(str(coordinate) for vertex in polygon for coordinate in vertex)
		for polygon in polygons)
	if initial_path is not None:
		lines.append(f'{len(initial_path)} ' + ' '.join(str(coordinate) for point in initial_path for coordinate in point))
	return '\n'.join(lines) + '\n'


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
) -> dict:
	command = [str(solver.resolve()), *arguments, *(['--initial-path'] if initial_path is not None else [])]
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
	try:
		try:
			stdout, stderr = process.communicate(
				encode_instance(start, target, polygons, max_calls, max_seconds, initial_path),
				timeout=process_timeout,
			)
		except KeyboardInterrupt:
			interrupt_running_solvers()
			process.communicate()
			raise
		except subprocess.TimeoutExpired:
			if os.name == 'posix':
				os.killpg(process.pid, signal.SIGKILL)
			else:
				process.kill()
			process.communicate()
			raise
	finally:
		with _ACTIVE_PROCESSES_LOCK:
			_ACTIVE_PROCESSES.discard(process)
	if process.returncode:
		raise RuntimeError(stderr.strip() or 'Free-order solver failed.')
	return json.loads(stdout)
