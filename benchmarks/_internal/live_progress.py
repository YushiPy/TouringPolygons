"""Live status of long solver runs: one line per report and a ``live.json`` snapshot.

The native solver prints a status every few seconds while it searches
(``--progress-interval``). ``LiveStatus`` turns those into
- a readable line on the terminal (or in the run's log),
- ``live.json`` next to the results, rewritten atomically, that
  ``tpp.py live`` shows from any other terminal without disturbing the run.

Nothing here influences a solve: a failing callback or an unwritable snapshot
is ignored. Only the standard library is used.
"""

from __future__ import annotations

import json
import math
import os
import signal
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

WINDOW = 12  # reports kept per running instance for trend and estimate
WRITE_EVERY_SECONDS = 5.0
LIVE_FILE = "live.json"
JOURNAL_FILE = "progress.jsonl"  # every report, appended: survives a crash and a lost terminal


def clock(seconds: float) -> str:
	seconds = max(0, int(seconds))
	days, rest = divmod(seconds, 86400)
	hours, rest = divmod(rest, 3600)
	minutes, secs = divmod(rest, 60)
	return (f"{days}d " if days else "") + f"{hours:02d}:{minutes:02d}:{secs:02d}"


def gap_fraction(lower: float | None, upper: float | None) -> float | None:
	"""(UB - LB) / UB, or None while there is no finite incumbent."""
	if lower is None or upper is None or not math.isfinite(upper) or upper <= 0:
		return None
	return max(0.0, (upper - lower) / upper)


def combine(workers: dict[int, dict]) -> dict:
	"""One status from the portfolio's searches: any worker's lower bound is valid, so the
	best lower and upper bounds win; work counters add up (calls are already shared)."""
	reports = list(workers.values())
	return {
		"elapsed_seconds": max(r["elapsed_seconds"] for r in reports),
		"lower_bound": max(
			(r["lower_bound"] for r in reports if r.get("lower_bound") is not None),
			default=None,
		),
		"upper_bound": min(
			(r["upper_bound"] for r in reports if r.get("upper_bound") is not None),
			default=None,
		),
		"calls": max(r["calls"] for r in reports),
		"nodes": sum(r["nodes"] for r in reports),
		"open_nodes": sum(r["open_nodes"] for r in reports),
		"max_sequence_depth": max(r["max_sequence_depth"] for r in reports),
		"regions": reports[0].get("regions"),
		"searches": len(reports),
	}


def estimate(history: list[dict], target_gap: float | None) -> dict:
	"""Rough trend from the recent reports; never a promise.

	``gap_per_hour`` is how fast the relative gap shrank over the window. The
	remaining time assumes the gap keeps shrinking geometrically, which branch
	and bound often does not, so it is None unless the gap clearly fell.
	"""
	result: dict = {"queue": None, "gap_per_hour": None, "remaining_seconds": None}
	if len(history) < 2:
		return result
	first, last = history[0], history[-1]
	span = last["elapsed_seconds"] - first["elapsed_seconds"]
	if span <= 0:
		return result
	if first["open_nodes"] and last["open_nodes"]:
		change = last["open_nodes"] / first["open_nodes"]
		result["queue"] = (
			"growing" if change > 1.05 else "shrinking" if change < 0.95 else "steady"
		)
	before, after = first.get("gap"), last.get("gap")
	if before is not None and after is not None:
		result["gap_per_hour"] = (before - after) / span * 3600
		if target_gap and 0 < target_gap < after < before:
			rate = (math.log(before) - math.log(after)) / span
			result["remaining_seconds"] = (
				math.log(after) - math.log(target_gap)
			) / rate
	return result


def describe(label: str, record: dict) -> str:
	"""The one-line status: bounds, gap, work done, queue trend and budget used."""
	parts = [f"[{label}] {clock(record['elapsed_seconds'])}"]
	lower, upper = record.get("lower_bound"), record.get("upper_bound")
	parts.append(f"LB {lower:.6g}" if lower is not None else "LB -")
	parts.append(f"UB {upper:.6g}" if upper is not None else "UB -")
	gap = record.get("gap")
	if gap is not None:
		parts.append(f"gap {100 * gap:.3g}%")
	parts.append(
		f"calls {record['calls']:,}  nodes {record['nodes']:,}  open {record['open_nodes']:,}"
	)
	trend = record.get("estimate", {})
	if trend.get("queue"):
		parts[-1] += f" ({trend['queue']})"
	if record.get("time_used") is not None:
		parts.append(f"time limit {100 * record['time_used']:.0f}% used")
	if record.get("calls_used") is not None:
		parts.append(f"call limit {100 * record['calls_used']:.0f}% used")
	if trend.get("remaining_seconds") is not None:
		parts.append(
			f"target gap in ~{clock(trend['remaining_seconds'])} if the pace holds (optimistic)"
		)
	return "  ".join(parts)


class LiveStatus:
	"""Collects the reports of the instances running now (several with ``--workers``)."""

	def __init__(
		self, path: Path | None = None, echo: Callable[[str], None] | None = None
	) -> None:
		self.path = path
		self.journal = path.with_name(JOURNAL_FILE) if path is not None else None
		self.echo = echo or (lambda line: print(line, flush=True))
		self.lock = threading.Lock()
		self.running: dict[str, dict] = {}
		self.last_write = 0.0

	def reporter(
		self,
		key: str,
		label: str,
		*,
		max_seconds: float | None = None,
		max_calls: int | None = None,
		target_gap: float | None = None,
		stop_signal: int = signal.SIGINT,
	) -> Callable[[dict], None]:
		"""A callback for ``run_unordered_solver(progress=...)`` that tracks one instance."""
		entry = {
			"label": label,
			"started_at": time.time(),
			"workers": {},
			"history": deque(maxlen=WINDOW),
			"max_seconds": max_seconds,
			"max_calls": max_calls,
			"target_gap": target_gap,
			"pid": None,
			"stop_signal": signal.Signals(stop_signal).name,
		}
		with self.lock:
			self.running[key] = entry

		def report(raw: dict) -> None:
			try:
				self._update(key, entry, raw)
			except Exception:
				pass

		return report

	def set_pid(self, key: str, pid: int) -> None:
		"""Record the solver process of ``key`` so ``tpp.py stop`` can ask it to stop."""
		with self.lock:
			if key in self.running:
				self.running[key]["pid"] = pid

	def last(self, key: str) -> dict | None:
		"""The latest combined report of ``key`` while it is still running."""
		with self.lock:
			entry = self.running.get(key)
			return dict(entry["latest"]) if entry and entry.get("latest") else None

	def finish(self, key: str) -> dict | None:
		"""Forget ``key`` and return its last combined report (the partial result if the solver died)."""
		with self.lock:
			entry = self.running.pop(key, None)
			self._write(force=True)
			return dict(entry["latest"]) if entry and entry.get("latest") else None

	def _update(self, key: str, entry: dict, raw: dict) -> None:
		with self.lock:
			entry["workers"][raw.get("worker", 0)] = raw
			record = combine(entry["workers"])
			record["gap"] = gap_fraction(record["lower_bound"], record["upper_bound"])
			entry["history"].append(record)
			record["estimate"] = estimate(list(entry["history"]), entry["target_gap"])
			if entry["max_seconds"] and math.isfinite(entry["max_seconds"]):
				record["time_used"] = min(
					1.0, record["elapsed_seconds"] / entry["max_seconds"]
				)
			if entry["max_calls"] and 0 < entry["max_calls"] < 10**15:
				record["calls_used"] = min(1.0, record["calls"] / entry["max_calls"])
			record["reported_at"] = time.time()
			entry["latest"] = record
			line = describe(entry["label"], record)
			self._journal(entry["label"], record)
			self._write()
		self.echo(line)

	def _journal(self, label: str, record: dict) -> None:
		if self.journal is None:
			return
		keep = ("elapsed_seconds", "lower_bound", "upper_bound", "gap", "calls", "nodes", "open_nodes")
		line = {"at": round(time.time(), 3), "label": label, **{name: record.get(name) for name in keep}}
		try:
			with self.journal.open("a") as handle:
				handle.write(json.dumps(line, allow_nan=False, default=str) + "\n")
		except (OSError, ValueError):
			pass

	def close(self) -> None:
		"""The run ended normally: drop the snapshot. After a crash it stays, and ``tpp.py live``
		flags it as belonging to a process that is gone."""
		with self.lock:
			self.running.clear()
		if self.path is not None:
			try:
				self.path.unlink()
			except OSError:
				pass

	def snapshot(self) -> dict:
		with self.lock:
			return self._snapshot()

	def _snapshot(self) -> dict:
		running = []
		for key, entry in self.running.items():
			latest = entry.get("latest")
			if latest is not None:
				running.append(
					{
						"key": key,
						"label": entry["label"],
						"started_at": entry["started_at"],
						"child_pid": entry.get("pid"),
						"stop_signal": entry.get("stop_signal"),
						**latest,
					}
				)
		return {
			"updated_at": time.time(),
			"pid": os.getpid(),
			"host": socket.gethostname(),
			"running": running,
		}

	def _write(self, force: bool = False) -> None:
		if self.path is None or (
			not force and time.time() - self.last_write < WRITE_EVERY_SECONDS
		):
			return
		self.last_write = time.time()
		try:
			temporary = self.path.with_name(self.path.name + ".tmp")
			temporary.write_text(
				json.dumps(self._snapshot(), allow_nan=False, default=str) + "\n"
			)
			temporary.replace(self.path)
		except (OSError, ValueError):
			pass


MAX_DEPTH = 6  # workspace/campaigns/NAME/results/RUN/live.json


def find_snapshots(root: Path) -> list[Path]:
	"""Every live.json under ``root`` (a campaign, a run, or its shards)."""
	if root.is_file():
		return [root]
	found = []
	base_depth = len(root.parts)
	for directory, subdirectories, files in os.walk(root):
		if len(Path(directory).parts) - base_depth >= MAX_DEPTH:
			subdirectories.clear()
		if LIVE_FILE in files:
			found.append(Path(directory) / LIVE_FILE)
	return sorted(found)


def _alive(pid: object, host: object = None) -> bool:
	"""True unless ``pid`` is known to be gone (a snapshot from another machine counts as alive)."""
	if not isinstance(pid, int) or (host and host != socket.gethostname()):
		return True
	try:
		os.kill(pid, 0)
	except ProcessLookupError:
		return False
	except OSError:
		return True
	return True


def read_status(
	root: Path,
	*,
	now: float | None = None,
	stale_after: float = 300.0,
	show_idle: bool = True,
	show_gone: bool = False,
	remove_gone: bool = False,
) -> tuple[list[tuple[str, float, str]], int]:
	"""What the live snapshots under ``root`` say now.

	Returns ``(entries, hidden)``: one ``(identity, reported_at, line)`` per running
	instance, and how many snapshots were left out because the process that wrote
	them is gone (killed, crashed or finished without cleaning up) unless ``show_gone``;
	with ``remove_gone`` those files are also deleted.
	"""
	now = time.time() if now is None else now
	entries: list[tuple[str, float, str]] = []
	hidden = 0
	for path in find_snapshots(root):
		try:
			data = json.loads(path.read_text())
		except (OSError, ValueError):
			continue
		age = now - data.get("updated_at", now)
		running = data.get("running", [])
		gone = not _alive(data.get("pid"), data.get("host"))
		if gone and remove_gone:
			try:
				path.unlink()
			except OSError:
				pass
			hidden += 1
			continue
		if gone and not show_gone:
			hidden += 1
			continue
		if not running:
			if show_idle:
				line = f"{path.parent.name}: nothing running (last update {clock(age)} ago)"
				entries.append((f"{path}:idle", data.get("updated_at", now), line))
			continue
		for record in running:
			silent = now - record.get("reported_at", now)
			note = ""
			if gone:
				note = f"  !! process {data['pid']} is no longer running on this machine (finished, stopped or crashed)"
			elif silent > stale_after:
				note = f"  !! no report for {clock(silent)}: a long oracle call, or the solver is stuck"
			entries.append(
				(
					f"{path}:{record['key']}",
					record.get("reported_at", now),
					describe(f"{path.parent.name} {record['label']}", record) + note,
				)
			)
	return entries, hidden


def running_children(root: Path) -> list[dict]:
	"""Solver processes of live runs on this machine that ``tpp.py stop`` can ask to stop."""
	children = []
	for path in find_snapshots(root):
		try:
			data = json.loads(path.read_text())
		except (OSError, ValueError):
			continue
		if data.get("host") not in (None, socket.gethostname()) or not _alive(data.get("pid"), data.get("host")):
			continue
		for record in data.get("running", []):
			if isinstance(record.get("child_pid"), int):
				children.append({"run": path.parent.name, **record})
	return children


def stop_child(child: dict) -> bool:
	"""Send the instance's stop signal (SIGINT for tpp-ours, SIGTERM for Fekete) to its process group."""
	number = getattr(signal, child.get("stop_signal") or "SIGINT", signal.SIGINT)
	for send, target in ((os.killpg, child["child_pid"]), (os.kill, child["child_pid"])):
		try:
			send(target, number)
			return True
		except ProcessLookupError:
			return False
		except OSError:
			continue
	return False


def render_snapshots(root: Path, **options) -> list[str]:
	"""Readable lines for the live snapshots under ``root``; flags runs that stopped reporting."""
	return [line for _, _, line in read_status(root, **options)[0]]
