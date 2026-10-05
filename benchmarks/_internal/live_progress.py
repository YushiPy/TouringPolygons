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
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

WINDOW = 12  # reports kept per running instance for trend and estimate
WRITE_EVERY_SECONDS = 5.0
LIVE_FILE = "live.json"


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
		}
		with self.lock:
			self.running[key] = entry

		def report(raw: dict) -> None:
			try:
				self._update(key, entry, raw)
			except Exception:
				pass

		return report

	def finish(self, key: str) -> None:
		with self.lock:
			self.running.pop(key, None)
			self._write(force=True)

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
			if entry["max_calls"] and entry["max_calls"] < 10**15:
				record["calls_used"] = min(1.0, record["calls"] / entry["max_calls"])
			record["reported_at"] = time.time()
			entry["latest"] = record
			line = describe(entry["label"], record)
			self._write()
		self.echo(line)

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
						**latest,
					}
				)
		return {"updated_at": time.time(), "pid": os.getpid(), "running": running}

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


def _alive(pid: object) -> bool:
	"""True unless ``pid`` is known to be gone (a snapshot from another machine counts as alive)."""
	if not isinstance(pid, int):
		return True
	try:
		os.kill(pid, 0)
	except ProcessLookupError:
		return False
	except OSError:
		return True
	return True


def render_snapshots(
	root: Path,
	*,
	now: float | None = None,
	stale_after: float = 300.0,
	show_idle: bool = True,
) -> list[str]:
	"""Readable lines for the live snapshots under ``root``; flags runs that stopped reporting."""
	now = time.time() if now is None else now
	lines = []
	for path in find_snapshots(root):
		try:
			data = json.loads(path.read_text())
		except (OSError, ValueError):
			continue
		age = now - data.get("updated_at", now)
		running = data.get("running", [])
		if not running:
			if show_idle:
				lines.append(
					f"{path.parent.name}: nothing running (last update {clock(age)} ago)"
				)
			continue
		gone = not _alive(data.get("pid"))
		for record in running:
			silent = now - record.get("reported_at", now)
			note = ""
			if gone:
				note = f"  !! process {data['pid']} is no longer running on this machine (finished, stopped or crashed)"
			elif silent > stale_after:
				note = f"  !! no report for {clock(silent)}: a long oracle call, or the solver is stuck"
			lines.append(
				describe(f"{path.parent.name} {record['label']}", record) + note
			)
	return lines
