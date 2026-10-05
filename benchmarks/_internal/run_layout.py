"""Where a campaign keeps its runs: ``results/<run-id>/``, whatever the problem.

Every run of a campaign has its own directory under ``results/``:

- free-order runs keep one file, ``report.json`` (plus ``live.json`` while they run);
- fixed-order runs keep ``run-index.csv`` and the per-input ``.csv``/``.md``/``.log``/``.done`` files;
- TSPN runs (``tpp.py tspn-compare``) keep ``config.json``, ``raw.jsonl``, the reports and, with
  ``--workers``, ``shard-N/`` directories.

The kind is told by what a run directory holds, not by its path. Earlier checkouts
kept free-order runs in ``results/free-order/<run-id>/`` and fixed-order files
directly in ``results/``; both layouts are still read, and ``migrate`` moves a
campaign to the current one. Only the standard library is used (the dashboard and
the TUI import this module).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

RESULTS = "results"
FREE_ORDER_REPORT = "report.json"
FIXED_ORDER_INDEX = "run-index.csv"
TSPN_CONFIG = "config.json"
LEGACY_FREE_ORDER_DIRECTORY = "free-order"
# Derived artifacts that are not runs and stay beside them.
NOT_RUNS = {"comparisons", LEGACY_FREE_ORDER_DIRECTORY}


def results_dir(campaign: Path) -> Path:
	return campaign / RESULTS


def new_run_id() -> str:
	return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def _newest_first(paths: list[Path]) -> list[Path]:
	return sorted(paths, key=lambda path: path.stat().st_mtime_ns, reverse=True)


def _run_directories(campaign: Path, marker: str) -> list[Path]:
	results = results_dir(campaign)
	if not results.is_dir():
		return []
	return _newest_first(
		[
			child
			for child in results.iterdir()
			if child.is_dir()
			and child.name not in NOT_RUNS
			and (child / marker).is_file()
		]
	)


# --- free order ---------------------------------------------------------------


def free_order_reports_in(results: Path) -> list[Path]:
	"""Every free-order report.json under a results directory, newest first (current and legacy layout)."""
	if not results.is_dir():
		return []
	current = [
		child / FREE_ORDER_REPORT
		for child in results.iterdir()
		if child.is_dir()
		and child.name not in NOT_RUNS
		and (child / FREE_ORDER_REPORT).is_file()
	]
	legacy = results / LEGACY_FREE_ORDER_DIRECTORY
	older = list(legacy.glob(f"*/{FREE_ORDER_REPORT}")) if legacy.is_dir() else []
	return _newest_first(current + older)


def free_order_reports(campaign: Path) -> list[Path]:
	"""Every free-order report.json of the campaign, newest first."""
	return free_order_reports_in(results_dir(campaign))


def latest_free_order_report(campaign: Path) -> Path | None:
	reports = free_order_reports(campaign)
	return reports[0] if reports else None


# --- fixed order --------------------------------------------------------------


def fixed_order_runs(campaign: Path) -> list[Path]:
	"""Fixed-order run directories, newest first."""
	return _run_directories(campaign, FIXED_ORDER_INDEX)


def latest_fixed_order_directory(campaign: Path) -> Path:
	"""The newest fixed-order run directory, or the legacy flat ``results/`` (which may be empty)."""
	runs = fixed_order_runs(campaign)
	return runs[0] if runs else results_dir(campaign)


def latest_fixed_order_index(campaign: Path) -> Path:
	"""Path of the newest run-index.csv; it does not exist before the first run."""
	return latest_fixed_order_directory(campaign) / FIXED_ORDER_INDEX


# --- TSPN ---------------------------------------------------------------------


def tspn_runs(campaign: Path) -> list[Path]:
	"""TSPN run directories, newest first. A sharded run that stopped before merging has only shard-N/config.json."""
	return _newest_first(
		list({*_run_directories(campaign, TSPN_CONFIG), *_run_directories(campaign, f"shard-0/{TSPN_CONFIG}")})
	)


def tspn_run_directory(campaign: Path, *, new: bool) -> Path:
	"""Where a ``tspn-compare`` run goes: a fresh ``results/<run-id>/``, or the one to resume.

	A campaign folder that predates this layout (``config.json`` directly in it) is resumed
	in place until ``workspace migrate-results`` moves it.
	"""
	if not new:
		runs = tspn_runs(campaign)
		if runs:
			return runs[0]
		if (campaign / TSPN_CONFIG).is_file():
			return campaign
	return results_dir(campaign) / new_run_id()


# --- migration ----------------------------------------------------------------


def _is_run_entry(path: Path) -> bool:
	return path.name in NOT_RUNS or (
		path.is_dir()
		and any(
			(path / marker).is_file()
			for marker in (FREE_ORDER_REPORT, FIXED_ORDER_INDEX, TSPN_CONFIG)
		)
	)


def migrate(
	campaign: Path,
	*,
	dry_run: bool = False,
	rewrite_paths: Callable[[Path, list[tuple[str, str]]], int] | None = None,
) -> list[str]:
	"""Move a campaign's results to ``results/<run-id>/`` and return what was (or would be) done.

	- ``results/free-order/<run>/`` becomes ``results/<run>/``;
	- fixed-order files directly in ``results/`` become one run directory named after the
	  time of their ``run-index.csv``, with the absolute paths they mention updated and the
	  index recorded in ``campaign.json`` pointing at the new place.
	"""
	actions: list[str] = []
	results = results_dir(campaign)
	actions.extend(_migrate_tspn(campaign, dry_run=dry_run))
	if not results.is_dir():
		return actions

	legacy = results / LEGACY_FREE_ORDER_DIRECTORY
	if legacy.is_dir():
		for run in sorted(legacy.iterdir()):
			target = results / run.name
			if target.exists():
				actions.append(
					f"SKIP {run.relative_to(campaign)}: {target.relative_to(campaign)} already exists"
				)
				continue
			actions.append(
				f"move {run.relative_to(campaign)} -> {target.relative_to(campaign)}"
			)
			if not dry_run:
				run.rename(target)
		if not dry_run and not any(legacy.iterdir()):
			legacy.rmdir()

	flat_index = results / FIXED_ORDER_INDEX
	if flat_index.is_file():
		run_id = (
			datetime.fromtimestamp(flat_index.stat().st_mtime, UTC).strftime(
				"%Y%m%d-%H%M%S-"
			)
			+ "legacy"
		)
		target = results / run_id
		leftovers = [
			child for child in sorted(results.iterdir()) if not _is_run_entry(child)
		]
		actions.append(
			f"move {len(leftovers)} fixed-order item(s) of results/ -> {target.relative_to(campaign)}"
		)
		if not dry_run:
			target.mkdir()
			for child in leftovers:
				child.rename(target / child.name)
			if rewrite_paths is not None:
				rewrite_paths(target, [(str(results), str(target))])
			_repoint_campaign_runs(campaign, run_id)
	return actions


def _migrate_tspn(campaign: Path, *, dry_run: bool) -> list[str]:
	"""A TSPN campaign folder written before the layout (config.json + raw.jsonl at its top) becomes one run."""
	config = campaign / TSPN_CONFIG
	if not config.is_file() or (campaign / "campaign.json").exists():
		return []
	run_id = (
		datetime.fromtimestamp(config.stat().st_mtime, UTC).strftime("%Y%m%d-%H%M%S-")
		+ "legacy"
	)
	target = results_dir(campaign) / run_id
	if target.exists():
		return [f"SKIP {campaign.name}: {target.relative_to(campaign)} already exists"]
	items = [child for child in sorted(campaign.iterdir()) if child.name != RESULTS]
	if not dry_run:
		target.mkdir(parents=True)
		for child in items:
			child.rename(target / child.name)
	return [f"move {len(items)} TSPN item(s) of {campaign.name}/ -> {target.relative_to(campaign)}"]


def _repoint_campaign_runs(campaign: Path, run_id: str) -> None:
	"""campaign.json records where each benchmark run's index is; follow the move."""
	path = campaign / "campaign.json"
	try:
		data = json.loads(path.read_text())
	except (OSError, ValueError):
		return
	changed = False
	for record in data.get("benchmark_runs", []):
		index = record.get("index")
		if isinstance(index, str) and re.fullmatch(
			r"results/run-index\.csv", index.replace("\\", "/")
		):
			record["index"] = f"results/{run_id}/run-index.csv"
			changed = True
	if changed:
		path.write_text(json.dumps(data, indent=2) + "\n")
