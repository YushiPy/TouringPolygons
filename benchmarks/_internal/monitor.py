"""Terminal monitor for detached jobs, running instances and their logs.

``tpp.py monitor`` is one curses screen over what ``jobs``, ``live`` and ``stop``
print: a Jobs tab (status, command, stop), an Instances tab (bounds, gap, calls
of every running solver instance, stop one) and a Logs tab (every job output
and solver/progress log, followed live). The data functions do not touch curses,
so ``--once`` prints the same overview as plain text.
"""

from __future__ import annotations

import argparse
import curses
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jobs
import live_progress
import process_guard
import workspace

LOG_NAMES = {"solver.log", "run.log", "output.log", "stderr.log", "stdout.log", live_progress.JOURNAL_FILE}
LOG_DEPTH = 9
LOG_LIMIT = 200
TAIL_BYTES = 256 * 1024
TABS = ("Jobs", "Instances", "Logs")


@dataclass
class Row:
	text: str
	detail: str = ""
	job: Path | None = None
	child: dict | None = None
	log: Path | None = None


def age(seconds: float) -> str:
	return live_progress.clock(max(0.0, seconds))


def collect_jobs() -> list[Row]:
	root = jobs.jobs_dir()
	rows = []
	for job in sorted(root.glob("*/job.json"), reverse=True) if root.exists() else []:
		data = jobs._read(job.parent)
		state = jobs.status(data)
		command = " ".join(data.get("command", [])[1:])
		detail = f"{data.get('id')}\n{command}\nstarted {data.get('started_at', '-')}  finished {data.get('finished_at', '-')}  exit {data.get('exit_code')}"
		rows.append(Row(f"{state:<11} {data.get('id', job.parent.name):<46} {command}", detail, job=job.parent,
			log=job.parent / "output.log"))
	return rows


def collect_instances(root: Path, cases: Sequence[int] = ()) -> list[Row]:
	rows = []
	now = time.time()
	for path in live_progress.find_snapshots(root):
		try:
			data = json.loads(path.read_text())
		except (OSError, ValueError):
			continue
		if not live_progress._alive(data.get("pid"), data.get("host")):
			continue
		journal = path.with_name(live_progress.JOURNAL_FILE)
		for record in data.get("running", []):
			if cases and live_progress.case_number(record["label"]) not in cases:
				continue
			line = live_progress.describe(f"{path.parent.name} {record['label']}", record)
			silent = now - record.get("reported_at", now)
			rows.append(Row(line, line + f"\nlast report {age(silent)} ago; solver pid {record.get('child_pid')}"
				f"; stop signal {record.get('stop_signal') or 'SIGINT'}",
				child={"run": path.parent.name, **record}, log=journal if journal.exists() else None))
	return rows


def find_logs(root: Path) -> list[Path]:
	found = []
	base = len(root.parts)
	for directory, subdirectories, files in os.walk(root):
		if len(Path(directory).parts) - base >= LOG_DEPTH:
			subdirectories.clear()
		found += [Path(directory) / name for name in files if name in LOG_NAMES]
	found.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
	return found[:LOG_LIMIT]


def collect_logs(root: Path) -> list[Row]:
	now = time.time()
	rows = []
	for path in find_logs(root):
		try:
			stat = path.stat()
		except OSError:
			continue
		name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
		rows.append(Row(f"{age(now - stat.st_mtime):>9} ago {stat.st_size / 1024:>9,.0f} KB  {name}", str(path), log=path))
	return rows


def tail(path: Path, max_bytes: int = TAIL_BYTES) -> list[str]:
	try:
		with path.open("rb") as file:
			file.seek(0, os.SEEK_END)
			size = file.tell()
			file.seek(max(0, size - max_bytes))
			text = file.read().decode(errors="replace")
	except OSError as error:
		return [f"(cannot read {path}: {error})"]
	lines = text.splitlines()
	return lines[1:] if size > max_bytes and lines else lines


def format_report(record: dict) -> str:
	"""One journal record as a readable line: wall-clock time, elapsed, bounds, gap and work done."""
	at = time.strftime("%H:%M:%S", time.localtime(record["at"])) if isinstance(record.get("at"), (int, float)) else "--:--:--"
	lower, upper, gap = record.get("lower_bound"), record.get("upper_bound"), record.get("gap")
	work = (f"iterations {record['calls']:,}" if record.get("nodes") is None
		else f"calls {record['calls']:,}  nodes {record['nodes']:,}  open {(record.get('open_nodes') or 0):,}")
	elapsed = live_progress.clock(record.get("elapsed_seconds") or 0)
	bounds = "  ".join(f"{name} {value:<12.8g}" if value is not None else f"{name} {'-':<12}"
		for name, value in (("LB", lower), ("UB", upper)))
	gap_text = f"gap {100 * gap:6.3g}%" if gap is not None else f"gap {'-':>6} "
	return f"{at}  {elapsed:>9}  {bounds}  {gap_text}  {work}"


def instance_lines(journal: Path | None, label: str, max_bytes: int = 4 * TAIL_BYTES) -> list[str]:
	"""Only the reports of one instance, from the run's shared progress journal."""
	if journal is None:
		return ["(no progress journal for this run)"]
	lines = []
	for raw in tail(journal, max_bytes):
		try:
			record = json.loads(raw)
		except ValueError:
			continue
		if record.get("label") == label:
			lines.append(format_report(record))
	return lines or [f"(no reports from {label} yet)"]


def snapshot(root: Path, cases: Sequence[int] = ()) -> dict[str, list[Row]]:
	return {"Jobs": collect_jobs(), "Instances": collect_instances(root, cases), "Logs": collect_logs(root)}


def stop_row(row: Row, force: bool = False) -> str:
	if row.job is not None:
		return jobs.request_stop(row.job, force)
	if row.child is not None and force:
		stubborn = process_guard.kill_tree(row.child["child_pid"])
		return f"Killed pid {row.child['child_pid']} (SIGINT first{', then SIGKILL' if stubborn else ''})."
	if row.child is not None:
		sent = live_progress.stop_child(row.child)
		return f"Asked pid {row.child['child_pid']} to stop; it reports its incumbent and bounds." if sent else "Already finished."
	return "Nothing to stop here."


def print_overview(root: Path, cases: Sequence[int] = ()) -> None:
	data = snapshot(root, cases)
	for tab in ("Jobs", "Instances"):
		print(f"== {tab} ({len(data[tab])})")
		for row in data[tab]:
			print(row.text)
	print(f"== Logs ({len(data['Logs'])}, newest first)")
	for row in data["Logs"][:10]:
		print(row.text)


# --- curses ------------------------------------------------------------------

HELP = "Tab/1-3 switch  ↑↓ move  Enter log  s stop  K kill  c case filter  r refresh  q quit"


def clip(window, y: int, text: str, attr: int = 0) -> None:
	height, width = window.getmaxyx()
	if 0 <= y < height:
		try:
			window.addnstr(y, 0, text.replace("\t", " "), width - 1, attr)
		except curses.error:
			pass


def view_log(screen, path: Path, title: str, read_lines=None) -> None:
	"""Scrollable log; follows the end until the user scrolls up (f resumes).

	``read_lines`` replaces reading ``path`` (used to show the lines of one instance only)."""
	follow, offset = True, 0
	screen.timeout(1000)
	while True:
		lines = read_lines() if read_lines is not None else tail(path)
		height, width = screen.getmaxyx()
		page = max(1, height - 2)
		top = max(0, len(lines) - page)
		offset = top if follow else min(max(0, offset), top)
		screen.erase()
		clip(screen, 0, f" {title}  [{'following' if follow else 'paused'}]", curses.A_REVERSE)
		for index, line in enumerate(lines[offset:offset + page]):
			clip(screen, 1 + index, line)
		clip(screen, height - 1, " ↑↓ PgUp/PgDn scroll  g/G start/end  f follow  q back ", curses.A_REVERSE)
		screen.refresh()
		key = screen.getch()
		if key in (ord("q"), 27, curses.KEY_LEFT):
			return
		if key in (curses.KEY_UP, ord("k")):
			follow, offset = False, offset - 1
		elif key in (curses.KEY_DOWN, ord("j")):
			offset += 1
			follow = offset >= top
		elif key == curses.KEY_PPAGE:
			follow, offset = False, offset - page
		elif key in (curses.KEY_NPAGE, ord(" ")):
			offset += page
			follow = offset >= top
		elif key == ord("g"):
			follow, offset = False, 0
		elif key in (ord("G"), ord("f")):
			follow = True


def confirm(screen, question: str) -> bool:
	height, _ = screen.getmaxyx()
	clip(screen, height - 1, f" {question} [y/N] ", curses.A_REVERSE | curses.A_BOLD)
	screen.refresh()
	screen.timeout(-1)
	return screen.getch() in (ord("y"), ord("Y"))


def ask(screen, prompt: str) -> str:
	"""One line of text typed at the bottom of the screen (Esc cancels with an empty answer)."""
	height, _ = screen.getmaxyx()
	text = ""
	screen.timeout(-1)
	curses.curs_set(1)
	try:
		while True:
			clip(screen, height - 1, " " * (screen.getmaxyx()[1] - 1))
			clip(screen, height - 1, f" {prompt}{text}", curses.A_REVERSE)
			screen.refresh()
			key = screen.getch()
			if key in (10, 13, curses.KEY_ENTER):
				return text
			if key == 27:
				return ""
			if key in (curses.KEY_BACKSPACE, 127, 8):
				text = text[:-1]
			elif 32 <= key < 127:
				text += chr(key)
	finally:
		curses.curs_set(0)


def parse_cases(text: str) -> list[int]:
	"""'131, 558 4-6' -> [131, 558, 4, 5, 6]; raises ValueError on anything else."""
	numbers: list[int] = []
	for part in text.replace(",", " ").split():
		low, _, high = part.partition("-")
		numbers += range(int(low), int(high or low) + 1)
	return numbers


def run_screen(screen, root: Path, interval: float, cases: Sequence[int] = ()) -> None:
	curses.curs_set(0)
	cases = list(cases)
	tab, selected, message = 0, [0, 0, 0], ""
	data: dict[str, list[Row]] = {}
	refreshed = 0.0
	while True:
		if time.time() - refreshed >= interval:
			data, refreshed = snapshot(root, cases), time.time()
		name = TABS[tab]
		rows = data[name]
		selected[tab] = min(selected[tab], max(0, len(rows) - 1))
		height, _ = screen.getmaxyx()
		screen.erase()
		headline = "  ".join(
			f"[{label} {len(data[label])}]" if index == tab else f" {label} {len(data[label])} "
			for index, label in enumerate(TABS)
		)
		if cases:
			headline += f"   case filter: {','.join(str(number) for number in cases)}"
		clip(screen, 0, " " + headline, curses.A_REVERSE)
		page = max(1, height - 6)
		start = min(max(0, selected[tab] - page // 2), max(0, len(rows) - page))
		if not rows:
			clip(screen, 2, {"Jobs": "No jobs. Start one with: tpp.py bench ... --detach",
				"Instances": "No running instance reports status.", "Logs": "No logs yet."}[name])
		for offset, row in enumerate(rows[start:start + page]):
			current = start + offset == selected[tab]
			clip(screen, 1 + offset, ("> " if current else "  ") + row.text, curses.A_BOLD | curses.A_REVERSE if current else 0)
		if rows:
			for offset, line in enumerate(rows[selected[tab]].detail.splitlines()[:3]):
				clip(screen, height - 5 + offset, line, curses.A_DIM)
		clip(screen, height - 2, message)
		clip(screen, height - 1, " " + HELP + " ", curses.A_REVERSE)
		screen.refresh()
		screen.timeout(500)
		key = screen.getch()
		if key == -1:
			continue
		message = ""
		if key in (ord("q"), 27):
			return
		if key in (9, curses.KEY_RIGHT):
			tab = (tab + 1) % len(TABS)
		elif key in (curses.KEY_BTAB, curses.KEY_LEFT):
			tab = (tab - 1) % len(TABS)
		elif key in (ord("1"), ord("2"), ord("3")):
			tab = key - ord("1")
		elif key in (curses.KEY_UP, ord("k")):
			selected[tab] = max(0, selected[tab] - 1)
		elif key in (curses.KEY_DOWN, ord("j")):
			selected[tab] += 1
		elif key == ord("c"):
			try:
				cases = parse_cases(ask(screen, "Cases to show (e.g. 131,558 or 4-6; empty = all): "))
				tab, refreshed = 1, 0.0
			except ValueError:
				message = "Not a list of case numbers."
		elif key in (ord("r"),):
			refreshed = 0.0
		elif rows and key in (10, 13, curses.KEY_ENTER, ord("l")):
			row = rows[selected[tab]]
			if row.child is not None and row.log is not None and row.log.exists():
				label = row.child["label"]
				view_log(screen, row.log, f"{label}   (run {row.child['run']})",
					lambda journal=row.log, label=label: instance_lines(journal, label))
			elif row.log is not None and row.log.exists():
				view_log(screen, row.log, str(row.log))
			else:
				message = "No log file for this row yet."
		elif rows and key in (ord("s"), ord("K")):
			row = rows[selected[tab]]
			force = key == ord("K")
			if row.job is None and row.child is None:
				message = "Nothing to stop here (use the Jobs or Instances tab)."
			elif force and not confirm(screen, f"Kill {row.detail.splitlines()[0][:60]}? SIGINT to its processes, then SIGKILL"):
				pass
			elif confirm(screen, f"{'Kill' if force else 'Stop'} {row.detail.splitlines()[0][:60]}?"):
				message = stop_row(row, force).splitlines()[0]
				refreshed = 0.0


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="tpp.py monitor", description=__doc__.split("\n\n")[0])
	parser.add_argument("path", nargs="?", help="a campaign NAME or a directory; default: the whole workspace")
	parser.add_argument("--once", action="store_true", help="print the overview as text and exit")
	parser.add_argument("--case", type=int, action="append", default=[], metavar="N",
		help="show only this case number in Instances; repeatable")
	parser.add_argument("--every", type=float, default=2.0, metavar="SECONDS", help="refresh interval")
	args = parser.parse_args(list(argv if argv is not None else []))
	root = workspace.campaign_path(args.path) if args.path else workspace.root()
	if args.once or not os.isatty(1):
		print_overview(root, args.case)
		return 0
	curses.wrapper(run_screen, root, max(0.5, args.every), args.case)
	return 0
