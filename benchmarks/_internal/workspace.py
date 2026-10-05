"""The local benchmark workspace: one ignored root for all generated data.

Layout (``TPP_WORKSPACE`` overrides the root, e.g. a scratch disk on a lab
machine with a small home quota)::

    benchmarks/workspace/
      campaigns/<name>/     instance sets with campaign.json, from the CLI,
                            the dashboard or another machine (<name>@<host>)
      runs/<name>/          outputs of commands that are not tied to a campaign
      experiments/<name>/   free-form investigations (notes, logs, ad hoc data)
      history.jsonl         one line per tpp.py invocation

Every run directory written by the CLI carries a ``run.json`` provenance
manifest (command, machine, Git revision, tool hashes, attempts).
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "benchmarks/workspace"
MANIFEST = "run.json"
CAMPAIGN_FILE = "campaign.json"
# Pre-workspace locations, migrated by `tpp.py workspace migrate`.
LEGACY_CAMPAIGNS = ROOT / "benchmarks/campaigns"
LEGACY_RESULTS = ROOT / "benchmarks/results"
LEGACY_RESULT_DIRS = ("generated-runs", "splits", "suite-results")
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".log", ".tsv"}


def root() -> Path:
	configured = os.environ.get("TPP_WORKSPACE")
	return Path(configured).expanduser().resolve() if configured else DEFAULT_ROOT


def origin() -> str:
	"""Who created an artifact: cli, dashboard or remote:<host> (TPP_ORIGIN)."""
	return os.environ.get("TPP_ORIGIN", "cli")


def campaigns_dir() -> Path:
	return root() / "campaigns"


def runs_dir() -> Path:
	return root() / "runs"


def experiments_dir() -> Path:
	return root() / "experiments"


def regions_dir() -> Path:
	"""Downloaded OpenStreetMap extracts (.osm.pbf) and their building caches."""
	return root() / "regions"


def history_file() -> Path:
	return root() / "history.jsonl"


def campaign_path(value: str | os.PathLike[str]) -> Path:
	"""A bare NAME is a workspace campaign; anything with a separator is a path."""
	path = Path(value)
	if path.is_absolute() or path.parent != Path("."):
		return path.resolve()
	return campaigns_dir() / path


def local_data(name: str) -> Path:
	"""A local campaign or experiment directory (campaigns win), for defaults."""
	for directory in (campaigns_dir(), experiments_dir()):
		if (directory / name).exists():
			return directory / name
	return campaigns_dir() / name


def run_path(name: str) -> Path:
	"""Stable output directory for a named, resumable command run."""
	return runs_dir() / name


def timestamped_run_path(kind: str) -> Path:
	return runs_dir() / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{kind}"


# --- provenance -------------------------------------------------------------


def _git(*arguments: str) -> str | None:
	try:
		result = subprocess.run(["git", "-C", str(ROOT), *arguments], capture_output=True, text=True, check=False)
	except OSError:
		return None
	return result.stdout.strip() if result.returncode == 0 else None


def git_state() -> dict:
	revision = _git("rev-parse", "HEAD")
	if revision is None:
		# A checkout copied by `tpp.py remote push` has no .git; it carries a stamp.
		stamp = _read_json(ROOT / ".tpp-source.json")
		if stamp:
			return {**stamp, "source": "pushed copy"}
	status = _git("status", "--porcelain", "--untracked-files=no")
	return {
		"revision": revision,
		"branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
		"dirty": None if status is None else bool(status),
	}


def _cpu_and_memory() -> tuple[str | None, int | None]:
	"""CPU model and physical memory in bytes, best effort per platform."""
	if platform.system() == "Darwin":
		def sysctl(name: str) -> str | None:
			result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
			return result.stdout.strip() or None
		memory = sysctl("hw.memsize")
		return sysctl("machdep.cpu.brand_string"), int(memory) if memory and memory.isdigit() else None
	model = memory = None
	with contextlib.suppress(OSError):
		for line in Path("/proc/cpuinfo").read_text().splitlines():
			if line.startswith("model name"):
				model = line.split(":", 1)[1].strip()
				break
	with contextlib.suppress(OSError, ValueError):
		for line in Path("/proc/meminfo").read_text().splitlines():
			if line.startswith("MemTotal:"):
				memory = int(line.split()[1]) * 1024
				break
	return model, memory


def machine_state() -> dict:
	"""Identify the machine of a run, so results from different hosts never mix silently."""
	cpu_model, memory = _cpu_and_memory()
	try:
		load = [round(value, 2) for value in os.getloadavg()]
	except OSError:
		load = None
	return {
		"host": socket.gethostname(),
		"user": getpass.getuser(),
		"system": f"{platform.system()} {platform.release()}",
		"machine": platform.machine(),
		"cpu_model": cpu_model,
		"cpus": os.cpu_count(),
		"memory_bytes": memory,
		"load_average": load,
		"python": sys.version.split()[0],
	}


def file_sha256(path: Path) -> str | None:
	try:
		return hashlib.sha256(path.read_bytes()).hexdigest()
	except OSError:
		return None


def now() -> str:
	return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_json(path: Path, data: dict) -> None:
	temporary = path.with_name(path.name + ".tmp")
	temporary.write_text(json.dumps(data, indent=2) + "\n")
	temporary.replace(path)


@contextlib.contextmanager
def recorded_run(
	directory: Path,
	*,
	kind: str,
	argv: Sequence[str] | None = None,
	tools: Iterable[Path] = (),
	inputs: Iterable[Path] = (),
	parameters: dict | None = None,
	manifest_name: str = MANIFEST,
) -> Iterator[dict]:
	"""Write/extend ``run.json`` around one attempt of a (resumable) run.

	Resuming the same directory appends an attempt, so a run executed over
	several sessions or machines keeps its whole history.
	"""
	directory.mkdir(parents=True, exist_ok=True)
	path = directory / manifest_name
	try:
		manifest = json.loads(path.read_text())
	except (OSError, json.JSONDecodeError):
		manifest = {"schema_version": 1, "kind": kind, "created_at": now(), "attempts": []}
	attempt = {
		"started_at": now(),
		"finished_at": None,
		"status": "running",
		"command": ["tpp.py", *(sys.argv[1:] if argv is None else argv)],
		"cwd": os.getcwd(),
		"origin": origin(),
		"machine": machine_state(),
		"git": git_state(),
		"tools": {str(tool): file_sha256(Path(tool)) for tool in tools},
		"inputs": {str(item): file_sha256(Path(item)) for item in inputs},
		"parameters": parameters or {},
	}
	manifest.setdefault("attempts", []).append(attempt)
	manifest["status"] = "running"
	_atomic_json(path, manifest)
	started = time.monotonic()
	try:
		yield attempt
	except KeyboardInterrupt:
		attempt["status"] = "interrupted"
		raise
	except BaseException:
		attempt["status"] = "failed"
		raise
	else:
		if attempt["status"] == "running":
			attempt["status"] = "completed"
	finally:
		attempt["finished_at"] = now()
		attempt["elapsed_seconds"] = round(time.monotonic() - started, 3)
		manifest["status"] = attempt["status"]
		manifest["updated_at"] = attempt["finished_at"]
		_atomic_json(path, manifest)


def append_history(argv: Sequence[str], exit_code: int | None, elapsed: float) -> None:
	"""Best-effort journal of every CLI invocation; never fails the command."""
	entry = {
		"at": now(),
		"command": ["tpp.py", *argv],
		"exit_code": exit_code,
		"elapsed_seconds": round(elapsed, 3),
		"host": socket.gethostname(),
		"origin": origin(),
		"revision": _git("rev-parse", "--short", "HEAD"),
	}
	try:
		history_file().parent.mkdir(parents=True, exist_ok=True)
		with history_file().open("a") as file:
			file.write(json.dumps(entry) + "\n")
	except OSError:
		pass


# --- listing ---------------------------------------------------------------


def _size(path: Path) -> int:
	if path.is_file():
		return path.stat().st_size
	return sum(item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def _human(size: int) -> str:
	value = float(size)
	for unit in ("B", "K", "M", "G"):
		if value < 1024 or unit == "G":
			return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
		value /= 1024
	return f"{size}B"


def _read_json(path: Path) -> dict:
	try:
		data = json.loads(path.read_text())
	except (OSError, json.JSONDecodeError):
		return {}
	return data if isinstance(data, dict) else {}


def jsonable(values: dict) -> dict:
	"""argparse values as JSON (paths as strings), for run.json parameters."""
	return {key: (str(value) if isinstance(value, Path) else value) for key, value in values.items()
		if isinstance(value, (str, int, float, bool, type(None), Path, list))}


def describe(directory: Path) -> dict:
	campaign = _read_json(directory / CAMPAIGN_FILE)
	manifest = _read_json(directory / MANIFEST)
	attempts = manifest.get("attempts") or [{}]
	return {
		"name": directory.name,
		"type": campaign.get("type") or manifest.get("kind") or "-",
		"origin": campaign.get("origin") or attempts[-1].get("origin") or "-",
		"status": manifest.get("status") or ("campaign" if campaign else "-"),
		"host": (attempts[-1].get("machine") or {}).get("host", "-"),
		"modified": datetime.fromtimestamp(directory.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
	}


def command_list(args: argparse.Namespace) -> int:
	sections = {"campaigns": campaigns_dir(), "runs": runs_dir(), "experiments": experiments_dir()}
	selected = [args.section] if args.section else list(sections)
	print(f"Workspace: {root()}")
	for section in selected:
		directory = sections[section]
		entries = sorted(path for path in directory.glob("*") if path.is_dir()) if directory.exists() else []
		print(f"\n{section} ({len(entries)})")
		for entry in entries:
			info = describe(entry)
			size = _human(_size(entry)) if args.sizes else ""
			print(f"  {info['name']:<48} {info['type']:<22} {info['origin']:<14} "
				f"{info['status']:<11} {info['modified']} {size}".rstrip())
	return 0


# --- migration from the pre-workspace layout --------------------------------


def rewrite_paths(directory: Path, replacements: list[tuple[str, str]]) -> int:
	"""Rewrite old absolute/relative location prefixes inside text outputs."""
	changed = 0
	for path in directory.rglob("*"):
		if not path.is_file() or path.is_symlink() or path.suffix not in TEXT_SUFFIXES:
			continue
		try:
			text = path.read_text()
		except (OSError, UnicodeDecodeError):
			continue
		updated = text
		for old, new in replacements:
			# Whole path components only: "run-1" must not rewrite "run-10".
			updated = re.sub(re.escape(old) + r"(?=[/\"'\s,;)\]]|$)", lambda _match, new=new: new, updated)
		if updated != text:
			path.write_text(updated)
			changed += 1
	return changed


def _fix_campaign_inputs(campaign: Path, old_location: Path) -> None:
	"""Keep relative input references that point outside the campaign valid."""
	path = campaign / CAMPAIGN_FILE
	data = _read_json(path)
	changed = False
	for record in data.get("inputs", []):
		file_value = record.get("file") if isinstance(record, dict) else None
		if not isinstance(file_value, str) or Path(file_value).is_absolute():
			continue
		target = (old_location / file_value).resolve()
		if target.is_relative_to(old_location.resolve()):
			continue
		record["file"] = os.path.relpath(target, campaign)
		changed = True
	if changed:
		_atomic_json(path, data)


def _move(source: Path, destination: Path, *, dry_run: bool) -> Path:
	final = destination
	suffix = 2
	while final.exists():
		final = destination.with_name(f"{destination.name}-{suffix}")
		suffix += 1
	print(f"  {source.relative_to(ROOT)} -> {final.relative_to(ROOT) if final.is_relative_to(ROOT) else final}")
	if not dry_run:
		final.parent.mkdir(parents=True, exist_ok=True)
		shutil.move(str(source), str(final))
	return final


def link_legacy_locations() -> None:
	"""Old commands and notes keep working through ignored compatibility links."""
	for legacy, target in ((LEGACY_CAMPAIGNS, campaigns_dir()), (LEGACY_RESULTS, runs_dir())):
		if legacy.is_dir() and not legacy.is_symlink() and all(item.name == ".DS_Store" for item in legacy.iterdir()):
			shutil.rmtree(legacy)
		if not legacy.exists():
			target.mkdir(parents=True, exist_ok=True)
			legacy.symlink_to(os.path.relpath(target.resolve(), legacy.parent.resolve()))
			print(f"  linked {legacy.relative_to(ROOT)} -> {os.readlink(legacy)}")


def command_migrate(args: argparse.Namespace) -> int:
	"""Move benchmarks/campaigns and benchmarks/results into the workspace."""
	dry = args.dry_run
	moves: list[tuple[Path, Path, Path]] = []  # (old, new, old parent prefix)
	if LEGACY_CAMPAIGNS.is_dir() and not LEGACY_CAMPAIGNS.is_symlink():
		for entry in sorted(LEGACY_CAMPAIGNS.iterdir()):
			if entry.name.startswith("."):
				continue
			kind = campaigns_dir() if (entry / CAMPAIGN_FILE).exists() else experiments_dir()
			moves.append((entry, kind / entry.name, LEGACY_CAMPAIGNS))
	if LEGACY_RESULTS.is_dir() and not LEGACY_RESULTS.is_symlink():
		for entry in sorted(LEGACY_RESULTS.iterdir()):
			if not entry.name.startswith("."):
				moves.append((entry, runs_dir() / entry.name, LEGACY_RESULTS))
	for name in LEGACY_RESULT_DIRS:
		legacy = ROOT / "benchmarks" / name
		if legacy.is_dir() and not legacy.is_symlink():
			moves.append((legacy, runs_dir() / f"legacy-{name}", legacy.parent))
	legacy_regions = ROOT / "packages/instance-generation/regions"
	if legacy_regions.is_dir():
		for entry in sorted(legacy_regions.iterdir()):
			moves.append((entry, regions_dir() / entry.name, legacy_regions))
	for extra in args.include:
		source = (ROOT / extra).resolve()
		if not source.is_dir() or not source.is_relative_to(ROOT):
			raise SystemExit(f"--include must name a directory inside the repository: {extra}")
		moves.append((source, experiments_dir() / source.name, source.parent))
	if not moves:
		print("Nothing to migrate.")
		link_legacy_locations()
		return 0
	print(("Would move" if dry else "Moving") + f" {len(moves)} entries into {root()}:")
	moved = [(old, _move(old, new, dry_run=dry)) for old, new, _ in moves]
	if dry:
		return 0
	replacements = []
	for old, new in moved:
		replacements.append((str(old), str(new)))
		replacements.append((os.path.relpath(old, ROOT), os.path.relpath(new, ROOT)))
	replacements.sort(key=lambda pair: len(pair[0]), reverse=True)
	for old, new in moved:
		if (new / CAMPAIGN_FILE).exists():
			_fix_campaign_inputs(new, old)
		count = rewrite_paths(new, replacements)
		if count:
			print(f"  rewrote stored paths in {count} file(s) of {new.name}")
	link_legacy_locations()
	return 0


def command_migrate_results(args: argparse.Namespace) -> int:
	"""Move every campaign's runs to results/<run>/ (see run_layout)."""
	import run_layout

	campaigns = sorted(path for path in campaigns_dir().iterdir() if (path / CAMPAIGN_FILE).exists()) if campaigns_dir().is_dir() else []
	total = 0
	for campaign in campaigns:
		actions = run_layout.migrate(campaign, dry_run=args.dry_run, rewrite_paths=rewrite_paths)
		if actions:
			print(f"{campaign.name}:")
			for action in actions:
				print(f"  {'would ' if args.dry_run else ''}{action}")
		total += len(actions)
	print(("Would apply" if args.dry_run else "Applied") + f" {total} change(s) in {len(campaigns)} campaign(s)." if total else "Every campaign already uses results/<run>/.")
	return 0


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="tpp.py workspace", description=__doc__.split("\n\n")[0])
	sub = parser.add_subparsers(dest="action")
	listing = sub.add_parser("list", help="list campaigns, runs and experiments")
	listing.add_argument("section", nargs="?", choices=("campaigns", "runs", "experiments"))
	listing.add_argument("--sizes", action="store_true", help="show disk usage (slower)")
	listing.set_defaults(func=command_list)
	migrate = sub.add_parser("migrate", help="move benchmarks/campaigns and benchmarks/results into the workspace")
	migrate.add_argument("--dry-run", action="store_true", help="only print the moves")
	migrate.add_argument("--include", action="append", default=[], metavar="DIR",
		help="also move this repository directory into experiments/ (repeatable)")
	migrate.set_defaults(func=command_migrate)
	layout = sub.add_parser("migrate-results", help="move each campaign's runs to results/<run>/")
	layout.add_argument("--dry-run", action="store_true", help="only print the moves")
	layout.set_defaults(func=command_migrate_results)
	where = sub.add_parser("path", help="print the workspace root")
	where.set_defaults(func=lambda _args: print(root()) or 0)
	args = parser.parse_args(argv)
	if not getattr(args, "func", None):
		args = parser.parse_args(["list", *([] if argv is None else argv)])
	return int(args.func(args) or 0)


if __name__ == "__main__":
	raise SystemExit(main())
