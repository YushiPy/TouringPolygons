"""Detached benchmark jobs that survive a closed terminal or SSH session.

A job runs one ``tpp.py`` command under a small supervisor process in its own
session. ``workspace/jobs/<id>/`` keeps ``job.json`` (command, PIDs, timing,
exit code) and ``output.log``. ``stop`` sends Ctrl+C to the job, which the
campaign runners handle cooperatively by checkpointing; ``--force`` kills it.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import workspace

CLI = workspace.ROOT / "benchmarks/tpp.py"


def jobs_dir() -> Path:
	return workspace.root() / "jobs"


def _now() -> str:
	return datetime.now(UTC).isoformat(timespec="seconds")


def _read(job: Path) -> dict:
	try:
		return json.loads((job / "job.json").read_text())
	except (OSError, json.JSONDecodeError):
		return {}


def _write(job: Path, data: dict) -> None:
	temporary = job / "job.json.tmp"
	temporary.write_text(json.dumps(data, indent=2) + "\n")
	temporary.replace(job / "job.json")


def _alive(pid: int | None) -> bool:
	if not pid:
		return False
	try:
		os.kill(pid, 0)
	except ProcessLookupError:
		return False
	except PermissionError:
		return True
	return True


def status(data: dict) -> str:
	code = data.get("exit_code")
	if code is not None:
		if code < 0:
			return {-signal.SIGINT: "interrupted", -signal.SIGKILL: "killed"}.get(code, f"signal {-code}")
		return "completed" if code == 0 else f"exit {code}"
	return "running" if _alive(data.get("supervisor_pid")) else "lost"


def resolve(name: str) -> Path:
	job = jobs_dir() / name
	if not (job / "job.json").exists():
		matches = sorted(jobs_dir().glob(f"*{name}*")) if jobs_dir().exists() else []
		if len(matches) != 1:
			raise SystemExit(f"Unknown job {name!r}; see 'tpp.py jobs list'.")
		job = matches[0]
	return job


def command_start(args: argparse.Namespace) -> int:
	command = list(args.command)
	if command and command[0] == "--":
		command = command[1:]
	if not command:
		raise SystemExit("usage: tpp.py jobs start [--name NAME] -- COMMAND [ARGS...]")
	label = args.name or command[0]
	job_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{label}"
	job = jobs_dir() / job_id
	job.mkdir(parents=True)
	_write(job, {
		"id": job_id,
		"command": ["tpp.py", *command],
		"cwd": str(workspace.ROOT),
		"created_at": _now(),
		"machine": workspace.machine_state(),
		"git": workspace.git_state(),
		"exit_code": None,
	})
	supervisor = subprocess.Popen(
		[sys.executable, __file__, "_supervise", str(job), *command],
		cwd=workspace.ROOT,
		stdin=subprocess.DEVNULL,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
		start_new_session=True,  # detached from this terminal: no SIGHUP on logout
	)
	data = _read(job)
	data["supervisor_pid"] = supervisor.pid
	_write(job, data)
	print(job_id)
	print(f"  log:  python3 benchmarks/tpp.py jobs log {job_id} --follow", file=sys.stderr)
	print(f"  stop: python3 benchmarks/tpp.py jobs stop {job_id}", file=sys.stderr)
	return 0


def supervise(job: Path, command: Sequence[str]) -> int:
	"""Run the job command, then record its exit status (internal)."""
	with (job / "output.log").open("ab", buffering=0) as log:
		child = subprocess.Popen(
			[sys.executable, str(CLI), *command],
			cwd=workspace.ROOT,
			stdin=subprocess.DEVNULL,
			stdout=log,
			stderr=subprocess.STDOUT,
			env=os.environ | {"PYTHONUNBUFFERED": "1"},
			# Own process group: `stop` signals the command and its solver
			# subprocesses together, like Ctrl+C in a terminal would.
			process_group=0,
		)
		data = _read(job)
		data.update(child_pid=child.pid, started_at=_now())
		_write(job, data)
		# A stop request reaches the child directly; the supervisor only waits.
		signal.signal(signal.SIGINT, signal.SIG_IGN)
		exit_code = child.wait()
	data = _read(job)
	data.update(exit_code=exit_code, finished_at=_now())
	_write(job, data)
	return exit_code


def command_list(args: argparse.Namespace) -> int:
	entries = sorted(jobs_dir().glob("*/job.json")) if jobs_dir().exists() else []
	if not entries:
		print("No jobs.")
		return 0
	for path in entries[-args.limit:]:
		data = _read(path.parent)
		command = " ".join(data.get("command", [])[1:])
		if len(command) > 70:
			command = command[:67] + "..."
		print(f"{data.get('id', path.parent.name):<44} {status(data):<10} {command}")
	return 0


def command_log(args: argparse.Namespace) -> int:
	job = resolve(args.job)
	log = job / "output.log"
	if not args.follow:
		lines = log.read_text(errors="replace").splitlines() if log.exists() else []
		print("\n".join(lines[-args.lines:]))
		return 0
	position = 0
	try:
		while True:
			if log.exists():
				with log.open("rb") as file:
					file.seek(position)
					chunk = file.read()
					position = file.tell()
				if chunk:
					sys.stdout.write(chunk.decode(errors="replace"))
					sys.stdout.flush()
			if status(_read(job)) != "running" and not chunk:
				print(f"\n[job {status(_read(job))}]")
				return 0
			time.sleep(1)
	except KeyboardInterrupt:
		return 0  # stop following; the job keeps running


def request_stop(job: Path, force: bool = False) -> str:
	"""Interrupt (or kill) a running job's process group; returns a message for the user."""
	data = _read(job)
	pid = data.get("child_pid")
	if status(data) != "running" or not _alive(pid):
		return f"Job is not running ({status(data)})."
	sent = signal.SIGKILL if force else signal.SIGINT
	os.killpg(pid, sent)
	message = f"Sent {sent.name} to {data['id']} (pid {pid})."
	if not force:
		message += "\nThe runner checkpoints and exits; use --force if it does not stop."
	return message


def command_stop(args: argparse.Namespace) -> int:
	print(request_stop(resolve(args.job), args.force))
	return 0


def main(argv: Sequence[str] | None = None) -> int:
	arguments = list(sys.argv[1:] if argv is None else argv)
	if arguments[:1] == ["_supervise"]:
		return supervise(Path(arguments[1]), arguments[2:])
	if arguments[:1] == ["start"] and "--" in arguments:
		# argparse.REMAINDER would swallow options; split on "--" explicitly.
		split = arguments.index("--")
		options = argparse.ArgumentParser(prog="tpp.py jobs start")
		options.add_argument("--name")
		parsed = options.parse_args(arguments[1:split])
		return command_start(argparse.Namespace(name=parsed.name, command=arguments[split + 1:]))
	parser = argparse.ArgumentParser(prog="tpp.py jobs", description=__doc__.split("\n\n")[0])
	sub = parser.add_subparsers(dest="action", required=True)
	start = sub.add_parser("start", help="start a detached tpp.py command")
	start.add_argument("--name", help="short label added to the job id")
	start.add_argument("command", nargs="*", help="-- COMMAND [ARGS...] (the -- is required)")
	start.set_defaults(func=command_start)
	listing = sub.add_parser("list", help="list jobs and their status")
	listing.add_argument("--limit", type=int, default=20)
	listing.set_defaults(func=command_list)
	log = sub.add_parser("log", help="show a job's output")
	log.add_argument("job")
	log.add_argument("--follow", "-f", action="store_true", help="keep printing new output")
	log.add_argument("--lines", "-n", type=int, default=40)
	log.set_defaults(func=command_log)
	stop = sub.add_parser("stop", help="interrupt a job (checkpointing runners stop cleanly)")
	stop.add_argument("job")
	stop.add_argument("--force", action="store_true", help="kill instead of interrupting")
	stop.set_defaults(func=command_stop)
	args = parser.parse_args(arguments)
	return int(args.func(args) or 0)


if __name__ == "__main__":
	sys.path.insert(0, str(Path(__file__).resolve().parent))
	raise SystemExit(main())
