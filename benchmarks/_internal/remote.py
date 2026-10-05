"""Run benchmarks on another machine (e.g. the IME network) over SSH.

Only ``ssh`` and ``rsync`` are needed locally; the remote needs Python 3.12+
(or uv), CMake and a C++23 compiler. Nothing is installed with sudo: missing
Eigen/Boost headers are fetched into the remote checkout's ``.cache/deps``.

    tpp.py remote push HOST [--dir ~/TouringPolygons] [--campaign NAME]
    tpp.py remote setup HOST            # venv, headers, build, doctor
    tpp.py remote run HOST -- free-order NAME --threads-per-instance 8 ...
    tpp.py remote jobs HOST             # list | log JOB [-f] | stop JOB
    tpp.py remote pull HOST [NAME...]   # into workspace/campaigns/NAME@HOST

HOST is anything ``ssh`` accepts, including aliases from ~/.ssh/config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import workspace

SETTINGS = "remotes.json"
SOURCE_STAMP = ".tpp-source.json"
DEFAULT_DIR = "~/TouringPolygons"


def _settings_path() -> Path:
	return workspace.root() / SETTINGS


def load_settings() -> dict:
	try:
		return json.loads(_settings_path().read_text())
	except (OSError, json.JSONDecodeError):
		return {}


def host_settings(host: str, args: argparse.Namespace | None = None) -> dict:
	"""Remote checkout and workspace for HOST, remembered from the last push."""
	saved = load_settings().get(host, {})
	directory = getattr(args, "dir", None) or saved.get("dir") or DEFAULT_DIR
	remote_workspace = getattr(args, "workspace", None) or saved.get("workspace") or f"{directory}/benchmarks/workspace"
	return {"dir": directory, "workspace": remote_workspace}


def save_settings(host: str, settings: dict) -> None:
	data = load_settings()
	data[host] = settings
	_settings_path().parent.mkdir(parents=True, exist_ok=True)
	_settings_path().write_text(json.dumps(data, indent=2) + "\n")


def _remote_path(path: str) -> str:
	"""Quote a remote path but let the remote shell expand a leading ~/."""
	if path == "~" or path.startswith("~/"):
		return "~/" + shlex.quote(path[2:]) if len(path) > 2 else "~"
	return shlex.quote(path)


def ssh(host: str, script: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
	return subprocess.run(["ssh", host, script], check=check, text=True, capture_output=capture)


def remote_cli(settings: dict, arguments: Sequence[str]) -> str:
	"""Shell snippet running tpp.py in the remote checkout and workspace."""
	return (
		f"cd {_remote_path(settings['dir'])} && "
		f"TPP_WORKSPACE={_remote_path(settings['workspace'])} "
		f"exec \"${{TPP_PYTHON:-python3}}\" benchmarks/tpp.py {shlex.join(arguments)}"
	)


# Not needed to run benchmarks; skipped unless --full (saves ~45 MB per push).
OPTIONAL_TREES = ("docs/", "apps/")


def source_files(*, with_external: bool, full: bool) -> list[str]:
	"""Tracked plus untracked-but-not-ignored files of the working tree."""
	command = ["git", "-C", str(workspace.ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
	files = subprocess.run(command, check=True, capture_output=True).stdout.decode().split("\0")
	if with_external:
		external = ["git", "-C", str(workspace.ROOT), "ls-files", "-z", "--recurse-submodules", "third_party"]
		files += subprocess.run(external, check=True, capture_output=True).stdout.decode().split("\0")
	selected = sorted({name for name in files if name and (workspace.ROOT / name).is_file()})
	if not full:
		selected = [name for name in selected if not name.startswith(OPTIONAL_TREES)]
	return selected


def write_source_stamp() -> Path:
	"""Record which local revision was pushed; the remote has no .git."""
	state = workspace.git_state()
	diff = subprocess.run(["git", "-C", str(workspace.ROOT), "diff", "HEAD"], capture_output=True, check=False).stdout
	state["diff_sha256"] = hashlib.sha256(diff).hexdigest() if diff else None
	state["pushed_from"] = workspace.machine_state()["host"]
	state["pushed_at"] = workspace.now()
	path = workspace.root() / SOURCE_STAMP
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(state, indent=2) + "\n")
	return path


def rsync(arguments: Sequence[str]) -> None:
	command = ["rsync", "-az", "--human-readable", *arguments]
	print("+", shlex.join(command), flush=True)
	subprocess.run(command, check=True)


def command_push(args: argparse.Namespace) -> int:
	settings = host_settings(args.host, args)
	files = source_files(with_external=args.with_external, full=args.full)
	listing = workspace.root() / ".push-files"
	listing.parent.mkdir(parents=True, exist_ok=True)
	listing.write_text("\n".join(files) + "\n")
	stamp = write_source_stamp()
	ssh(args.host, f"mkdir -p {_remote_path(settings['dir'])} {_remote_path(settings['workspace'])}/campaigns")
	# Sends the working tree (including uncommitted edits), never ignored files.
	rsync(["--files-from", str(listing), f"{workspace.ROOT}/", f"{args.host}:{settings['dir']}/"])
	rsync([str(stamp), f"{args.host}:{settings['dir']}/{SOURCE_STAMP}"])
	for name in args.campaign:
		source = workspace.campaign_path(name)
		if not (source / workspace.CAMPAIGN_FILE).exists():
			raise SystemExit(f"Not a campaign: {source}")
		rsync([f"{source}/", f"{args.host}:{settings['workspace']}/campaigns/{source.name}/"])
	save_settings(args.host, settings)
	print(f"Pushed {len(files)} files to {args.host}:{settings['dir']} (workspace {settings['workspace']}).")
	return 0


def command_setup(args: argparse.Namespace) -> int:
	settings = host_settings(args.host, args)
	steps = [["setup"], ["build", "--fetch-deps"] if args.fetch_deps else ["build"], ["doctor"]]
	if args.gurobi:
		steps.insert(2, ["build", "--gurobi", "tpp-bnb-workload-benchmark"])
	for step in steps:
		print(f"== {args.host}: tpp.py {' '.join(step)}", flush=True)
		result = ssh(args.host, remote_cli(settings, step), check=False)
		if result.returncode and step != ["doctor"]:
			print(f"Remote step failed: tpp.py {' '.join(step)}", file=sys.stderr)
			if step == ["setup"]:
				print("Python 3.12+ with uv is required remotely. Without root, uv installs into ~/.local/bin;\n"
					"see https://docs.astral.sh/uv/getting-started/installation/ (or set TPP_PYTHON).", file=sys.stderr)
			return result.returncode
	return 0


def command_run(args: argparse.Namespace) -> int:
	settings = host_settings(args.host, args)
	command = list(args.command)
	if command[:1] == ["--"]:
		command = command[1:]
	if not command:
		raise SystemExit("usage: tpp.py remote run HOST [--name LABEL] -- COMMAND [ARGS...]")
	start = ["jobs", "start", *(["--name", args.name] if args.name else []), "--", *command]
	return ssh(args.host, remote_cli(settings, start), check=False).returncode


def command_jobs(args: argparse.Namespace) -> int:
	settings = host_settings(args.host, args)
	arguments = list(args.arguments) or ["list"]
	return ssh(args.host, remote_cli(settings, ["jobs", *arguments]), check=False).returncode


def command_pull(args: argparse.Namespace) -> int:
	"""Copy remote campaigns/runs into the local workspace as NAME@HOST."""
	settings = host_settings(args.host, args)
	listing = ssh(args.host, f"cd {_remote_path(settings['workspace'])} 2>/dev/null && "
		"for d in campaigns/* runs/*; do [ -d \"$d\" ] && echo \"$d\"; done; echo; pwd", capture=True)
	lines = listing.stdout.splitlines()
	remote_root = lines[-1] if lines else ""
	entries = [line for line in lines[:-1] if line]
	if args.names:
		entries = [entry for entry in entries if entry.split("/", 1)[1] in args.names]
	if not entries:
		print("Nothing to pull.")
		return 0
	label = args.host.split("@")[-1].split(".")[0]
	for entry in entries:
		section, name = entry.split("/", 1)
		local = workspace.root() / section / f"{name}@{label}"
		local.mkdir(parents=True, exist_ok=True)
		rsync([f"{args.host}:{settings['workspace']}/{entry}/", f"{local}/"])
		# Stored absolute remote paths point into the local copy afterwards.
		workspace.rewrite_paths(local, [(f"{remote_root}/{entry}", str(local))])
		campaign = local / workspace.CAMPAIGN_FILE
		if campaign.exists():
			data = json.loads(campaign.read_text())
			data["origin"] = f"remote:{args.host}"
			campaign.write_text(json.dumps(data, indent=2) + "\n")
	print(f"Pulled {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} from {args.host}.")
	return 0


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="tpp.py remote", description=__doc__.split("\n\n")[0],
		epilog="Run 'tpp.py remote ACTION --help' for the options of each action.")
	sub = parser.add_subparsers(dest="action", required=True)

	def add(name: str, help_text: str, func) -> argparse.ArgumentParser:
		command = sub.add_parser(name, help=help_text)
		command.add_argument("host", help="SSH destination, e.g. user@machine or an ~/.ssh/config alias")
		command.add_argument("--dir", help=f"remote checkout directory (default: last used or {DEFAULT_DIR})")
		command.add_argument("--workspace", help="remote workspace, e.g. a scratch disk (default: DIR/benchmarks/workspace)")
		command.set_defaults(func=func)
		return command

	push = add("push", "copy tracked files (and optionally campaigns) to HOST", command_push)
	push.add_argument("--campaign", action="append", default=[], help="also send this local campaign (repeatable)")
	push.add_argument("--with-external", action="store_true", help="include the third_party/tspn-socg submodule")
	push.add_argument("--full", action="store_true", help="also send docs/ and apps/")
	setup = add("setup", "prepare the remote venv, headers and native tools", command_setup)
	setup.add_argument("--fetch-deps", action="store_true", help="download Eigen/Boost headers (no root needed)")
	setup.add_argument("--gurobi", action="store_true", help="also build the Gurobi-enabled tools")
	run = add("run", "start a detached tpp.py command on HOST", command_run)
	run.add_argument("--name", help="short label for the job id")
	run.add_argument("command", nargs="*", help="-- COMMAND [ARGS...] (the -- is required)")
	jobs = add("jobs", "list | log JOB [-f] | stop JOB [--force] on HOST", command_jobs)
	jobs.add_argument("arguments", nargs=argparse.REMAINDER)
	pull = add("pull", "copy remote campaigns/runs into the local workspace", command_pull)
	pull.add_argument("names", nargs="*", help="only these campaign/run names")
	arguments = list(sys.argv[1:] if argv is None else argv)
	command: list[str] | None = None
	if arguments[:1] == ["run"] and "--" in arguments:
		# argparse.REMAINDER would swallow options such as --name; split on "--".
		split = arguments.index("--")
		arguments, command = arguments[:split], arguments[split + 1:]
	args = parser.parse_args(arguments)
	if command is not None:
		args.command = command
	return int(args.func(args) or 0)


if __name__ == "__main__":
	raise SystemExit(main())
