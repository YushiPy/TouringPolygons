"""Own, locked Python environment for the public benchmark CLI."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "benchmarks"
VENV = PROJECT / ".venv"
READY = VENV / ".tpp-environment-ready"


def environment_fingerprint() -> str:
	digest = hashlib.sha256()
	for name in ("pyproject.toml", "uv.lock", ".python-version"):
		digest.update(name.encode())
		digest.update((PROJECT / name).read_bytes())
	return digest.hexdigest()


def environment_ready() -> bool:
	try:
		return READY.read_text().strip() == environment_fingerprint()
	except OSError:
		return False


def environment_python(directory: Path | None = None) -> Path:
	# Keep the entry point itself, not its resolved Homebrew interpreter target.
	if directory is None:
		directory = VENV
	return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_environment() -> dict[str, str]:
	environment = os.environ.copy()
	environment["PATH"] = (
		str(environment_python().parent) + os.pathsep + environment.get("PATH", "")
	)
	environment["VIRTUAL_ENV"] = str(VENV)
	environment.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
	return environment


def enter_environment(argv: Sequence[str]) -> None:
	"""Use the project's interpreter without installing anything during a run."""
	if not argv or argv[0] == "setup" or any(flag in argv for flag in ("--help", "-h")):
		return
	python = environment_python()
	if not python.is_file() or not environment_ready():
		raise SystemExit(
			"Ambiente de benchmarks não preparado. Execute:\n"
			"  python3 benchmarks/tpp.py setup\n"
			"Depois repita o comando da campanha."
		)
	environment = runtime_environment()
	if Path(sys.prefix).resolve() == VENV.resolve():
		os.environ.update(environment)
		return
	# exec preserves Ctrl+C, process identity and exit status. Import-only callers
	# (dashboard/tests) retain their environment; only the public script enters it.
	os.execve(str(python), [str(python), str(PROJECT / "tpp.py"), *argv], environment)


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		description="Prepare benchmarks/.venv from the committed uv.lock, without running solvers."
	)
	parser.add_argument(
		"--python", default="3.12", help="Python version or executable (default: 3.12)."
	)
	parser.add_argument(
		"--offline",
		action="store_true",
		help="Use only cached dependencies; fail if an item is missing.",
	)
	args = parser.parse_args(argv)
	uv = shutil.which("uv")
	if uv is None:
		print(
			"uv não encontrado. Execute ./scripts/install_dependencies.sh para preparar as dependências do projeto.",
			file=sys.stderr,
		)
		return 2
	environment = os.environ.copy()
	# Ignore an unrelated active venv or UV_PROJECT_ENVIRONMENT override.
	environment["UV_PROJECT_ENVIRONMENT"] = str(VENV)
	environment.setdefault("UV_CACHE_DIR", str(ROOT / ".cache/uv"))
	command = [
		uv,
		"sync",
		"--project",
		str(PROJECT),
		"--locked",
		"--python",
		args.python,
	]
	if args.offline:
		command.append("--offline")
	READY.unlink(missing_ok=True)
	result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
	if result.returncode:
		return result.returncode
	# Import the libraries used by generation and independent validation. This
	# verifies the actual new interpreter; it never builds or runs either solver.
	check = subprocess.run(
		[
			str(environment_python()),
			"-c",
			(
				"import sys; import shapely; import osmium; import matplotlib; "
				"print('Python:', sys.version.split()[0]); print('Ambiente:', sys.prefix); "
				"print('Shapely:', shapely.__version__); print('Matplotlib:', matplotlib.__version__)"
			),
		],
		cwd=ROOT,
		env=runtime_environment(),
		check=False,
	)
	if check.returncode:
		return check.returncode
	READY.write_text(environment_fingerprint() + "\n")
	print(
		"Setup concluído. Use python3 benchmarks/tpp.py COMANDO; a CLI seleciona benchmarks/.venv automaticamente."
	)
	return 0
