"""Fekete et al.'s published solver, unmodified, on the 558 instances of their paper.

``tpp.py fekete-setup`` prepares it, ``tpp.py fekete-paper`` runs configuration A
and ``tpp.py fekete-single-thread`` configuration B. The upstream repository
(tubs-alg/TSPN-SoCG-2026) is checked out at the pinned commit, apart from our
fork in third_party/tspn-socg, and built as its README says (``setup.py
develop``, then ``pip install -e .``). Its tracked files are never changed. The
dependency versions its README and paper name are fixed by a Conan lockfile in
an isolated Conan home (Boost 1.83, CGAL 6.0.1, fmt 11.2.0, Gurobi 12.0.0, which
its own recipe downloads). Each run calls ``solve_annotated_instance`` exactly as the paper's
experiment script ``evaluation/rq4_scalability/01_run_solver.py`` and times the
call the same way (``fekete_reference_child.py``), one process per run.

  A  fekete-paper          convex-hull-order root (CHR, ``OrderRoot``), one run at a
                           time with one thread per logical CPU of the performance
                           cores: the paper's 24 threads filled its 12-core, 24-CPU
                           Ryzen; ``--threads N`` chooses another count
  B  fekete-single-thread  CHR and LEFP (``LongestEdgePlusFurthestSite``) roots,
                           1 thread, one run per performance core
  C  ours-single-thread    our tpp-unordered with the options of the 16/10
                           comparison, 1 thread, one run per performance core

``ours-setup`` builds and tests our solver for C. Its directory carries our commit,
so a later run with changed code never mixes with an earlier one. Every tour is
validated independently at 1e-7, outside the measurement.

All use three repetitions and eps 0.001 (the paper's 0.1%). The commands run as
a background job, resume after an interruption, and wait for each other, so
only one configuration uses the machine at a time. Results stay in
``workspace/experiments/fekete-reference/``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cpu_affinity
import workspace

ROOT = workspace.ROOT
UPSTREAM_URL = "https://github.com/tubs-alg/TSPN-SoCG-2026.git"
# The head of the upstream main branch since 2026-05-15; our fork is based on it.
UPSTREAM_COMMIT = "5cd84507c2409caf47c524f60bdd3b96453f0e15"
INSTANCE_ARCHIVE = "instances/instances_socg_simplified.zip"
INSTANCE_ARCHIVE_SHA256 = "210841184500cb444f332537c4291aeff502dde4e3adc4a4ee307a80db21711f"
EXPECTED_INSTANCES = 558
EPS = 0.001
PAPER_THREADS = 24  # num_threads of the paper's experiment scripts
REPETITIONS = 3
EXTEND_REPETITIONS = 1  # the 300 s repetitions agreed on every count and within 1% in time
DEFAULT_SECONDS = 300
ORDER_SEED = 20261017
SMOKE_INSTANCES = 6

BUILD = ROOT / ".build/fekete-upstream"
SOURCE = BUILD / "source"
VENV = BUILD / "venv"
CONAN_HOME = BUILD / "conan"
SETUP_RECORD = BUILD / "setup.json"
CHILD = Path(__file__).with_name("fekete_reference_child.py")

# The [build-system] tools of the upstream pyproject, at versions that build it.
BUILD_TOOLS = (
	"conan==2.33.0", "scikit-build==0.19.1", "skbuild-conan==1.6.0", "cmake==3.31.10",
	"ninja==1.13.2", "setuptools==84.0.0", "wheel==0.48.0",
)
# setup.py asks Conan for version ranges, which resolve to the newest versions (or to
# whatever is cached). A lockfile fixes the whole graph instead: the versions the paper
# reports (Boost 1.83, CGAL 6.0.1, Gurobi 12.0.0, the version its own recipe downloads)
# and otherwise the README's minimums. fmt 12 no longer provides fmt::format through
# fmt/core.h, which the bindings include. Conan reads ./conan.lock by itself, so the
# lockfile is an untracked file in the checkout; the tracked files stay unmodified.
PINNED_REQUIRES = (
	"nlohmann_json/3.11.3", "gurobi/12.0.0", "cgal/6.0.1", "boost/1.83.0",
	"mpfr/4.2.1", "fmt/11.2.0", "gtest/1.17.0", "pybind11/2.13.6",
)
LOCKFILE = BUILD / "conan.lock"
TRANSIENT_MARKERS = (
	"too many 502 error responses", "too many 503 error responses", "too many 504 error responses",
	"connection timed out", "read timed out", "temporary failure in name resolution", "connection reset",
)
ROOT_LABELS = {"OrderRoot": "CHR", "LongestEdgePlusFurthestSite": "LEFP", "ours": "nosso"}
VALIDATION_TOLERANCE = 1e-7
# Ours as in the 16/10 comparison (tspn-compare's defaults): the binary's relative gap
# is (UB-LB)/UB, so eps/(1+eps) closes exactly when UB <= (1+eps)*LB, a stricter
# criterion than Fekete's (UB-LB) <= eps*UB + eps.
OUR_MAX_CALLS = 10**8
OUR_ARGUMENTS = (
	"--cycle", "--absolute-gap", "0", "--relative-gap", repr(EPS / (1 + EPS)), "--feasibility-tolerance", "1e-08",
	"--cycle-optimization", "cache", "--cycle-optimization", "features", "--cycle-optimization", "root",
	"--cycle-optimization", "interval",
)


@dataclass(frozen=True)
class Configuration:
	key: str
	command: str
	roots: tuple[str, ...]
	threads: int | None  # None: one per logical CPU of the performance cores
	summary: str
	solver: str = "fekete"


CONFIGURATIONS = {
	"A": Configuration("A", "fekete-paper", ("OrderRoot",), None,
		"configuração do artigo: raiz CHR, uma thread por CPU lógica dos núcleos de performance, uma execução por vez"),
	"B": Configuration("B", "fekete-single-thread", ("OrderRoot", "LongestEdgePlusFurthestSite"), 1,
		"uma thread: raízes CHR e LEFP, uma execução por núcleo de performance"),
	"C": Configuration("C", "ours-single-thread", ("ours",), 1,
		"nosso solver: uma thread, uma execução por núcleo de performance, como a B", solver="ours"),
}


def fail(message: str) -> None:
	print(f"Erro: {message}", file=sys.stderr, flush=True)
	raise SystemExit(2)


def say(message: str = "") -> None:
	print(message, flush=True)


# --- setup -------------------------------------------------------------------


def find_uv() -> str | None:
	found = shutil.which("uv")
	if found:
		return found
	for directory in (Path.home() / ".local/bin", Path.home() / ".cargo/bin"):
		if os.access(directory / "uv", os.X_OK):
			return str(directory / "uv")
	return None


def missing_system_tools() -> list[str]:
	"""Tools the build needs that cannot be installed without an administrator."""
	missing = [tool for tool in ("git", "make") if shutil.which(tool) is None]
	if shutil.which("c++") is None and shutil.which("g++") is None and shutil.which("clang++") is None:
		missing.append("c++")
	return missing


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
	return subprocess.run(["git", "-C", str(SOURCE), *arguments], capture_output=True, text=True, check=check)


def tracked_changes() -> str:
	return git("status", "--porcelain", "--untracked-files=no").stdout.strip()


def checkout_upstream() -> None:
	"""The upstream checkout at the pinned commit; tracked files are never modified."""
	if not (SOURCE / ".git").exists():
		if SOURCE.exists() and any(SOURCE.iterdir()):
			fail(f"{SOURCE} existe mas não é um checkout Git; mova-o e rode de novo.")
		SOURCE.parent.mkdir(parents=True, exist_ok=True)
		say(f"Clonando {UPSTREAM_URL}")
		subprocess.run(["git", "clone", "--quiet", UPSTREAM_URL, str(SOURCE)], check=True)
	if git("cat-file", "-e", f"{UPSTREAM_COMMIT}^{{commit}}", check=False).returncode:
		git("fetch", "--quiet", "origin", UPSTREAM_COMMIT)
	head = git("rev-parse", "HEAD", check=False).stdout.strip()
	if head != UPSTREAM_COMMIT:
		if head and tracked_changes():
			fail(f"o checkout {SOURCE} tem alterações em arquivos versionados; preserve-as e resolva manualmente.")
		git("checkout", "--quiet", "--detach", UPSTREAM_COMMIT)
	if git("rev-parse", "HEAD").stdout.strip() != UPSTREAM_COMMIT:
		fail(f"o checkout {SOURCE} não está em {UPSTREAM_COMMIT}.")
	if tracked_changes():
		fail(f"o checkout {SOURCE} tem alterações em arquivos versionados; ele deve ficar idêntico ao original.")


def instance_archive() -> Path:
	return SOURCE / INSTANCE_ARCHIVE


def verify_archive() -> None:
	archive = instance_archive()
	digest = hashlib.sha256(archive.read_bytes()).hexdigest()
	if digest != INSTANCE_ARCHIVE_SHA256:
		fail(f"SHA-256 inesperado em {archive}: {digest}")


def conan_environment() -> dict[str, str]:
	environment = os.environ.copy()
	environment["CONAN_HOME"] = str(CONAN_HOME)
	environment["PATH"] = f"{VENV / 'bin'}{os.pathsep}{environment.get('PATH', '')}"
	environment["VIRTUAL_ENV"] = str(VENV)
	environment.pop("CONAN_DEFAULT_PROFILE", None)
	return environment


def prepare_venv(uv: str) -> None:
	python = VENV / "bin/python"
	environment = os.environ | {"UV_CACHE_DIR": str(ROOT / ".cache/uv")}
	if not os.access(python, os.X_OK):
		say("Criando o ambiente Python 3.12 do solver (uv baixa o Python se faltar)")
		subprocess.run([uv, "venv", "--quiet", "--seed", "--python", "3.12", str(VENV)], check=True, env=environment)
	marker = VENV / ".fekete-build-tools"
	wanted = "\n".join(BUILD_TOOLS)
	if not marker.is_file() or marker.read_text().strip() != wanted:
		subprocess.run([uv, "pip", "install", "--quiet", "--python", str(python), *BUILD_TOOLS], check=True, env=environment)
		marker.write_text(wanted + "\n")


def conan_profile() -> str:
	"""The detected default profile of the isolated Conan home."""
	conan = str(VENV / "bin/conan")
	subprocess.run([conan, "profile", "detect", "--force", "--name", "detected"],
		capture_output=True, text=True, check=True, env=conan_environment())
	profile = (CONAN_HOME / "profiles/detected").read_text().strip() + "\n"
	path = CONAN_HOME / "profiles/default"
	if not path.is_file() or path.read_text() != profile:
		path.write_text(profile)
	return profile


def conan_lockfile() -> str:
	"""The lockfile of the pinned graph, created once per profile and set of pins."""
	conan = str(VENV / "bin/conan")
	environment = conan_environment()
	key = LOCKFILE.with_name(LOCKFILE.name + ".requires")
	wanted = "\n".join(PINNED_REQUIRES) + "\n" + (CONAN_HOME / "profiles/default").read_text()
	if not LOCKFILE.is_file() or not key.is_file() or key.read_text() != wanted:
		say("Fixando as versões das dependências do solver (lockfile do Conan)")
		# setup.py exports the Gurobi recipe of the checkout before installing; the lockfile needs it first.
		subprocess.run([conan, "export", str(SOURCE / "cmake/conan/gurobi_public")], check=True, env=environment,
			capture_output=True, text=True)
		command = [conan, "lock", "create", "--lockfile-out", str(LOCKFILE), "-pr", "default", "-s", "build_type=Release",
			"--build=missing"]
		for reference in PINNED_REQUIRES:
			command += ["--requires", reference]
		result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
		if result.returncode:
			fail(f"conan lock create falhou:\n{(result.stdout + result.stderr)[-3000:]}")
		key.write_text(wanted)
	text = LOCKFILE.read_text()
	target = SOURCE / "conan.lock"
	if not target.is_file() or target.read_text() != text:
		target.write_text(text)
	return text


def run_build_step(command: Sequence[str], environment: dict[str, str]) -> None:
	"""One build command in the upstream checkout, retried after a transient download failure."""
	for attempt in range(1, 4):
		process = subprocess.Popen(list(command), cwd=SOURCE, env=environment,
			stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
		transient = False
		assert process.stdout is not None
		for line in process.stdout:
			print(line, end="", flush=True)
			transient = transient or any(marker in line.lower() for marker in TRANSIENT_MARKERS)
		code = process.wait()
		if code == 0:
			return
		if not transient or attempt == 3:
			fail(f"o build do solver de Fekete falhou ({' '.join(command)}); veja a saída acima.")
		say(f"Falha temporária de download; tentando de novo ({attempt + 1}/3) em {5 * attempt} s")
		time.sleep(5 * attempt)


def compiler_version() -> str:
	for compiler in ("c++", "g++", "clang++"):
		if shutil.which(compiler):
			output = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=False).stdout
			return output.splitlines()[0] if output else compiler
	return "unknown"


def binding_path() -> Path | None:
	return next(iter(sorted((SOURCE / "python/tspn_bnb2/core").glob("_tspn_bindings*"))), None)


def build_fingerprint(profile: str, lockfile: str) -> str:
	payload = {"commit": UPSTREAM_COMMIT, "tools": BUILD_TOOLS, "lockfile": lockfile, "profile": profile,
		"compiler": compiler_version(), "python": str(VENV / "bin/python")}
	return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build(profile: str) -> None:
	environment = conan_environment()
	python = str(VENV / "bin/python")
	say("Compilando o solver de Fekete como o README dele indica (a primeira vez leva de 20 a 60 minutos)")
	# Untracked build output of an earlier configuration; CMake must not reuse it.
	for stale in (SOURCE / "_skbuild", SOURCE / ".conan"):
		shutil.rmtree(stale, ignore_errors=True)
	run_build_step([python, "setup.py", "develop"], environment)
	run_build_step([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation",
		"--editable", "."], environment)
	# The package imports its plotting modules (matplotlib, seaborn) when it is imported;
	# setup.py lists neither. Its requirements.txt has matplotlib; seaborn is in no list.
	run_build_step([python, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements.txt",
		"seaborn"], environment)
	if binding_path() is None:
		fail("o build não produziu o módulo _tspn_bindings")
	if tracked_changes():
		fail(f"o build alterou arquivos versionados em {SOURCE}; o código deve ficar idêntico ao original.")


def resolved_packages(lockfile: str) -> dict[str, str]:
	"""Package -> version of everything in the lockfile (requirements and build tools)."""
	data = json.loads(lockfile)
	versions = {}
	for section in ("requires", "build_requires", "python_requires"):
		for reference in data.get(section, []):
			name, _, rest = reference.partition("/")
			versions[name] = rest.split("#", 1)[0]
	return dict(sorted(versions.items()))


# --- the Gurobi license -----------------------------------------------------


def license_candidates() -> list[Path]:
	"""License files to try when Gurobi does not find one by itself."""
	candidates = [Path.home() / "gurobi.lic", Path("/opt/gurobi/gurobi.lic"), Path("/Library/gurobi/gurobi.lic")]
	for name in ("GUROBI_HOME",):
		if os.environ.get(name):
			home = Path(os.environ[name])
			candidates += [home / "gurobi.lic", home.parent / "gurobi.lic"]
	for pattern in ("opt/gurobi*/gurobi.lic", "opt/gurobi*/*/gurobi.lic", "Library/gurobi*/gurobi.lic"):
		candidates += sorted(Path("/").glob(pattern), reverse=True)
	unique: list[Path] = []
	for candidate in candidates:
		if candidate.is_file() and candidate not in unique:
			unique.append(candidate)
	return unique


def child_environment(license_file: str | None) -> dict[str, str]:
	environment = {key: value for key, value in os.environ.items() if not key.startswith("CONAN_")}
	environment["PYTHONUNBUFFERED"] = "1"
	environment.pop("PYTHONPATH", None)
	if license_file:
		environment["GRB_LICENSE_FILE"] = license_file
	return environment


def check_gurobi(license_file: str | None) -> tuple[bool, str]:
	with tempfile.TemporaryDirectory(prefix="fekete-check-") as scratch:
		output = Path(scratch) / "result.json"
		try:
			process = subprocess.run([str(VENV / "bin/python"), str(CHILD), "--check", "--output", str(output)],
				capture_output=True, text=True, timeout=300, env=child_environment(license_file), cwd=scratch, check=False)
		except subprocess.TimeoutExpired:
			return False, "the warm-up solve did not finish in 300 s"
		try:
			result = json.loads(output.read_text())
		except (OSError, json.JSONDecodeError):
			result = {"ok": False, "error": f"exit {process.returncode}"}
		log = (process.stdout + process.stderr)[-3000:]
		return bool(result.get("ok")), (result.get("error") or "") + "\n" + log


def license_problem(message: str) -> bool:
	return any(word in message.lower() for word in ("license", "licence", "grb_license_file"))


def find_license() -> str | None:
	"""``None`` when Gurobi finds a license by itself, else the file that works."""
	configured = os.environ.get("GRB_LICENSE_FILE")
	ok, message = check_gurobi(configured)
	if ok:
		return configured
	if not license_problem(message):
		fail(f"o solver de Fekete compilou, mas o teste com um problema pequeno falhou:\n{message.strip()}")
	tried = []
	for candidate in license_candidates():
		tried.append(str(candidate))
		ok, attempt = check_gurobi(str(candidate))
		if ok:
			say(f"Licença do Gurobi: {candidate}")
			return str(candidate)
		message = attempt
	grbgetkey = next(iter(sorted(CONAN_HOME.glob("p/**/bin/grbgetkey"))), None)
	fail(
		"o solver de Fekete compilou, mas o Gurobi não aceitou nenhuma licença:\n"
		f"{message.strip()}\n\n"
		+ (f"Arquivos de licença testados: {', '.join(tried)}\n" if tried else "Nenhum arquivo gurobi.lic encontrado.\n")
		+ "O Gurobi precisa de uma licença acadêmica (https://www.gurobi.com/academia/), não da licença\n"
		"restrita que vem com o pip. Depois de obtê-la (grbgetkey CHAVE, numa rede da universidade"
		+ (f"; o programa está em {grbgetkey}" if grbgetkey else "")
		+ "), coloque o gurobi.lic em ~/gurobi.lic ou defina GRB_LICENSE_FILE e rode de novo.")
	return None


# --- prepare -----------------------------------------------------------------


def read_setup() -> dict | None:
	try:
		record = json.loads(SETUP_RECORD.read_text())
	except (OSError, json.JSONDecodeError):
		return None
	return record if binding_path() is not None and record.get("commit") == UPSTREAM_COMMIT else None


def prepare(force_check: bool = True) -> dict:
	"""Check out, build (when anything changed) and verify the upstream solver."""
	missing = missing_system_tools()
	if missing:
		hint = ("No macOS, instale as Command Line Tools com 'xcode-select --install' (não pede sudo) "
			"e rode de novo." if platform.system() == "Darwin" else
			"Peça a um administrador para instalá-los (Debian/Ubuntu: sudo apt-get install build-essential git).")
		fail(f"faltam ferramentas de sistema que exigem administrador: {', '.join(missing)}. {hint}")
	uv = find_uv()
	if uv is None:
		fail("uv não encontrado; rode ./scripts/install_dependencies.sh primeiro.")
	BUILD.mkdir(parents=True, exist_ok=True)
	checkout_upstream()
	verify_archive()
	prepare_venv(uv)
	profile = conan_profile()
	lockfile = conan_lockfile()
	fingerprint = build_fingerprint(profile, lockfile)
	record = read_setup()
	if record is None or record.get("fingerprint") != fingerprint:
		build(profile)
		record = None
	else:
		say("Build do solver de Fekete em dia.")
	if record is None or force_check:
		license_file = find_license()
	else:
		license_file = record.get("license_file")
	if record is None:
		binding = binding_path()
		assert binding is not None
		freeze = subprocess.run([str(VENV / "bin/python"), "-m", "pip", "freeze", "--disable-pip-version-check"],
			capture_output=True, text=True, check=False).stdout.split()
		record = {
			"commit": UPSTREAM_COMMIT,
			"repository": UPSTREAM_URL,
			"fingerprint": fingerprint,
			"tracked_files_unmodified": True,
			"conan_profile": profile,
			"conan_lockfile": json.loads(lockfile),
			"conan_packages": resolved_packages(lockfile),
			"build_tools": list(BUILD_TOOLS),
			"compiler": compiler_version(),
			"python": subprocess.run([str(VENV / "bin/python"), "--version"], capture_output=True, text=True,
				check=False).stdout.strip(),
			"python_packages": freeze,
			"binding": str(binding.relative_to(SOURCE)),
			"binding_sha256": workspace.file_sha256(binding),
			"instance_archive_sha256": INSTANCE_ARCHIVE_SHA256,
			"built_at": workspace.now(),
			"machine": workspace.machine_state(),
		}
	record["license_file"] = license_file
	temporary = SETUP_RECORD.with_name(SETUP_RECORD.name + ".tmp")
	temporary.write_text(json.dumps(record, indent=2) + "\n")
	temporary.replace(SETUP_RECORD)
	return record


def setup_main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="tpp.py fekete-setup", description=(
		"Clone, build and verify Fekete et al.'s unmodified solver for fekete-paper and fekete-single-thread. "
		"Never uses sudo; idempotent."))
	parser.parse_args(list(argv or []))
	record = prepare()
	packages = record.get("conan_packages", {})
	say(f"Solver de Fekete pronto: {UPSTREAM_COMMIT[:7]} (original, sem modificações), "
		f"Gurobi {packages.get('gurobi', '?')}, Boost {packages.get('boost', '?')}, CGAL {packages.get('cgal', '?')}, "
		f"{record['compiler']}.")
	return 0


# --- our solver ---------------------------------------------------------------


def our_revision() -> str:
	state = workspace.git_state()
	revision = (state.get("revision") or "unknown")[:7]
	return revision + ("-dirty" if state.get("dirty") else "")


def prepare_ours() -> dict:
	"""Headers and GMP without root when missing, then tpp-unordered and its tests (once per build)."""
	import paula_campaign

	binary, arithmetic = paula_campaign.prepare_native()
	return {"binary": str(binary), "binary_sha256": workspace.file_sha256(binary), "arithmetic": arithmetic,
		"revision": our_revision(), "git": workspace.git_state(), "arguments": list(OUR_ARGUMENTS),
		"max_calls": OUR_MAX_CALLS, "compiler": compiler_version()}


def ours_setup_main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="tpp.py ours-setup", description=(
		"Build and test our solver for ours-single-thread (no sudo; idempotent)."))
	parser.parse_args(list(argv or []))
	ours = prepare_ours()
	say(f"Nosso solver pronto: {ours['binary']} (commit {ours['revision']}, aritmética {ours['arithmetic']}).")
	return 0


# --- instances and plan ------------------------------------------------------


@dataclass(frozen=True)
class Task:
	repetition: int
	instance: str
	root: str

	@property
	def key(self) -> tuple[int, str, str]:
		return (self.repetition, self.instance, self.root)


def load_instances(archive: Path) -> dict[str, int]:
	"""Instance name -> polygon count, from the upstream archive."""
	sizes = {}
	with zipfile.ZipFile(archive) as file:
		for member in file.namelist():
			if member.endswith(".json"):
				sizes[Path(member).stem] = len(json.loads(file.read(member))["polygons"])
	if len(sizes) != EXPECTED_INSTANCES:
		fail(f"{archive} tem {len(sizes)} instâncias; esperadas {EXPECTED_INSTANCES}")
	return dict(sorted(sizes.items()))


def load_polygons(archive: Path) -> dict[str, list]:
	"""Instance name -> polygons as our solver and the validator read them (outer rings; the archive has no holes)."""
	from convert_instances import convert_json_case

	polygons = {}
	with zipfile.ZipFile(archive) as file:
		for member in file.namelist():
			if member.endswith(".json"):
				polygons[Path(member).stem] = convert_json_case(member, file.read(member)).polygons
	return polygons


def trajectory_points(wkt: str | None) -> list[tuple[float, float]]:
	"""Points of the WKT LINESTRING in which Fekete's solver returns its tour."""
	match = re.match(r"\s*LINESTRING\s*\((.*)\)\s*$", wkt or "", re.S)
	if match is None:
		return []
	return [(float(x), float(y)) for x, y, *_ in (pair.split() for pair in match.group(1).split(","))]


def finite(value) -> bool:
	return isinstance(value, (int, float)) and math.isfinite(value)


def criteria(lower, upper) -> dict:
	"""Both stopping rules on the reported bounds: Fekete's and ours (UB <= (1+eps) LB)."""
	if not (finite(lower) and finite(upper)) or lower < 0 or upper < lower:
		return {"fekete": False, "strict": False}
	return {"fekete": upper - lower <= EPS * upper + EPS, "strict": upper <= (1 + EPS) * lower}


def smoke_instances(sizes: dict[str, int]) -> list[str]:
	return sorted(sorted(sizes, key=lambda name: (sizes[name], name))[:SMOKE_INSTANCES])


def plan_tasks(instances: Sequence[str], roots: Sequence[str], repetitions: int) -> list[Task]:
	"""Each repetition in its own seeded order; roots alternate between consecutive instances."""
	tasks = []
	for repetition in range(repetitions):
		order = sorted(instances)
		random.Random(ORDER_SEED + repetition).shuffle(order)
		for position, name in enumerate(order):
			ordered = roots if (position + repetition) % 2 == 0 else tuple(reversed(roots))
			tasks += [Task(repetition, name, root) for root in ordered]
	return tasks


def performance_cpus() -> list[int]:
	"""Logical CPUs of the performance cores: all of them on a CPU without efficiency cores."""
	if platform.system() == "Linux" and hasattr(os, "sched_getaffinity"):
		return sorted(cpu for core in cpu_affinity.detect_cores() if core.performance for cpu in core.cpus)
	if platform.system() == "Darwin":
		value = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"], capture_output=True, text=True,
			check=False).stdout.strip()
		if value.isdigit():
			return list(range(int(value)))
	return list(range(os.cpu_count() or 1))


def default_workers(configuration: Configuration) -> int:
	"""One run at a time with several threads; otherwise one run per performance core."""
	if configuration.threads > 1:
		return 1
	if platform.system() == "Linux" and hasattr(os, "sched_getaffinity"):
		cores = cpu_affinity.detect_cores()
		return max(1, sum(core.performance for core in cores))
	if platform.system() == "Darwin":
		for name in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
			value = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False).stdout.strip()
			if value.isdigit():
				return max(1, int(value))
	return max(1, (os.cpu_count() or 2) // 2)


def run_directory(configuration: Configuration, seconds: int, smoke: bool) -> Path:
	label = configuration.command.removeprefix("fekete-")
	if configuration.solver == "ours":
		label = f"ours-{our_revision()}"
	elif configuration.threads != 1:
		label += f"-{configuration.threads}threads"
	name = f"{configuration.key}-{label}-{seconds}s"
	return experiment_root() / (name + ("-smoke" if smoke else ""))


def experiment_root() -> Path:
	return workspace.experiments_dir() / "fekete-reference"


def base_run(configuration: Configuration, setup: dict) -> Path:
	"""The finished DEFAULT_SECONDS run of this configuration with this very build."""
	pattern = run_directory(configuration, DEFAULT_SECONDS, False).name
	if configuration.solver == "ours":
		pattern = f"{configuration.key}-ours-*-{DEFAULT_SECONDS}s"  # any commit, as long as the binary is the same
	candidates = []
	for directory in experiment_root().glob(pattern):
		try:
			plan = json.loads((directory / "plan.json").read_text())
		except (OSError, json.JSONDecodeError):
			continue
		if plan.get("setup_fingerprint") == setup["fingerprint"]:
			candidates.append((directory.stat().st_mtime, directory, plan))
	if not candidates:
		fail(f"não há rodada de {DEFAULT_SECONDS} s da configuração {configuration.key} com este mesmo build em "
			f"{experiment_root()}; --extend estende uma rodada existente.")
	_, directory, plan = max(candidates)
	if len(completed_keys(read_rows(directory / "runs.jsonl"))) < plan["tasks"]:
		fail(f"a rodada {directory.name} ainda não terminou; rode --extend depois dela.")
	return directory


def unsolved_tasks(base: Path, repetitions: int) -> list[Task]:
	"""Every instance and root that some repetition of the base run left open."""
	open_runs = sorted({(row["instance"], row["root"]) for row in latest_rows(read_rows(base / "runs.jsonl")).values()
		if not solved(row)})
	return [Task(repetition, instance, root) for repetition in range(repetitions) for instance, root in open_runs]


def seconds_text(seconds: float) -> str:
	return f"{seconds:.1f} s" if seconds >= 1 else f"{1000 * seconds:.0f} ms"


def hours(seconds: float) -> str:
	minutes = round(seconds / 60)
	return f"{minutes // 60}h{minutes % 60:02d}min"


# --- rows ---------------------------------------------------------------------


FINAL_STATUSES = ("ok", "killed")


def read_rows(path: Path) -> list[dict]:
	rows = []
	if path.exists():
		for line in path.read_text().splitlines():
			try:
				rows.append(json.loads(line))
			except json.JSONDecodeError:
				continue  # a line cut by a crash; the run is repeated
	return rows


def row_key(row: dict) -> tuple[int, str, str]:
	return (row["repetition"], row["instance"], row["root"])


def latest_rows(rows: Sequence[dict]) -> dict[tuple[int, str, str], dict]:
	latest = {}
	for row in rows:
		latest[row_key(row)] = row
	return latest


def completed_keys(rows: Sequence[dict]) -> set[tuple[int, str, str]]:
	return {key for key, row in latest_rows(rows).items() if row.get("status") in FINAL_STATUSES}


def solved(row: dict) -> bool:
	"""By the solver's own rule (Fekete: is_optimal; ours: UB <= (1+eps) LB)."""
	if row.get("status") != "ok":
		return False
	if "solved" in row:
		return bool(row["solved"])
	return bool((row.get("solution") or {}).get("is_optimal"))


# --- machine load -------------------------------------------------------------


def busy_cpu_seconds() -> float | None:
	"""CPU-seconds spent busy by the whole machine so far (Linux)."""
	try:
		fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
	except (OSError, IndexError):
		return None
	values = [int(value) for value in fields[:8]]
	idle = values[3] + values[4]
	return (sum(values) - idle) / os.sysconf("SC_CLK_TCK")


def foreign_processes(threshold: float = 50.0) -> list[str]:
	"""Other processes using at least ``threshold`` percent of a CPU right now."""
	if shutil.which("ps") is None:
		return []
	result = subprocess.run(["ps", "-A", "-o", "pid=,user=,pcpu=,comm="], capture_output=True, text=True, check=False)
	ours = {os.getpid(), os.getppid()}
	busy = []
	for line in result.stdout.splitlines():
		parts = line.split(None, 3)
		if len(parts) == 4 and parts[0].isdigit() and int(parts[0]) not in ours:
			try:
				if float(parts[2]) >= threshold:
					busy.append(f"{parts[1]} {parts[3].strip()} ({parts[2]}% CPU, pid {parts[0]})")
			except ValueError:
				continue
	return busy


# --- locks --------------------------------------------------------------------


def alive(pid: int | None) -> bool:
	if not pid:
		return False
	try:
		os.kill(pid, 0)
	except ProcessLookupError:
		return False
	except PermissionError:
		return True
	return True


def lock_holder(path: Path) -> dict | None:
	try:
		data = json.loads(path.read_text())
	except (OSError, json.JSONDecodeError):
		return None
	if data.get("host") == platform.node() and not alive(data.get("pid")):
		return None
	return data


def try_lock(path: Path, data: dict) -> bool:
	path.parent.mkdir(parents=True, exist_ok=True)
	for _ in range(2):
		try:
			descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
		except FileExistsError:
			if lock_holder(path) is not None:
				return False
			path.unlink(missing_ok=True)  # left by a process that no longer exists
			continue
		with os.fdopen(descriptor, "w") as file:
			json.dump(data, file)
		return True
	return False


def release_lock(path: Path) -> None:
	holder = lock_holder(path)
	if holder is None or holder.get("pid") == os.getpid():
		path.unlink(missing_ok=True)


def configuration_lock(configuration: Configuration) -> Path:
	return experiment_root() / f"{configuration.key}.lock"


def machine_lock() -> Path:
	return experiment_root() / "machine.lock"


# --- one run ------------------------------------------------------------------


class Runner:
	"""Runs the tasks with ``workers`` concurrent solver processes, appending one row per run."""

	def __init__(self, configuration: Configuration, seconds: int, workers: int, directory: Path,
			sizes: dict[str, int], license_file: str | None, total: int, done: int, *,
			polygons: dict[str, list] | None = None, binary: Path | None = None) -> None:
		self.configuration = configuration
		self.polygons = polygons or {}
		self.binary = binary
		self.seconds = seconds
		self.workers = workers
		self.directory = directory
		self.sizes = sizes
		self.license_file = license_file
		self.total = total
		self.done = done
		self.lock = threading.Lock()
		self.stop = threading.Event()
		self.running: dict[int, bool] = {}
		self.durations: list[float] = []
		self.started = time.monotonic()
		self.pool = (cpu_affinity.CorePool(workers, 1, "p-cores", log=lambda text, **_: say(text))
			if configuration.threads == 1 else None)
		# A run with several threads stays on the performance cores when they suffice (Linux only).
		cpus = performance_cpus() if configuration.threads != 1 and hasattr(os, "sched_setaffinity") else []
		self.cpus = cpus if 0 < (configuration.threads or 0) <= len(cpus) else []

	def hard_limit(self) -> float:
		# The solver's limit is cooperative; a run still going well past it is killed.
		return self.seconds + max(120.0, 0.5 * self.seconds)

	def run(self, task: Task) -> dict | None:
		held = {"lease": self.pool.acquire() if self.pool else None}
		try:
			return self._run(task, held)
		finally:
			if held["lease"] is not None and self.pool is not None:
				self.pool.release(held["lease"])

	def _run(self, task: Task, held: dict) -> dict | None:
		row: dict = {
			"configuration": self.configuration.key, "instance": task.instance, "polygons": self.sizes[task.instance],
			"root": task.root, "root_label": ROOT_LABELS.get(task.root, task.root), "threads": self.configuration.threads,
			"repetition": task.repetition, "time_limit": self.seconds, "eps": EPS, "host": platform.node(),
			"started_at": workspace.now(),
		}
		with tempfile.TemporaryDirectory(prefix="fekete-run-") as scratch:
			output, log_path = Path(scratch) / "result.json", Path(scratch) / "output.log"
			ours = self.configuration.solver == "ours"
			if ours:
				from unordered_runner import encode_instance

				input_path = Path(scratch) / "input.txt"
				input_path.write_text(encode_instance((0.0, 0.0), (0.0, 0.0), self.polygons[task.instance],
					OUR_MAX_CALLS, self.seconds))
				command = [str(self.binary), *OUR_ARGUMENTS]
			else:
				command = [str(VENV / "bin/python"), str(CHILD), "--archive", str(instance_archive()),
					"--instance", task.instance, "--root", task.root, "--threads", str(self.configuration.threads),
					"--time-limit", str(self.seconds), "--eps", str(EPS), "--output", str(output)]
			busy_before = busy_cpu_seconds()
			load_before = os.getloadavg()[0]
			began = time.monotonic()
			with contextlib.ExitStack() as files:
				log = files.enter_context(log_path.open("wb"))
				# Ours reads the instance on stdin and prints its result as JSON on stdout.
				stdin = files.enter_context(input_path.open("rb")) if ours else subprocess.DEVNULL
				stdout = files.enter_context(output.open("wb")) if ours else log
				process = subprocess.Popen(command, stdin=stdin, stdout=stdout,
					stderr=log if ours else subprocess.STDOUT, env=child_environment(self.license_file), cwd=scratch,
					start_new_session=True)
			with self.lock:
				self.running[process.pid] = False
			if self.pool is not None:
				self.pool.pin(process.pid, held["lease"])
			elif self.cpus:
				os.sched_setaffinity(process.pid, set(self.cpus))
			timer = threading.Timer(self.hard_limit(), self._kill, (process.pid, True))
			timer.start()
			_, status, usage = os.wait4(process.pid, 0)
			timer.cancel()
			process.returncode = os.waitstatus_to_exitcode(status)
			wall = time.monotonic() - began
			with self.lock:
				killed_by_timer = self.running.pop(process.pid, False)
			own_cpus = (usage.ru_utime + usage.ru_stime) / wall if wall > 0 else None
			affinity = self.pool.release(held["lease"], own_cpus) if self.pool is not None else None
			held["lease"] = None
			if self.stop.is_set() and not killed_by_timer:
				return None  # interrupted: the run is repeated when the command is run again
			busy_after = busy_cpu_seconds()
			rss = usage.ru_maxrss / (1024 * 1024 if platform.system() == "Darwin" else 1024)
			row["process"] = {"exit_code": process.returncode, "wall_seconds": round(wall, 3),
				"user_seconds": round(usage.ru_utime, 3), "system_seconds": round(usage.ru_stime, 3),
				"max_rss_mb": round(rss, 1)}
			if affinity is not None:
				row["cpu_affinity"] = affinity
			elif self.cpus:
				row["cpu_affinity"] = {"cpus": self.cpus, "performance": True}
			own = usage.ru_utime + usage.ru_stime
			row["machine_load"] = {"load_average_before": round(load_before, 2),
				"load_average_after": round(os.getloadavg()[0], 2)}
			if busy_before is not None and busy_after is not None and wall > 0 and self.workers == 1:
				# Other processes' CPU use during the run, in CPUs (only exact with one run at a time).
				row["machine_load"]["other_cpus"] = round(max(0.0, busy_after - busy_before - own) / wall, 3)
			try:
				result = json.loads(output.read_text())
			except (OSError, json.JSONDecodeError):
				result = {}
			if ours and result and process.returncode == 0:
				result = {"ok": True, "solve_time": result.get("seconds"), "result": result}
			if killed_by_timer:
				row["status"] = "killed"
				row["error"] = f"killed after {self.hard_limit():.0f} s without finishing"
			elif result.get("ok"):
				row["status"] = "ok"
			else:
				row["status"] = "error"
				row["error"] = result.get("error") or f"solver process exited with {process.returncode}"
			for field in ("solve_time", "solve_time_perf", "solution", "result", "type", "traceback"):
				if field in result:
					row[field] = result[field]
			if row["status"] == "ok":
				self.judge(row, task)
			else:
				row["log_tail"] = log_path.read_bytes()[-4000:].decode("utf-8", "replace")
		row["finished_at"] = workspace.now()
		return row

	def judge(self, row: dict, task: Task) -> None:
		"""Bounds, both stopping rules, the solver's own verdict and an independent tour check (unmeasured)."""
		from unordered_validation import validate_cycle

		if self.configuration.solver == "ours":
			data = row.get("result") or {}
			lower, upper, path = data.get("lower_bound"), data.get("upper_bound"), data.get("path") or []
		else:
			solution = row.get("solution") or {}
			lower, upper = solution.get("lower_bound"), solution.get("upper_bound")
			path = trajectory_points(solution.get("trajectory"))
		row["lower_bound"], row["upper_bound"] = lower, upper
		row["criteria"] = criteria(lower, upper)
		# Each solver is judged by its own rule: Fekete's is_optimal, ours UB <= (1+eps) LB.
		row["solved"] = (row["criteria"]["strict"] if self.configuration.solver == "ours"
			else bool((row.get("solution") or {}).get("is_optimal")))
		polygons = self.polygons.get(task.instance)
		if polygons is not None:
			row["validation"] = (validate_cycle(polygons, path, VALIDATION_TOLERANCE) if len(path) >= 2
				else {"valid": False, "status": "no tour"})

	def _kill(self, pid: int, by_timer: bool = False) -> None:
		with self.lock:
			if pid in self.running and by_timer:
				self.running[pid] = True
		try:
			os.killpg(pid, signal.SIGKILL)
		except (ProcessLookupError, PermissionError):
			pass

	def kill_all(self) -> None:
		with self.lock:
			pids = list(self.running)
		for pid in pids:
			self._kill(pid)

	def record(self, row: dict) -> None:
		with self.lock:
			with (self.directory / "runs.jsonl").open("a") as file:
				file.write(json.dumps(row) + "\n")
			self.done += 1
			self.durations.append(row["process"]["wall_seconds"])
			remaining = self.total - self.done
			mean = statistics.fmean(self.durations)
			eta = remaining * mean / self.workers
			if row["status"] == "ok":
				lower, upper = row.get("lower_bound"), row.get("upper_bound")
				if row.get("solved"):
					outcome = "resolvida"
				elif finite(lower) and finite(upper) and upper > 0:
					outcome = f"não fechou, gap {100 * (upper - lower) / upper:.2f}%"
				else:
					outcome = "não fechou, sem solução"
				if row.get("validation", {}).get("valid") is False:
					outcome += ", tour inválido a 1e-7"
				timing = seconds_text(row["solve_time"])
			else:
				outcome, timing = row["status"], f"{row['process']['wall_seconds']:.0f} s"
			say(f"[{self.done:>5}/{self.total}] rep {row['repetition'] + 1} {row['instance']:<36} "
				f"{row['root_label']:<4} {timing:>8}  {outcome}  | decorrido {hours(time.monotonic() - self.started)}, "
				f"restante ~{hours(eta)}")


def work(runner: Runner, tasks: list[Task]) -> None:
	queue = list(reversed(tasks))
	queue_lock = threading.Lock()

	def worker() -> None:
		while not runner.stop.is_set():
			with queue_lock:
				if not queue:
					return
				task = queue.pop()
			row = runner.run(task)
			if row is not None:
				runner.record(row)

	threads = [threading.Thread(target=worker, daemon=True) for _ in range(runner.workers)]
	for thread in threads:
		thread.start()
	try:
		while any(thread.is_alive() for thread in threads):
			for thread in threads:
				thread.join(timeout=0.5)
	except KeyboardInterrupt:
		runner.stop.set()
		runner.kill_all()
		for thread in threads:
			thread.join()
		raise


# --- summary -----------------------------------------------------------------


def summarize(configuration: Configuration, rows: Sequence[dict], tasks: Sequence[Task]) -> dict:
	latest = latest_rows(rows)
	summary: dict = {"configuration": configuration.key, "planned_runs": len(tasks), "recorded_runs": len(latest),
		"roots": {}}
	for root in configuration.roots:
		mine = [row for key, row in latest.items() if key[2] == root]
		per_repetition = {}
		for repetition in sorted({row["repetition"] for row in mine}):
			subset = [row for row in mine if row["repetition"] == repetition]
			solved_rows = [row for row in subset if solved(row)]
			per_repetition[repetition + 1] = {
				"runs": len(subset), "solved": len(solved_rows),
				"not_solved_in_time": sum(row["status"] == "ok" and not solved(row) for row in subset),
				"killed": sum(row["status"] == "killed" for row in subset),
				"errors": sum(row["status"] == "error" for row in subset),
				"valid_tours": sum(bool(row.get("validation", {}).get("valid")) for row in subset),
				"solved_seconds_sum": round(sum(row["solve_time"] for row in solved_rows), 1),
			}
		by_instance = defaultdict(list)
		for row in mine:
			by_instance[row["instance"]].append(row)
		median_times = [statistics.median(row["solve_time"] for row in group) for group in by_instance.values()
			if all(solved(row) for row in group)]
		summary["roots"][ROOT_LABELS.get(root, root)] = {
			"repetitions": per_repetition,
			"instances_solved_in_every_repetition": sum(all(solved(row) for row in group) for group in by_instance.values()),
			"instances_solved_in_some_repetition": sum(any(solved(row) for row in group) for group in by_instance.values()),
			"median_of_instance_medians_seconds": round(statistics.median(median_times), 3) if median_times else None,
		}
	return summary


def render(configuration: Configuration, seconds: int, summary: dict, setup: dict, repetitions: int = REPETITIONS) -> str:
	if configuration.solver == "ours":
		title = f"# Nosso solver, configuração {configuration.key} ({configuration.command})"
		provenance = (f"Commit `{setup.get('revision', '?')}`, `tpp-unordered {' '.join(OUR_ARGUMENTS)}`, aritmética "
			f"{setup.get('arithmetic', '?')}, {setup.get('compiler', '?')}. \"Resolvida\" é `UB ≤ (1+eps)·LB` com limites provados.")
		column = "Variante"
	else:
		packages = setup.get("conan_packages", {})
		title = f"# Fekete et al., configuração {configuration.key} ({configuration.command})"
		provenance = (f"Código original `{UPSTREAM_COMMIT[:7]}` de {UPSTREAM_URL} sem modificações; Gurobi "
			f"{packages.get('gurobi', '?')}, Boost {packages.get('boost', '?')}, CGAL {packages.get('cgal', '?')}, "
			f"{setup.get('compiler', '?')}. \"Resolvida\" é `is_optimal` do solver: `(UB−LB) ≤ eps·UB + eps`.")
		column = "Raiz"
	lines = [
		title, "",
		f"{configuration.summary}; {configuration.threads} thread(s) por execução, limite de {seconds} s, eps {EPS}, "
		f"{repetitions} repetição(ões). {provenance} Tours validados a {VALIDATION_TOLERANCE:g}, fora da medição.", "",
		f"Execuções registradas: {summary['recorded_runs']} de {summary['planned_runs']}.", "",
		f"| {column} | Repetição | Execuções | Resolvidas | Sem fechar no tempo | Mortas | Erros | Tours válidos | "
		"Soma dos tempos das resolvidas (s) |",
		"|---|---|---|---|---|---|---|---|---|",
	]
	for label, data in summary["roots"].items():
		for repetition, counts in data["repetitions"].items():
			lines.append(f"| {label} | {repetition} | {counts['runs']} | {counts['solved']} | {counts['not_solved_in_time']} | "
				f"{counts['killed']} | {counts['errors']} | {counts['valid_tours']} | {counts['solved_seconds_sum']} |")
	lines.append("")
	for label, data in summary["roots"].items():
		lines.append(f"- {label}: {data['instances_solved_in_every_repetition']} instâncias resolvidas em todas as "
			f"repetições, {data['instances_solved_in_some_repetition']} em alguma; mediana das medianas por instância "
			f"{data['median_of_instance_medians_seconds']} s.")
	return "\n".join(lines) + "\n"


def write_summary(configuration: Configuration, seconds: int, directory: Path, tasks: Sequence[Task], setup: dict) -> str:
	summary = summarize(configuration, read_rows(directory / "runs.jsonl"), tasks)
	(directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
	text = render(configuration, seconds, summary, setup, 1 + max(task.repetition for task in tasks))
	(directory / "summary.md").write_text(text)
	return text


# --- commands ---------------------------------------------------------------


def running_job(configuration: Configuration) -> str | None:
	"""Id of the background job running this configuration, if any."""
	import jobs

	if not jobs.jobs_dir().exists():
		return None
	for job in sorted(jobs.jobs_dir().iterdir(), reverse=True):
		data = jobs._read(job)
		if data.get("command", [None, None])[1:2] == [configuration.command] and jobs.status(data) == "running":
			return job.name
	return None


def job_hints(job: str) -> None:
	import jobs

	cli = f"{ROOT / 'benchmarks/.venv/bin/python'} {ROOT / 'benchmarks/tpp.py'}"
	say(f"  acompanhar: tail -f {jobs.jobs_dir() / job / 'output.log'}")
	say(f"  parar:      {cli} jobs stop {job}   (rodar o comando de novo retoma)")


def describe_plan(configuration: Configuration, seconds: int, workers: int, total: int, pending: int,
		directory: Path) -> None:
	say(f"Configuração {configuration.key} ({configuration.command}): {configuration.summary}.")
	say(f"{total} execuções ({pending} pendentes), {workers} por vez, {configuration.threads} thread(s) cada, "
		f"limite de {seconds} s cada.")
	say(f"Pior caso, se todas esgotassem o tempo: ~{hours(pending * (seconds + 5) / workers)}. "
		"O log mostra a estimativa do tempo restante a cada execução.")
	say(f"Resultados: {directory}")


def run_main(key: str, argv: Sequence[str] | None = None) -> int:
	configuration = CONFIGURATIONS[key]
	solver = "do nosso solver" if configuration.solver == "ours" else "do solver original de Fekete et al."
	parser = argparse.ArgumentParser(prog=f"tpp.py {configuration.command}", description=(
		f"Configuração {key} {solver} nas 558 instâncias do artigo de Fekete et al.: {configuration.summary}. "
		"Roda em segundo plano, retoma de onde parou e espera as outras configurações terminarem."))
	parser.add_argument("seconds", nargs="?", type=int, default=DEFAULT_SECONDS,
		help=f"limite de tempo por execução, em segundos (padrão {DEFAULT_SECONDS}, o do artigo)")
	parser.add_argument("--plan", action="store_true", help="só mostra o plano e a estimativa")
	parser.add_argument("--smoke", action="store_true",
		help=f"teste rápido: as {SMOKE_INSTANCES} menores instâncias, uma repetição, em diretório separado")
	parser.add_argument("--foreground", action="store_true", help="roda neste terminal em vez de em segundo plano")
	parser.add_argument("--extend", action="store_true", help=(
		f"roda com este limite (maior que {DEFAULT_SECONDS} s) só o que não fechou na rodada de {DEFAULT_SECONDS} s "
		f"desta configuração, com o mesmo build, em {EXTEND_REPETITIONS} repetição e num diretório próprio"))
	parser.add_argument("--repetitions", type=int, help=argparse.SUPPRESS)
	parser.add_argument("--workers", type=int, help=argparse.SUPPRESS)
	if configuration.threads is None:
		parser.add_argument("--threads", type=int, help=(
			"threads por execução (padrão: uma por CPU lógica dos núcleos de performance desta máquina; "
			f"o artigo usou {PAPER_THREADS})"))
	args = parser.parse_args(list(argv or []))
	if args.seconds < 1:
		parser.error("o limite de tempo deve ser positivo")
	if configuration.threads is None:
		threads = getattr(args, "threads", None) or len(performance_cpus())
		if threads < 1:
			parser.error("--threads deve ser positivo")
		configuration = replace(configuration, threads=threads)
		if threads > (os.cpu_count() or threads):
			say(f"ATENÇÃO: {threads} threads em {os.cpu_count()} CPUs lógicas; as threads vão disputar a CPU.")
	workers = args.workers or default_workers(configuration)
	if workers < 1 or (args.repetitions is not None and args.repetitions < 1):
		parser.error("--workers e --repetitions devem ser positivos")
	if args.extend and (args.smoke or args.seconds <= DEFAULT_SECONDS):
		parser.error(f"--extend pede um limite maior que {DEFAULT_SECONDS} s e não combina com --smoke")
	directory = run_directory(configuration, args.seconds, args.smoke)

	holder = lock_holder(configuration_lock(configuration))
	if holder is not None and holder.get("pid") == os.getpid():
		holder = None
	# A job started a moment ago may not hold its lock yet.
	job = None if args.foreground else running_job(configuration)
	if holder is not None or job is not None:
		if args.foreground:
			fail(f"a configuração {key} já está em andamento.")
		if holder is None:
			say(f"A configuração {key} acabou de ser iniciada.")
		else:
			done = len(completed_keys(read_rows(Path(holder["directory"]) / "runs.jsonl")))
			say(f"A configuração {key} já está em andamento ({Path(holder['directory']).name}, {done} execuções "
				f"concluídas, {'rodando' if holder.get('state') == 'running' else 'aguardando a outra configuração'}).")
		if job:
			job_hints(job)
		return 0

	if configuration.solver == "ours":
		if not instance_archive().is_file():
			checkout_upstream()  # the instances come from the upstream archive, as for Fekete's runs
		verify_archive()
		setup = prepare_ours()
		setup["fingerprint"] = setup["binary_sha256"]
	else:
		setup = read_setup()
		if setup is None:
			say("O solver de Fekete ainda não foi preparado; preparando agora.")
			setup = prepare()
	sizes = load_instances(instance_archive())
	if args.extend:
		base = base_run(configuration, setup)
		directory = experiment_root() / f"{base.name.removesuffix(f'-{DEFAULT_SECONDS}s')}-{args.seconds}s-unsolved"
		tasks = unsolved_tasks(base, args.repetitions or EXTEND_REPETITIONS)
		if not tasks:
			say(f"Tudo fechou em {base.name}; não há o que estender.")
			return 0
		say(f"Estende {base.name}: {len({(task.instance, task.root) for task in tasks})} execuções que não fecharam em "
			f"{DEFAULT_SECONDS} s, agora com {args.seconds} s.")
	else:
		instances = smoke_instances(sizes) if args.smoke else list(sizes)
		repetitions = args.repetitions or (1 if args.smoke else REPETITIONS)
		tasks = plan_tasks(instances, configuration.roots, repetitions)
	rows = read_rows(directory / "runs.jsonl")
	done = completed_keys(rows)
	pending = [task for task in tasks if task.key not in done]
	describe_plan(configuration, args.seconds, workers, len(tasks), len(pending), directory)
	if args.plan:
		return 0
	if not pending:
		say("\nTodas as execuções já foram concluídas.\n")
		say(write_summary(configuration, args.seconds, directory, tasks, setup))
		return 0
	plan_path = directory / "plan.json"
	if plan_path.exists():
		earlier = json.loads(plan_path.read_text())
		if earlier.get("setup_fingerprint") != setup["fingerprint"]:
			fail(f"o build do solver mudou desde o início de {directory}; mova o diretório para começar de novo.")

	if not args.foreground:
		ok, message = check_gurobi(setup.get("license_file")) if configuration.solver == "fekete" else (True, "")
		if not ok:
			fail(f"o Gurobi não funciona agora; rode ./scripts/install_dependencies.sh de novo.\n{message.strip()}")
		busy = foreign_processes()
		if busy:
			say("\nATENÇÃO: há outros processos usando a CPU agora; os tempos ficam piores enquanto eles rodarem:")
			for line in busy[:10]:
				say(f"  {line}")
		import jobs

		command = [configuration.command, str(args.seconds), *(["--smoke"] if args.smoke else []),
			*(["--extend"] if args.extend else []), *([f"--repetitions={args.repetitions}"] if args.repetitions else []),
			*([f"--workers={args.workers}"] if args.workers else []),
			*([f"--threads={configuration.threads}"] if CONFIGURATIONS[key].threads is None else []), "--foreground"]
		with contextlib.redirect_stdout(io.StringIO()) as printed, contextlib.redirect_stderr(io.StringIO()):
			code = jobs.command_start(argparse.Namespace(name=configuration.command, command=command))
		job = printed.getvalue().split()[0] if printed.getvalue().split() else None
		say(f"\nRodando em segundo plano (job {job}); pode fechar o terminal ou a sessão SSH.")
		say("Rodar o mesmo comando de novo mostra o andamento.")
		if job:
			job_hints(job)
		return code

	lock_data = {"pid": os.getpid(), "host": platform.node(), "configuration": key, "directory": str(directory),
		"state": "waiting", "since": workspace.now()}
	if not try_lock(configuration_lock(configuration), lock_data):
		fail(f"a configuração {key} já está em andamento.")
	try:
		announced = False
		while not try_lock(machine_lock(), lock_data | {"state": "running"}):
			other = lock_holder(machine_lock()) or {}
			if not announced:
				say(f"Aguardando a configuração {other.get('configuration', '?')} terminar para usar a máquina sozinha...")
				announced = True
			time.sleep(30)
		try:
			lock_data["state"] = "running"
			configuration_lock(configuration).write_text(json.dumps(lock_data))
			return execute(configuration, args.seconds, workers, directory, tasks, pending, sizes, setup,
				extends=base.name if args.extend else None)
		finally:
			release_lock(machine_lock())
	finally:
		release_lock(configuration_lock(configuration))


def execute(configuration: Configuration, seconds: int, workers: int, directory: Path, tasks: list[Task],
		pending: list[Task], sizes: dict[str, int], setup: dict, extends: str | None = None) -> int:
	directory.mkdir(parents=True, exist_ok=True)
	plan_path = directory / "plan.json"
	if not plan_path.exists():
		plan = {
			"configuration": configuration.key, "command": configuration.command, "roots": list(configuration.roots),
			"threads": configuration.threads, "time_limit": seconds, "eps": EPS,
			"repetitions": 1 + max(task.repetition for task in tasks), "extends": extends,
			"workers": workers, "order_seed": ORDER_SEED, "instances": len({task.instance for task in tasks}),
			"tasks": len(tasks), "setup_fingerprint": setup["fingerprint"], "setup": setup,
			"validation_tolerance": VALIDATION_TOLERANCE, "machine": workspace.machine_state(),
		}
		if configuration.solver == "ours":
			plan["call"] = f"tpp-unordered {' '.join(OUR_ARGUMENTS)}, instance on stdin (as tspn-compare)"
			plan["timing"] = ("the solver's own 'seconds' (the whole solve after reading the input), as in the "
				"16/10 comparison; one process per run")
		else:
			plan["call"] = ("evaluation/rq4_scalability/01_run_solver.py: solve_annotated_instance(..., "
				"branching='FarthestPoly', search='DfsBfs', rules=[], node_simplification=False, MAX_TE_TOGGLE_COUNT=1, "
				"MAX_GEOM_TOGGLE_COUNT=1, decomposition_branch=True, callback=None, initial_solution=None)")
			plan["timing"] = ("time.time() around solve_annotated_instance, as in the paper's script; one process per "
				"run, Gurobi started by a warm-up solve outside the measurement")
		plan_path.write_text(json.dumps(plan, indent=2) + "\n")
	busy = foreign_processes()
	if busy:
		say("ATENÇÃO: outros processos usando a CPU: " + "; ".join(busy[:10]))
	runner = Runner(configuration, seconds, workers, directory, sizes, setup.get("license_file"), len(tasks),
		len(tasks) - len(pending), polygons=load_polygons(instance_archive()),
		binary=Path(setup["binary"]) if configuration.solver == "ours" else None)
	say(f"Começando: {len(pending)} execuções pendentes de {len(tasks)}, {workers} por vez.")

	def interrupt(*_) -> None:
		raise KeyboardInterrupt

	# The solver processes run in their own sessions; a plain kill must stop them too.
	signal.signal(signal.SIGTERM, interrupt)
	try:
		with workspace.recorded_run(directory, kind="fekete-reference", argv=[configuration.command, str(seconds)],
				inputs=[instance_archive()], parameters={"configuration": configuration.key, "workers": workers}):
			work(runner, pending)
	except KeyboardInterrupt:
		say("\nInterrompido; as execuções concluídas estão salvas. Rodar o comando de novo retoma.")
		return 130
	say("")
	say(write_summary(configuration, seconds, directory, tasks, setup))
	say(f"Resumo em {directory / 'summary.md'}; uma linha por execução em runs.jsonl.")
	return 0


def paper_main(argv: Sequence[str] | None = None) -> int:
	return run_main("A", argv)


def single_thread_main(argv: Sequence[str] | None = None) -> int:
	return run_main("B", argv)


def ours_single_thread_main(argv: Sequence[str] | None = None) -> int:
	return run_main("C", argv)


if __name__ == "__main__":
	raise SystemExit(setup_main(sys.argv[1:]))
