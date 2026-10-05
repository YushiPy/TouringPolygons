"""``tpp.py free-compare``: prepare and run the 558-case Fekete free-order comparison.

Builds the selected solver(s), creates the campaign for the pinned Fekete suite and runs or
resumes it with ``tpp.py free-order``. It replaces ``scripts/run_comparison.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import native_build
import workspace

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_SOURCE = ROOT / "third_party/tspn-socg"
SUITE = ROOT / "benchmarks/suites/fekete-instances.bin"
EXPECTED_SUITE_SHA256 = "aa442e0546567461621b7fcdb9596ba7b3cc4094929d23fb9bb38d1093c88737"
EXPECTED_CASES = 558
DEFAULT_CAMPAIGN = "fekete-free-order-comparison-v1"
EPS = 0.001
PATCHES = (
	ROOT / "patches/tspn-socg-fmt-format-header.patch",
	ROOT / "patches/tspn-socg-directional-oracle-variant.patch",
)
# `git status --porcelain` of the submodule: only the managed patches may show.
FMT_PATCH_STATUS = " M python/tspn_bnb2/core/_tspn_bindings.cpp"
PATCHED_STATUS = " M CMakeLists.txt\n M python/tspn_bnb2/core/_tspn_bindings.cpp"
PIP_PACKAGES = (
	"conan>=2.0.0", "setuptools", "scikit-build>=0.18.0", "skbuild-conan", "cmake>=3.23,<4", "ninja",
)
TRANSIENT_MARKERS = (
	"too many 502 error responses",
	"too many 503 error responses",
	"too many 504 error responses",
	"connection timed out",
	"read timed out",
	"temporary failure in name resolution",
)
CAMPAIGN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SOLVERS = {"tpp-ours": (True, False), "tpp-fekete": (False, True), "both": (True, True)}


def fail(message: str) -> None:
	print(f"Error: {message}", file=sys.stderr)
	raise SystemExit(2)


def git(*arguments: str, cwd: Path = ROOT, check: bool = False) -> subprocess.CompletedProcess:
	return subprocess.run(["git", "-C", str(cwd), *arguments], capture_output=True, text=True, check=check)


def git_output(*arguments: str, cwd: Path = ROOT) -> str:
	"""Output of a git command with only the trailing newline removed (a leading space is meaningful)."""
	return git(*arguments, cwd=cwd).stdout.rstrip("\n")


# --- inputs -----------------------------------------------------------------


def verify_suite() -> None:
	if not SUITE.is_file():
		fail(f"Fekete suite is missing: {SUITE}")
	actual = hashlib.sha256(SUITE.read_bytes()).hexdigest()
	if actual != EXPECTED_SUITE_SHA256:
		fail(f"Fekete suite SHA-256 mismatch: expected {EXPECTED_SUITE_SHA256}, got {actual}")


def pinned_revision() -> str:
	for line in git_output("ls-tree", "HEAD", "--", "third_party/tspn-socg").splitlines():
		fields = line.split()
		if fields and fields[0] == "160000":
			return fields[2]
	return ""


def prepare_submodule() -> str:
	"""Submodule at the pinned revision with exactly the managed compatibility patches applied."""
	expected = pinned_revision()
	if not expected:
		fail("third_party/tspn-socg is not pinned as a Git submodule in HEAD")
	initialized = False
	if (EXTERNAL_SOURCE / ".git").exists():
		current = git_output("rev-parse", "HEAD", cwd=EXTERNAL_SOURCE)
		changes = git_output("status", "--porcelain", "--untracked-files=all", cwd=EXTERNAL_SOURCE)
		if current == expected:
			if changes not in ("", FMT_PATCH_STATUS, PATCHED_STATUS):
				fail("the Fekete submodule has local changes beyond the managed compatibility patches; preserve them and resolve manually")
			initialized = True
		elif changes:
			fail("the Fekete submodule has local changes; preserve them and check it out to the pinned revision manually")
	if not initialized:
		git("submodule", "update", "--init", "--recursive", "--", "third_party/tspn-socg", check=True)
	current = git_output("rev-parse", "HEAD", cwd=EXTERNAL_SOURCE)
	if current != expected:
		fail(f"Fekete submodule is at {current}; expected pinned revision {expected}")
	for patch in PATCHES:
		if git("apply", "--reverse", "--check", str(patch), cwd=EXTERNAL_SOURCE).returncode == 0:
			print(f"Fekete compatibility patch already applied: {patch.name}")
		elif git("apply", "--check", str(patch), cwd=EXTERNAL_SOURCE).returncode == 0:
			git("apply", str(patch), cwd=EXTERNAL_SOURCE, check=True)
			print(f"Applied local Fekete compatibility patch: {patch.name}")
		else:
			fail(f"the pinned Fekete source does not match compatibility patch {patch.name}; preserve it and inspect the submodule revision")
	if git_output("status", "--porcelain", "--untracked-files=all", cwd=EXTERNAL_SOURCE) != PATCHED_STATUS:
		fail("the Fekete submodule has local changes beyond the managed compatibility patches; preserve them and resolve manually")
	return current


# --- the Fekete Python binding ----------------------------------------------


def run_quiet(command: Sequence[str], **options) -> bool:
	return subprocess.run(command, capture_output=True, **options).returncode == 0


def _build_with_retries(python: Path) -> None:
	"""``setup.py develop``, retrying when a dependency download fails for a transient network reason."""
	for attempt in range(1, 4):
		process = subprocess.Popen(
			[str(python), "setup.py", "develop"], cwd=EXTERNAL_SOURCE,
			stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
		)
		transient = False
		assert process.stdout is not None
		for line in process.stdout:
			print(line, end="", flush=True)
			transient = transient or any(marker in line.lower() for marker in TRANSIENT_MARKERS)
		code = process.wait()
		if code == 0:
			return
		if not transient or attempt == 3:
			raise SystemExit(code or 1)
		delay = 5 * attempt
		print(f"Transient dependency download failure; retrying setup ({attempt + 1}/3) in {delay}s...", file=sys.stderr, flush=True)
		time.sleep(delay)
	fail("Fekete dependency setup failed after three attempts.")


def build_fekete(revision: str, build_jobs: int) -> None:
	"""Virtual environment, Conan dependencies and the compiled binding, skipped when nothing changed."""
	venv = EXTERNAL_SOURCE / ".venv"
	python = venv / "bin/python"
	if not os.access(python, os.X_OK):
		subprocess.run([getattr(sys, "_base_executable", sys.executable), "-m", "venv", str(venv)], check=True)
	tools_present = all(os.access(venv / "bin" / name, os.X_OK) for name in ("conan", "cmake", "ninja"))
	if not tools_present or not run_quiet([str(python), "-c", "import skbuild, skbuild_conan"]):
		subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", *PIP_PACKAGES], check=True)
	os.environ["PATH"] = f"{venv / 'bin'}{os.pathsep}{os.environ['PATH']}"
	os.environ["TPP_BUILD_JOBS"] = os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(build_jobs)
	conan = str(venv / "bin/conan")
	if not run_quiet([conan, "profile", "show", "-pr", "default"]):
		subprocess.run([conan, "profile", "detect"], check=True)

	profile = subprocess.run([conan, "profile", "show", "-pr", "default"], capture_output=True, text=True, check=True).stdout.rstrip("\n")
	cmake_version = subprocess.run([str(venv / "bin/cmake"), "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
	conan_version = subprocess.run([conan, "--version"], capture_output=True, text=True, check=True).stdout.strip()
	fingerprint = subprocess.run(
		[str(python), str(Path(__file__).with_name("fekete_fingerprint.py")), str(EXTERNAL_SOURCE), revision, str(ROOT), profile, cmake_version, conan_version],
		capture_output=True, text=True, check=True,
	).stdout.strip()
	marker = venv / ".touring-polygons-fekete-build-fingerprint"
	binding = next(iter((EXTERNAL_SOURCE / "python/tspn_bnb2/core").glob("_tspn_bindings*.so")), None)
	needed = not (binding and marker.is_file() and marker.read_text().strip() == fingerprint)
	if needed:
		_build_with_retries(python)
	else:
		print("Fekete build is up to date; skipping Conan and CMake setup.")
	editable = venv / ".touring-polygons-editable-installed"
	if not editable.exists():
		subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--editable", "."], cwd=EXTERNAL_SOURCE, check=True)
		editable.touch()
	binding = next(iter((EXTERNAL_SOURCE / "python/tspn_bnb2/core").glob("_tspn_bindings*.so")), None)
	if binding is None:
		fail("the Fekete Python binding was not produced by the build")
	subprocess.run([str(python), "-c", (
		"import importlib.util, pathlib, sys\n"
		"binding = pathlib.Path(sys.argv[1])\n"
		"spec = importlib.util.spec_from_file_location('_tspn_bindings', binding)\n"
		"module = importlib.util.module_from_spec(spec)\n"
		"spec.loader.exec_module(module)\n"
		"callable(module.branch_and_bound) or sys.exit('Fekete binding is missing branch_and_bound')\n"
		"print(f'Verified Fekete binding: {binding}')\n"
	), str(binding)], check=True)
	if needed:
		temporary = marker.with_name(marker.name + ".tmp")
		temporary.write_text(fingerprint + "\n")
		temporary.replace(marker)
	subprocess.run([str(python), "-c", (
		"import gurobipy as gp\n"
		"with gp.Env(empty=True) as environment:\n"
		"    environment.setParam('OutputFlag', 0)\n"
		"    environment.start()\n"
		"print(f'Verified Gurobi runtime and license: gurobipy {gp.gurobi.version()}')\n"
	)], check=True)


def use_conan_packages(reason: str) -> None:
	"""Reuse the C++ packages Fekete's Conan setup generated (Eigen3, Boost, CGAL) for our build."""
	prefix = EXTERNAL_SOURCE / ".conan/release"
	for name in ("Eigen3Config.cmake", "BoostConfig.cmake"):
		if not (prefix / name).is_file():
			fail(f"{reason} {name} is missing at {prefix}; prepare the C++ dependencies first.")
	if not ((prefix / "cgal-config.cmake").is_file() or (prefix / "CGALConfig.cmake").is_file()):
		fail(f"{reason} CGAL CMake package is missing at {prefix}; prepare the C++ dependencies first.")
	os.environ["CMAKE_PREFIX_PATH"] = os.pathsep.join(filter(None, [str(prefix), os.environ.get("CMAKE_PREFIX_PATH", "")]))


# --- our solver --------------------------------------------------------------


def build_ours() -> None:
	runner_bin = ROOT / "benchmarks/.venv/bin"
	if (runner_bin / "cmake").exists():
		os.environ["PATH"] = f"{runner_bin}{os.pathsep}{os.environ['PATH']}"
	if not shutil.which("cmake"):
		fail("CMake is required to build tpp-ours")
	print(f"Using CMake: {shutil.which('cmake')}")
	toolchain = native_build.select_toolchain()
	os.environ.update(toolchain)
	print(f"Verified C++ toolchain: CC={toolchain['CC']} CXX={toolchain['CXX']} (C++{toolchain['TPP_CXX_STANDARD']})")
	binary = native_build.ensure_tool("tpp-unordered")
	print(f"Verified our solver build: {binary}")
	subprocess.run([str(binary), "--help"], capture_output=True, check=True)


# --- the campaign ------------------------------------------------------------


def ensure_campaign(name: str) -> Path:
	campaign = workspace.campaigns_dir() / name
	campaign.mkdir(parents=True, exist_ok=True)
	metadata_path = campaign / workspace.CAMPAIGN_FILE
	suite = SUITE.resolve()
	if metadata_path.exists():
		metadata = json.loads(metadata_path.read_text())
		if not isinstance(metadata, dict):
			fail(f"Invalid campaign metadata; preserving it: {metadata_path}")
		inputs = metadata.get("inputs", [])
		source_file = inputs[0].get("file") if len(inputs) == 1 and isinstance(inputs[0], dict) else None
		if not source_file or (campaign / source_file).resolve() != suite:
			fail(f"Campaign input differs from the Fekete suite: {metadata_path}")
		source = metadata.get("source")
		if not isinstance(source, dict) or source.get("sha256") != EXPECTED_SUITE_SHA256:
			fail(f"Campaign records a different Fekete suite hash: {metadata_path}")
	else:
		if any(child for child in campaign.iterdir() if child.name != workspace.CAMPAIGN_FILE):
			fail(f"Campaign directory has files but no campaign.json; preserving it: {campaign}")
		metadata = {
			"schema_version": 1,
			"name": f"Fekete {EXPECTED_CASES}-case fixed-endpoint free-order campaign ({name})",
			"type": "free_order_comparison",
			"inputs": [{"file": os.path.relpath(suite, campaign)}],
			"source": {"file": "benchmarks/suites/fekete-instances.bin", "sha256": EXPECTED_SUITE_SHA256, "case_count": EXPECTED_CASES},
		}
		temporary = metadata_path.with_suffix(".tmp")
		temporary.write_text(json.dumps(metadata, indent=2) + "\n")
		temporary.replace(metadata_path)
	return campaign


# --- command ----------------------------------------------------------------


def positive_integer(text: str) -> int:
	if not re.fullmatch(r"[1-9][0-9]*", text):
		raise argparse.ArgumentTypeError("must be a positive integer")
	return int(text)


def time_limit(text: str) -> int:
	if text != "-1" and not re.fullmatch(r"[1-9][0-9]*", text):
		raise argparse.ArgumentTypeError("must be -1 or a positive integer")
	return int(text)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="tpp.py free-compare",
		description="Build the selected solver(s), then run or resume the 558-case Fekete fixed-endpoint campaign. "
		"The default per-instance time limit is unlimited. Building tpp-fekete needs a valid Gurobi academic license.",
	)
	parser.add_argument("--solver", choices=sorted(SOLVERS), default="both", help="default: both")
	parser.add_argument("--workers", type=positive_integer, default=1, help="concurrent queued solver cases (default: 1)")
	parser.add_argument("--threads-per-instance", type=positive_integer, default=1, help="solver threads per instance (default: 1)")
	parser.add_argument("--build-jobs", type=positive_integer, default=int(os.environ.get("TPP_BUILD_JOBS", "8")),
		help="parallel compiler jobs (default: TPP_BUILD_JOBS or 8)")
	parser.add_argument("--campaign", help="override the thread-count campaign name")
	parser.add_argument("--max-seconds", type=time_limit, default=-1, help="per-instance limit; -1 means unlimited (default: -1)")
	parser.add_argument("--max-calls", type=positive_integer, default=100_000_000, help="our solver's call limit")
	parser.add_argument("--setup-only", action="store_true", help="check the selected dependencies and builds, then exit")
	parser.add_argument("--force", action="store_true", help="start a new report instead of resuming/reusing one")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
	name = args.campaign
	if name is not None and (not CAMPAIGN_NAME.match(name) or name in (".", "..")):
		parser.error("--campaign must be a simple name without path separators")
	if name is None:
		name = DEFAULT_CAMPAIGN if args.threads_per_instance == 1 else f"fekete-free-order-comparison-{args.threads_per_instance}threads"
	run_ours, run_fekete = SOLVERS[args.solver]
	if not shutil.which("git"):
		fail("git is required")

	verify_suite()
	revision = ""
	if run_fekete:
		revision = prepare_submodule()
		build_fekete(revision, args.build_jobs)
		use_conan_packages("Fekete Conan setup did not generate")
	elif run_ours:
		use_conan_packages("Cached C++ dependency")
	if run_ours:
		build_ours()

	print(f"Pinned Fekete suite: {EXPECTED_CASES} cases ({EXPECTED_SUITE_SHA256})")
	if run_fekete:
		print(f"Fekete revision: {revision}")
	if args.setup_only:
		print(f"Setup complete for solver selection: {args.solver}.")
		options = f" --solver {args.solver}" + (f" --campaign {name}" if name != DEFAULT_CAMPAIGN else "")
		print(f"Run the campaign with: tpp.py free-compare{options}")
		return 0

	campaign = ensure_campaign(name)
	print(f"Solver selection: {args.solver}\nCampaign: {campaign}")
	print(f"Run settings: workers={args.workers}, threads/instance={args.threads_per_instance}, "
		f"max-seconds={args.max_seconds}, max-calls={args.max_calls}")
	import free_order_campaign

	command = [
		name, "--max-instances", str(EXPECTED_CASES), "--max-calls", str(args.max_calls),
		"--max-seconds", str(args.max_seconds), "--threads-per-instance", str(args.threads_per_instance),
		"--workers", str(args.workers), "--absolute-gap", "0", "--relative-gap", format(EPS / (1.0 + EPS), ".17g"),
		"--eps", str(EPS), "--feasibility-tolerance", "0.001", "--validation-tolerance", "1e-7",
	]
	if run_ours:
		command += ["--solver", "tpp-ours"]
	if run_fekete:
		command += ["--solver", "tpp-fekete"]
	if args.force:
		command.append("--force")
	return free_order_campaign.main(command)


if __name__ == "__main__":
	raise SystemExit(main())
