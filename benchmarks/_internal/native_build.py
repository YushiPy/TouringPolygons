"""Build the native TPP tools in shared, lock-protected CMake directories.

Every ``packages/*/cpp/src/main-*.cpp`` is a named CMake target (see the
package CMakeLists). All tools share one build directory per configuration, so
building one tool never reconfigures or overwrites another one. Gurobi tools
use a second directory because they change the ``tpp_convex`` library.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "packages/nonconvex-tpp/cpp"
BUILD_ROOT = ROOT / ".build"
DEPS_ROOT = ROOT / ".cache/deps"
CONFIG_FILE = ".tpp-build-config.json"
TOOL_PACKAGES = {"nonconvex-tpp": "tpp-", "convex-tpp": "tpp-convex-"}
GUROBI_ONLY_TOOLS = {"tpp-convex-cycle-benchmark", "tpp-convex-gurobi-fallback-audit"}

# Pinned header-only releases used when the machine has no installed copy.
HEADER_DEPENDENCIES = {
	"eigen": {
		"url": "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
		"sha256": "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
		"marker": "Eigen/Core",
		"extract": "eigen-3.4.0/Eigen/",
		"cmake": "TPP_EIGEN_INCLUDE_DIR",
	},
	"boost": {
		"url": "https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz",
		"sha256": "2575e74ffc3ef1cd0babac2c1ee8bdb5782a0ee672b1912da40e5b4b591ca01f",
		"marker": "boost/version.hpp",
		"extract": "boost_1_86_0/boost/",
		"cmake": "TPP_BOOST_INCLUDE_DIR",
	},
}

# Pinned GNU releases (checksums of the Homebrew formulae) built into .cache/deps when a Linux
# machine has no GMP headers: GMP is static, and m4 is only needed to configure GMP.
SOURCE_DEPENDENCIES = {
	"m4": {
		"version": "1.4.21",
		"urls": ("https://ftp.gnu.org/gnu/m4/m4-1.4.21.tar.xz", "https://ftpmirror.gnu.org/m4/m4-1.4.21.tar.xz"),
		"sha256": "f25c6ab51548a73a75558742fb031e0625d6485fe5f9155949d6486a2408ab66",
		"marker": "bin/m4",
	},
	"gmp": {
		"version": "6.3.0",
		"urls": ("https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz", "https://ftpmirror.gnu.org/gmp/gmp-6.3.0.tar.xz"),
		"sha256": "a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898",
		"marker": "lib/libgmp.a",
	},
}


def tool_name(package: str, source: Path) -> str:
	"""Mirror the CMake naming rule: main-foo_bar.cpp -> tpp[-convex]-foo-bar."""
	stem = re.sub(r"^main-", "", source.stem)
	if package == "nonconvex-tpp":
		stem = re.sub(r"^tpp_", "", stem)
	return TOOL_PACKAGES[package] + stem.replace("_", "-")


def available_tools() -> dict[str, Path]:
	tools = {}
	for package in TOOL_PACKAGES:
		for source in sorted((ROOT / "packages" / package / "cpp/src").glob("main-*.cpp")):
			tools[tool_name(package, source)] = source
	return tools


def build_directory(*, gurobi: bool = False) -> Path:
	return BUILD_ROOT / ("tools-gurobi" if gurobi else "tools")


def tool_path(tool: str, *, gurobi: bool = False) -> Path:
	return build_directory(gurobi=gurobi) / "bin" / tool


def build_jobs() -> int:
	text = os.environ.get("TPP_BUILD_JOBS", str(min(os.cpu_count() or 4, 8)))
	try:
		jobs = int(text)
	except ValueError as error:
		raise SystemExit("TPP_BUILD_JOBS must be a positive integer.") from error
	if jobs < 1:
		raise SystemExit("TPP_BUILD_JOBS must be a positive integer.")
	return jobs


def gurobi_home() -> Path | None:
	configured = os.environ.get("GUROBI_HOME")
	if configured:
		return Path(configured)
	patterns = ("/Library/gurobi*/macos_universal2", "/opt/gurobi*/linux64", "/opt/gurobi*/armlinux64")
	for pattern in patterns:
		found = sorted(Path("/").glob(pattern.lstrip("/")), reverse=True)
		if found:
			return found[0]
	return None


def header_dependency_dirs() -> dict[str, Path]:
	"""Include directories for header-only dependencies fetched into .cache/deps."""
	found = {}
	for name, spec in HEADER_DEPENDENCIES.items():
		configured = os.environ.get(spec["cmake"])
		if configured:
			found[spec["cmake"]] = Path(configured)
			continue
		for candidate in sorted((DEPS_ROOT / name).glob("*")):
			if (candidate / spec["marker"]).exists():
				found[spec["cmake"]] = candidate
	return found


def probe_compiler(cc: str, cxx: str, standard: str) -> str | None:
	"""Return None when CC/CXX build a C++<standard> std::format program."""
	with tempfile.TemporaryDirectory(prefix="tpp-cxx-probe-") as directory:
		root = Path(directory)
		(root / "CMakeLists.txt").write_text(
			"cmake_minimum_required(VERSION 3.20)\n"
			"project(tpp_probe LANGUAGES CXX)\n"
			"add_executable(tpp_probe main.cpp)\n"
			# Same requirement as packages/common-geometry: this fails when the
			# compiler or this CMake version does not know the standard.
			f"target_compile_features(tpp_probe PRIVATE cxx_std_{standard})\n"
		)
		(root / "main.cpp").write_text(
			"#include <format>\nint main() { return std::format(\"{}\", 23) == \"23\" ? 0 : 1; }\n"
		)
		environment = os.environ | {"CC": cc, "CXX": cxx}
		for command in (["cmake", "-S", str(root), "-B", str(root / "build")],
				["cmake", "--build", str(root / "build")]):
			result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
			if result.returncode:
				return "\n".join((result.stdout + result.stderr).splitlines()[-12:])
	return None


def compiler_candidates() -> Iterator[tuple[str, str]]:
	if os.environ.get("CXX"):
		yield os.environ.get("CC", "cc"), os.environ["CXX"]
		return
	default_cxx = shutil.which("c++")
	if default_cxx:
		yield os.environ.get("CC", "cc"), default_cxx
	if platform.system() == "Darwin":
		for prefix in ("/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"):
			if Path(prefix, "clang++").exists():
				yield f"{prefix}/clang", f"{prefix}/clang++"
	else:
		for version in (16, 15, 14):
			cxx = shutil.which(f"g++-{version}")
			if cxx:
				yield shutil.which(f"gcc-{version}") or "cc", cxx
		for version in (20, 19, 18):
			cxx = shutil.which(f"clang++-{version}")
			if cxx:
				yield shutil.which(f"clang-{version}") or "cc", cxx


def select_toolchain() -> dict[str, str]:
	"""Pick the first compiler that builds the project's C++ standard."""
	requested = os.environ.get("TPP_CXX_STANDARD")
	standards = [requested] if requested else ["26", "23"]
	diagnostics = []
	for cc, cxx in compiler_candidates():
		for standard in standards:
			error = probe_compiler(cc, cxx, standard)
			if error is None:
				return {"CC": cc, "CXX": cxx, "TPP_CXX_STANDARD": standard}
			diagnostics.append(f"CC={cc} CXX={cxx} C++{standard}:\n{error}")
	raise SystemExit(
		"No usable C++23 compiler found. Install one, or set CC/CXX explicitly.\n\n"
		+ "\n\n".join(diagnostics[-2:])
	)


def requested_toolchain() -> list[str | None]:
	return [os.environ.get(name) for name in ("CC", "CXX", "TPP_CXX_STANDARD")]


def configuration(*, gurobi: bool, toolchain: dict[str, str] | None = None) -> dict:
	toolchain = toolchain or select_toolchain()
	arguments = [
		"-DCMAKE_BUILD_TYPE=Release",
		f"-DTPP_CXX_STANDARD={toolchain['TPP_CXX_STANDARD']}",
		f"-DTPP_ENABLE_GUROBI={'ON' if gurobi else 'OFF'}",
	]
	if gurobi:
		home = gurobi_home()
		if home is None:
			raise SystemExit("Gurobi tools need GUROBI_HOME (or a standard /Library or /opt install).")
		arguments.append(f"-DGUROBI_HOME={home}")
	for variable, directory in sorted(header_dependency_dirs().items()):
		arguments.append(f"-D{variable}={directory}")
	if (gmp := source_dependency_prefix("gmp")) is not None:
		arguments += [f"-DTPP_GMP_INCLUDE_DIR={gmp / 'include'}", f"-DTPP_GMP_LIBRARY={gmp / 'lib/libgmp.a'}"]
	arguments.extend(shlex.split(os.environ.get("TPP_CMAKE_ARGS", "")))
	return {
		"source": str(SOURCE.resolve()),
		"requested": requested_toolchain(),
		"toolchain": toolchain,
		"arguments": arguments,
	}


@contextmanager
def build_lock(directory: Path) -> Iterator[None]:
	directory.parent.mkdir(parents=True, exist_ok=True)
	with open(directory.with_name(directory.name + ".lock"), "w") as handle:
		fcntl.flock(handle, fcntl.LOCK_EX)
		try:
			yield
		finally:
			fcntl.flock(handle, fcntl.LOCK_UN)


def cache_matches_checkout(directory: Path, source: Path = SOURCE) -> bool:
	"""False when a CMake cache was created for another checkout or directory."""
	cache = directory / "CMakeCache.txt"
	if not cache.exists():
		return True
	values = {}
	try:
		for line in cache.read_text().splitlines():
			for name in ("CMAKE_HOME_DIRECTORY", "CMAKE_CACHEFILE_DIR"):
				if line.startswith(f"{name}:"):
					values[name] = line.split("=", 1)[1]
	except OSError:
		return False
	# CMake records the paths as it was given them, symbolic links included
	# (on macOS, /tmp rather than /private/tmp), so both sides are resolved.
	def same(name: str, path: Path) -> bool:
		return name in values and Path(values[name]).resolve() == path.resolve()
	return same("CMAKE_HOME_DIRECTORY", source) and same("CMAKE_CACHEFILE_DIR", directory)


def _configured(directory: Path) -> dict | None:
	try:
		return json.loads((directory / CONFIG_FILE).read_text())
	except (OSError, json.JSONDecodeError):
		return None


def ensure_tools(
	tools: Sequence[str],
	*,
	gurobi: bool = False,
	no_build: bool = False,
	quiet: bool = False,
) -> list[Path]:
	"""Build the named tools if needed and return their executable paths."""
	known = available_tools()
	unknown = [tool for tool in tools if tool not in known]
	if unknown:
		raise SystemExit(f"Unknown native tool(s): {', '.join(unknown)}. Run 'tpp.py build --list'.")
	gurobi = gurobi or any(tool in GUROBI_ONLY_TOOLS for tool in tools)
	directory = build_directory(gurobi=gurobi)
	paths = [tool_path(tool, gurobi=gurobi) for tool in tools]
	if no_build:
		missing = [path for path in paths if not path.exists()]
		if missing:
			raise SystemExit(f"--no-build was passed, but these tools are not built: {', '.join(map(str, missing))}")
		return paths
	output = subprocess.DEVNULL if quiet else None
	with build_lock(directory):
		if not cache_matches_checkout(directory):
			print(f"Build: discarding a relocated CMake cache in {directory}", flush=True)
			shutil.rmtree(directory)
		previous = _configured(directory)
		# Probing compilers costs seconds; reuse the probed toolchain while the
		# checkout and the CC/CXX/standard request are unchanged.
		reuse = (previous is not None and previous.get("source") == str(SOURCE.resolve())
			and previous.get("requested") == requested_toolchain())
		wanted = configuration(gurobi=gurobi, toolchain=previous["toolchain"] if reuse else None)
		if previous != wanted or not (directory / "CMakeCache.txt").exists():
			if previous is not None and previous.get("toolchain") != wanted["toolchain"]:
				shutil.rmtree(directory, ignore_errors=True)
			if not quiet:
				print(f"Build: configuring {directory} "
					f"({Path(wanted['toolchain']['CXX']).name}, C++{wanted['toolchain']['TPP_CXX_STANDARD']})", flush=True)
			environment = os.environ | {key: wanted["toolchain"][key] for key in ("CC", "CXX")}
			try:
				subprocess.run(["cmake", "-S", wanted["source"], "-B", str(directory), *wanted["arguments"]],
					env=environment, check=True, stdout=output)
			except subprocess.CalledProcessError:
				print("\nIf the error says Eigen or Boost was not found and you cannot install system packages "
					"(no root needed for this): python3 benchmarks/tpp.py build --fetch-deps", file=sys.stderr)
				raise
			directory.mkdir(parents=True, exist_ok=True)
			(directory / CONFIG_FILE).write_text(json.dumps(wanted, indent=2) + "\n")
		subprocess.run(["cmake", "--build", str(directory), "--parallel", str(build_jobs()), "--target", *tools],
			check=True, stdout=output)
	for path in paths:
		if not path.exists():
			raise SystemExit(f"Build succeeded without producing {path}")
	return paths


def ensure_tool(tool: str, **options) -> Path:
	return ensure_tools([tool], **options)[0]


def fetch_header_dependencies(names: Sequence[str]) -> int:
	"""Download pinned header-only releases into .cache/deps (no root needed)."""
	import hashlib
	import tarfile
	import urllib.request

	for name in names:
		spec = HEADER_DEPENDENCIES[name]
		target = DEPS_ROOT / name
		if any((candidate / spec["marker"]).exists() for candidate in target.glob("*")):
			print(f"{name}: already available in {target}")
			continue
		target.mkdir(parents=True, exist_ok=True)
		archive = target / Path(spec["url"]).name
		print(f"{name}: downloading {spec['url']}", flush=True)
		with urllib.request.urlopen(spec["url"]) as response, archive.open("wb") as file:
			shutil.copyfileobj(response, file)
		digest = hashlib.sha256(archive.read_bytes()).hexdigest()
		if digest != spec["sha256"]:
			archive.unlink()
			raise SystemExit(f"{name}: SHA-256 mismatch ({digest}); refusing to use the download.")
		print(f"{name}: extracting headers", flush=True)
		with tarfile.open(archive) as bundle:
			# Only the header tree is needed; the full Boost sources are ~10x larger.
			members = [member for member in bundle if member.name.startswith(spec["extract"])]
			bundle.extractall(target, members=members, filter="data")
		archive.unlink()
	return 0


def source_dependency_prefix(name: str) -> Path | None:
	"""Install prefix of a source dependency already built into .cache/deps, if any."""
	spec = SOURCE_DEPENDENCIES[name]
	prefix = DEPS_ROOT / name / spec["version"]
	return prefix if (prefix / spec["marker"]).exists() else None


def system_gmp_available(cc: str = "cc") -> bool:
	"""True when the C compiler finds gmp.h and links -lgmp without extra paths."""
	with tempfile.TemporaryDirectory(prefix="tpp-gmp-probe-") as directory:
		source = Path(directory) / "probe.c"
		source.write_text("#include <gmp.h>\nint main(void) { mpz_t x; mpz_init_set_ui(x, 3); mpz_clear(x); return 0; }\n")
		result = subprocess.run([cc, str(source), "-lgmp", "-o", str(Path(directory) / "probe")],
			capture_output=True, check=False)
		return result.returncode == 0


def build_uses_gmp(directory: Path | None = None) -> bool | None:
	"""Whether a configured build links GMP (None before the first configure)."""
	cache = (directory or build_directory()) / "CMakeCache.txt"
	try:
		lines = cache.read_text().splitlines()
	except OSError:
		return None
	values = [line.split("=", 1)[1] for line in lines if line.startswith("TPP_GMP_LIBRARY:")]
	return bool(values) and not values[0].endswith("NOTFOUND")


def _build_source_dependency(name: str, work: Path, configure: Sequence[str], environment: dict[str, str],
		check: bool = False) -> Path:
	import hashlib
	import tarfile
	import urllib.request

	spec = SOURCE_DEPENDENCIES[name]
	prefix = DEPS_ROOT / name / spec["version"]
	archive = work / Path(spec["urls"][0]).name
	for url in spec["urls"]:
		try:
			print(f"{name}: downloading {url}", flush=True)
			with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as file:
				shutil.copyfileobj(response, file)
			break
		except OSError as error:
			print(f"{name}: download failed ({error})", flush=True)
	else:
		raise SystemExit(f"{name}: no source URL could be downloaded.")
	digest = hashlib.sha256(archive.read_bytes()).hexdigest()
	if digest != spec["sha256"]:
		raise SystemExit(f"{name}: SHA-256 mismatch ({digest}); refusing to use the download.")
	with tarfile.open(archive) as bundle:
		bundle.extractall(work, filter="data")
	source = work / f"{name}-{spec['version']}"
	log = work / f"{name}.log"
	steps = [["./configure", f"--prefix={prefix}", *configure], ["make", f"-j{build_jobs()}"]]
	if check:
		steps.append(["make", f"-j{build_jobs()}", "check"])
	steps.append(["make", "install"])
	for step in steps:
		print(f"{name}: {' '.join(step[:2])}", flush=True)
		with log.open("a") as output:
			result = subprocess.run(step, cwd=source, env=environment, stdout=output, stderr=subprocess.STDOUT, check=False)
		if result.returncode:
			print("\n".join(log.read_text().splitlines()[-30:]), file=sys.stderr)
			raise SystemExit(f"{name}: '{' '.join(step)}' failed")
	return prefix


def build_gmp() -> Path:
	"""Build the pinned GMP as a static library into .cache/deps/gmp (no root), with its test suite.

	m4 is built first when the machine has none (GMP's configure needs it). GMP 6.3.0
	predates C23, the default of GCC 15, so its C code is compiled as gnu17.
	"""
	if (prefix := source_dependency_prefix("gmp")) is not None:
		print(f"gmp: already built in {prefix}")
		return prefix
	DEPS_ROOT.mkdir(parents=True, exist_ok=True)
	environment = dict(os.environ)
	with tempfile.TemporaryDirectory(dir=DEPS_ROOT, prefix="build-") as directory:
		work = Path(directory)
		if not shutil.which("m4"):
			m4 = source_dependency_prefix("m4") or _build_source_dependency("m4", work, [], environment)
			environment["PATH"] = f"{m4 / 'bin'}{os.pathsep}{environment.get('PATH', '')}"
		environment["CFLAGS"] = "-O2 -std=gnu17 -fPIC"
		prefix = _build_source_dependency("gmp", work, ["--disable-shared", "--enable-static", "--with-pic"],
			environment, check=True)
	print(f"gmp: installed in {prefix}")
	return prefix


def doctor() -> int:
	"""Report what this machine can build and run, without changing anything."""
	ok = True

	def report(label: str, value: str | None, required: bool = True) -> None:
		nonlocal ok
		mark = "ok " if value else ("-- " if not required else "!! ")
		ok = ok and (bool(value) or not required)
		print(f"  {mark}{label:<22} {value or 'missing'}")

	print(f"Machine: {platform.node()} ({platform.system()} {platform.machine()}), {os.cpu_count()} CPUs")
	report("python", f"{sys.version.split()[0]} ({sys.executable})")
	report("uv", shutil.which("uv"), required=False)
	report("cmake", shutil.which("cmake"))
	try:
		toolchain = select_toolchain()
		report("C++ compiler", f"{toolchain['CXX']} (C++{toolchain['TPP_CXX_STANDARD']})")
	except SystemExit:
		report("C++ compiler", None)
	headers = header_dependency_dirs()
	for name, spec in HEADER_DEPENDENCIES.items():
		report(f"{name} (fetched)", str(headers[spec["cmake"]]) if spec["cmake"] in headers else None, required=False)
	gmp = source_dependency_prefix("gmp")
	report("gmp (built)", str(gmp) if gmp else None, required=False)
	report("gmp in tools build", {True: "yes", False: "no (Boost cpp_rational)"}.get(build_uses_gmp()), required=False)
	home = gurobi_home()
	report("gurobi", str(home) if home and home.exists() else None, required=False)
	for directory in (build_directory(), build_directory(gurobi=True)):
		built = sorted(path.name for path in (directory / "bin").glob("tpp*")) if directory.exists() else []
		report(f"built in {directory.name}", f"{len(built)} tools" if built else None, required=False)
	print("\nIf the build cannot find Eigen or Boost, run: python3 benchmarks/tpp.py build --fetch-deps")
	return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(
		prog="tpp.py build",
		description="Build native tools into .build/tools (or .build/tools-gurobi) and print their paths.",
	)
	parser.add_argument("tools", nargs="*", help="tool names, e.g. tpp-unordered; default: the benchmark solvers")
	parser.add_argument("--list", action="store_true", help="list buildable tools")
	parser.add_argument("--gurobi", action="store_true", help="build with the Gurobi baseline enabled")
	parser.add_argument("--fetch-deps", action="store_true",
		help="download pinned Eigen/Boost headers into .cache/deps when they are not installed (then build only the tools you name)")
	parser.add_argument("--fetch-gmp", action="store_true",
		help="build the pinned GMP (and m4 if missing) into .cache/deps when the system has no GMP headers")
	parser.add_argument("--doctor", action="store_true", help="check compilers and dependencies")
	args = parser.parse_args(argv)
	if args.list:
		for name, source in available_tools().items():
			suffix = "  (Gurobi)" if name in GUROBI_ONLY_TOOLS else ""
			print(f"{name:<38} {source.relative_to(ROOT)}{suffix}")
		return 0
	if args.doctor:
		return doctor()
	if args.fetch_gmp:
		build_gmp()
	if args.fetch_deps:
		fetch_header_dependencies(list(HEADER_DEPENDENCIES))
	if args.fetch_deps or args.fetch_gmp:
		if not args.tools:
			print("Dependencies ready; build with: python3 benchmarks/tpp.py build [TOOL...]")
			return 0
	if args.tools:
		for path in ensure_tools(args.tools, gurobi=args.gurobi):
			print(path)
		return 0
	# Default set: the free-order solver is required; the fixed-order benchmark uses <print>, which
	# needs GCC 14 or newer, so on an older compiler it is skipped instead of failing the build.
	for path in ensure_tools(["tpp-unordered"], gurobi=args.gurobi):
		print(path)
	try:
		for path in ensure_tools(["tpp-bnb-workload-benchmark"], gurobi=args.gurobi):
			print(path)
	except subprocess.CalledProcessError:
		print("\nSkipped tpp-bnb-workload-benchmark (the fixed-order benchmark): it did not compile with this "
			"compiler, usually because <print> needs GCC 14+. Free-order and TSPN runs do not use it.", file=sys.stderr)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
