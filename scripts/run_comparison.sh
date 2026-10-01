#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_SOURCE="$ROOT/third_party/tspn-socg"
SUITE="$ROOT/benchmarks/suites/german-instances.bin"
EXPECTED_SUITE_SHA256="aa442e0546567461621b7fcdb9596ba7b3cc4094929d23fb9bb38d1093c88737"
EXPECTED_CASES=558

workers=1
threads_per_instance=1
build_jobs="${TPP_BUILD_JOBS:-8}"
campaign_name="german-free-order-comparison-v1"
max_seconds=-1
max_calls=100000000
force=0
setup_only=0

usage() {
	cat <<'EOF'
Usage: scripts/run_comparison.sh [options]

Build both free-order solvers, then run or resume the 558-case German
fixed-endpoint comparison. The default per-instance time limit is unlimited.

Options:
  --workers N                 Concurrent queued solver cases (default: 1)
  --threads-per-instance N    Solver threads per instance (default: 1)
  --build-jobs N              Parallel compiler jobs (default: TPP_BUILD_JOBS or 8)
  --campaign NAME             Local campaign name (default: german-free-order-comparison-v1)
  --max-seconds N             Per-instance limit; -1 means unlimited (default: -1)
  --max-calls N               Our solver's call limit (default: 100000000)
  --setup-only                Check dependencies and compile both solvers, then exit
  --force                     Start a new report instead of resuming/reusing one
  -h, --help                  Show this help

Python 3.12+ is required. Set TPP_PYTHON to select its executable. The host
needs a C++23-capable compiler, OpenMP, Eigen3, Boost headers, and a valid
Gurobi academic license for the Fekete solver.
EOF
}

fail() {
	printf 'Error: %s\n' "$*" >&2
	exit 2
}

while (($#)); do
	case "$1" in
		--workers)
			(($# >= 2)) || fail '--workers requires a value'
			workers="$2"
			shift 2
			;;
		--threads-per-instance)
			(($# >= 2)) || fail '--threads-per-instance requires a value'
			threads_per_instance="$2"
			shift 2
			;;
		--build-jobs)
			(($# >= 2)) || fail '--build-jobs requires a value'
			build_jobs="$2"
			shift 2
			;;
		--campaign)
			(($# >= 2)) || fail '--campaign requires a value'
			campaign_name="$2"
			shift 2
			;;
		--max-seconds)
			(($# >= 2)) || fail '--max-seconds requires a value'
			max_seconds="$2"
			shift 2
			;;
		--max-calls)
			(($# >= 2)) || fail '--max-calls requires a value'
			max_calls="$2"
			shift 2
			;;
		--force)
			force=1
			shift
			;;
		--setup-only)
			setup_only=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			fail "unknown option: $1"
			;;
	esac
done

[[ "$workers" =~ ^[1-9][0-9]*$ ]] || fail '--workers must be a positive integer'
[[ "$threads_per_instance" =~ ^[1-9][0-9]*$ ]] || fail '--threads-per-instance must be a positive integer'
[[ "$build_jobs" =~ ^[1-9][0-9]*$ ]] || fail '--build-jobs must be a positive integer'
[[ "$max_seconds" == '-1' || "$max_seconds" =~ ^[1-9][0-9]*$ ]] || fail '--max-seconds must be -1 or a positive integer'
[[ "$max_calls" =~ ^[1-9][0-9]*$ ]] || fail '--max-calls must be a positive integer'
[[ "$campaign_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && "$campaign_name" != '.' && "$campaign_name" != '..' ]] \
	|| fail '--campaign must be a simple name without path separators'

command -v git >/dev/null 2>&1 || fail 'git is required'

python_bin="${TPP_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
	if command -v python3.12 >/dev/null 2>&1; then
		python_bin="$(command -v python3.12)"
	else
		python_bin="$(command -v python3)"
	fi
elif [[ ! -x "$python_bin" ]]; then
	python_bin="$(command -v "$python_bin" || true)"
fi
[[ -n "$python_bin" ]] || fail 'could not find the selected Python executable'
"$python_bin" -c 'import sys; sys.version_info >= (3, 12) or sys.exit("Python 3.12 or newer is required; set TPP_PYTHON to its executable.")' \
	|| fail 'Python 3.12 or newer is required; set TPP_PYTHON to its executable'

[[ -f "$SUITE" ]] || fail "German suite is missing: $SUITE"
actual_suite_sha256="$("$python_bin" - "$SUITE" <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()
print(digest)
PY
)"
[[ "$actual_suite_sha256" == "$EXPECTED_SUITE_SHA256" ]] \
	|| fail "German suite SHA-256 mismatch: expected $EXPECTED_SUITE_SHA256, got $actual_suite_sha256"

expected_submodule="$(git -C "$ROOT" ls-tree HEAD -- third_party/tspn-socg | awk '$1 == "160000" {print $3}')"
[[ -n "$expected_submodule" ]] || fail 'third_party/tspn-socg is not pinned as a Git submodule in HEAD'
external_patches=(
	"$ROOT/patches/tspn-socg-fmt-format-header.patch"
	"$ROOT/patches/tspn-socg-directional-oracle-variant.patch"
)
fmt_patch_status=' M python/tspn_bnb2/core/_tspn_bindings.cpp'
patched_source_status=' M CMakeLists.txt
 M python/tspn_bnb2/core/_tspn_bindings.cpp'
submodule_initialized=0
if [[ -e "$EXTERNAL_SOURCE/.git" ]]; then
	current_submodule="$(git -C "$EXTERNAL_SOURCE" rev-parse HEAD 2>/dev/null || true)"
	submodule_changes="$(git -C "$EXTERNAL_SOURCE" status --porcelain --untracked-files=all 2>/dev/null || true)"
	if [[ "$current_submodule" == "$expected_submodule" ]]; then
		[[ -z "$submodule_changes" || "$submodule_changes" == "$fmt_patch_status" \
			|| "$submodule_changes" == "$patched_source_status" ]] \
			|| fail 'the Fekete submodule has local changes beyond the managed compatibility patches; preserve them and resolve manually'
		submodule_initialized=1
	else
		[[ -z "$submodule_changes" ]] || fail 'the Fekete submodule has local changes; preserve them and check it out to the pinned revision manually'
	fi
fi
if ((!submodule_initialized)); then
	git -C "$ROOT" submodule update --init --recursive -- third_party/tspn-socg
fi
current_submodule="$(git -C "$EXTERNAL_SOURCE" rev-parse HEAD)"
[[ "$current_submodule" == "$expected_submodule" ]] \
	|| fail "Fekete submodule is at $current_submodule; expected pinned revision $expected_submodule"
	for external_patch in "${external_patches[@]}"; do
		if git -C "$EXTERNAL_SOURCE" apply --reverse --check "$external_patch" >/dev/null 2>&1; then
			printf 'Fekete compatibility patch already applied: %s\n' "$(basename "$external_patch")"
		elif git -C "$EXTERNAL_SOURCE" apply --check "$external_patch" >/dev/null 2>&1; then
			git -C "$EXTERNAL_SOURCE" apply "$external_patch"
			printf 'Applied local Fekete compatibility patch: %s\n' "$(basename "$external_patch")"
		else
			fail "the pinned Fekete source does not match compatibility patch $(basename "$external_patch"); preserve it and inspect the submodule revision"
		fi
	done
submodule_changes="$(git -C "$EXTERNAL_SOURCE" status --porcelain --untracked-files=all 2>/dev/null || true)"
[[ "$submodule_changes" == "$patched_source_status" ]] \
	|| fail 'the Fekete submodule has local changes beyond the managed compatibility patches; preserve them and resolve manually'

external_venv="$EXTERNAL_SOURCE/.venv"
external_python="$external_venv/bin/python"
if [[ ! -x "$external_python" ]]; then
	"$python_bin" -m venv "$external_venv"
fi

if [[ ! -x "$external_venv/bin/conan" || ! -x "$external_venv/bin/cmake" \
	|| ! -x "$external_venv/bin/ninja" ]] \
	|| ! "$external_python" -c 'import skbuild, skbuild_conan' >/dev/null 2>&1; then
	"$external_python" -m pip install --disable-pip-version-check \
		'conan>=2.0.0' 'setuptools' 'scikit-build>=0.18.0' 'skbuild-conan' \
		'cmake>=3.23,<4' 'ninja'
fi
export PATH="$external_venv/bin:$PATH"
export TPP_BUILD_JOBS="$build_jobs"
export CMAKE_BUILD_PARALLEL_LEVEL="$build_jobs"
if ! "$external_venv/bin/conan" profile show -pr default >/dev/null 2>&1; then
	"$external_venv/bin/conan" profile detect
fi

editable_marker="$external_venv/.touring-polygons-editable-installed"
conan_profile="$($external_venv/bin/conan profile show -pr default)"
cmake_version="$($external_venv/bin/cmake --version | head -n 1)"
conan_version="$($external_venv/bin/conan --version)"
fekete_build_fingerprint="$($external_python - "$EXTERNAL_SOURCE" "$current_submodule" "$ROOT" "$conan_profile" "$cmake_version" "$conan_version" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import sysconfig

root = pathlib.Path(sys.argv[1]).resolve()
revision = sys.argv[2]
project_root = pathlib.Path(sys.argv[3]).resolve()
conan_profile = sys.argv[4]
cmake_version = sys.argv[5]
conan_version = sys.argv[6]
digest = hashlib.sha256()
excluded = {".git", ".venv", ".conan", "_skbuild", "__pycache__", ".cache", "build", "dist"}
native_suffixes = {".cpp", ".cc", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".ipp", ".tpp", ".cmake", ".txt"}
input_roots = (
	("fekete", root, True),
	("embedded-convex", project_root / "packages/convex-tpp/cpp", False),
	("embedded-geometry", project_root / "packages/common-geometry/cpp", False),
)
for label, source_root, include_metadata in input_roots:
	for directory, subdirectories, filenames in os.walk(source_root):
		subdirectories[:] = sorted(
			name for name in subdirectories
			if name not in excluded and not name.endswith(".egg-info")
		)
		for filename in sorted(filenames):
			path = pathlib.Path(directory) / filename
			relative = path.relative_to(source_root)
			under_conan_recipe = relative.parts[:2] == ("cmake", "conan")
			build_metadata = path.name in {"setup.py", "pyproject.toml", "conanfile.py", "conanfile.txt", "conan.lock", "conandata.yml"}
			if path.suffix.lower() not in native_suffixes and not (include_metadata and build_metadata) and not (include_metadata and under_conan_recipe):
				continue
			digest.update(label.encode())
			digest.update(b"/")
			digest.update(relative.as_posix().encode())
			digest.update(b"\0")
			digest.update(path.read_bytes())
			digest.update(b"\0")
configuration = {
	"revision": revision,
	"python": sys.version_info[:2],
	"python_abi": {key: sysconfig.get_config_var(key) for key in ("SOABI", "EXT_SUFFIX", "INCLUDEPY", "LIBDIR")},
	"python_prefix": sys.prefix,
	"platform": platform.system(),
	"machine": platform.machine(),
	"conan_profile": conan_profile,
	"cmake_version": cmake_version,
	"conan_version": conan_version,
	"environment": {key: os.environ.get(key, "") for key in (
		"CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CMAKE_ARGS",
		"CMAKE_BUILD_TYPE", "CMAKE_GENERATOR", "CMAKE_TOOLCHAIN_FILE", "CMAKE_PREFIX_PATH",
		"CMAKE_OSX_ARCHITECTURES", "MACOSX_DEPLOYMENT_TARGET",
	)},
}
configuration["tool_versions"] = {}
for name, command in (("c_compiler", os.environ.get("CC", "cc")),
		("cxx_compiler", os.environ.get("CXX", "c++"))):
	try:
		version = subprocess.run(shlex.split(command) + ["--version"], capture_output=True, text=True)
		configuration["tool_versions"][name] = (version.stdout + version.stderr).splitlines()[:2]
	except (OSError, ValueError):
		configuration["tool_versions"][name] = command
digest.update(json.dumps(configuration, sort_keys=True).encode())
print(digest.hexdigest())
PY
)"
fekete_build_marker="$external_venv/.touring-polygons-fekete-build-fingerprint"
binding="$(find "$EXTERNAL_SOURCE/python/tspn_bnb2/core" -maxdepth 1 -type f -name '_tspn_bindings*.so' -print -quit)"
fekete_build_needed=1
if [[ -n "$binding" && -f "$fekete_build_marker" ]] \
	&& [[ "$(cat "$fekete_build_marker")" == "$fekete_build_fingerprint" ]]; then
	fekete_build_needed=0
	printf 'Fekete build is up to date; skipping Conan and CMake setup.\n'
fi

if ((fekete_build_needed)); then
	(
		cd "$EXTERNAL_SOURCE"
		"$external_python" - <<'PY'
import subprocess
import sys
import time

transient_markers = (
    "too many 502 error responses",
    "too many 503 error responses",
    "too many 504 error responses",
    "connection timed out",
    "read timed out",
    "temporary failure in name resolution",
)

for attempt in range(1, 4):
    process = subprocess.Popen(
        [sys.executable, "setup.py", "develop"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    transient_network_failure = False
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lowered = line.lower()
        if any(marker in lowered for marker in transient_markers):
            transient_network_failure = True
    returncode = process.wait()
    if returncode == 0:
        break
    if not transient_network_failure or attempt == 3:
        raise SystemExit(returncode or 1)
    delay = 5 * attempt
    print(
        f"Transient dependency download failure; retrying setup "
        f"({attempt + 1}/3) in {delay}s...",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(delay)
else:
    raise SystemExit("Fekete dependency setup failed after three attempts.")
PY
	)
fi
if [[ ! -f "$editable_marker" ]]; then
	(
		cd "$EXTERNAL_SOURCE"
		"$external_python" -m pip install --disable-pip-version-check --editable .
	)
	touch "$editable_marker"
fi
binding="$(find "$EXTERNAL_SOURCE/python/tspn_bnb2/core" -maxdepth 1 -type f -name '_tspn_bindings*.so' -print -quit)"
[[ -n "$binding" ]] || fail 'the Fekete Python binding was not produced by the build'

"$external_python" - "$binding" <<'PY'
import importlib.util
import pathlib
import sys

binding = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_tspn_bindings", binding)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if not callable(module.branch_and_bound):
    raise SystemExit("Fekete binding is missing branch_and_bound")
print(f"Verified Fekete binding: {binding}")
PY

if ((fekete_build_needed)); then
	temporary_fekete_marker="$fekete_build_marker.tmp"
	printf '%s\n' "$fekete_build_fingerprint" > "$temporary_fekete_marker"
	mv -f "$temporary_fekete_marker" "$fekete_build_marker"
fi

"$external_python" - <<'PY'
import gurobipy as gp

with gp.Env(empty=True) as environment:
    environment.setParam("OutputFlag", 0)
    environment.start()

print(f"Verified Gurobi runtime and license: gurobipy {gp.gurobi.version()}")
PY

# The Fekete Conan setup provides Eigen3, Boost, and CGAL for its own build.
# Reuse those generated CMake package configs when configuring our solver so
# Linux users do not need system-wide development packages.
conan_cmake_prefix="$EXTERNAL_SOURCE/.conan/release"
for dependency_config in Eigen3Config.cmake BoostConfig.cmake; do
	[[ -f "$conan_cmake_prefix/$dependency_config" ]] \
		|| fail "Fekete Conan setup did not generate $dependency_config in $conan_cmake_prefix"
done
[[ -f "$conan_cmake_prefix/cgal-config.cmake" || -f "$conan_cmake_prefix/CGALConfig.cmake" ]] \
	|| fail "Fekete Conan setup did not generate a CGAL CMake package in $conan_cmake_prefix"
export CMAKE_PREFIX_PATH="$conan_cmake_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
printf 'Reusing Conan C++ dependencies for our solver: %s\n' "$conan_cmake_prefix"

last_cpp23_probe_diagnostic=''
probe_cpp23_toolchain() {
	local probe_cc="$1"
	local probe_cxx="$2"
	local probe_dir status
	probe_dir="$(mktemp -d "${TMPDIR:-/tmp}/tpp-cxx23-probe.XXXXXX")"
	cat > "$probe_dir/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.20)
project(tpp_cpp23_probe LANGUAGES CXX)
add_executable(tpp_cpp23_probe main.cpp)
target_compile_features(tpp_cpp23_probe PRIVATE cxx_std_23)
EOF
	cat > "$probe_dir/main.cpp" <<'EOF'
#include <format>
int main() { return std::format("{}", 23) == "23" ? 0 : 1; }
EOF
	status=0
	CC="$probe_cc" CXX="$probe_cxx" cmake -S "$probe_dir" -B "$probe_dir/build" > "$probe_dir/output.log" 2>&1 || status=$?
	if ((status == 0)); then
		CC="$probe_cc" CXX="$probe_cxx" cmake --build "$probe_dir/build" >> "$probe_dir/output.log" 2>&1 || status=$?
	fi
	if ((status != 0)); then
		last_cpp23_probe_diagnostic="C++23 toolchain probe failed for CC=$probe_cc CXX=$probe_cxx
$(tail -n 12 "$probe_dir/output.log")"
		rm -rf "$probe_dir"
		return 1
	fi
	rm -rf "$probe_dir"
	return 0
}

if [[ -n "${CXX:-}" ]]; then
	probe_cc="${CC:-cc}"
	if ! probe_cpp23_toolchain "$probe_cc" "$CXX"; then
		printf '%s\n' "$last_cpp23_probe_diagnostic" >&2
		fail 'the selected CC/CXX toolchain cannot build the comparison C++23 requirement; choose a compatible compiler or unset CC/CXX for automatic selection'
	fi
	printf 'Verified C++23 toolchain: CC=%s CXX=%s\n' "$probe_cc" "$CXX"
else
	configured_cc="${CC:-}"
	default_cc="${CC:-cc}"
	default_cxx="$(command -v c++ || true)"
	selected=0
	if [[ -n "$default_cxx" ]] && probe_cpp23_toolchain "$default_cc" "$default_cxx"; then
		CXX="$default_cxx"
		CC="$default_cc"
		selected=1
	fi
	if ((!selected)); then
		candidate_compilers=()
		candidate_c_compilers=()
		case "$(uname -s)" in
			Darwin)
				for candidate_cxx in /opt/homebrew/opt/llvm/bin/clang++ /usr/local/opt/llvm/bin/clang++; do
					[[ -x "$candidate_cxx" ]] || continue
					candidate_compilers+=("$candidate_cxx")
					candidate_c_compilers+=("${candidate_cxx%clang++}clang")
				done
				;;
			Linux)
				for version in 16 15 14; do
					candidate_cxx="$(command -v "g++-$version" || true)"
					[[ -n "$candidate_cxx" ]] || continue
					candidate_cc="$(command -v "gcc-$version" || true)"
					[[ -n "$candidate_cc" ]] || candidate_cc="$default_cc"
					candidate_compilers+=("$candidate_cxx")
					candidate_c_compilers+=("$candidate_cc")
				done
				for version in 20 19 18; do
					candidate_cxx="$(command -v "clang++-$version" || true)"
					[[ -n "$candidate_cxx" ]] || continue
					candidate_cc="$(command -v "clang-$version" || true)"
					[[ -n "$candidate_cc" ]] || candidate_cc="$default_cc"
					candidate_compilers+=("$candidate_cxx")
					candidate_c_compilers+=("$candidate_cc")
				done
				;;
		esac
		for index in "${!candidate_compilers[@]}"; do
			candidate_cxx="${candidate_compilers[$index]}"
			candidate_cc="$configured_cc"
			if [[ -z "$candidate_cc" ]]; then
				candidate_cc="${candidate_c_compilers[$index]}"
			fi
			if probe_cpp23_toolchain "$candidate_cc" "$candidate_cxx"; then
				CXX="$candidate_cxx"
				CC="$candidate_cc"
				selected=1
				printf 'Default compiler is incompatible with the comparison C++23 check; using verified compiler: %s\n' "$CXX"
				break
			fi
		done
	fi
	if ((!selected)); then
		[[ -z "$last_cpp23_probe_diagnostic" ]] || printf '%s\n' "$last_cpp23_probe_diagnostic" >&2
		fail 'no usable C++23 compiler found; install a compiler and CMake that support cxx_std_23, or set CC and CXX explicitly'
	fi
	export CC CXX
	printf 'Verified C++23 toolchain: CC=%s CXX=%s\n' "$CC" "$CXX"
fi

# This campaign's solver targets only use C++23. Keep the repository's default
# standard unchanged for other targets, some of which use std::print.
export TPP_CXX_STANDARD=23

"$external_python" - "$ROOT" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "benchmarks/_internal"))
from free_order_campaign import ensure_binary

binary = ensure_binary()
print(f"Verified our solver build: {binary}")
PY
"$ROOT/.build/unordered/tpp" --help >/dev/null

if ((setup_only)); then
	printf 'Pinned German suite: %s cases (%s)\n' "$EXPECTED_CASES" "$EXPECTED_SUITE_SHA256"
	printf 'Fekete revision: %s\n' "$current_submodule"
	printf 'Setup complete: dependencies verified and both solvers compiled.\n'
	printf 'Run the comparison with: scripts/run_comparison.sh'
	if [[ "$campaign_name" != 'german-free-order-comparison-v1' ]]; then
		printf ' --campaign %s' "$campaign_name"
	fi
	if ((threads_per_instance != 1 || workers != 1 || max_seconds != -1 || max_calls != 100000000)); then
		printf ' --threads-per-instance %s --workers %s --max-seconds %s --max-calls %s' \
			"$threads_per_instance" "$workers" "$max_seconds" "$max_calls"
	fi
	if [[ "$build_jobs" != "${TPP_BUILD_JOBS:-8}" ]]; then printf ' --build-jobs %s' "$build_jobs"; fi
	if ((force)); then printf ' --force'; fi
	printf '\n'
	exit 0
fi

campaign_dir="$ROOT/benchmarks/campaigns/$campaign_name"
mkdir -p "$campaign_dir"
"$external_python" - "$campaign_dir" "$SUITE" "$campaign_name" "$EXPECTED_SUITE_SHA256" "$EXPECTED_CASES" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

campaign = pathlib.Path(sys.argv[1])
suite = pathlib.Path(sys.argv[2]).resolve()
name = sys.argv[3]
expected_hash = sys.argv[4]
expected_cases = int(sys.argv[5])
metadata_path = campaign / "campaign.json"
relative_suite = os.path.relpath(suite, campaign)
actual_hash = hashlib.sha256(suite.read_bytes()).hexdigest()
if actual_hash != expected_hash:
    raise SystemExit(f"Suite hash changed during setup: {actual_hash}")

if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise SystemExit(f"Invalid campaign metadata; preserving it: {metadata_path}")
    inputs = metadata.get("inputs", [])
    source_file = inputs[0].get("file") if len(inputs) == 1 and isinstance(inputs[0], dict) else None
    if not source_file or (campaign / source_file).resolve() != suite:
        raise SystemExit(f"Campaign input differs from the German suite: {metadata_path}")
    source = metadata.get("source")
    if not isinstance(source, dict) or source.get("sha256") != expected_hash:
        raise SystemExit(f"Campaign records a different German suite hash: {metadata_path}")
else:
    if any(campaign.iterdir()):
        raise SystemExit(f"Campaign directory has files but no campaign.json; preserving it: {campaign}")
    metadata = {
        "schema_version": 1,
        "name": f"German 558-case fixed-endpoint free-order comparison ({name})",
        "type": "free_order_comparison",
        "inputs": [{"file": relative_suite}],
        "source": {
            "file": "benchmarks/suites/german-instances.bin",
            "sha256": expected_hash,
            "case_count": expected_cases,
        },
    }
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
PY

printf 'Pinned German suite: %s cases (%s)\n' "$EXPECTED_CASES" "$EXPECTED_SUITE_SHA256"
printf 'Fekete revision: %s\n' "$current_submodule"
printf 'Campaign: %s\n' "$campaign_dir"
printf 'Run settings: workers=%s, threads/instance=%s, max-seconds=%s, max-calls=%s\n' \
	"$workers" "$threads_per_instance" "$max_seconds" "$max_calls"

relative_gap="$("$external_python" -c 'eps = 0.001; print(format(eps / (1.0 + eps), ".17g"))')"
command=("$external_python" "$ROOT/benchmarks/tpp.py" free-order "$campaign_name"
	--solver unordered --solver tspn --max-instances "$EXPECTED_CASES"
	--max-calls "$max_calls" --max-seconds "$max_seconds"
	--threads-per-instance "$threads_per_instance" --workers "$workers"
	--absolute-gap 0 --relative-gap "$relative_gap" --eps 0.001
	--feasibility-tolerance 0.001 --validation-tolerance 1e-7)
if ((force)); then
	command+=(--force)
fi
"${command[@]}"
