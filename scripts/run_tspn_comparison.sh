#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_SOURCE="$ROOT/third_party/tspn-socg"
SUITE="$EXTERNAL_SOURCE/instances/instances_socg_simplified.zip"
EXPECTED_SUITE_SHA256="210841184500cb444f332537c4291aeff502dde4e3adc4a4ee307a80db21711f"
EXPECTED_CASES=558

campaign_name="tspn-fekete-comparison-v1"
campaign_explicit=0
solver_choice="both"
seconds=60
external_timeout=75
repetitions=1
workers=1
relative_gap="1e-6"
cycle_optimizations="cache,features,root,interval"
portfolio_flag=""
search_strategy=""
capture_oracles=0
force=0
setup_only=0
dry_run=0

usage() {
	cat <<'EOF'
Usage: scripts/run_tspn_comparison.sh [options]

Build the solvers, then run or resume the 558-case Fekete TSPN campaign
(closed tour, free cyclic order, no fixed point) comparing our solver with
the pinned Fekete SOCP B&B. Results go to
benchmarks/campaigns/tspn-fekete-comparison-v1/ (raw JSONL, summary CSV,
per-stratum CSV, progress.json and analysis.md report).

Options:
  --campaign NAME             Override the campaign directory name
  --seconds N                 Per-instance native limit (default: 60)
  --external-timeout N        Per-instance wall-clock watchdog (default: 75)
  --repetitions N             Repetitions per instance per solver (default: 1)
  --relative-gap X            Target relative gap (default: 1e-6)
  --cycle-optimizations LIST  Comma-separated opt-ins
                              (default: cache,features,root,interval)
  --portfolio                 Cooperative portfolio mode (implies memo opt-in)
  --portfolio-no-sharing      Independent portfolio race
  --search-strategy NAME      best-bound or dfs-bfs (default: solver default)
  --capture-oracles           Record per-call oracle captures (diagnostic)
  --solver NAME               tpp-ours, tpp-fekete, or both (default: both).
                              tpp-ours does not need Gurobi.
  --workers N                 Run N parallel case shards (default: 1). Each
                              instance's real solve keeps 1 thread).
  --force                     Move the existing campaign directory aside
  --setup-only                Check dependencies and exit
  --dry-run                   Print the plan and exit
  -h, --help                  Show this help

Python 3.12+ is required (TPP_PYTHON overrides). The Fekete submodule must be
initialized at the pinned revision with its local patches; run
scripts/run_comparison.sh --setup-only --solver tpp-fekete once first when the
Conan dependencies in third_party/tspn-socg/.conan/release are missing. On
remote machines export GUROBI_HOME before running when Gurobi is not in the
macOS default location. Re-run the same command with --resume semantics built
in: the runner skips completed records and continues the campaign.
EOF
}

fail() {
	printf 'Error: %s\n' "$*" >&2
	exit 2
}

while (($#)); do
	case "$1" in
		--campaign)
			(($# >= 2)) || fail '--campaign requires a value'
			campaign_name="$2"; campaign_explicit=1; shift 2 ;;
		--seconds)
			(($# >= 2)) || fail '--seconds requires a value'
			seconds="$2"; shift 2 ;;
		--external-timeout)
			(($# >= 2)) || fail '--external-timeout requires a value'
			external_timeout="$2"; shift 2 ;;
		--repetitions)
			(($# >= 2)) || fail '--repetitions requires a value'
			repetitions="$2"; shift 2 ;;
		--relative-gap)
			(($# >= 2)) || fail '--relative-gap requires a value'
			relative_gap="$2"; shift 2 ;;
		--cycle-optimizations)
			(($# >= 2)) || fail '--cycle-optimizations requires a value'
			cycle_optimizations="$2"; shift 2 ;;
		--portfolio)
			portfolio_flag="--portfolio"; shift ;;
		--portfolio-no-sharing)
			portfolio_flag="--portfolio-no-sharing"; shift ;;
		--search-strategy)
			(($# >= 2)) || fail '--search-strategy requires a value'
			search_strategy="$2"; shift 2 ;;
		--capture-oracles)
			capture_oracles=1; shift ;;
		--solver)
			(($# >= 2)) || fail '--solver requires a value'
			solver_choice="$2"; shift 2 ;;
		--workers)
			(($# >= 2)) || fail '--workers requires a value'
			workers="$2"; shift 2 ;;
		--force)
			force=1; shift ;;
		--setup-only)
			setup_only=1; shift ;;
		--dry-run)
			dry_run=1; shift ;;
		-h|--help)
			usage; exit 0 ;;
		*)
			fail "unknown option: $1" ;;
	esac
done

[[ "$seconds" =~ ^(0\.[0-9]+|[1-9][0-9]*(\.[0-9]+)?)$ ]] || fail '--seconds must be a positive number'
[[ "$external_timeout" =~ ^(0\.[0-9]+|[1-9][0-9]*(\.[0-9]+)?)$ ]] || fail '--external-timeout must be a positive number'
[[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || fail '--repetitions must be a positive integer'
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || fail '--workers must be a positive integer'
[[ "$campaign_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && "$campaign_name" != '.' && "$campaign_name" != '..' ]] \
	|| fail '--campaign must be a simple name without path separators'
case "$search_strategy" in
	''|best-bound|dfs-bfs) ;;
	*) fail '--search-strategy must be best-bound or dfs-bfs' ;;
esac
case "$solver_choice" in
	tpp-ours) solver_args=(--solver ours) ;;
	tpp-fekete) solver_args=(--solver fekete) ;;
	both) solver_args=() ;;
	*) fail '--solver must be tpp-ours, tpp-fekete, or both' ;;
esac

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

[[ -f "$SUITE" ]] || fail "Fekete instance archive is missing: $SUITE"
actual_suite_sha256="$("$python_bin" - "$SUITE" <<'PY'
import hashlib
import pathlib
import sys

print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
[[ "$actual_suite_sha256" == "$EXPECTED_SUITE_SHA256" ]] \
	|| fail "Fekete instance archive SHA-256 mismatch: expected $EXPECTED_SUITE_SHA256, got $actual_suite_sha256"

expected_submodule="$(git -C "$ROOT" ls-tree HEAD -- third_party/tspn-socg | awk '$1 == "160000" {print $3}')"
[[ -n "$expected_submodule" ]] || fail 'third_party/tspn-socg is not pinned as a Git submodule in HEAD'
if [[ ! -e "$EXTERNAL_SOURCE/.git" ]]; then
	git -C "$ROOT" submodule update --init --recursive -- third_party/tspn-socg
fi
current_submodule="$(git -C "$EXTERNAL_SOURCE" rev-parse HEAD)"
[[ "$current_submodule" == "$expected_submodule" ]] \
	|| fail "Fekete submodule is at $current_submodule; expected pinned revision $expected_submodule"

conan_cmake_prefix="$EXTERNAL_SOURCE/.conan/release"
if [[ ! -d "$conan_cmake_prefix" ]]; then
	printf 'Conan C++ dependencies not found at %s\n' "$conan_cmake_prefix" >&2
	printf 'Run scripts/run_comparison.sh --setup-only --solver tpp-fekete first to prepare Fekete.\n' >&2
	exit 2
fi
for dependency in "cgal-config.cmake cgalConfig.cmake" "fmt-config.cmake fmtConfig.cmake" "BoostConfig.cmake" "nlohmann_json-config.cmake nlohmann_jsonConfig.cmake" "Eigen3Config.cmake"; do
	required=0
	case "$dependency" in
		"cgal-config.cmake cgalConfig.cmake"|"fmt-config.cmake fmtConfig.cmake") [[ "$solver_choice" != 'tpp-ours' ]] && required=1 ;;
		*) required=1 ;;
	esac
	((required)) || continue
	found=0
	for candidate in $dependency; do
		if [[ -f "$conan_cmake_prefix/$candidate" ]]; then found=1; break; fi
	done
	((found)) || fail "Conan setup did not generate a CMake package for: $dependency"
done
export CMAKE_PREFIX_PATH="$conan_cmake_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
printf 'Reusing Conan C++ dependencies: %s\n' "$conan_cmake_prefix"

if [[ -x "$EXTERNAL_SOURCE/.venv/bin/cmake" ]]; then
	export PATH="$EXTERNAL_SOURCE/.venv/bin:$PATH"
elif ! command -v cmake >/dev/null 2>&1; then
	printf 'CMake is not on PATH; install CMake 3.23+ or use the Fekete comparison environment.\n' >&2
	exit 2
fi

output_dir="$ROOT/benchmarks/campaigns/$campaign_name"

# Choose the newest C++ standard the default compiler's CMake supports.
# Remote Ubuntu 24.04 ships GCC 13 (cxx_std_23 only); newer toolchains may
# accept cxx_std_26. AppleClang's -std=c++26 works while its CMake feature
# cxx_std_26 is unknown, so probe at the CMake level like run_comparison.sh.
if [[ -z "${TPP_CXX_STANDARD:-}" ]]; then
	probe_cxx_standard() {
		local meta="$1" probe_dir status=0
		probe_dir="$(mktemp -d "${TMPDIR:-/tmp}/tpp-cxx-probe.XXXXXX")"
		cat > "$probe_dir/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.20)
project(tpp_cxx_probe LANGUAGES CXX)
add_executable(tpp_cxx_probe main.cpp)
target_compile_features(tpp_cxx_probe PRIVATE cxx_std_${meta})
EOF
		echo '#include <format>
int main(){return std::format("{}",0)=="0"?0:1;}' > "$probe_dir/main.cpp"
		cmake -S "$probe_dir" -B "$probe_dir/build" > "$probe_dir/config.log" 2>&1 || status=$?
		if ((status == 0)); then
			cmake --build "$probe_dir/build" > "$probe_dir/build.log" 2>&1 || status=$?
		fi
		if ((status == 0)) && "$probe_dir/build/tpp_cxx_probe"; then
			rm -rf "$probe_dir"
			return 0
		fi
		rm -rf "$probe_dir"
		return 1
	}
	if probe_cxx_standard 26; then
		TPP_CXX_STANDARD=26
	elif probe_cxx_standard 23; then
		TPP_CXX_STANDARD=23
	else
		fail 'the default c++ compiler supports neither cxx_std_26 nor cxx_std_23; export TPP_CXX_STANDARD and/or CC/CXX, or install a newer toolchain'
	fi
	export TPP_CXX_STANDARD
	printf 'Detected C++ standard: %s\n' "$TPP_CXX_STANDARD"
fi
if ((force)) && [[ -d "$output_dir" ]]; then
	backup="$output_dir.previous-$(date +%Y%m%d-%H%M%S)"
	mv "$output_dir" "$backup"
	printf 'Moved existing campaign to %s\n' "$backup"
fi

args=(--all --instances-zip "$SUITE" --output "$output_dir"
	--seconds "$seconds" --external-timeout "$external_timeout"
	--repetitions "$repetitions" --relative-gap "$relative_gap")
if ((${#solver_args[@]} > 0)); then args+=("${solver_args[@]}"); fi
if [[ -f "$output_dir/config.json" ]]; then
	args+=(--resume)
fi
if [[ -n "$portfolio_flag" ]]; then args+=("$portfolio_flag"); fi
if [[ -n "$search_strategy" ]]; then args+=(--search-strategy "$search_strategy"); fi
if ((capture_oracles)); then args+=(--capture-oracles); fi
IFS=',' read -ra optimizations <<< "$cycle_optimizations"
for opt in "${optimizations[@]}"; do
	opt="${opt// /}"
	[[ -n "$opt" ]] && args+=(--cycle-optimization "$opt")
done
if ((dry_run)); then args+=(--dry-run); fi

if ((setup_only)); then
	printf 'Setup complete: %s cases (%s), Fekete %s, GUROBI_HOME=%s\n' \
		"$EXPECTED_CASES" "$EXPECTED_SUITE_SHA256" "$current_submodule" "${GUROBI_HOME:-<unset>}"
	printf 'Run with: scripts/run_tspn_comparison.sh'
	if ((campaign_explicit)); then printf ' --campaign %s' "$campaign_name"; fi
	printf '\n'
	exit 0
fi

mkdir -p "$output_dir"
printf 'Archive: %s cases; campaign: %s\n' "$EXPECTED_CASES" "$output_dir"
printf 'Settings: seconds=%s, external-timeout=%s, repetitions=%s, gap=%s, portfolio=%s, search=%s, opts=%s\n' \
	"$seconds" "$external_timeout" "$repetitions" "$relative_gap" "${portfolio_flag:-none}" "${search_strategy:-default}" "$cycle_optimizations"

if (( workers == 1 || dry_run )); then
	"$python_bin" "$ROOT/benchmarks/tpp.py" tspn-benchmark "${args[@]}"
	exit $?
fi

# Parallel mode: build once, shard the archive, merge the finished shards.
fekete_flag=ON
build_targets=(tpp-fekete-cycle tpp-unordered)
if [[ "$solver_choice" == 'tpp-ours' ]]; then
	fekete_flag=OFF; build_targets=(tpp-unordered)
elif [[ "$solver_choice" == 'tpp-fekete' ]]; then
	build_targets=(tpp-fekete-cycle)
fi
build_dir="$ROOT/.build/tspn-comparison"
configure_cmd=(cmake -S "$ROOT/benchmarks/_internal/tspn_native" -B "$build_dir"
	-DFEKETE_SOURCE="$EXTERNAL_SOURCE" -DTARGET=main-unordered
	-DWITH_TSPN_FEKETE="$fekete_flag" -DTPP_CXX_STANDARD="$TPP_CXX_STANDARD")
if [[ -n "${GUROBI_HOME:-}" ]]; then configure_cmd+=(-DGUROBI_HOME="$GUROBI_HOME"); fi
nlohmann_header="$(find "$HOME/.conan2/p" -path '*/p/include/nlohmann/json.hpp' 2>/dev/null | head -1 || true)"
if [[ -n "$nlohmann_header" ]]; then configure_cmd+=(-DNLOHMANN_INCLUDE_DIR="$(dirname "$(dirname "$nlohmann_header")")"); fi
printf 'Building shared binaries in %s (%s)\n' "$build_dir" "${build_targets[*]}"
"${configure_cmd[@]}" > "$output_dir/build.txt" 2>&1 \
	|| { tail -40 "$output_dir/build.txt" >&2; fail 'configure failed'; }
cmake --build "$build_dir" --target "${build_targets[@]}" -j "${TPP_BUILD_JOBS:-8}" >> "$output_dir/build.txt" 2>&1 \
	|| { tail -40 "$output_dir/build.txt" >&2; fail 'build failed'; }

ours_binary="$build_dir/touring_polygons/tpp-unordered"
fekete_binary="$build_dir/tpp-fekete-cycle"
"$python_bin" - "$SUITE" "$workers" "$output_dir" "$ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[4])
sys.path.insert(0, str(root / 'benchmarks' / '_internal'))
import tspn_benchmark

archive = pathlib.Path(sys.argv[1])
workers = int(sys.argv[2])
out = pathlib.Path(sys.argv[3])
cases, counts = tspn_benchmark.select_all_socg_inputs(archive)
for i in range(workers):
    shard = [c for idx, c in enumerate(cases) if idx % workers == i]
    shard_dir = out / f'shard-{i}'
    shard_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'formulation': 'TSPN, free cyclic order, no fixed point, closed polygon regions',
        'selection': {'archive': str(archive), 'policy': 'all archive cases', 'shard': f'{i+1}/{workers}'},
        'instances': shard,
    }
    (shard_dir / 'instances.json').write_text(json.dumps(payload))
print(f'Shard instances written for {workers} workers: {len(cases)} cases total')
PY

shard_pids=()
for ((i=0; i<workers; i++)); do
	shard_dir="$output_dir/shard-$i"
	shard_args=(--inputs "$shard_dir/instances.json" --output "$shard_dir"
		--seconds "$seconds" --external-timeout "$external_timeout"
		--repetitions "$repetitions" --relative-gap "$relative_gap"
		--skip-build --build-dir "$build_dir" --ours-binary "$ours_binary")
	if [[ "$solver_choice" != 'tpp-ours' ]]; then shard_args+=(--fekete-binary "$fekete_binary"); fi
	if ((${#solver_args[@]} > 0)); then shard_args+=("${solver_args[@]}"); fi
	if [[ -f "$shard_dir/config.json" ]]; then shard_args+=(--resume); fi
	if [[ -n "$portfolio_flag" ]]; then shard_args+=("$portfolio_flag"); fi
	if [[ -n "$search_strategy" ]]; then shard_args+=(--search-strategy "$search_strategy"); fi
	if ((capture_oracles)); then shard_args+=(--capture-oracles); fi
	for opt in "${optimizations[@]}"; do
		opt="${opt// /}"
		[[ -n "$opt" ]] && shard_args+=(--cycle-optimization "$opt")
	done
	echo "Shard $i: ${#shard_args[@]} args; log $shard_dir/run.log"
	"$python_bin" "$ROOT/benchmarks/tpp.py" tspn-benchmark "${shard_args[@]}" \
		> "$shard_dir/run.log" 2>&1 &
	shard_pids+=($!)
done

rc=0
for pid in "${shard_pids[@]}"; do
	wait "$pid" || rc=$?
done
if ((rc != 0)); then printf 'Warning: at least one shard exited with status %s; merging available records.\n' "$rc" >&2; fi

"$python_bin" - "$output_dir" "$workers" "$ROOT" <<'PY'
import json
import sys
import pathlib

out = pathlib.Path(sys.argv[1])
workers = int(sys.argv[2])
root = pathlib.Path(sys.argv[3])
sys.path.insert(0, str(root / 'benchmarks' / '_internal'))
from tspn_diagnostics import digest

shards = [out / f'shard-{i}' for i in range(workers)]
rows = []
for shard in shards:
    raw = shard / 'raw.jsonl'
    if raw.exists():
        rows.extend(line for line in raw.read_text().splitlines() if line.strip())
(out / 'raw.jsonl').write_text('\n'.join(rows) + ('\n' if rows else ''))
instances = []
formulation = None
for shard in shards:
    payload = json.loads((shard / 'instances.json').read_text())
    formulation = formulation or payload['formulation']
    instances.extend(payload['instances'])
config = json.loads((shards[0] / 'config.json').read_text())
full = {'formulation': formulation, 'instances': instances,
        'selection': {'policy': 'all archive cases', 'shards': workers}}
(out / 'instances.json').write_text(json.dumps(full))
config['inputs_sha256'] = digest(full)
config['plan']['cases'] = len(instances)
config['plan']['planned_runs'] = len(instances) * config['repetitions'] * len(config.get('solvers', ['ours', 'fekete']))
config['plan']['shards'] = workers
(out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
print(f'Merged {len(rows)} rows from {workers} shards into {out}')
PY
if [[ $? -ne 0 ]]; then echo 'Warning: merge failed' >&2; fi
"$python_bin" "$ROOT/benchmarks/tpp.py" tspn-benchmark --report-only --output "$output_dir" || true
exit $rc
