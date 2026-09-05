#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

install_missing=1
threads=""
benchmark_instances=30
benchmark_max_calls=50000

while [[ $# -gt 0 ]]; do
	case "$1" in
		--no-install)
			install_missing=0
			shift
			;;
		--threads)
			threads="$2"
			shift 2
			;;
		--benchmark-instances)
			benchmark_instances="$2"
			shift 2
			;;
		--benchmark-max-calls)
			benchmark_max_calls="$2"
			shift 2
			;;
		-h|--help)
			cat <<'USAGE'
usage: scripts/sanity_check.sh [options]

Options:
  --no-install                 Report missing dependencies instead of installing them.
  --threads N                  Benchmark worker threads. Default: all hardware threads.
  --benchmark-instances N      Number of dev-suite cases to benchmark. Default: 30.
  --benchmark-max-calls N      Convex-call cap per benchmark case. Default: 50000.
USAGE
			exit 0
			;;
		*)
			echo "Unknown option: $1" >&2
			exit 2
			;;
	esac
done

have() {
	command -v "$1" >/dev/null 2>&1
}

install_with_homebrew() {
	local packages=("$@")
	if (( ${#packages[@]} == 0 )); then
		return
	fi
	if ! have brew; then
		echo "Missing Homebrew; install these dependencies manually: ${packages[*]}" >&2
		exit 1
	fi
	echo "+ brew install ${packages[*]}"
	brew install "${packages[@]}"
}

install_with_apt() {
	local packages=("$@")
	if (( ${#packages[@]} == 0 )); then
		return
	fi
	if ! have apt-get; then
		echo "Missing apt-get; install these dependencies manually: ${packages[*]}" >&2
		exit 1
	fi
	echo "+ sudo apt-get update"
	sudo apt-get update
	echo "+ sudo apt-get install -y ${packages[*]}"
	sudo apt-get install -y "${packages[@]}"
}

ensure_dependencies() {
	local missing=()
	have python3 || missing+=(python3)
	have cmake || missing+=(cmake)
	have c++ || missing+=(cxx)
	have make || missing+=(make)

	if [[ "$(uname -s)" == "Darwin" ]]; then
		if ! have brew || ! brew --prefix libomp >/dev/null 2>&1; then
			missing+=(libomp)
		fi
	else
		if ! ldconfig -p 2>/dev/null | grep -q 'libomp'; then
			missing+=(libomp)
		fi
	fi

	if (( ${#missing[@]} == 0 )); then
		echo "Dependencies: ok"
		return
	fi

	if (( install_missing == 0 )); then
		echo "Missing dependencies: ${missing[*]}" >&2
		exit 1
	fi

	if [[ "$(uname -s)" == "Darwin" ]]; then
		local brew_packages=()
		for item in "${missing[@]}"; do
			case "$item" in
				python3) brew_packages+=(python) ;;
				cxx|make) ;;
				*) brew_packages+=("$item") ;;
			esac
		done
		install_with_homebrew "${brew_packages[@]}"
	else
		local apt_packages=()
		for item in "${missing[@]}"; do
			case "$item" in
				cxx|make) apt_packages+=(build-essential) ;;
				libomp) apt_packages+=(libomp-dev) ;;
				*) apt_packages+=("$item") ;;
			esac
		done
		install_with_apt "${apt_packages[@]}"
	fi
}

run_step() {
	echo
	echo "==> $*"
	"$@"
}

ensure_dependencies

run_step python3 benchmarks/tpp.py generate-suites
run_step packages/convex-tpp/cpp/run_generated_tests.sh

run_step cmake --preset nonconvex-release -DTARGET=main-bnb_workload_benchmark
run_step cmake --build --preset nonconvex-release

benchmark_dir="benchmarks/results/sanity"
mkdir -p "$benchmark_dir"
timestamp="$(date +%Y%m%d-%H%M%S)"
csv_output="$benchmark_dir/dev-${timestamp}.csv"
summary_output="$benchmark_dir/dev-${timestamp}.md"

echo
echo "==> Non-convex benchmark smoke run"
benchmark_command=(
	.build/nonconvex-release/packages/nonconvex-tpp/cpp/tpp
	benchmarks/suites/algorithm-dev-v1.bin
	-1 "$benchmark_instances" "$benchmark_max_calls" -1 1
	"$csv_output" "$summary_output"
)

if [[ -n "$threads" ]]; then
	TPP_BENCH_THREADS="$threads" "${benchmark_command[@]}"
else
	"${benchmark_command[@]}"
fi

echo
echo "Benchmark summary: $summary_output"
sed -n '1,120p' "$summary_output"
