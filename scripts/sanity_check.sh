#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

install_missing=1
full=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		--no-install)
			install_missing=0
			shift
			;;
		--full)
			full=1
			shift
			;;
		-h|--help)
			cat <<'USAGE'
usage: scripts/sanity_check.sh [options]

Builds the solvers and runs their tests in about a minute once the build is warm
(the first build compiles everything and takes longer). Benchmarks are not part
of it: use python3 benchmarks/tpp.py.

Options:
  --no-install  Report missing dependencies instead of installing them.
  --full        Also run the generated convex stress suites (about 25 minutes more).
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

# Nothing here is installed with sudo: on Linux the missing system packages are only listed.
report_missing_packages() {
	local packages=("$@")
	if (( ${#packages[@]} == 0 )); then
		return
	fi
	echo "Missing system packages: ${packages[*]}" >&2
	echo "This script never uses sudo. Ask an administrator to install them (on Debian/Ubuntu: sudo apt-get install ${packages[*]})." >&2
	echo "Eigen and Boost headers do not need root: python3 benchmarks/tpp.py build --fetch-deps" >&2
	exit 1
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
		for package in eigen boost; do
			if ! have brew || ! brew --prefix "$package" >/dev/null 2>&1; then missing+=("$package"); fi
		done
	else
		[[ -f /usr/include/eigen3/Eigen/Core || -f /usr/local/include/eigen3/Eigen/Core ]] || missing+=(eigen)
		[[ -f /usr/include/boost/multiprecision/cpp_bin_float.hpp || -f /usr/local/include/boost/multiprecision/cpp_bin_float.hpp ]] || missing+=(boost)
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
				eigen) apt_packages+=(libeigen3-dev) ;;
				boost) apt_packages+=(libboost-dev) ;;
				*) apt_packages+=("$item") ;;
			esac
		done
		report_missing_packages "${apt_packages[@]}"
	fi
}

run_step() {
	echo
	echo "==> $*"
	local started=$SECONDS
	"$@"
	echo "    ($((SECONDS - started)) s)"
}

unordered_example() {
	.build/tools/bin/tpp-unordered < packages/nonconvex-tpp/cpp/tests/unordered-example.txt >/dev/null
}

started=$SECONDS
ensure_dependencies

tests=(
	tpp-convex-intersection-tests
	tpp-convex-directional-tests
	tpp-convex-cycle-tests
	tpp-convex-cycle-certificate-tests
	tpp-convex-disk-cycle-tests
	tpp-unordered-tests
	tpp-tspn-tests
)

# tpp-convex-cetsp: for the end-to-end checks in benchmarks/tests/test_cetsp_cli.py.
run_step python3 benchmarks/tpp.py build "${tests[@]}" tpp-unordered tpp-convex-cetsp
for test in "${tests[@]}"; do
	run_step ".build/tools/bin/$test"
done
run_step unordered_example
if (( full )); then
	run_step packages/convex-tpp/cpp/run_generated_tests.sh
else
	run_step packages/convex-tpp/cpp/run_generated_tests.sh --quick
fi
run_step python3 benchmarks/tpp.py generate-suites
# TPP_REQUIRE_NATIVE: a test whose native tool is missing fails rather than skips.
run_step env TPP_REQUIRE_NATIVE=1 benchmarks/.venv/bin/python -m unittest discover -b -s benchmarks/tests

echo
echo "Sanity check passed in $((SECONDS - started)) s."
