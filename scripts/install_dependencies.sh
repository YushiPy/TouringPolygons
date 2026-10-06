#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

web=1
for argument in "$@"; do
	case "$argument" in
		--solvers-only|--no-web) web=0 ;;
		-h|--help)
			cat <<'EOF'
Usage: scripts/install_dependencies.sh [--solvers-only]

Install what the project needs, never with sudo (Homebrew on macOS needs none; on Linux
missing system packages are only listed). With --solvers-only (alias --no-web) only what the
two solvers and the benchmark CLI need is installed: no Node.js, no dashboard
environment, no npm packages and no Playwright browser.
EOF
			exit 0
			;;
		*) echo "Unknown option: $argument (see --help)" >&2; exit 2 ;;
	esac
done

have() {
	command -v "$1" >/dev/null 2>&1
}

install_system_dependencies() {
	if [[ "$(uname -s)" == "Darwin" ]]; then
		if ! have brew; then
			echo "Homebrew is required. Install it from https://brew.sh and rerun this script." >&2
			exit 1
		fi
		if ! have c++ || ! have make; then
			echo "Apple Command Line Tools are required. Run 'xcode-select --install' and rerun this script." >&2
			exit 1
		fi

		local formulae=()
		have python3 || formulae+=(python)
		have cmake || formulae+=(cmake)
		have uv || formulae+=(uv)
		if (( web )) && { ! have node || ! have npm; }; then formulae+=(node); fi
		for formula in libomp eigen boost; do
			brew --prefix "$formula" >/dev/null 2>&1 || formulae+=("$formula")
		done

		if (( ${#formulae[@]} > 0 )); then
			echo "+ brew install ${formulae[*]}"
			brew install "${formulae[@]}"
		else
			echo "System dependencies: already installed"
		fi
		return
	fi

	if have apt-get; then
		# Nothing is installed with sudo here: only what cannot be had without root is reported.
		# CMake and Ninja come from the Python environments; Eigen and Boost headers come from
		# Fekete's Conan setup or `tpp.py build --fetch-deps`.
		local packages=()
		have python3 || packages+=(python3)
		have git || packages+=(git)
		python3 -c 'import venv, ensurepip' >/dev/null 2>&1 || packages+=(python3-venv)
		have c++ || packages+=(build-essential)
		if (( web )); then have node || packages+=(nodejs npm); fi
		if (( ${#packages[@]} > 0 )); then
			echo "Missing system packages: ${packages[*]}" >&2
			echo "This script never uses sudo. Ask an administrator to install them (Debian/Ubuntu: sudo apt-get install ${packages[*]})." >&2
			exit 1
		fi
		echo "System tools: present (a C++23 compiler such as g++ 13 or newer is also required; checked by the build)"

		if ! have uv; then
			echo "uv is required and installs in your home directory without root: see https://docs.astral.sh/uv/ and rerun this script." >&2
			exit 1
		fi
		return
	fi

	echo "Unsupported operating system: install Python, uv, CMake, a C++ compiler, Node.js, OpenMP, Eigen, and Boost manually." >&2
	exit 1
}

sync_python_app() {
	local directory="$1"
	echo
	echo "==> Python dependencies: $directory"
	(
		cd "$directory"
		uv sync --frozen
	)
}

sync_node_app() {
	local directory="$1"
	echo
	echo "==> Node dependencies: $directory"
	(
		cd "$directory"
		npm ci
	)
}

install_system_dependencies

echo
echo "==> Git submodules"
git submodule update --init --recursive

python3 benchmarks/tpp.py setup

# Eigen and Boost headers: without system packages (no root), fetch the pinned releases
# (SHA-256 checked) into .cache/deps, where the build finds them.
if [[ "$(uname -s)" != "Darwin" ]] \
	&& { [[ ! -f /usr/include/eigen3/Eigen/Core && ! -f /usr/local/include/eigen3/Eigen/Core ]] \
		|| [[ ! -f /usr/include/boost/multiprecision/cpp_bin_float.hpp && ! -f /usr/local/include/boost/multiprecision/cpp_bin_float.hpp ]]; }; then
	echo
	echo "==> Eigen/Boost headers (no system packages found; fetching pinned releases, no root needed)"
	python3 benchmarks/tpp.py build --fetch-deps
fi

if (( web )); then
	sync_python_app apps/benchmark-dashboard
	sync_node_app apps/benchmark-dashboard

	echo
	echo "==> Playwright Chromium"
	(
		cd apps/benchmark-dashboard
		npx playwright install chromium
	)

	echo
	echo "All project dependencies are installed."
else
	echo
	echo "==> Check"
	python3 benchmarks/tpp.py doctor || true
	echo
	echo "Solver dependencies are installed (web dependencies skipped)."
	echo "Next: python3 benchmarks/tpp.py free-compare --setup-only --solver both"
	echo "      (builds both solvers; Fekete also needs a valid Gurobi license)."
fi
