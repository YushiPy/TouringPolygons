#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

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
		if ! have node || ! have npm; then formulae+=(node); fi
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
		local packages=()
		have python3 || packages+=(python3)
		have cmake || packages+=(cmake)
		have c++ || packages+=(build-essential)
		have node || packages+=(nodejs npm)
		[[ -f /usr/include/eigen3/Eigen/Core ]] || packages+=(libeigen3-dev)
		[[ -f /usr/include/boost/multiprecision/cpp_bin_float.hpp ]] || packages+=(libboost-dev)
		ldconfig -p 2>/dev/null | grep -q libomp || packages+=(libomp-dev)

		if (( ${#packages[@]} > 0 )); then
			echo "+ sudo apt-get update"
			sudo apt-get update
			echo "+ sudo apt-get install -y ${packages[*]}"
			sudo apt-get install -y "${packages[@]}"
		else
			echo "System dependencies: already installed"
		fi

		if ! have uv; then
			echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun this script." >&2
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
