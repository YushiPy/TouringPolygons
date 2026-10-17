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

Install what the project needs without sudo: what is missing and can be had without root is
downloaded into your home directory or the repository (uv, Python 3.12, CMake, C++ headers,
Fekete's original solver and its Conan packages); what needs an administrator is only
listed, and the script stops so you can install it and run it again. On macOS, Homebrew is
used when present but is not required. With --solvers-only (alias --no-web) Node.js, the
dashboard environment and its Playwright browser are skipped.
EOF
			exit 0
			;;
		*) echo "Unknown option: $argument (see --help)" >&2; exit 2 ;;
	esac
done

have() {
	command -v "$1" >/dev/null 2>&1
}

skipped=()

# Tools that only an administrator can install. Everything else is downloaded below.
check_system_tools() {
	local missing=()
	have git || missing+=(git)
	have make || missing+=(make)
	{ have c++ || have g++ || have clang++; } || missing+=(c++)
	have curl || have wget || missing+=(curl)
	if (( ${#missing[@]} == 0 )); then
		return
	fi
	echo "Missing system tools: ${missing[*]}" >&2
	if [[ "$(uname -s)" == "Darwin" ]]; then
		echo "Install Apple's Command Line Tools with 'xcode-select --install' (no sudo needed) and rerun this script." >&2
	else
		echo "This script never uses sudo. Ask someone with administrator rights to install them" >&2
		echo "(Debian/Ubuntu: sudo apt-get install build-essential git curl), then rerun this script." >&2
	fi
	exit 1
}

install_homebrew_formulae() {
	local formulae=()
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
		echo "Homebrew packages: already installed"
	fi
}

# uv provides Python and the locked environments; it installs in ~/.local/bin without root.
ensure_uv() {
	for directory in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
		[[ -x "$directory/uv" ]] && export PATH="$directory:$PATH"
	done
	have uv && return
	echo
	echo "==> uv (official installer, into ~/.local/bin; no root)"
	if have curl; then
		curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
	else
		wget -qO- https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
	fi
	export PATH="$HOME/.local/bin:$PATH"
	have uv || { echo "uv could not be installed; see https://docs.astral.sh/uv/" >&2; exit 1; }
}

# A Python to start benchmarks/tpp.py (3.11 or newer); uv downloads 3.12 when the system has none.
# The benchmark environment itself always gets the pinned 3.12 (benchmarks/.python-version).
find_python() {
	local candidate
	for candidate in python3.12 python3.13 python3.14 python3; do
		if have "$candidate" && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
			command -v "$candidate"
			return
		fi
	done
	UV_CACHE_DIR="$repo_root/.cache/uv" uv python install 3.12 >&2
	uv python find 3.12
}

check_system_tools
if [[ "$(uname -s)" == "Darwin" ]]; then
	if have brew; then
		install_homebrew_formulae
	else
		echo "Homebrew not found: continuing without it (installing Homebrew needs an administrator)."
		echo "Without it our solver cannot be built on macOS (it needs Homebrew's libomp); Fekete's solver can."
		skipped+=("our solver on macOS (needs Homebrew's libomp: brew install libomp)")
	fi
fi
ensure_uv
python_bin="$(find_python)"
echo "Python for the benchmark CLI: $python_bin"

echo
echo "==> Git submodules"
git submodule update --init --recursive

"$python_bin" benchmarks/tpp.py setup
cli=("$repo_root/benchmarks/.venv/bin/python" benchmarks/tpp.py)

# Eigen and Boost headers: without system packages (no root), fetch the pinned releases
# (SHA-256 checked) into .cache/deps, where the build finds them.
if ! have brew \
	&& { [[ ! -f /usr/include/eigen3/Eigen/Core && ! -f /usr/local/include/eigen3/Eigen/Core ]] \
		|| [[ ! -f /usr/include/boost/multiprecision/cpp_bin_float.hpp && ! -f /usr/local/include/boost/multiprecision/cpp_bin_float.hpp ]]; }; then
	echo
	echo "==> Eigen/Boost headers (no system packages found; fetching pinned releases, no root needed)"
	"${cli[@]}" build --fetch-deps
fi

echo
echo "==> Fekete et al.'s original solver (upstream code, unmodified; Conan packages and Gurobi 12 into .build/)"
"${cli[@]}" fekete-setup

if [[ "$(uname -s)" != "Darwin" ]] || have brew; then
	echo
	echo "==> Our solver (GMP and headers without root when missing; builds and runs its tests once)"
	"${cli[@]}" ours-setup
fi

if (( web )); then
	if have node && have npm; then
		echo
		echo "==> Python dependencies: apps/benchmark-dashboard"
		(cd apps/benchmark-dashboard && uv sync --frozen)
		echo
		echo "==> Node dependencies: apps/benchmark-dashboard"
		(cd apps/benchmark-dashboard && npm ci)
		echo
		echo "==> Playwright Chromium"
		(cd apps/benchmark-dashboard && npx playwright install chromium)
	else
		skipped+=("the dashboard (needs Node.js and npm; not needed for benchmarks)")
	fi
fi

echo
echo "==> Check"
"${cli[@]}" doctor || true
echo
if (( ${#skipped[@]} > 0 )); then
	echo "Installed everything except:"
	for item in "${skipped[@]}"; do
		echo "  - $item"
	done
else
	echo "All project dependencies are installed."
fi
echo "Reference runs: scripts/run_fekete_paper.sh (A), scripts/run_fekete_single_thread.sh (B) and"
echo "scripts/run_ours_single_thread.sh (C); add --plan to see the plan first."
