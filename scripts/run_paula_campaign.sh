#!/usr/bin/env bash
# Atalho para `tpp.py paula-campaign`: prepara tudo sem sudo (uv, ambiente Python,
# headers, GMP e o solver) e executa ou retoma a campanha longa nas instâncias da
# Paula. A lógica vive em benchmarks/_internal/paula_campaign.py; veja --help.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# uv prepara benchmarks/.venv (CMake vem dele). Sem uv, instala-o em ~/.local/bin, sem root.
for directory in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
	[[ -x "$directory/uv" ]] && export PATH="$directory:$PATH"
done
if ! command -v uv >/dev/null 2>&1; then
	echo "==> uv (official installer, into ~/.local/bin; no root)"
	if command -v curl >/dev/null 2>&1; then
		curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
	elif command -v wget >/dev/null 2>&1; then
		wget -qO- https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
	else
		echo 'Error: uv is missing and neither curl nor wget is available to install it.' >&2
		exit 2
	fi
	export PATH="$HOME/.local/bin:$PATH"
fi

python_bin="${TPP_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
	python_bin="$(command -v python3.12 || command -v python3 || true)"
elif [[ ! -x "$python_bin" ]]; then
	python_bin="$(command -v "$python_bin" || true)"
fi
[[ -n "$python_bin" ]] || { echo 'Error: could not find Python 3.12 or newer (set TPP_PYTHON)' >&2; exit 2; }

"$python_bin" "$ROOT/benchmarks/tpp.py" setup --python "$python_bin"
exec "$ROOT/benchmarks/.venv/bin/python" "$ROOT/benchmarks/tpp.py" paula-campaign "$@"
