#!/usr/bin/env bash
# Atalho para `tpp.py free-compare`: toda a lógica (checagens, build do Fekete e do nosso solver,
# campanha, execução) vive em benchmarks/_internal/free_order_comparison.py.
# Veja `scripts/run_comparison.sh --help`. TPP_PYTHON escolhe o Python (3.12 ou mais novo).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
python_bin="${TPP_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
	python_bin="$(command -v python3.12 || command -v python3)"
elif [[ ! -x "$python_bin" ]]; then
	python_bin="$(command -v "$python_bin" || true)"
fi
[[ -n "$python_bin" ]] || { echo 'Error: could not find the selected Python executable' >&2; exit 2; }

"$python_bin" "$ROOT/benchmarks/tpp.py" setup --python "$python_bin"
exec "$ROOT/benchmarks/.venv/bin/python" "$ROOT/benchmarks/tpp.py" free-compare "$@"
