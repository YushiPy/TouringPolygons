#!/usr/bin/env bash
# Atalho para `tpp.py tspn-compare`: toda a lógica da campanha TSPN vive em
# benchmarks/_internal/tspn_campaign.py. Veja `scripts/run_tspn_comparison.sh --help`.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TPP_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
	python_bin="$(command -v python3.12 || command -v python3)"
elif [[ ! -x "$python_bin" ]]; then
	python_bin="$(command -v "$python_bin" || true)"
fi
[[ -n "$python_bin" ]] || { echo 'Error: could not find the selected Python executable' >&2; exit 2; }

"$python_bin" "$ROOT/benchmarks/tpp.py" setup --python "$python_bin"
exec "$ROOT/benchmarks/.venv/bin/python" "$ROOT/benchmarks/tpp.py" tspn-compare "$@"
