#!/usr/bin/env bash
# Configuração E: o nosso solver no protocolo da A (uma execução por vez com uma thread por
# CPU lógica dos núcleos de performance; --threads N escolhe outro número), com as opções da
# comparação de 16/10 e o modo paralelo escolhido em fekete_reference.OUR_PARALLEL. O
# argumento é o limite por execução em segundos (padrão 300). Roda em segundo plano,
# retoma de onde parou e espera as outras configurações terminarem; o diretório leva o
# commit do nosso código. Prepare antes com scripts/install_dependencies.sh.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$ROOT/benchmarks/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Rode antes ./scripts/install_dependencies.sh' >&2; exit 2; }
exec "$python_bin" "$ROOT/benchmarks/tpp.py" ours-threads "$@"
