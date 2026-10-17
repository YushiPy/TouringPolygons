#!/usr/bin/env bash
# Configuração D: como a A (solver original de Fekete et al., uma execução por vez com uma
# thread por CPU lógica dos núcleos de performance; --threads N escolhe outro número), mas
# com a raiz LEFP. O argumento é o limite por execução em segundos (padrão 300). Roda em
# segundo plano, retoma de onde parou e espera as outras configurações terminarem. A lógica
# vive em benchmarks/_internal/fekete_reference.py; prepare antes com
# scripts/install_dependencies.sh.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$ROOT/benchmarks/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Rode antes ./scripts/install_dependencies.sh' >&2; exit 2; }
exec "$python_bin" "$ROOT/benchmarks/tpp.py" fekete-paper-lefp "$@"
