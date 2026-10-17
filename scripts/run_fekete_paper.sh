#!/usr/bin/env bash
# Configuração A do solver original de Fekete et al. (código do repositório deles, sem
# modificações) nas 558 instâncias do artigo: raiz CHR, uma execução por vez com uma thread
# por CPU lógica dos núcleos de performance (o artigo usou as 24 do Ryzen deles; --threads N
# escolhe outro número), três repetições. O argumento é o limite por execução em segundos
# (padrão 300).
# Roda em segundo plano e retoma de onde parou; --plan mostra só o plano. A lógica vive em
# benchmarks/_internal/fekete_reference.py; prepare antes com scripts/install_dependencies.sh.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$ROOT/benchmarks/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Rode antes ./scripts/install_dependencies.sh' >&2; exit 2; }
exec "$python_bin" "$ROOT/benchmarks/tpp.py" fekete-paper "$@"
