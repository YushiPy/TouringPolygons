#!/usr/bin/env bash
# Configuração B do solver original de Fekete et al. (código do repositório deles, sem
# modificações) nas 558 instâncias do artigo: uma thread, raízes CHR e LEFP, uma execução
# por núcleo de performance, três repetições. O único argumento é o limite por execução em
# segundos (padrão 300). Roda em segundo plano, retoma de onde parou e, se a configuração A
# estiver rodando, espera ela terminar; --plan mostra só o plano. A lógica vive em
# benchmarks/_internal/fekete_reference.py; prepare antes com scripts/install_dependencies.sh.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$ROOT/benchmarks/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Rode antes ./scripts/install_dependencies.sh' >&2; exit 2; }
exec "$python_bin" "$ROOT/benchmarks/tpp.py" fekete-single-thread "$@"
