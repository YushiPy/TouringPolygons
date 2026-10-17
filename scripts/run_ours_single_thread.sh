#!/usr/bin/env bash
# Configuração C: o nosso solver nas 558 instâncias do artigo de Fekete et al., no mesmo
# protocolo da configuração B (uma thread, uma execução por núcleo de performance, três
# repetições), com as opções da comparação de 16/10. O único argumento é o limite por
# execução em segundos (padrão 300). Roda em segundo plano, retoma de onde parou e espera
# as configurações A e B terminarem; o diretório leva o commit do nosso código. A lógica
# vive em benchmarks/_internal/fekete_reference.py; prepare antes com
# scripts/install_dependencies.sh.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$ROOT/benchmarks/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Rode antes ./scripts/install_dependencies.sh' >&2; exit 2; }
exec "$python_bin" "$ROOT/benchmarks/tpp.py" ours-single-thread "$@"
