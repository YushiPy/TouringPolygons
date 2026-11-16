#!/usr/bin/env bash
# Artigo 1, Seção 7: todos os experimentos em sequência nesta máquina, num job em
# segundo plano que sobrevive ao fim da sessão SSH. Rodar de novo retoma de onde parou
# (cada etapa grava done.json). Os argumentos passam adiante, por exemplo --smoke (tamanhos
# pequenos, minutos) ou --plan (só mostra as etapas); --profile paper roda só o que o
# artigo usa desde 9/11 (caminhos, construções e ciclos, sem B&B), e --profile scaling
# só a varredura de escala (polígonos regulares em linha e em zigue-zague). A lógica vive em
# benchmarks/_internal/paper1_experiments.py; prepare antes com
# scripts/install_dependencies.sh. Precisa da licença do Gurobi para este usuário.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$ROOT/benchmarks/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Rode antes ./scripts/install_dependencies.sh' >&2; exit 2; }
exec "$python_bin" "$ROOT/benchmarks/tpp.py" paper1-experiments "$@"
