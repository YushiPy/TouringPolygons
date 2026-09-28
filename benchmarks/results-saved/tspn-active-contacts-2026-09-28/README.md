# TSPN: contatos ativos e inicialização tardia do OpenMP

Esta campanha investiga as perdas de desempenho da comparação de 27/09 e
preserva entradas, medições brutas, parâmetros, hashes das fontes e análise.
O resultado final está em [analysis.md](analysis.md); as 33 comparações
individuais estão em [tspn-final/analysis.md](tspn-final/analysis.md).

As alterações partem de `924e8b6ecf74767a8a6321ff88a71e2b7eb8eb2d` e usam
o worktree `codex/convex-cycle-disjoint`. O submódulo externo permanece
inalterado em `f4aa78c631545e4e894732a0fe8aef45f455c34c`.

## Evidência

- `oracle-before/`: 16 relaxações convexas normalizadas, executável anterior,
  um aquecimento e uma repetição medida por backend. Diagnóstico, sem pretensão
  de comparação estatística para diferenças pequenas.
- `oracle-after/`: mesmas entradas, contatos ativos corrigidos, cinco repetições.
  APIs racional e double e SOCP Gurobi; certificados independentes.
- `tspn-after/`: somente a correção de contatos; 33 casos, três repetições.
- `tspn-final/`: também inicialização tardia do OpenMP; mesmos 33 casos,
  cinco repetições. Esta é a comparação final contra o B&B de Fekete.
- `comparison-inputs.json`: 25 casos da campanha anterior e oito adicionais,
  selecionados antes da comparação, sem usar seus tempos para escolhê-los.
- `comparison-summary.json`, `provenance.json` e `tests/`: agregados,
  proveniência e saídas dos testes afetados.

Os casos adicionais são os próximos dois de cada tamanho 5, 10, 15 e 20
na ordem do corpus `benchmarks/suites/german-instances.bin`. Os 16 subproblemas
usam os **fechos convexos**, inclusive quando o polígono original é não convexo.
As coordenadas são normalizadas pela caixa da instância completa, como no B&B.

## Reprodução

No checkout que contém estas alterações, usando diretórios de saída novos:

```bash
python3 benchmarks/tpp.py cycle-benchmark \
  --inputs benchmarks/results-saved/tspn-active-contacts-2026-09-28/oracle-after/instances.json \
  --output benchmarks/results/cycle-active-repeat --repetitions 5
python3 benchmarks/tpp.py tspn-benchmark \
  --inputs benchmarks/results-saved/tspn-active-contacts-2026-09-28/comparison-inputs.json \
  --output benchmarks/results/tspn-active-repeat --repetitions 5 --seconds 3
```

Se o submódulo estiver inicializado em outro checkout, forneça
`--fekete-source CAMINHO`. O baseline do oráculo deve ser reproduzido no commit
anterior, conforme `oracle-before/README.md`. Os parâmetros e comandos de
compilação efetivamente usados estão nos `config.json` de cada etapa.

Os tempos são da chamada nativa completa, excluindo lançamento do processo e
inicialização do ambiente Gurobi. O custo OpenMP era interno à nossa chamada e
continua sendo contabilizado quando o paralelismo é realmente usado.
