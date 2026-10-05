# Aritmética exata do oráculo convexo — 2026-10-05

Formulação: TPP euclidiano de ordem livre com extremos fixos
(`tpp_nonconvex_unordered_solve`). Comparação do nosso solver: binário `final`
da rodada anterior (limites de visita ativos) × branch `free-order-oracle-perf`.
Registro completo das tentativas:
[`unordered-tpp-experiments.md`](../../../docs/algorithms/unordered-tpp-experiments.md).

## Mudanças (todas preservam a busca)

- Arena por thread para os nós do DAG de `FilteredRational`.
- Sinais dos predicados do mapa filtrado decididos só por intervalos; DAG
  construído apenas se o intervalo não decide.
- KKT, materialização e `set_exact_bounds` em inteiros homogêneos, sem MDC.
- Caixa por polígono antes do teste de caixas aresta a aresta do mapa.

## Protocolo

Corpus `fekete-comparison/instances.bin`; gap absoluto 0, relativo
`0.0009990009990009992`; tolerância de visita 1e-8; validação Shapely 1e-7;
teto 600 s; uma thread; variantes intercaladas; 3 repetições. macOS arm64
(M4 Pro), **na bateria** (`powermode 1`): tempos absolutos ~60% acima dos da
rodada anterior; só as razões intercaladas são comparáveis.

## Resultados

| Conjunto | Casos | Média geom. | Mediana | Mín. | Máx. | Soma das medianas | Gap fechado | Busca idêntica |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Difícil | 19 | **1,302×** | 1,312× | 0,990× | 2,056× | 466 → 295 s | 114/114 | 19/19 |
| Validação | 18 | **1,339×** | 1,365× | 0,983× | 2,069× | 234 → 176 s | 108/108 | 18/18 |

Busca idêntica: 144 campos determinísticos (caminho, ordem, limites,
chamadas, nós, contadores) iguais em todas as 222 execuções. Maiores ganhos:
419 2,02×, 408 2,06×, 415 2,07×, 406 1,97×, 411 1,90×, 540 1,85×, 261 1,84×,
417 1,79×. Casos dominados por consultas de visita ficam neutros
(0,98–1,01×). A validação foi medida uma única vez, no fim, sem orientar
nenhuma escolha.

## Limitações

- Uma máquina, na bateria, sem isolamento; casos < 0,5 s têm ruído alto.
- O conjunto difícil orientou o desenvolvimento.
- `exact=true` é fechamento do gap configurado, não otimalidade algébrica.
