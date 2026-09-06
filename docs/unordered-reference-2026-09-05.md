# Manifesto reproduzível, TPP não convexo com ordem livre

Este manifesto identifica a referência histórica registrada em 5 de setembro de
2026, sem substituir nenhum resultado. Ele foi consolidado em 5 de setembro de
2026 no checkout abaixo.

## Escopo e artefatos preservados

A referência mede a suíte `algorithm-dev-v1.bin`, com 60 instâncias, extremos
fixos, ordem livre, uma thread e limite de dois segundos por instância. Os tempos
de resolução não incluem a inicialização dos processos.

| Artefato | SHA-256 |
| --- | --- |
| `benchmarks/results/unordered/final-dev.jsonl` | `8ad93f47533ba98b3fe7fb6707fc3d2bd22528f250da6d7e17cff3a17247785c` |
| `benchmarks/results/unordered/final-comparison/comparison.csv` | `6013e4dcecaf7ec2192f16be09feb026c34a8c591b9817596336bf76093e8a69` |
| `benchmarks/results/unordered/final-comparison/summary.json` | `1d89a26bf7fe344e87fe3056054ea6f177c7b0fa037f2286cb9ffddcc45a45e0` |
| `tspn-comparison/results/unordered-final/20260905-123147/algorithm-dev-v1-tspn-path.csv` | `c88e364cbd13f31c23e05c2f2f6015fdfd1044646a942adf1788da4c5fbdb946` |

Os resultados são locais e ignorados pelo Git. O executável local que os produziu
está em `.build/unordered/tpp`, tem data de modificação `2026-09-05 12:26:56 -0300`,
anterior aos resultados, e SHA-256
`a378f45b7cace5cfa17d86703c53b1f0f354a16b1f9e6ba7a9422b8f8ebb034a`.

## Identificação das fontes e da máquina

- Revisão Git: `8063dc953398a7d4b451be2911c086b1b181a176` (`main`),
  `2026-09-05 18:29:34 -0300`.
- Alterações locais na consolidação: somente
  `docs/plano-de-trabalho-tpp.md`, não rastreado. Não havia diff nas fontes,
  scripts ou dashboard usados pela referência.
- Identificador das fontes relevantes: SHA-256 de `git ls-files -s` para
  `packages/{common-geometry,convex-tpp,nonconvex-tpp,optimal-convex-partition}/cpp`,
  `benchmarks/scripts`, `apps/benchmark-dashboard` e `scripts/verify_unordered.sh`:
  `882bffec6c2e093c0efc482d0d38fba7115604914e79cd85a9e197d9ba3cc08a`.
- Máquina: MacBook Pro M4 Pro, arm64, macOS 26.6.2 (build `25G83`).
- Ferramentas de compilação: Apple Clang 17.0.0, CMake 4.4.3, configuração
  `Release`, C++26, `-O3`, OpenMP via Homebrew `libomp 22.1.8`.
- Dependências de headers: Eigen 5.0.1 e Boost 1.92.0. CGAL 6.2 está instalado
  para a decomposição. Gurobi não é necessário para o solver de produção.
- Ambiente do dashboard: `apps/benchmark-dashboard/.venv/bin/python`, Python
  3.14.6 e Shapely 2.1.2.
- Ambiente do externo: `tspn-comparison/solver/.venv/bin/python`, Python
  3.12.13, GurobiPy 12.0.3 e Shapely 2.1.2. O binário desse interpretador tem
  SHA-256 `71720f1fc66989ebd691e81c96111b47ae6ff3f1a478666084d1cacbf0fccbf2`.

O commit foi criado depois dos arquivos de resultado, por isso a identificação do
executável e do índice das fontes é mantida além da revisão Git. A igualdade exata
entre o executável histórico e uma recompilação nova não foi inferida apenas pela
revisão.

## Comandos reproduzíveis

Execute todos os comandos a partir da raiz do repositório. A compilação e os testes
do solver não requerem ambiente virtual Python:

```bash
cmake -S packages/nonconvex-tpp/cpp -B .build/unordered -DTARGET=main-unordered
cmake --build .build/unordered --target tpp tpp-unordered-tests -j 8
scripts/verify_unordered.sh
```

O último comando recompila e executa a suíte C++ e o exemplo, portanto pode ser
usado sozinho para a verificação padrão. A validação independente exige uma licença
Gurobi ativa, além de `gurobipy` e Shapely, no interpretador que a executar:

```bash
tspn-comparison/solver/.venv/bin/python \
	packages/nonconvex-tpp/cpp/tests/validate_unordered_gurobi.py
```

Para uma nova rodada própria, escreva em um diretório novo e nunca em
`final-dev.jsonl`:

```bash
python3 benchmarks/scripts/unordered_benchmark.py \
	--suite benchmarks/suites/algorithm-dev-v1.bin \
	--solver .build/unordered/tpp --seconds 2 --max-calls 10000000 \
	--output benchmarks/results/unordered/RUN-ID/ours.jsonl

tspn-comparison/solver/.venv/bin/python \
	tspn-comparison/benchmarks/run_comparison.py \
	--suite benchmarks/suites/algorithm-dev-v1.bin --mode path --threads 1 \
	--time-limit 2 --eps 0.000001 \
	--output benchmarks/results/unordered/RUN-ID/external

python3 benchmarks/scripts/summarize_unordered.py \
	benchmarks/results/unordered/RUN-ID/ours.jsonl \
	benchmarks/results/unordered/RUN-ID/external/RUN-SUBDIR/algorithm-dev-v1-tspn-path.csv \
	--output benchmarks/results/unordered/RUN-ID/comparison
```

Substitua `RUN-ID` e `RUN-SUBDIR` por diretórios novos criados pela rodada externa.
O resumidor rejeita hashes de instâncias distintos, modo diferente de `path` e
objetivos incompatíveis nos casos ótimos de ambos.

Como alternativa integrada ao dashboard, use seu ambiente, que fornece Shapely para
a validação independente dos caminhos próprios:

```bash
apps/benchmark-dashboard/.venv/bin/python benchmarks/tpp.py free-order CAMPAIGN \
	--solver unordered --solver tspn --threads 1 --max-seconds 2
```

Esse comando mantém relatórios por execução em
`benchmarks/campaigns/CAMPAIGN/results/free-order/`. Ele requer tanto o checkout
externo quanto seu ambiente Python quando `--solver tspn` é usado. As tolerâncias
da referência são `1e-7 + 1e-9 * UB` no nosso solver, `--eps 0.000001` no externo
e tolerância geométrica padrão `0.001` no externo, portanto os resultados devem ser
reportados como comparáveis com essa limitação explícita.

## Resultado histórico e limites

O resumo preservado informa 41/60 instâncias declaradas ótimas pelo nosso solver,
40/60 pelo externo, 47,80 s e 46,52 s de tempo total, respectivamente. Todos os
60 caminhos próprios foram verificados de forma independente; os caminhos externos
não foram validados pelo mesmo validador. O teste de extremos do adaptador externo
sinalizou 27 falhas. Esses números descrevem somente esta configuração.

Não foram reexecutados benchmarks durante a consolidação. As tarefas de ampliar
benchmarks e mostrar métricas no dashboard dependem das tarefas 3 e 2 do plano,
respectivamente, e permanecem pendentes.
