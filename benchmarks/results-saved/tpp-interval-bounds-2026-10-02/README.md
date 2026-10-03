# Limites intervalares antes do replay racional — 2026-10-02

Formulação: TPP euclidiano de ordem livre com extremos fixos e polígonos simples, mantendo a decomposição convexa e o B&B existentes. Comparação **nosso baseline da etapa de fronteiras × nosso solver modificado**, sem nova execução de Fekete. O corpus histórico permanece intacto.

## Mudança e correção

O B&B já fornecia uma tolerância por chamada, mas `tpp_convex_solve_certified` a ignorava e sempre tentava provar otimalidade algébrica. A nova versão tenta antes um caminho binário exatamente factível e limites primal-dual intervalares nos **polígonos originais**. Esse certificado só encerra antecipadamente se a subtração dirigida provar que o gap cabe na tolerância recebida, ou se o LB certificado alcançar o cutoff. Nenhuma tolerância global, orçamento ou restrição foi relaxada.

O replay do traço existente foi generalizado por escalar para reutilizar sua construção em double. O reparador existente fornece candidatas; orientações intervalares com racional binário na ambiguidade provam o pertencimento de cada contato exportado. Os vetores duais são diferenças exatas de coordenadas binárias divididas por limites superiores de suas normas, portanto pertencem ao disco unitário. Os suportes mínimos e o comprimento do caminho são arredondados para fora. Políticas de elos curtos só propõem vetores duais factíveis, sem declarar coincidência geométrica.

Quando a primeira proposta não basta e a heurística sugere fronteiras compartilhadas, a mesma contração `2^-20` da etapa anterior fornece outro traço. Ele é remapeado para a geometria original e submetido ao certificado barato. A implementação mede diretamente `[L,U]` do problema original; não usa o ótimo perturbado nem precisa aplicar a correção de continuidade `2 eta sum R_i`.

Falha em qualquer prova conserva o replay racional, KKT completo e recuperações anteriores. A API híbrida sem opções continua exigindo otimalidade exata; os retornos por limites têm estatística separada, sem afirmar ótimo algébrico. `TPP_INTERVAL_PRIMAL_DUAL=OFF` permite ablação. Pertencimento intervalar é compartilhado com o certificado de ciclo em `binary_certificate.h`, sem outro solver geométrico. Contrato e argumento: [oráculo certificado](../../../docs/algorithms/certified-convex-oracle.md).

## Protocolo

- Corpus `fekete-comparison/instances.bin`, SHA-256 `aa442e0546567461621b7fcdb9596ba7b3cc4094929d23fb9bb38d1093c88737`.
- Release, `-O3`, AppleClang 21, C++26, macOS arm64, GMP; filtro direcional e recuperação de fronteiras ativados. Uma thread e um processo de solver por vez; nenhuma compilação ou teste desta sessão concorrente. Atividade externa da máquina não foi monitorada.
- CLI pública `benchmarks/tpp.py free-order-ablation`, variantes intercaladas por caso/repetição. Três repetições, orçamento de 2 s nativos/caso e 10.000.000 chamadas. Startup do processo e validação externa não entram no tempo nativo.
- Gap global absoluto `1e-7`, relativo `1e-9`, factibilidade `1e-8`; `oracle_relative_gap=1e-6` já existente. O B&B fornece gap mais estrito nos refinamentos. Validação independente Shapely a `1e-7`, objetivo recomputado e LB ≤ UB.
- Speedup = mediana das três execuções baseline / mediana das três execuções novas. Somente pares válidos com gap fechado em todas as seis execuções entram nas razões; intervalos finais são conferidos. Timeouts não são tempos de solução.
- Amostra de desenvolvimento: mesmos 48 casos estratificados da [etapa anterior](../tpp-certified-oracle-2026-10-02/README.md), quatro por OSM/random/tessellation × 5–10/11–20/21–40/41–60 polígonos. Essa amostra foi reutilizada durante desenvolvimento.
- Validação separada: 48 casos sem reposição, mesmos 12 estratos e quatro por estrato, seed 20261003. Exclui os 48 anteriores e os focais 0/445. Sorteio sobre o CSV de classificação preservado, grupos em ordem lexicográfica, antes da medição dessa amostra; nenhuma configuração foi escolhida pelos seus resultados.

Índices da validação separada: `4, 5, 8, 31, 60, 66, 73, 82, 132, 137, 167, 173, 195, 207, 215, 231, 234, 266, 278, 287, 299, 307, 312, 319, 340, 356, 361, 379, 383, 386, 403, 407, 409, 440, 449, 452, 453, 459, 473, 480, 489, 491, 494, 496, 506, 513, 519, 525`.

## Resultados finais

| Amostra / fonte | Pares concluídos | Speedup geométrico | Mediana |
|---|---:|---:|---:|
| Desenvolvimento, todas | 43 | **1,726×** | 1,708× |
| Desenvolvimento, OSM | 14 | 1,913× | 1,983× |
| Desenvolvimento, aleatórias | 16 | 1,579× | 1,666× |
| Desenvolvimento, Voronoi/tessellation | 13 | **1,723×** | 1,678× |
| Validação separada, todas | 43 | **1,694×** | 1,630× |
| Validação separada, OSM | 13 | 2,006× | 1,941× |
| Validação separada, aleatórias | 15 | 1,431× | 1,456× |
| Validação separada, Voronoi/tessellation | 15 | **1,731×** | 1,644× |

Na amostra de desenvolvimento, o baseline concluiu **43/48** casos em todas as repetições (129/144 execuções), a nova versão **44/48** (132/144); o caso adicional foi 555. Na validação separada, ambos concluíram **43/48** (129/144 cada). As **576 trajetórias validaram**, sem erros ou intervalos finais incompatíveis nos pares concluídos.

Em conjunto, 86 pares concluídos: 1,710× geométrico; melhora em 85/86. A única razão abaixo de um foi 0,999× no caso 447. Essa agregação continua condicionada aos pares concluídos e à distribuição estratificada, não estima o corpus inteiro. Soma das medianas: desenvolvimento 6,770 → 4,415 s; validação separada 9,843 → 5,962 s. Somas não são médias geométricas.

O certificado barato encerrou 174.417/197.904 chamadas perfiladas (88,1%) no desenvolvimento e 279.507/297.928 (93,8%) na validação separada. Entre esses retornos, 5.457 e 2.523 vieram da proposta contraída. Isso não mede isoladamente seu speedup: falhas e custo da segunda proposta também contam no tempo total.

## Casos focais

Três repetições adicionais, 30 s nativos/caso, mesmas tolerâncias e thread. Os três casos fecharam o gap em **todas as 18 execuções**, com caminhos válidos e intervalos compatíveis.

| case_index | Mediana baseline → nova | Speedup | Chamadas baseline → nova |
|---|---:|---:|---:|
| 0 | 8,012 → 7,301 s | **1,097×** | 15991 → 15354 |
| 445 | 3,376 → 2,634 s | **1,281×** | 4531 → 4673 |
| 451 | 24,795 → 19,187 s | **1,292×** | 28915 → 29397 |

No caso 451, materialização + certificado passaram de 18,459 para 11,610 s; a construção geométrica subiu de 1,350 para 1,993 s. A proposta/certificação adicional tem custo, mas o ganho líquido permanece positivo. Os tempos absolutos dessa rodada diferem bastante das etapas anteriores, inclusive para o mesmo executável baseline. Não atribuímos esse desvio à mudança nova nem combinamos tempos de rodadas diferentes: os speedups acima usam somente pares desta sessão.

## Validação e limites

- GMP: **20.288 verificações**, 1.000 caixas e 200 convexos afins; zero falhas, gaps não resolvidos ou divergências com a referência racional. Limites novos são comparados com a referência exata e contatos exportados são testados em racional. Inclui cutoff sem gap permitido, gap demasiado estrito, arredondamento incompatível, orientação, escalas e fronteiras compartilhadas.
- Boost sem GMP: **4.692 verificações**, 100 caixas e 20 convexos afins; zero falhas, gaps não resolvidos ou divergências.
- Ablação com `TPP_INTERVAL_PRIMAL_DUAL=OFF`: **3.753 verificações** com GMP, 100 caixas e 20 convexos afins; zero falhas, gaps não resolvidos ou divergências, nenhum retorno intervalar.
- Ordem livre: 86 casos por enumeração, 344 buscas interrompidas e avaliação multithread; passou. TSPN: 19 casos enumerados, 152 buscas interrompidas, portfólios, budgets compartilhados, hints/duals e concorrência; passou.
- Certificado de ciclo compartilhado: 13.500 comparações com suporte em todos os vértices e 1.040 casos de elos zero; passou. Smoke WASM existente: seis casos e extração de mapa passaram; módulo não reconstruído para esta alteração.
- `RUN_BROWSER=0 npm run test:all` continua interrompendo nos mesmos 11 erros de Ruff em `tests/test_tspn_benchmark.py`, não alterado. O sanity completo passou na primeira etapa; não foi repetido nesta etapa. Os testes diretamente afetados acima foram executados.
- As 594 execuções finais, incluindo os focais, validaram. `exact` do B&B significa fechamento do gap numérico declarado; retornos primal-dual do oráculo não afirmam igualdade algébrica do ótimo. Casos incompletos continuam fora das razões de tempo de solução.
- Há ruído em instâncias rápidas, censura dos casos difíceis e reutilização da amostra de desenvolvimento. A validação separada confirma o ganho nessa distribuição, sem garantir ganho uniforme em outras instâncias. Chamadas que não obtêm um dual suficientemente forte ainda pagam o caminho exato; sua frequência e custo dependem da geometria e da busca.
- Não houve nova execução de Fekete. Estes ganhos adicionais não devem multiplicar automaticamente seus speedups históricos. Uma afirmação atualizada do paper requer nova comparação no mesmo ambiente, com gaps, orçamentos, status e censura comparáveis.

## Identificação

- Baseline SHA-256: `21a962466f92a28f8123dd24a678f440fe4c196239e95893eb20a0c220a6fc5e`.
- Nova versão SHA-256: `34316d6c3324cb813b227ca62b267e328723c9188ac58e5a6ad31836248d7b85`.

Campanhas, dados brutos, executáveis, seleção e hashes de fontes/configuração ficam em `benchmarks/results/tpp-interval-20261002/`, ignorado. Este diretório preserva somente este resumo.
