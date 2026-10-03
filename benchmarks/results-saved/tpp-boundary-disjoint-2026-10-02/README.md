# Recuperação para fronteiras compartilhadas — 2026-10-02

Formulação: TPP euclidiano de ordem livre com extremos fixos e polígonos simples. Comparação **nosso solver da etapa anterior × nosso solver modificado**, sem nova execução de Fekete. A campanha histórica de 558 casos permanece intacta.

A implementação mantém a construção inicial em double e o cutoff dual. Para candidatas rejeitadas com geometria sugerindo interiores disjuntos, contrai cada polígono em direção à média de seus vértices por `2^-20` e reutiliza a recorrência disjunta existente. Somente o traço combinatório é aproveitado: vértices, reflexões, pertencimento e KKT são reconstruídos/certificados em racional na **entrada original**. Uma proposta direta usa a geometria original se a contração falha; depois continuam os mapas filtrados/racionais. A recuperação fica em função sem inlining para preservar o caminho comum.

Também são testados dois testemunhos exatos baratos para blocos de elos zero: concentrar a mudança de direção no primeiro ou no último contato. A propagação completa pelo disco continua quando ambos falham. Nenhuma tolerância de otimização ou factibilidade foi relaxada. Contratos e argumento de continuidade: [oráculo certificado](../../../docs/algorithms/certified-convex-oracle.md).

## Protocolo

- Corpus preservado `fekete-comparison/instances.bin`, SHA-256 `aa442e0546567461621b7fcdb9596ba7b3cc4094929d23fb9bb38d1093c88737`.
- Mesmos 48 índices estratificados da [etapa anterior](../tpp-certified-oracle-2026-10-02/README.md): quatro por fonte OSM/random/tessellation × faixas 5–10/11–20/21–40/41–60; seed 20261002. Três repetições, execução intercalada, um solver por vez e uma thread. Testes desta sessão ficaram suspensos; atividade externa da máquina não foi monitorada.
- Release, `-O3`, AppleClang 21, macOS arm64, GMP; filtros e recuperação de fronteiras ativados. Baseline já contém o certificado completo de elos zero e a recuperação filtrada da etapa anterior.
- Orçamento de 2 s nativos/caso e 10.000.000 chamadas. Gap absoluto `1e-7`, relativo `1e-9`, factibilidade `1e-8`; validação independente Shapely a `1e-7` e conferência do objetivo/LB ≤ UB.
- Speedup = mediana das três execuções baseline / mediana das três execuções novas. Razões somente nos pares válidos com gap fechado em todas as seis execuções; timeouts não viram tempos de solução. Desenvolvimento e escolha da configuração são exploratórios, com reutilização da amostra.

## Resultado final

| Fonte | Pares concluídos | Speedup geométrico | Mediana do speedup |
|---|---:|---:|---:|
| Todas | 39 | 1.015× | 1.031× |
| OSM | 14 | 0.971× | 1.013× |
| Aleatórias | 13 | 1.033× | 1.073× |
| Voronoi/tessellation | 12 | 1.047× | 1.190× |

Ambas as variantes fecharam o gap em **39/48 casos em todas as repetições**, 117/144 execuções cada. Todas as 288 trajetórias validaram, sem erros nem intervalos finais incompatíveis nos pares concluídos. Soma das medianas nos 39 pares: 13,072 s → 11,781 s; essa soma não é a média geométrica.

Nos Voronoi com **11+ polígonos**, oito pares concluídos, a média geométrica foi **1,228×**. Acima de 30 polígonos, apenas dois pares concluídos, foi 1,569×. Esses recortes por tamanho são análises exploratórias; a amostra/censura não permite generalizar o resultado para todos os Voronoi grandes. Instâncias rápidas tiveram regressões e ruído expressivo; não há ganho uniforme, e o ganho geométrico global é pequeno.

## Casos focais

Uma repetição separada, 30 s nativos/caso, mesmas tolerâncias/thread. Todos os seis caminhos validaram.

| case_index | Tempo baseline → novo | Chamadas baseline → novo | Gap relativo baseline → novo |
|---|---:|---:|---:|
| 0 | 30,00 s → 30,00 s | 11302 → 12436 | 0,7025% → 0,4163% |
| 445 | **23,13 s → 17,12 s** | 4473 → 4531 | fechado → fechado |
| 451 | 30,01 s → 30,01 s | **5683 → 8641** | **4,1440% → 3,1939%** |

Caso 445: 1,351× entre duas soluções com gap numérico fechado. Casos 0/451 continuam incompletos; mais chamadas em orçamento fixo e menor gap não são speedup de solução completa. Rodadas exploratórias anteriores variaram bastante em tempos/status, inclusive no caso 0: esta única repetição não estima uma distribuição de tempo de solução.

## Replay corrigido e ablações

203 sequências extraídas sistematicamente por id (stride 2) das 406 chamadas concluídas em uma captura anterior de 2 s do caso 451. Extremos e polígonos estão no **mesmo sistema normalizado**. Replay sem cutoff de busca, três repetições alternadas por variante; diagnóstico de retenção de candidata ativado nas duas variantes. Soma das medianas por sequência: **0,605 s → 0,394 s (1,534×)**. As 41 propostas rejeitadas pela tentativa inicial foram recuperadas e certificadas com a contração; nenhum fallback racional completo e nenhum intervalo incompatível. Não é uma estimativa global do oráculo ou do B&B.

O antigo número de **5,92×** foi retirado das conclusões: aquele replay combinava polígonos normalizados com extremos não normalizados e resolvia outro problema. O [resumo anterior](../tpp-certified-oracle-2026-10-02/README.md) registra a correção. Suas 288 execuções de B&B e medições focais usaram entradas completas e não dependem desse replay.

Tentativas de aplicar a recorrência disjunta antes da construção inicial acrescentaram custo em casos fáceis e foram descartadas. A versão mantida só tenta recuperar candidatas rejeitadas. Contração sem os dois atalhos KKT teve ganhos agregados pequenos/variáveis nas rodadas exploratórias; não atribuímos diferenças entre rodadas a uma única mudança, pois houve variação de execução. Não foram alterados critérios de poda, gap ou orçamento.

## Validação e limitações

- GMP: **12.558 verificações direcionais**, 1.000 caixas e 200 convexos afins; zero falhas, gaps não resolvidos ou divergências com a referência racional. Inclui reflexão analítica em aresta compartilhada e regressão reduzida do caso 451 (três regiões, doze vértices), escalas `1e-9/1/1e9`, orientação e sequência invertidas.
- Boost sem GMP: **3.753 verificações**, 100 caixas e 20 convexos afins; zero falhas, gaps não resolvidos ou divergências.
- Ordem livre: 86 comparações por enumeração, 344 buscas interrompidas e avaliação multithread; passaram.
- TSPN: 19 casos enumerados, buscas interrompidas, portfólios, budgets compartilhados, hints/duals e concorrência; passou após a última alteração. Smoke WASM existente: seis casos e extração de mapa passaram; o módulo não foi reconstruído para esta otimização.
- `RUN_BROWSER=0 npm run test:all` voltou a interromper nos 11 erros de Ruff em `tests/test_tspn_benchmark.py`, não alterado. As três falhas Python identificadas na etapa anterior estão no resumo anterior; essa suíte Python não foi repetida nesta etapa.
- O sanity check completo passou na etapa anterior; não foi repetido após esta alteração. Foram executados os testes diretamente afetados acima.
- `exact` significa fechamento do gap numérico declarado, não igualdade algébrica. Perturbação é somente proposta: continuidade não prova estabilidade do traço nem autoriza usar seu objetivo como LB do problema original.
- Estes ganhos são sobre nosso baseline recente; não podem ser multiplicados automaticamente pelos speedups históricos contra Fekete. Uma afirmação nova no paper requer corpus inteiro, ambiente controlado e gaps/orçamentos comparáveis.

## Identificação

- Baseline SHA-256: `dd10f181f2f044acdb6936d16711600ffcf57a2f84cf0239af882f9c6c996fb4`.
- Versão nova SHA-256: `21a962466f92a28f8123dd24a678f440fe4c196239e95893eb20a0c220a6fc5e`.

Raw, executáveis, entradas de replay, manifestos de fontes/configuração e logs ficam em `benchmarks/results/tpp-touching-20261002/`, ignorado. Este diretório preserva somente o resumo.
