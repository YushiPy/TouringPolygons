# Resultados finais — comparação dos dois solvers

Este diretório contém dois arquivos canônicos, ambos separados por `;`:

- [`free-order-results-final.csv`](./free-order-results-final.csv): solver de
  ordem livre, 558 instâncias.
- [`german-results-final.csv`](./german-results-final.csv): solver alemão/de
  ordem fixa, 557 instâncias comparáveis.

Os demais CSVs são rodadas intermediárias, checkpoints ou resultados parciais.

## Arquivo principal

O arquivo consolidado é [`free-order-results-final.csv`](./free-order-results-final.csv).
Ele usa `;` como separador e contém uma linha para cada uma das 558 instâncias.
Os artefatos versionados neste diretório são somente os dois CSVs canônicos e
este README; JSONL, checkpoints e resultados intermediários permanecem locais.

O relatório auditável em PDF está em
`output/pdf/touring-polygons-benchmark-report/touring-polygons-benchmark-report.pdf`.

Todos os 558 casos terminaram exatamente. Os casos 129 e 541 foram finalmente
resolvidos na execução sem limite de tempo, com limite de 1 bilhão de chamadas.
Os outros casos usam o melhor resultado certificado da campanha progressiva.

O arquivo alemão contém 498 casos exatos e 59 casos limitados por tempo. O caso
554 foi excluído porque o solver alemão sofreu uma falha determinística `SIGBUS`.
Todas as 557 linhas finais têm `branch_limited=false`.

## Como os testes foram feitos

- Suíte: `benchmarks/suites/german-instances.bin`, com 558 instâncias.
- Solver: `.build/unordered/tpp-unordered`.
- Campanha progressiva: 10 s, 60 s, 600 s e 3600 s por instância.
- A cada etapa, somente as instâncias ainda não exatas eram executadas novamente.
- Os casos 129 e 541 continuaram depois de 3600 s com `--seconds inf`,
  `--max-calls 1000000000` e dois workers em paralelo.
- `best_updates` conta apenas melhorias encontradas durante a busca; a
  aproximação/incumbente inicial não é contabilizada.
- `selected_run`, `selected_time_limit_seconds` e `selected_max_calls` indicam
  qual execução forneceu a linha final. Para uma execução sem limite de tempo,
  `selected_time_limit_seconds` fica vazio.

As instâncias foram executadas independentemente e em paralelo entre workers.
O `--resume` do runner é apenas um checkpoint por instância: ele preserva
linhas já gravadas, mas não retoma o estado interno da busca B&B de uma
instância que terminou por limite de chamadas.

## Critério comum de exatidão

No solver de ordem livre, `exact=true` corresponde a `termination=optimal`.
No solver alemão, a coluna foi normalizada como:

```text
exact = exhausted and not time_limited and not branch_limited
```

Assim, resultados que haviam sido marcados como `exhausted=true` mas tinham
branching artificialmente limitado não são considerados exatos.

## Colunas

### Proveniência e instância

| Coluna(s) | Significado |
|---|---|
| `selected_run` | Campanha que forneceu a linha final: progressiva ou ilimitada. |
| `selected_time_limit_seconds` | Limite de tempo usado nessa execução; vazio quando ilimitado. |
| `selected_max_calls` | Limite de chamadas usado nessa execução. |
| `schema_version` | Versão do esquema JSON produzido pelo solver. |
| `source` | Identificação do exportador/campanha. |
| `case_index` | Índice da instância na suíte. |
| `repeat_index` | Índice da repetição; aqui é 0. |
| `polygons` | Número de polígonos da instância. |
| `polygon_vertices_total` | Número total de vértices dos polígonos originais. |
| `polygon_vertices_min`, `polygon_vertices_max` | Menor e maior número de vértices em um polígono. |
| `order_space_log2` | Logaritmo na base 2 do espaço estimado de ordens possíveis. |
| `sha256` | Hash da representação codificada da instância. |

### Chamadas, estados e árvore de busca

| Coluna(s) | Significado |
|---|---|
| `calls` | Número total de chamadas de busca/relaxação. |
| `relaxation_calls` | Chamadas de relaxação. |
| `refinement_calls` | Chamadas de refinamento. |
| `complete_order_oracle_calls` | Chamadas ao oráculo de ordem completa. |
| `complete_piece_oracle_calls` | Chamadas ao oráculo completo de peças. |
| `oracle_cutoff_calls` | Chamadas interrompidas pelo limite interno do oráculo. |
| `nodes` | Nós/estados expandidos na busca. |
| `partial_states_created` | Estados parciais criados. |
| `children_generated` | Filhos gerados. |
| `children_queued` | Filhos colocados na fila. |
| `screened_nodes` | Nós filtrados antes da expansão. |
| `pruned_nodes` | Nós podados. |
| `pruned_states` | Estados podados. |
| `bound_prunes` | Podas causadas por limites. |
| `incumbent_prunes` | Podas causadas pelo incumbente atual. |
| `peak_queue` | Maior tamanho observado da fila. |

### Ramificação e profundidade

| Coluna(s) | Significado |
|---|---|
| `insertion_branches` | Ramificações por inserção de um polígono/peça. |
| `decomposition_branches` | Ramificações introduzidas pela decomposição. |
| `insertion_positions_considered` | Posições de inserção examinadas. |
| `insertion_positions_pruned` | Posições de inserção podadas. |
| `branch_events` | Número de eventos de ramificação. |
| `total_branching` | Soma das quantidades de filhos gerados nos eventos. |
| `mean_branching_factor` | Média de filhos por evento de ramificação. |
| `max_observed_branching` | Maior ramificação observada. |
| `sequence_depth_sum` | Soma das profundidades das sequências visitadas. |
| `sequence_depth_samples` | Número de amostras usadas nessa soma. |
| `mean_sequence_depth` | Profundidade média da sequência. |
| `max_sequence_depth` | Maior profundidade observada. |

### Decomposição e peças convexas

| Coluna(s) | Significado |
|---|---|
| `decomposed_polygons` | Polígonos que foram decompostos. |
| `convex_pieces_generated` | Número total de peças convexas geradas. |
| `convex_pieces_min`, `convex_pieces_max` | Menor e maior número de peças por polígono decomposto. |

### Incumbente e qualidade das soluções

| Coluna(s) | Significado |
|---|---|
| `incumbent_updates` | Número total de atualizações do incumbente, incluindo a inicial. |
| `best_updates` | Melhorias do incumbente durante a busca, excluindo a aproximação inicial. |
| `initial_lower_bound`, `initial_upper_bound` | Limites antes da busca principal. |
| `initial_length` | Comprimento da solução incumbente inicial. |
| `incumbent_length` | Comprimento do incumbente registrado durante a execução. |
| `first_best_update_length` | Comprimento após a primeira melhoria durante a busca. |
| `final_length` | Comprimento da solução final registrada. |
| `first_incumbent_seconds` | Tempo até o primeiro incumbente. |
| `initial_gap_percent` | Gap percentual inicial entre limites/solução. |
| `lower_bound`, `upper_bound` | Limites finais inferior e superior. |
| `final_absolute_gap` | Diferença absoluta entre os limites finais. |
| `final_relative_gap` | Gap final relativo. |

### Status e tempo

| Coluna(s) | Significado |
|---|---|
| `exact` | Indica que a instância foi certificada como ótima. |
| `termination` | Motivo textual de término: `optimal`, `call_limit`, `time_limit` ou `numerical_limit`. |
| `exhausted` | Forma booleana derivada: busca exaurida/ótima. |
| `time_limited` | A execução terminou por limite de tempo. |
| `call_limited` | A execução terminou por limite de chamadas. |
| `numerical_limited` | A execução terminou por uma condição numérica. |
| `seconds` | Tempo total medido pelo solver. |
| `solver_seconds` | Alias do tempo total do solver. |
| `search_seconds` | Tempo da busca. |
| `bnb_seconds` | Alias do tempo de busca B&B. |
| `seconds_per_call` | Tempo médio por chamada. |
| `calls_per_expanded_node` | Chamadas por nó expandido. |
| `decomposition_percent` | Percentual do tempo total gasto na decomposição. |

### Fallbacks e certificação geométrica

| Coluna(s) | Significado |
|---|---|
| `fallback_calls` | Total de chamadas que usaram fallback. |
| `fallback_geometric_path_invalid_calls` | Fallback por caminho geométrico inválido. |
| `fallback_certificate_gap_calls` | Fallback por gap na certificação. |
| `fallback_locator_exception_calls` | Fallback por exceção no localizador. |
| `fallback_nonfinite_calls` | Fallback por valor não finito. |
| `fallback_contact_construction_calls` | Fallback na construção de contato. |
| `fallback_membership_ordering_calls` | Fallback na ordenação por pertinência. |
| `fallback_local_optimality_calls` | Fallback na verificação de otimalidade local. |
| `fallback_coincident_contact_calls` | Fallback por contato coincidente. |
| `predicate_exact_evaluations` | Avaliações exatas de predicados geométricos. |
| `extended_precision_calls` | Chamadas usando precisão estendida. |
| `oracle_time_limit_calls` | Chamadas internas do oráculo limitadas por tempo. |
| `repaired_geometric_path_calls` | Caminhos geométricos reparados. |

### Perfil temporal detalhado

| Coluna(s) | Significado |
|---|---|
| `preprocessing_seconds` | Tempo de pré-processamento. |
| `initial_heuristic_seconds` | Tempo da heurística inicial. |
| `finalization_seconds` | Tempo de finalização e validação. |
| `convex_oracle_seconds` | Tempo total dos oráculos convexos. |
| `convex_geometric_solver_seconds` | Tempo do solver geométrico convexo. |
| `convex_certificate_verification_seconds` | Tempo de verificação de certificados convexos. |
| `convex_contact_materialization_seconds` | Tempo de materialização de contatos convexos. |
| `convex_fallback_seconds` | Tempo total em fallbacks convexos. |
| `convex_fallback_long_double_seconds` | Tempo em fallback com `long double`. |
| `convex_fallback_extended_precision_seconds` | Tempo em fallback com precisão estendida. |
| `decomposition_seconds` | Tempo da decomposição geométrica. |
| `visit_check_seconds` | Tempo total das verificações de visita. |
| `heuristic_visit_check_seconds` | Tempo de verificações de visita da heurística. |
| `search_visit_check_seconds` | Tempo de verificações de visita durante a busca. |
| `finalization_visit_check_seconds` | Tempo de verificações de visita na finalização. |
| `search_maintenance_seconds` | Tempo de manutenção da estrutura de busca. |

Valores vazios ou nulos indicam que a métrica não se aplica ou não foi
produzida naquela execução.

## Como o solver alemão foi testado

O solver alemão é o solver de ordem fixa em `.build/bnb/tpp`. A campanha foi
feita progressivamente com limites de 10 s, 60 s, 600 s e 1800 s. Depois:

- as 55 instâncias ainda não resolvidas receberam 7200 s;
- as 153 instâncias que tinham `branch_limited=true` foram identificadas;
- 33 delas já estavam na rodada de 7200 s e as 120 restantes foram executadas
  novamente com 7200 s;
- essas duas rodadas usaram `--max-branching -1`,
  `--max-calls 100000000` e 12 workers;
- `caffeinate -i` foi usado para impedir o repouso automático do Mac.

O arquivo [`german-results-final.csv`](./german-results-final.csv) combina a
melhor linha anterior para cada caso com as novas execuções sem limite de
branching. Portanto, ele tem exatamente uma linha por caso comparável e não
contém certificações baseadas em branching artificialmente limitado.

## Colunas do solver alemão

As colunas do arquivo alemão são as métricas nativas do benchmark, acrescidas
de `exact` e `termination` para ficarem comparáveis com o outro arquivo.

| Colunas | Significado |
|---|---|
| `source`, `case_index`, `repeat_index`, `checksum` | Proveniência, índice da instância, repetição e hash da entrada. |
| `polygons` | Número de polígonos originais. |
| `decomposed_pieces`, `grouped_pieces` | Quantidade de peças após decomposição e agrupamento. |
| `total_combinations` | Número estimado de combinações de peças. |
| `total_vertices_min`, `total_vertices_max` | Menor e maior número de vértices entre os polígonos. |
| `calls` | Chamadas totais ao solver. |
| `incumbent_solves`, `bound_solves`, `leaf_solves` | Chamadas usadas para incumbente, limites e folhas. |
| `visited_nodes`, `pruned_nodes` | Nós visitados e podados. |
| `best_updates` | Melhorias do incumbente durante a busca, sem contar a aproximação inicial. |
| `first_best_update_length` | Comprimento após a primeira melhoria da busca. |
| `mean_selected` | Média de peças selecionadas por estado. |
| `initial_length`, `incumbent_length`, `final_length` | Comprimentos inicial, do incumbente e final. |
| `lower_bound`, `upper_bound` | Melhor limite inferior global da fronteira B&B e melhor solução viável conhecida ao final da execução. |
| `optimality_gap_percent` | Gap global real, calculado como `100 * (upper_bound / lower_bound - 1)`. É a métrica comparável ao artigo. |
| `optimality_tolerance_percent` | Tolerância configurada para parada antecipada; zero significa que a execução busca exaurir a árvore. |
| `bounds_consistent` | Verifica que `lower_bound <= upper_bound` dentro da tolerância numérica. |
| `tolerance_reached` | Indica que o gap global atingiu a tolerância configurada. |
| `initial_gap_percent`, `incumbent_gap_percent` | Ganhos relativos de comprimento em relação à solução inicial/incumbente; não são o gap global UB/LB. |
| `prune_rate_percent` | Percentual de nós podados. |
| `calls_per_visited_node`, `bound_calls_per_leaf` | Razões de chamadas por nó e por folha. |
| `decomposition_seconds`, `approximation_seconds`, `bnb_seconds`, `solver_seconds`, `instance_seconds` | Tempos da decomposição, aproximação, B&B, chamadas convexas e tempo total por instância (`decomposition + approximation + B&B`). |
| `decomposition_percent`, `approximation_percent`, `bnb_percent`, `solver_percent` | Percentuais correspondentes do tempo total. |
| `incumbent_solver_seconds`, `bound_solver_seconds`, `leaf_solver_seconds` | Tempo nos três tipos de chamadas do solver. |
| `seconds_per_call` | Tempo médio por chamada. |
| `piece_graph_*`, `port_bound_*`, `refinement_bound_*`, `contact_bound_*` | Tempos, chamadas e podas dos respectivos limites. |
| `hull_bound_prunes`, `piece_graph_extra_prunes`, `piece_graph_dominates`, `port_extra_prunes`, `port_dominates` | Estatísticas dos limites geométricos e dominância. |
| `exhausted` | Indica que a árvore visitada foi esgotada. Sozinho, não basta se houve limite de branching. |
| `time_limited` | A execução atingiu o limite de tempo. |
| `call_limited` | A execução atingiu o limite de chamadas convexas. |
| `branch_limited` | Algum nó teve seus filhos artificialmente limitados. No arquivo final é sempre `false`. |
| `exact` | Certificação normalizada de otimalidade. |
| `termination` | `optimal`, `optimality_tolerance`, `time_limit`, `call_limit` ou `branch_limit`. |
| `max_observed_branching` | Maior branching observado. |
| `failed_prune_count`, `failed_prune_ratio_mean`, `failed_prune_gap_mean`, `failed_prune_depth_mean` | Estatísticas de podas que falharam nos testes de validação. |

Valores vazios indicam que a métrica não se aplica ou não foi produzida pela
execução correspondente.

Para reproduzir a métrica do artigo, o runner aceita
`--optimality-tolerance-percent 0.1`. O benchmark nativo também pode receber
`TPP_BENCH_OPTIMALITY_TOLERANCE_PERCENT=0.1`. A tolerância é aplicada somente
depois que existe uma fronteira B&B com limites válidos; uma execução exaurida
continua sendo marcada como `optimal`, e não como parada por tolerância.

Exemplo de rodada comparável ao artigo:

```sh
python3 benchmarks/scripts/run_german_per_case.py \
  --suite benchmarks/suites/german-instances.bin \
  --solver .build/bnb/tpp \
  --seconds 300 \
  --optimality-tolerance-percent 0.1 \
  --max-calls 100000000 \
  --max-branching -1 \
  --workers 12 \
  --output benchmarks/results/free-order-vs-german-20260919/german-round-300s-tol01.csv \
  --resume
```

Use um nome de saída novo para a primeira rodada com o esquema novo; `--resume`
é um checkpoint de linhas já concluídas, não uma migração de resultados antigos.
O runner também gera automaticamente um arquivo irmão `.meta.json` com hashes da
suíte e do binário, limites, tolerância, plataforma e hash do CSV final.
