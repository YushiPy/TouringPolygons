# Recuperação certificada do oráculo TPP — 2026-10-02

Formulação: caminho euclidiano de ordem livre com extremos fixos, polígonos simples possivelmente não convexos. Comparação **nosso baseline × nosso solver modificado**, sem nova execução de Fekete. A campanha histórica de 558 casos não foi substituída.

Baseline: revisão `e759fca9605256efd6e9c727f332c882beecc918`. Mudança: certificado completo de contatos coincidentes compartilhado com o ciclo, mais recuperação de traços rejeitados por predicados de intervalos com avaliação racional sob demanda. O código geométrico é o mesmo `intersecting_maps.cpp`, com outra aritmética; não há outro algoritmo geométrico.

Motivação: o diagnóstico local anterior dos 58 casos historicamente mais rápidos em Fekete indicava medianas de 99,2% do nosso tempo no oráculo, 73,3% no fallback racional e apenas 0,77% na construção geométrica em double. Esses dados orientaram a otimização; não integram a estatística da campanha nova. Um solver geométrico especializado não garante menor tempo total quando reconstrução e certificação exatas dominam o custo.

## Protocolo

- Corpus: `fekete-comparison/instances.bin`, SHA-256 `aa442e0546567461621b7fcdb9596ba7b3cc4094929d23fb9bb38d1093c88737`.
- Amostra uniforme sem reposição: quatro casos por OSM/random/tessellation × 5–10/11–20/21–40/41–60 polígonos, seed 20261002; 48 casos. Seleção anterior à medição.
- Três repetições; variantes intercaladas por caso/repetição; uma thread por solver, um processo por vez. Os outros testes desta sessão ficaram suspensos durante a medição; a atividade externa da máquina não foi monitorada.
- Release, `-O3`, C++26, AppleClang 21, macOS arm64, GMP. Filtro ativado somente quando a candidata em double foi rejeitada e o cutoff barato não resolveu a chamada.
- Orçamento: 2 s nativos/caso e 10.000.000 chamadas. Gap absoluto 1e-7, relativo 1e-9, factibilidade 1e-8. Validação geométrica independente com Shapely a 1e-7; conferência do objetivo e LB ≤ UB.
- Speedup = mediana das três execuções baseline / mediana das três execuções novas. Somente casos válidos com gap fechado em todas as seis execuções entram nas razões; os limites finais precisam ser compatíveis. Timeouts não entram como tempos de solução.

Índices zero-based selecionados: `2, 44, 72, 87, 144, 156, 186, 197, 199, 206, 208, 214, 227, 249, 267, 268, 273, 303, 317, 322, 328, 337, 345, 349, 351, 376, 391, 392, 397, 405, 435, 437, 446, 447, 448, 451, 458, 461, 483, 497, 500, 503, 509, 516, 517, 523, 544, 555`.

## Resultado da amostra

| Fonte | Pares concluídos | Speedup geométrico | Mediana do speedup |
|---|---:|---:|---:|
| Todas | 37 | 1.210× | 1.157× |
| OSM | 14 | 1.104× | 1.085× |
| Aleatórias | 12 | 1.290× | 1.354× |
| Voronoi/tessellation | 11 | 1.269× | 1.185× |

Baseline fechou o gap em **37/48 casos em todas as repetições** (111/144 execuções); a versão nova, em **39/48** (118/144 execuções). Os 37 pares comuns tiveram zero trajetórias inválidas, erros ou intervalos finais incompatíveis. Todas as 288 execuções validaram suas trajetórias, incluindo as incompletas.

Nos pares comuns, a soma das medianas caiu de **10.670 s para 7.888 s**. A soma das medianas de fallbacks completos caiu de **369 para zero**. Essa soma de tempos não é uma média geométrica nem uma estimativa do corpus inteiro.

## Diagnóstico de chamadas caras e casos focais

Correção do replay diagnóstico: a medição inicialmente registrada como 0,767 s → 0,130 s (5,92×) combinava polígonos normalizados com extremos não normalizados. Ela resolve outro problema e não representa as chamadas capturadas do B&B. Esse número foi retirado das conclusões; os dados permanecem locais para auditoria. A campanha pareada de 288 execuções e os casos focais abaixo usam a entrada completa corretamente e não dependem desse replay.

Três casos historicamente desfavoráveis foram medidos separadamente, uma repetição, 30 s nativos/caso, mesmas tolerâncias e uma thread. Todos os seis resultados são factíveis e incompletos; não há speedup de solução completa para esses casos.

| case_index | Chamadas baseline → novo | Gap relativo baseline → novo | Fallbacks baseline → novo |
|---|---:|---:|---:|
| 0 | 6490 → 11705 | 1.9315% → 0.6409% | 1364 → 0 |
| 445 | 3178 → 4396 | 0.2358% → 0.0241% | 1000 → 0 |
| 451 | 2983 → 5618 | 5.6273% → 4.2054% | 1166 → 0 |

O aumento de chamadas em orçamento fixo mede trabalho realizado, não um speedup de prova: os traços podem mudar e os gaps continuam abertos. O perfil da nova versão concentra custo em reconstrução/certificação racional e, nos casos 445/451, também na construção filtrada. Esses são os próximos alvos de medição.

## Ablações e limitações

- Apenas compartilhar o certificado completo de elos zero não deu ganho agregado consistente (≈0,99× geométrico na amostra pareada).
- Construção filtrada em todas as chamadas intersectantes custou mais nos casos já certificados em double: 108/144 execuções fecharam o gap contra 111/144 do baseline. Essa configuração não foi mantida; o padrão é recuperação sob demanda.
- A amostra dá o mesmo peso às fontes/tamanhos; não representa a distribuição das 558 instâncias. As razões excluem os casos difíceis incompletos. Tempos de instâncias abaixo de um milissegundo têm ruído expressivo, mesmo com três repetições.
- O ganho 1,21× é adicional sobre nosso baseline atual; não pode multiplicar automaticamente os 10,40× aritméticos ou 4,80× geométricos históricos contra Fekete, medidos com outro protocolo e tolerância externa de 0,1%.
- Os limites aceitos pelo oráculo usam os mesmos predicados exatos e arredondamentos dirigidos. O rótulo `exact` do B&B significa fechamento do gap numérico declarado, não igualdade algébrica do ótimo global.
- Para uma nova afirmação no paper, repetir o corpus inteiro e Fekete no mesmo ambiente, com orçamentos e gaps comparáveis, incluindo status e censura dos casos não concluídos.

## Validação

- 12.512 verificações direcionais: 1.000 casos de caixas e 200 convexos afins, reconstrução, certificados, orientação, degenerescências e 2.000 comparações independentes de aritmética filtrada (incluindo cancelamento, underflow, overflow e arredondamento não suportado); zero falhas, gaps não resolvidos ou divergências com a referência racional.
- Ordem livre: 86 comparações por enumeração de ordens, 344 verificações de busca interrompida e avaliação paralela; passaram.
- TSPN: enumeração, interrupções, portfólios, budgets compartilhados, duals e concorrência; passaram.
- Certificado de ciclo compartilhado: 13.500 testes contra todos os vértices e 1.040 casos independentes de elos zero; passaram.
- JavaScript: 40 testes, sintaxe e lint passaram; smoke WASM existente passou (sem recompilar o módulo para esta otimização).
- `RUN_BROWSER=0 npm run test:all` interrompeu em 11 erros de Ruff em `tests/test_tspn_benchmark.py`, não alterado. Execução separada de 91 unittests Python: três falhas em verificações não modificadas (URL de script SIICUSP, retomada de campanha e hash de partições congeladas), uma ignorada.
- Backend Boost sem GMP: 3.707 verificações direcionais, com 100 casos de caixas e 20 convexos afins; zero falhas, gaps não resolvidos ou divergências com a referência racional.
- Regressões nativas de interseções passaram. `./scripts/sanity_check.sh --no-install` terminou com código 0: fixtures básicos/degenerados, seis suítes geométricas geradas e smoke B&B de 30 casos com limite de 50.000 chamadas/caso. Os executáveis de geração/verificação foram configurados em Release antes da execução.

## Identificação e dados locais

- Executável baseline: SHA-256 `0e234b5b73ae080934dafecb87be261dba25d4c23f1a75eec352a07284a39de0`.
- Executável recovery: SHA-256 `dd10f181f2f044acdb6936d16711600ffcf57a2f84cf0239af882f9c6c996fb4`.

Dados brutos, captures, replay, manifestos, hashes de fontes e logs permanecem em `benchmarks/results/tpp-zero-contact-20261002/`, ignorado. Este diretório preserva somente este resumo.
