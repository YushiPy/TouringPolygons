# Benchmarks preservados

Esta pasta guarda resumos compactos dos resultados, não cópias de campanhas.
Execuções completas, entradas repetidas, trajetórias, logs, builds e patches de
experimento ficam nos diretórios locais ignorados (`benchmarks/workspace/`).

Pastas com dados:

| Pasta | Conteúdo |
|---|---|
| [fekete-comparison](fekete-comparison/README.md) | Comparação original (558 casos, nosso solver antigo × Fekete et al., teto de 6 h, uma thread por caso, até 10 processos em paralelo) usada pelo material SIICUSP; CSVs e corpus são dependências do exportador. |
| [free-order-dantzig-2026-10-06](free-order-dantzig-2026-10-06/per-case.csv) | Nova comparação na dantzig (nosso solver atual × Fekete), uma linha por caso; resumo [abaixo](#free-order-dantzig-2026-10-06). |
| [convex-cycle-gurobi-reference-2026-09-25](convex-cycle-gurobi-reference-2026-09-25/README.md) | `instances.json`, fixture pequeno exigido pelos testes e benchmarks. |

Todo o restante está resumido neste arquivo, uma seção por tentativa, na ordem
inversa de data. Cada seção informa formulação, orçamento e tolerâncias,
status de exatidão e limitações. Um gap numérico fechado nos B&B não é ótimo
algébrico; os bounds SOCP de Fekete são numéricos. Os READMEs originais
(protocolos completos, listas de índices, hashes de binários, validações) estão
no histórico do Git: `git show 85e1ed4:benchmarks/results-saved/<pasta>/README.md`.

## tspn-paula-lower-bound-2026-10-07

**Formulação.** O mesmo TSPN de ciclo livre e o mesmo protocolo da seção
[`tspn-paula-cycle-2026-10-06`](#tspn-paula-cycle-2026-10-06): gap relativo
1e-6, uma thread, `cache,features,root,interval`, macOS arm64 na tomada,
8 processos simultâneos (10 na rodada de 600 s), um por execução. As variantes
de uma mesma instância rodam juntas. Alvo: o limite inferior (LB) nas 38
abertas. Para medir só o LB, as variantes partem do mesmo incumbente
(`--initial-path`): o melhor tour de uma execução de 120 s de
`--primal-ils 0.95`, que empata com o melhor da Paula em 32 das 38. A métrica é
a fração do gap da base que a variante fecha, `(LB_v - LB_b)/(UB - LB_b)`.
Seleção feita antes de medir: as 13 de desenvolvimento são uma a cada três
abertas por tamanho e as 25 restantes são de validação. Dados locais:
`campaigns/paula-tspn-cycle-20261006/screen/lb`.

- **Diagnóstico.** Nas abertas, os fechos convexos quase não se tocam e o
  polígono cobre 80–90% do seu fecho, então o gargalo é combinatório. A
  fronteira tem profundidade média de ~15 regiões. Mais tempo ajuda pouco:
  dobrar o tempo fecha ~1/3 do gap restante (42rat99: 0,891 → 0,914 → 0,935
  em 30/60/120 s). Com `sample`, o oráculo leva > 95% do tempo, quase todo em
  aritmética racional; no caminho com pontos, toda chamada usa a construção
  racional.
- **Opções existentes (triagem, 13 de desenvolvimento, 120 s).** Sem
  mergulhos: +29%. `lazy` (filhos na fila com o limite de inserção e oráculo
  só quando saem): +29%. `--insertion-lookahead 8`: +9%. One-tree, `dual` com
  `dual-screen` e `bound-first`: ~0. Strong branching: −7%. Com o incumbente do
  ILS, os mergulhos e a avaliação imediata dos filhos deixam de compensar.
- **Limite de inserções múltiplas** (`--multi-insertion-bound`, novo; contrato
  em [`unordered-tpp.md`](../../docs/algorithms/unordered-tpp.md#limite-de-inserções-múltiplas-2026-10-07)).
  Na triagem: +20% sozinho, +59% com `lazy` sem mergulhos (contra +51% sem
  ele). Com `--insertion-lookahead 8` por cima, nada muda. Precificar também em
  ordem decrescente rende +0,2% de LB pelo dobro do custo (rejeitado).
- **Validação (25 instâncias, 120 s, três repetições; mediana por
  instância).** As três repetições diferem em no máximo
  0,0015 de LB/UB. 50pcb442 fecha em todas as variantes; nas outras 24:

  | Variante | LB/UB médio | Gap fechado: média / mediana / mín–máx | Melhora em |
  |---|---:|---:|---:|
  | base | 0,854 | — | — |
  | `m` (`--multi-insertion-bound`) | 0,868 | 10,7% / 8,6% / 4,5–42% | 24/24 |
  | `ln` (`lazy` sem mergulhos) | 0,917 | 47,0% / 44,4% / 34–73% | 24/24 |
  | `mln` (os dois) | **0,924** | **52,5% / 47,3% / 36–82%** | 24/24 |

  O limite novo ajuda mais quando há pontos (50kroB100: 42% sozinho) e menos
  nos ciclos de 100 polígonos (5–9%).
- **197 fechadas** (60 s, sem tour inicial, uma repetição; correção e
  regressão): base 196, `m` 197, `ln` 196, `mln` 197 fechadas. Nenhum ótimo
  diverge e nenhum LB passa do ótimo da base. Em média geométrica, `m` é
  1,03×, `ln` 1,19× e `mln` 1,23× mais rápido que a base; nenhuma instância
  ficou mais de 1,5× (+0,05 s) mais lenta. Soma dos tempos: 311 → 257 s.
- **Pipeline completo nas 38 (600 s, `--primal-ils 0.5`, uma repetição).**
  Mesmas condições para as duas configurações (10 processos), sem tour
  fornecido. Antes é `--primal-ils 0.5`; agora é ele mais
  `--multi-insertion-bound --cycle-optimization lazy --lazy-oracles
  --dive-interval 0`.

  | | Antes | Agora |
  |---|---:|---:|
  | Gap certificado: mín / mediana / máx | 0 / 11,0% / 32,2% | 0 / **5,7%** / **20,5%** |
  | Mediana em ≤ 60 / 61–99 / 100 polígonos | 5,2% / 18,9% / 14,1% | 0,9% / 8,7% / 8,0% |
  | Gap fechado (1e-6) | 3 | 4 (+49gil262-7x7) |
  | Tour ≤ melhor da Paula + 0,005 | 33 | 34 |

  Nas 35 que a configuração antiga não fecha, a nova fecha 36–100% do gap
  restante (mediana 53%) e melhora todas. Os tours são os mesmos, salvo o
  ruído do ILS: 100i3000-803 empatou com a Paula desta vez. Todos os tours
  validaram. Abaixo de 0,4% ficam também 42rat99-6x7 (0,19%), 50lin318 (0,14%)
  e 50kroB100 (0,35%).
- **Limitações.** Uma máquina, com 8 a 10 processos simultâneos. A triagem
  orientou a escolha. O limite novo usa binary64 com margem, não aritmética
  racional (ver o contrato). O LB continua sendo o gargalo: 34 das 38 seguem
  abertas aos 600 s, com até 20% de gap nas instâncias de 61–100 regiões.

## trust-double-2026-10-08

**Pergunta.** O que acontece se o nosso solver jogar "nos termos do Fekete", isto
é, aceitar valores numéricos sem prova? **Formulação.** Os 558 casos de
`fekete-comparison/instances.bin`, gap `UB ≤ 1,001·LB`, validação Shapely 1e-7.
Mesmo binário (branch `oracle-float-polish`, macOS arm64), duas variantes
intercaladas por caso, 4 casos simultâneos, uma repetição: padrão certificado ×
`--trust-double`. Esta opção é de diagnóstico e insegura: o oráculo de caminho
devolve o caminho do traço em `double` (replay e reparo, sem certificado), e o
comprimento dele vale como limite inferior e superior. Os limites de inserção e
de visita continuam rigorosos. Se o traço falha, a chamada usa o oráculo
certificado (135.743 de 27,9 M chamadas).

**Resultado.**
- **Correção.** As 558 execuções sem certificado declararam o gap fechado e
  devolveram caminhos viáveis (Shapely). Mas em **46 casos o limite inferior
  declarado passa do comprimento de um caminho viável conhecido**, uma
  afirmação provadamente falsa (até 3,2% acima; caso 132: LB 14.197,9 com um
  caminho de 13.753,7). Em **44 casos (7,9%) o caminho final está mais de 0,1%
  acima do ótimo certificado** (mediana 0,24%, máximo 3,27%). A causa é a poda
  com valores superestimados, que descarta a ordem ótima. Por geometria
  (Shapely, polígonos originais): disjuntos 0/179 (incluindo 19 com fechos
  convexos que se cruzam), só toques 20/231, sobreposição com área 26/148;
  todos os erros estão em instâncias com polígonos que se tocam ou se cruzam.
  Por fonte: aleatórias
  25/160, tesselação 12/78, OSM 7/320. Por tamanho: 4–10 1/128, 11–20 7/136,
  21–40 11/149, 41–60 25/145.
- **Velocidade.** Certificado/sem certificado (média geométrica): 1,26× nos
  casos ≥ 1 ms (n = 344), 1,18× nos ≥ 0,1 s e 1,21× nos ≥ 1 s. Soma dos
  tempos: 746 → 728 s (os casos longos são dominados por consultas de visita).

**Leitura.** A certificação custa ~20% do tempo e evita respostas erradas em
~8% dos casos. A comparação com o Fekete continua a favor dele, porque ele não
paga por garantias. Mas "confiar" não é uma alternativa equivalente para o nosso
solver: o erro de um algoritmo combinatório com uma decisão errada é
descontínuo (pontos percentuais), enquanto o de um solver cônico numérico
cresce suavemente com a tolerância. **Limitações.** Uma máquina compartilhada
com 4 processos, uma repetição; os erros dependem só da aritmética e se repetem.
Dados em `experiments/trust-double` do workspace local.

## float-recovery-dantzig-2026-10-08

**Formulação.** Os 558 casos de `fekete-comparison/instances.bin`, gap
`UB ≤ 1,001·LB` (absoluto 0), validação Shapely 1e-7, sem limite de chamadas,
teto de 3600 s. Binário da branch `oracle-float-polish` (`ded35f2`, GCC,
C++23) na dantzig (i9-12900K). Duas variantes do mesmo binário: padrão
(polimento em ponto flutuante após a prova intervalar) × `--no-float-recovery`
(replay exato e recuperações racionais). Uma repetição, as duas variantes em
sequência por caso, 4 casos simultâneos, cada solver fixado num núcleo P.
A máquina tinha outros usuários (load average ~7): em 171 casos as duas
variantes rodaram com o hyperthread vizinho ocupado, em 327 com ele livre e em
60 com cargas diferentes.

**Resultado.** 1116/1116 execuções fecharam o gap e validaram.
- **Aritmética racional.** Padrão: **0 chamadas racionais** em 27,5 M
  chamadas (27,15 M pela prova intervalar, 379.405 pelo polimento, 0 fallbacks),
  12.984 sinais de pertencimento decididos em racional (em 85 casos) e 16.360
  conversões de polígonos para o cache. Referência: 470.841 chamadas racionais
  (262.444 replay exato/KKT, 155.575 filtradas, 52.822 fronteira disjunta).
- **Efeito da carga.** Nos casos de busca idêntica (nenhuma chamada no
  polimento, ≥ 10 ms), a razão é 0,997 (n = 8) e 1,001 (n = 54) com a mesma
  carga nas duas variantes, e 0,84 ou 1,38 com cargas diferentes. Os 60 pares
  com cargas diferentes ficam fora das razões abaixo. Eles incluem o caso 129
  (Dubai, 707 → 978 s com a mesma busca), que sozinho inverte a soma bruta dos
  tempos (1832 → 1910 s).
- **Ganho com a mesma carga** (média geométrica referência/padrão):

  | Casos | n | Todos | Só os que usaram o polimento |
  |---|---:|---:|---:|
  | referência ≥ 1 ms | 330 | 1,37× (mín. 0,54×) | 1,59× (n = 213) |
  | referência ≥ 0,1 s | 125 | **1,51×** (mín. 0,97×) | 1,77× (n = 90) |
  | referência ≥ 1 s | 40 | **1,52×** (mín. 0,98×) | 1,85× (n = 27) |

  Soma dos tempos dos 498 pares com a mesma carga: 468,8 → 283,7 s (1,65×).
  Por número de polígonos (≥ 1 ms): 11–20 1,34×, 21–40 1,32×, 41–60 1,44×.
  As seis razões abaixo de 0,9 são casos de 1–13 ms, quatro deles com busca
  idêntica: ruído na escala de milissegundos.

**Limitações.** Máquina compartilhada e uma única repetição: só as razões com a
mesma carga são confiáveis, e a medição limpa (máquina ociosa, `--workers 1`)
continua pendente. O checkout da dantzig estava marcado como sujo. Dados brutos
em `experiments/float-recovery-dantzig` do workspace da dantzig.

## tpp-float-oracle-2026-10-08

**Formulação.** TPP de ordem livre com extremos fixos, corpus
`fekete-comparison/instances.bin`; protocolo padrão de
`docs/algorithms/unordered-tpp-experiments.md`: gap `UB ≤ 1,001·LB`
(absoluto 0), uma thread, um processo por vez, variantes intercaladas por caso,
duas repetições, validação Shapely 1e-7, teto de 900 s (nenhum caso chegou
perto). Mesmo binário (`d854b27`, branch `oracle-float-polish`, macOS arm64,
AppleClang, C++23, GMP), com três configurações do oráculo convexo: híbrido
atual, `--float-oracle` (só ponto flutuante) e `--float-recovery` (prova
intervalar do híbrido e polimento em ponto flutuante no lugar do replay exato e
das recuperações). Conjuntos: difícil (19, seed 20261005) e validação
(18, seed 20261006), os mesmos de 05/10.

**Resultado.** As 222 execuções fecharam o gap e validaram o caminho. Com
`--float-oracle`, 6,6 M chamadas por repetição usaram só ponto flutuante, com 0
fallbacks ao híbrido. Média geométrica das medianas por caso:

| Conjunto | Híbrido → `--float-oracle` | Híbrido → `--float-recovery` | Soma dos tempos (híbrido / float / recovery) |
|---|---:|---:|---:|
| Difícil (19) | 1,32× (0,76–3,37×) | **1,47×** (0,97–3,23×) | 150,5 / 105,6 / 95,5 s |
| Validação (18) | 1,54× (0,81–4,24×) | **1,68×** (0,99–4,00×) | 91,2 / 62,0 / 53,6 s |

As duas repetições diferem em menos de 0,5% nas médias. Os maiores ganhos estão
nos casos dominados pela cauda de chamadas caras (215: 4,0×; 540, 461, 214:
~3,2×; 419: 2,6–2,8×). `--float-oracle` fica 15–25% mais lento nos casos em que
quase toda chamada já fecha pela prova intervalar do híbrido (63, 95, 97, 173,
476, 65); `--float-recovery` não perde em nenhum caso além do ruído
(mínimo 0,97×). Replay de 55.454 chamadas capturadas (156, 417, 419): todas
fecharam em ponto flutuante, sem limites incompatíveis com o híbrido.

**Padrão adotado.** A busca passou a usar o recovery por padrão, com o polimento
partindo dos contatos que a prova intervalar tentou, e o modo só em ponto
flutuante saiu dela. Confirmação com o binário da branch (uma repetição,
`--no-float-recovery` como referência): difícil **1,495×** (0,99–3,57×;
149,4 → 93,8 s), validação **1,716×** (0,98–4,46×; 89,9 → 51,9 s), 74/74
fecharam e validaram, 279.509 chamadas fechadas pelo polimento e 0 fallbacks ao
racional. No replay, o ponto de partida do polimento (μ inicial 10²–10⁴ × o
final; fração para o interior 2⁻⁷ ou 2⁻¹²) não mudou o tempo.

**Limitações.** Uma máquina (Mac, sem isolamento térmico). Os tempos
absolutos variaram ~1,5× entre sessões, mas as razões intercaladas ficaram
estáveis (1,33×/1,34× em sessões diferentes). O conjunto difícil orientou o
desenvolvimento; a validação foi medida uma única vez. Não houve medição no
corpus completo nem na dantzig. Os limites são rigorosos no ambiente IEEE
binary64 verificado por `cycle_interval_environment()`; gaps muito mais
apertados que os do B&B podem ficar abertos e cair no híbrido.

## free-order-history-2026-10-07

**Formulação.** Mesmo problema e corpus de `free-order-dantzig-2026-10-06`. São
12 revisões do nosso solver, de `bb1c44a` (base de `fekete-comparison`, 21/09) a
`8f8241a` (`main` de 06/10), compiladas com o mesmo GCC/C++23 e rodadas por
`tpp.py free-order-history` (revisão `2e80f4e`) na dantzig (i9-12900K). Amostra
estratificada de 36 casos: 3/4/5 por fonte nas faixas de 11–20/21–40/41–60
polígonos, em quantis do tempo de 21/09, excluídos os casos com mais de 60 s em
21/09 e os com menos de 2 ms em 06/10. Três repetições, uma thread, revisões
intercaladas por caso, 4 casos simultâneos, cada solver num núcleo P fixo,
gap `UB ≤ 1,001·LB` (absoluto 0), validação independente 1e-7. As revisões
anteriores a 01/10 receberam só edições de build: C++26→C++23 e ordem dos
inicializadores designados, idêntica ao `7c6e29c`.

**Resultado.** As 1296 execuções fecharam o gap e validaram. As chamadas ao
oráculo foram idênticas entre repetições. Ganho em média geométrica das medianas
por caso:

| Etapa | Commit | Sobre a anterior | Menos chamadas | Chamadas mais baratas | Parcela do ganho (log) | Sobre 21/09 |
|---|---|---:|---:|---:|---:|---:|
| Mergulho a cada expansão + poda de irmãos | `e3ac002` | 1,34× | 1,34× | 1,00× | 9% | 1,34× |
| Corte dual certificado | `3e1b4eb` | 1,06× | 1,00× | 1,06× | 2% | 1,41× |
| Cache de polígonos exatos | `0447f77` | 1,11× | 1,00× | 1,11× | 3% | 1,58× |
| Ciclos certificados no B&B compartilhado | `924e8b6` | 1,05× | 1,00× | 1,05× | 1% | 1,65× |
| GMP no lugar de `cpp_rational` | `63a5124` | 1,67× | 1,00× | 1,67× | 16% | 2,75× |
| Certificados intervalares rigorosos | `e759fca` | 1,00× | 1,00× | 1,00× | 0% | 2,76× |
| Limites intervalares + recuperação filtrada + cache de pares | `d004d64` | 3,89× | 0,99× | 3,93× | 42% | 10,7× |
| Geometria emprestada, cache por segmento, provas compactas | `31a208b` | 1,59× | 1,00× | 1,59× | 14% | 17,1× |
| Limites de visita por âncora | `f4c99cb` | 1,08× | 1,00× | 1,08× | 2% | 18,4× |
| Arena e KKT em inteiros homogêneos (solver de 06/10) | `82cb31e` | 1,26× | 1,00× | 1,26× | 7% | 23,2× |
| Alocação e conversões | `8f8241a` | 1,10× | 1,00× | 1,10× | 3% | 25,6× |

No total, `bb1c44a` → `8f8241a` deu **25,6×** (IC 95% por bootstrap nos casos:
22,3–29,5×), com 1,33× menos chamadas e 19,3× menos tempo por chamada. Cerca de
91% do ganho em escala logarítmica veio do custo por chamada. Por faixa:
11–20 deu 18,0×, 21–40 deu 27,7× e 41–60 deu 29,8×. A divisão por data ficou
assim: de 21/09 a `e759fca` (02/10), 2,76× (1,34× em chamadas, 2,06× por
chamada); de `e759fca` a `82cb31e`, 8,4×, todo por chamada. A rodada local de
02/10 (`fekete-full-1h-8workers-20261002`) não usou `e759fca` puro: suas linhas
trazem `oracle_interval_bound_calls`, contador que só existe a partir de
`d004d64`. Isso explica a divisão antiga, de 4,5× por chamada até 02/10.

**Limitações.** A dantzig estava carregada (load average ~19). Em quase todas
as execuções, o outro hyperthread do núcleo estava ocupado, o que infla os
tempos absolutos: os de 06/10 ficaram 1,27× mais lentos que na comparação de
06/10. As razões ficaram estáveis: entre repetições, cada etapa variou no máximo
±4%, e o total ficou em 25,4×, 26,4× e 25,6×. A amostra exclui os casos acima de
60 s em 21/09, onde o ganho tende a ser maior. Os ganhos por etapa são
sequenciais e dependem das etapas anteriores. O checkout da dantzig estava sujo
(`dirty`). Dados brutos em `experiments/free-order-history` do workspace da
dantzig.

## free-order-dantzig-2026-10-06

**Formulação.** TPP euclidiano de ordem livre, extremos fixos, polígonos simples;
os 558 casos do corpus `fekete-comparison/instances.bin`. **Nosso solver**
(revisão `470a0c9`/`bcd234e`, mesmo solver) × **Fekete et al.** (B&B SOCP, Gurobi),
rodados na dantzig (Intel i9-12900K, Linux). Sem limite de tempo nem de chamadas;
uma thread por caso; gap relativo alvo 0,1% nos dois (`UB ≤ 1,001·LB`), validação
independente 1e-7. O Fekete usa a configuração original (`FEASIBILITY_TOLERANCE`
0,001 e `SPANNING_TOLERANCE` 0,0009, os padrões da biblioteca, que os scripts de
avaliação do artigo não alteram). Dados por caso:
[`free-order-dantzig-2026-10-06/per-case.csv`](free-order-dantzig-2026-10-06/per-case.csv).

**Resultado.**

| | Nosso solver | Fekete |
|---|---:|---:|
| Casos com gap fechado | **558/558** | **553/558** |
| Soma dos tempos nos 553 casos fechados por ambos | 580 s | 329.791 s (maior: caso 558, 91.145 s ≈ 25,3 h) |
| Soma dos tempos nos 558 casos | 1.899 s (maior: caso 130, 722 s) | — |

Nos 553 casos fechados por ambos, o nosso solver foi mais rápido em **553/553**; nos 5 abertos o Fekete ficou pelo menos 6 h (21.657–21.720 s na campanha de teto de 6 h, `fekete-comparison`) sem fechar, contra no máximo 722 s do nosso, logo o nosso foi mais rápido também neles (**558/558**, com essa ressalva de censura)
(menor razão 3,4×, caso 452; maior 2.265×). Razão Fekete/nosso: média aritmética
**111,6×**, média geométrica 49,8×, mediana **41,3×**. Por fonte (média
geométrica): OSM (315) 72,6×; aleatórias (160) 39,6×; tessellation (78) 17,3×.
Por número de polígonos: 4–10 (128) 20,9×; 11–20 (136) 38,2×; 21–40 (149) 63,0×;
41–60 (140) 110,4×. Todas as 558 trajetórias nossas validaram.
Nosso solver no `470a0c9` × a campanha anterior dele na mesma pasta: 26.116 s →
1.899 s no total (5,6× geométrico).

**Casos em aberto no Fekete (5):** 65, 66, 130, 493, 542 (numeração a partir
de 1). Foram interrompidos manualmente depois de dias sem fechar o gap (o tempo por
caso não foi gravado nessa versão). Os limites parciais do Fekete são compatíveis
com o ótimo do nosso solver (LB do Fekete ≤ nosso UB e UB do Fekete ≥ nosso LB).
Na antiga comparação (`fekete-comparison`, teto de 6 h) os abertos eram 65, 66, 130,
131, 231, 493, 542 e 558: o 231 fechou aqui em 13,9 h. Uma nova execução do Fekete
nos 8 casos então abertos (revisão `7f4c8dd`, 4 casos em paralelo nos núcleos de
desempenho, configuração original: `FEASIBILITY_TOLERANCE` 0,001, gap relativo
0,1%) fechou o **131 em 41.347 s (≈ 11,5 h; nosso solver 29,5 s)**, o **420 em
24.631 s (≈ 6,8 h; nosso 144,2 s)** e, depois de mais tempo, o **558 em 91.145 s
(≈ 25,3 h; nosso 60,8 s)**; as razões são 1.403×, 170,8× e 1.499×. Os caminhos
do Fekete nesses três casos não passam na validação a 1e-7 (`valid=false`, desvio
de cobertura da ordem de 4e-6 no 131), como esperado com a tolerância de 1e-3. Uma
execução anterior do 131 fechou em 42.500 s, mas com `FEASIBILITY_TOLERANCE` 1e-8 e
`SPANNING_TOLERANCE` 0,0009 (nunca ajustada junto, apesar de a biblioteca
recomendá-la logo abaixo da primeira); não é a configuração original e ficou fora
da comparação.

**Limitações — leia antes de citar a razão.**
- A campanha do Fekete é de 2 de outubro, com 8 casos em paralelo na máquina; o
  nosso solver rodou 1 caso por vez (2 nos casos 65 e 66). Com uma thread por
  caso em 24 núcleos o efeito esperado é pequeno, mas não foi medido.
- Os caminhos do Fekete dessa campanha não têm validação independente
  registrada (`valid` ausente), só `endpoint_valid`; com a tolerância de
  cobertura de 1e-3 do Fekete, a validação a 1e-7 pode não passar em todos.
- Os 6 casos abertos do Fekete são censurados: as razões os excluem e, portanto,
  **subestimam** a vantagem. Uma afirmação para publicação exige reexecutar o
  Fekete, na configuração original, no mesmo ambiente.
- `exact=true` é fechamento do gap configurado, não otimalidade algébrica.

## tpp-visit-bounds-lns-2026-10-05

Nosso solver (commit `5adad03`) × modificado, sem Fekete; TPP de ordem livre,
corpus de 558 casos, gap relativo 0,0999%, visita 1e-8, validação Shapely 1e-7,
teto de 600 s, uma thread, três repetições intercaladas, macOS arm64 (M4 Pro).
Conjunto difícil (19 casos, orientou o desenvolvimento) e validação (18, sorteada
antes de medir, medida uma vez).

- **Limites superiores de visita** (ativo; `--no-visit-upper-bounds` desliga):
  âncora por região limita a distância; a busca é idêntica (37/37 casos). Média
  geométrica 1,289× (difícil) e 1,218× (validação); caso 97: contatos exatos de
  9,6 M para 0,93 M.
- **LNS exata por janelas** (`--window-lns`, **OFF**): 1,410× / 1,252×, muda a
  trajetória da busca; ganho extra de ~1% no caso Dubai (129; 507 → 333 s com os
  limites, 330 s com a LNS). Ativar por padrão exige campanha maior.
- Todas as 333 execuções fecharam o gap e validaram. Limitações: uma máquina, sem
  isolamento, casos < 0,5 s ruidosos; não substitui a comparação com Fekete nem
  multiplica os speedups históricos. Registro das tentativas:
  [`unordered-tpp-experiments.md`](../../docs/algorithms/unordered-tpp-experiments.md).

## tspn-paula-cycle-2026-10-06

TSPN de ciclo livre (sem depósito) nas 235 instâncias `npol <= 100` da coleção
local da Paula (GTSP-Lib/MOM-Lib convertidas, polígonos não convexos, 66 com
pontos e 40 só com segmentos além de polígonos), contra o Fekete et al. fixado
e as referências do rascunho do paper (CPLEX MIQCP, 1 h, nas 87 com até 15
polígonos; comprimento euclidiano do tour ótimo do GTSP, factível nas 235 e
portanto limite superior). Protocolo TSPN mantido: 60 s, gap relativo 1e-6,
uma thread, otimizações de ciclo `cache,features,root,interval`, Gurobi com
tolerâncias apertadas; nosso solver e o Fekete em execuções separadas (às vezes
simultâneas, um processo cada), macOS arm64 na bateria. Os dados e a campanha
são locais (sem autorização de redistribuição).

- **Antes**: o modo ciclo rejeitava pontos e segmentos (106/235 instâncias).
- **Depois** (`paula-tspn-evaluation`): 235/235 tours válidos, **197** com gap
  fechado; Fekete 213 válidos, **124** fechados. Nos 213 comuns: ambos 124, só
  nós 59, só Fekete 0; tempo 7,0× menor (média geométrica; 0,57–344×).
- CPLEX: nas 80 instâncias que ele provou ótimas, nosso valor difere no máximo
  3,2e-5 (dentro da tolerância dele); nas 7 que ele deixou abertas após 1 h
  (gaps 11–89%), provamos o ótimo em < 0,05 s.
- Correções exigidas pelo corpus: pontos e segmentos no oráculo de ciclo;
  âncora de ponto (45 vs 42 fechadas, 2,67× sem ela); contatos exatos da
  recuperação como caminho (o gap de relaxação de `numerical_limit`); e pino
  de canto da região comum (uma relaxação de 4 regiões passou de > 60 s a
  6 ms; 10i400-206 e 10i45-18 de abertas a 0,57 s e 0,24 s).
- Regressão SoCG (48 casos, 4 por estrato, 60 s): 47/47 fechados em ambos, 42
  buscas idênticas, as 6 diferentes com menos chamadas; 1,09× geométrico.
- Mesmo gap da campanha alemã (ε = 0,001, 60 s): nós 197 fechadas (235 válidas),
  Fekete 172 (216 válidas); onde ambos fecham, 7,5× geométrico (≤ 15 polígonos
  7,2×, 16–40 8,2×, 41–60 5,8×). Com 600 s para o Fekete nas 10 instâncias de
  41–60 que só nós fechamos em 60 s: ≥ 12,1× (4,1–34,8×; uma falha e um tour
  inválido dele). A vantagem é menor que nas instâncias alemãs (49×) porque o
  limite de 60 s corta justamente os casos longos, e acima de 60 polígonos
  nenhum dos dois fecha.
- Direções testadas nas 36 de 41–60 polígonos (60 s, ε = 0,001; base fecha 20):
  ramificação forte (ciclo 16, caminho 17), lookahead 8 (20), DFS/BFS (19), ciclo
  sem âncora com ramificação forte (13), partidas primais (20) e LNS exata (20;
  UB médio 0,998 do tour do GTSP contra 1,001). Fora do solver, Held–Karp com
  distâncias entre regiões dá 0,22–0,81 do tour do GTSP e um limite por triplas
  (dois elos saindo do mesmo ponto) 0,73–0,94 mesmo na ordem ótima; o LB do B&B
  já alcança 0,72–0,98. Nenhuma foi adotada.
- Limitações: uma repetição; 38 instâncias abertas (42–100 polígonos, sobretudo
  com pontos ou de 100 regiões), onde o Fekete também não fecha. Os resultados
  do ILS da Paula vieram no rascunho de 2026-10-07 (abaixo); tempos de CPU em
  máquinas diferentes, e o tempo dela é o da execução inteira, não o de achar
  o melhor tour (em média 23 s contra 214 s).
- TPP com depósito no centro da bbox (`convert-paula`, mesmo binário, 60 s,
  gap 1e-6): 196 fechadas contra 197 do TSPN, 2× mais rápido onde ambos fecham,
  mesma fronteira (0 de 22 acima de 60 polígonos).
- **Contra o ILS-BCD da Paula** (rascunho de 2026-10-07, tabela do apêndice:
  melhor de 10 execuções por instância, cada uma até 3.000 iterações sem melhora
  ou 1.200 s de CPU, com comprimentos em duas casas; outra máquina). Nas 197 que
  fechamos, os tours coincidem: com gap 1e-6, nosso valor e o dela diferem no
  máximo 0,003% (arredondamento dela) e o tour dela fica no máximo 0,0024%
  acima do nosso LB, ou seja, nosso certificado mostra que ela acha o ótimo.
  Nosso tempo para fechar é 1.150× menor que o tempo médio de uma execução dela
  (média geométrica; ≤ 15 polígonos 3.600×, 16–40 680×, 41–60 97×, mínimo 6×;
  lemos a coluna de CPU da tabela como a média por execução).
- **Nas 38 abertas, com a busca local iterada opcional** (`--primal-ils 0.95`,
  padrões validados: Or-opt de até 3, listas de 10 candidatos, aceitação com
  reaquecimento dela e reotimização exata de janelas de 8 regiões pelo próprio
  B&B; ver `docs/algorithms/unordered-tpp.md`). Seleção: 19 instâncias de
  desenvolvimento e 19 de validação sorteadas por tamanho antes de medir; na
  validação (60 s, uma execução) a média geométrica do nosso tour sobre o dela
  foi 1,0020 sem a reotimização exata, 1,0007 com polimento só de contatos e
  **1,00015** com ela (17/19 empates). Avaliação (300 s, uma execução, semente
  0): **33/38** chegam ao tour dela (comprimento ≤ o dela + 0,005), em mediana
  352× antes do tempo médio de uma execução dela (média geométrica 287×,
  mínimo 7×; tipicamente 0,1–30 s); as 5 restantes ficam 0,05–0,37% acima
  do melhor dela, nunca acima do pior dos 10 dela e no máximo 0,2% acima da
  média deles. Com 10 sementes de 120 s nessas 5, o melhor nosso empata com o
  melhor dela em 4; 100i1000-410 fica 0,19% acima. Todos os tours validaram.
  O B&B sozinho (600 s, com ou sem LNS) ficava até 4% acima dela nas de 100
  regiões. Gap certificado aos 600 s nas abertas: 2–35% (o LB é o limite;
  caiu para 0–20%, mediana 5,7%, em
  [`tspn-paula-lower-bound-2026-10-07`](#tspn-paula-lower-bound-2026-10-07)).
- No TPP de ordem livre com extremos fixos (corpus do Fekete, 11 casos
  difíceis, `--primal-ils 0.5 --primal-ils-stagnation 200`) o ILS não acelera
  o B&B: 0,07–0,82× nos que fecham em até 3 s e 0,91× e 0,99× nos de 16 e 51 s,
  apesar do UB inicial melhor. Fica desligado por padrão.

## tpp-oracle-allocation-2026-10-06

`main` (`7f4c8dd`) × branch `oracle-shared-vertex-speedup`; mesmo protocolo
(gap relativo 0,000999, gap absoluto 0, 1e8 chamadas, 600 s, uma thread,
variantes intercaladas, três repetições; macOS arm64/GMP, na bateria: só as
razões valem). Mudanças que preservam a busca: sinais da materialização
decididos antes por intervalos `double` (inteiros homogêneos só quando o sinal
fica aberto), vetores do mapa direcional reaproveitados/reservados, valores
exatos do DAG filtrado atribuídos em posições retidas da arena e conversão
racional → `double` com o mesmo arredondamento do Boost (10⁷ casos, inclusive
empates, idênticos bit a bit). Difícil (19): **1,142×** geométrico (mediana
1,120; 0,999–1,414; 283 → 236 s); validação (18): **1,164×** (mediana 1,167;
0,987–1,444; 176 → 143 s). Gap fechado em 222/222 execuções e 144 campos
determinísticos idênticos ao `main`; testes nativos também sob ASan/UBSan.
Limitações: uma máquina, na bateria; o conjunto difícil e o caso 417 orientaram
o desenvolvimento.

## tpp-oracle-exact-arithmetic-2026-10-05

Binário `final` anterior × branch `free-order-oracle-perf`; mesmo protocolo (na
bateria: tempos absolutos ~60% acima, só as razões intercaladas valem). Mudanças
que preservam a busca: arena por thread para o DAG de `FilteredRational`, sinais
decididos por intervalos, KKT/materialização em inteiros homogêneos sem MDC e
caixa por polígono. Difícil (19): **1,302×** geométrico (466 → 295 s);
validação (18): **1,339×** (234 → 176 s). Gap fechado em 222/222 execuções e 144
campos determinísticos idênticos. Casos dominados por consultas de visita ficam
neutros (0,98–1,01×). Limitações: uma máquina, na bateria; o conjunto difícil
orientou o desenvolvimento.

## tpp-interval-bounds-2026-10-02

Nosso baseline da etapa de fronteiras × modificado, sem Fekete. `tpp_convex_solve_certified`
ignorava a tolerância por chamada e sempre provava otimalidade algébrica; a nova
versão tenta antes um caminho binário exatamente factível e limites primal-dual
intervalares nos polígonos originais, encerrando só se a subtração dirigida
provar que o gap cabe na tolerância (`TPP_INTERVAL_PRIMAL_DUAL=OFF` desliga).
Protocolo: 2 s por caso, 10 M chamadas, gap abs. 1e-7 / rel. 1e-9, factibilidade
1e-8, Shapely 1e-7, três repetições, uma thread, macOS arm64/GMP. Amostras de 48
casos estratificados (OSM/aleatório/tessellation × 5–10/11–20/21–40/41–60):
desenvolvimento (reutilizada) 43 pares, **1,726×**; validação separada (seed
20261003) 43 pares, **1,694×**; juntas 1,710×, melhora em 85/86 (pior 0,999×).
Focais (30 s): casos 0, 445 e 451 em 1,10×, 1,28× e 1,29×. 594 execuções
validaram. Ruído em casos rápidos, censura dos difíceis; os ganhos são sobre o
nosso baseline e não multiplicam os speedups históricos contra Fekete.

## tpp-boundary-disjoint-2026-10-02

Recuperação para fronteiras compartilhadas: para candidatas rejeitadas com
geometria sugerindo interiores disjuntos, contrai cada polígono por `2^-20` e
reutiliza a recorrência disjunta, aproveitando só o traço combinatório (tudo é
reconstruído e certificado em racional na entrada original), mais dois
testemunhos exatos para blocos de elos zero. Mesma amostra de 48 casos; 2 s, 10 M
chamadas, gap 1e-7/1e-9. Resultado: **1,015×** geométrico em 39 pares (OSM
0,971×, aleatórias 1,033×, Voronoi 1,047×; Voronoi com 11+ polígonos 1,228×,
exploratório). Caso 445: 1,351× (23,1 → 17,1 s); casos 0 e 451 seguem
incompletos. Um replay antigo de 5,92× foi retirado (polígonos normalizados com
extremos não normalizados); o replay corrigido do caso 451 deu 1,534× e não
estima o B&B. Ganho global pequeno, sem ganho uniforme.

## tpp-certified-oracle-2026-10-02

Baseline `e759fca9` × recuperação certificada: certificado de contatos
coincidentes compartilhado com o ciclo e recuperação, por avaliação racional sob
demanda, de traços rejeitados por predicados intervalares (mesma geometria,
outra aritmética). 48 casos estratificados (seed 20261002), três repetições,
2 s, gap 1e-7/1e-9. **1,210×** geométrico em 37 pares (OSM 1,104×, aleatórias
1,290×, Voronoi 1,269×); o baseline fechou 37/48 e a nova versão 39/48; fallbacks
completos 369 → 0; 288 execuções validaram. O certificado de elos zero sozinho
não deu ganho agregado (≈0,99×); construção filtrada em todas as chamadas foi
descartada. Os casos 0, 445 e 451 continuam incompletos (30 s). Motivação: nos
58 casos historicamente mais rápidos em Fekete, 99,2% do tempo ia para o oráculo.

## tspn-held-karp-learning-2026-09-30

Triagem TSPN (tour fechado, ordem livre) com 2 s, gap 1e-6. Cada variante: 6/6
tours válidos nos dois conjuntos, 6/6 gaps fechados em small6 e 3/6 em large6. O
one-tree melhorou o bound inicial em 4/6 casos grandes (~335 ms) sem fechar mais
casos; o branching aprendido piorou OSM39 (580 → 1.297 chamadas, 0,751 → 1,235 s).
Sem benefício suficiente: as duas opções ficam **OFF**.

## tspn-dual-interval-sharing-2026-09-30

TSPN, gap 1e-6, factibilidade 1e-8, validação 1e-7; 96 resultados nativos
válidos (60 com gap fechado, 36 no limite). O **certificado intervalar** venceu
as 9 comparações pareadas do large6 (1,263×) e, no OSM39 com portfólio, passou de
0,754 s para 0,507 s (1,481×). `dual-screen` não melhorou casos fechados;
`share-bounds` teve hits e podas, mas não reduziu o tempo de parede. Ganhos não
universais; opções experimentais ficam desligadas.

## tspn-cycle-reuse-cutoff-2026-09-30

Memo de uma busca: 6.124 consultas, zero repetições. `bound-first` registrou
podas certificadas sem ganho de tempo estável. Com portfólio e compartilhamento,
OSM39 (10 s, três repetições) caiu de 0,8243 s para 0,7023 s (14,8%) com ~370
hits; evidência restrita a esse caso. Conjunto auditado: 90 trajetórias válidas,
66 gaps fechados, três casos grandes sem objetivo fechado. Memo segue opt-in.

## tspn-oracle-optimizations-2026-09-29

Triagem de otimizações do oráculo (2 s, teto 8 s, gap 1e-6). A shortlist
cache + features + root (CFR): 12/12 trajetórias válidas, 9/12 gaps fechados;
nos sete casos com ambos fechados venceu 7/7, mediana **3,686×**. Candidate E
fechou OSM39 em 3/3 repetições (mediana 0,736 s). Uma relaxação de 19 regiões
foi de >8 s para ~110 ms (diagnóstico; fixture em
`packages/convex-tpp/cpp/tests/cycle_active_contacts.json`). OSM50, random59 e
tessellation60 seguiram abertos; `lazy` deixou OSM39 com ~8,5% de gap em 2 s.

## tspn-portfolio-large-2026-09-29

Seis casos de 30–60 regiões selecionados antes de medir (5 s, teto 12 s); OSM39
com três repetições (10 s). Todas as trajetórias validaram. Cada modo nativo
fechou 3/6 gaps e Fekete 2/6; nos dois casos com ambos fechados (random30 e
tessellation30) o nativo foi mais rápido em todos os modos. OSM39: medianas
default 4,998 s, DFS/BFS 1,501 s, corrida 1,654 s, cooperativo 1,705 s; Fekete
ficou aberto. Triagem pequena e estratificada, sem alegação universal.

## tspn-portfolio-holdout-2026-09-29

Seis casos pré-selecionados (hash de nome, sem Bangalore), três repetições, 1 s
nativo, teto 8 s. O nativo venceu 4/5 pares em todos os modos (speedup mediano:
default 2,26×, DFS/BFS 3,16×, corrida 2,00×, cooperativo 2,19×); fechou 18/18
execuções e Fekete 15/18 (a tesselação de 20 regiões favoreceu Fekete). Sem
vantagem consistente do portfólio cooperativo sobre DFS/BFS ou corrida.

## tspn-socg-stratified-2026-09-28

Triagem de 12 casos SOCG escolhidos lexicograficamente (viés: quatro casos OSM
de Bangalore), uma repetição, 2 s. GMP + cutoff + contatos herdados: 8/12 gaps
nativos e 5/12 no Fekete; nos cinco pares fechados o nativo venceu 3/5 (mediana
1,68×). Histórica, anterior às correções de contatos ativos; substituída por
`tspn-active-contacts`.

## tspn-active-contacts-2026-09-28

TSPN, tour fechado, ordem livre; 33 casos, cinco repetições; gap 1e-6,
factibilidade 1e-8, validação 1e-7. As 165 execuções de cada solver validaram. O
B&B nativo fechou 33/33 gaps e Fekete 25/33; nos 25 pares fechados o nativo
venceu 17/25 (speedup geométrico mediano 1,70×). `fekete_3_n10`: ~19,0 s →
15,7 ms após reutilizar contatos ativos (com certificado independente e fallback
completo); `fekete_6_n15` passou de gap aberto a fechado em ~108 ms. O B&B segue
exponencial; algumas instâncias ainda perdem.

## tspn-fekete-2026-09-27 e tspn-fekete-final-2026-09-27

Comparação inicial (25 casos sintéticos e do corpus de Fekete et al., três
repetições) e sua recaptura: o nativo retornou tours válidos nos 25 casos (um de
15 regiões ficou com gap aberto); Fekete também, mas sem fechar o gap em vários.
Tempos só valem em pares com gap fechado por ambos. Superadas por
`tspn-active-contacts`.

## convex-cycle-final-2026-09-27

Ciclo euclidiano fechado, ordem fixa, 20 instâncias sintéticas convexas
(separação, interseções, contenção, degenerados); 15 repetições, um aquecimento
por backend, Gurobi com uma thread (tempos C++ incluem validação e certificado;
Gurobi é numérico). As 20 saídas racionais foram certificadas ótimas, as 20
double factíveis e compatíveis com Gurobi. Medianas: racional 1,29–21,33× e
double 1,11–22,38× mais rápidos que a chamada Gurobi completa. O certificado usa
predicados exatos; double permite recuperação racional local. Suíte finita: não
prova dominância universal. As medições preliminares `complete` (20 instâncias),
`optimized` e `performance` (18 instâncias) deram os mesmos resultados
qualitativos e foram consolidadas aqui.
