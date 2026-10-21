# Benchmarks preservados

Esta pasta guarda resumos compactos dos resultados, não cópias de campanhas.
Execuções completas, entradas repetidas, trajetórias, logs, builds e patches de
experimento ficam nos diretórios locais ignorados (`benchmarks/workspace/`).

Pastas com dados:

| Pasta | Conteúdo |
|---|---|
| [fekete-reference-dantzig-2026-10-20](fekete-reference-dantzig-2026-10-20/per-case.csv) | Réplica das configurações A, B e C da armijo na dantzig (Intel), 300 s, mesmo formato; resumo [abaixo](#fekete-reference-dantzig-2026-10-20). |
| [fekete-reference-armijo-2026-10-17](fekete-reference-armijo-2026-10-17/per-case.csv) | Medição de referência na armijo: o Fekete original (configuração do artigo, raiz LEFP e uma thread) × nosso solver com 1 e 16 threads, 558 casos, 3 repetições de 300 s mais 900 s nas abertas; uma linha por caso, configuração, raiz e limite; resumo [abaixo](#fekete-reference-armijo-2026-10-17). |
| [fekete-comparison](fekete-comparison/README.md) | Comparação original (558 casos, nosso solver antigo × Fekete et al., teto de 6 h, uma thread por caso, até 10 processos em paralelo) usada pelo material SIICUSP; CSVs e corpus são dependências do exportador. |
| [float-recovery-dantzig-2026-10-08](float-recovery-dantzig-2026-10-08/per-case.csv) | Comparação com o Fekete usando nossos tempos da rodada de 08/10 (polimento em ponto flutuante), uma linha por caso, no mesmo formato da de 06/10; usada pelo material SIICUSP; resumo [abaixo](#float-recovery-dantzig-2026-10-08). |
| [free-order-dantzig-2026-10-06](free-order-dantzig-2026-10-06/per-case.csv) | Comparação na dantzig (nosso solver de 06/10 × Fekete), uma linha por caso; resumo [abaixo](#free-order-dantzig-2026-10-06). |
| [convex-cycle-gurobi-reference-2026-09-25](convex-cycle-gurobi-reference-2026-09-25/README.md) | `instances.json`, fixture pequeno exigido pelos testes e benchmarks. |

Todo o restante está resumido neste arquivo, uma seção por tentativa, na ordem
inversa de data. Cada seção informa formulação, orçamento e tolerâncias,
status de exatidão e limitações. Um gap numérico fechado nos B&B não é ótimo
algébrico; os bounds SOCP de Fekete são numéricos. Os READMEs originais
(protocolos completos, listas de índices, hashes de binários, validações) estão
no histórico do Git: `git show 85e1ed4:benchmarks/results-saved/<pasta>/README.md`.

## tspn-smoothed-vertex-scaling-2026-10-21

**Pergunta.** Depois do oráculo em O(log m) por chamada
([abaixo](#oracle-vertex-scaling-2026-10-21)), o B&B do TSPN inteiro também
escala com o número de vértices por região?

**Formulação.**
- **Instâncias.** Seis instâncias do corpus do Fekete (`bangalore`,
  `random_mixed_square`, `vienna`, `istanbul`, `warsaw` e `tokyo`; n = 50 ou
  60). Cada região foi trocada pelo seu fecho convexo somado a um disco de raio
  igual a 1% da mediana dos diâmetros, com m pontos nos arcos e coordenadas na
  grade 2^-30.
- **Variantes.** m = 256, 2.048 e 16.384, mais o fecho sem disco. As regiões se
  sobrepõem como no corpus.
- **Execução.** TSPN cíclico, com os argumentos da configuração C da referência
  do Fekete: gap relativo 0,1%, tolerância de factibilidade 1e-8, `cache`,
  `features`, `root` e `interval`, uma thread. Mac M4 Pro, uma execução por
  instância, limite de 600 s.
- **Antes.** A branch antes das mudanças no B&B (7c50537): o oráculo já era
  O(log m), mas o B&B ainda era linear em m.

**Resultado.** Todas as 24 execuções fecham o gap de 0,1% (é um gap numérico,
não ótimo algébrico), com os mesmos valores nas três variantes suavizadas.
Tempo total em segundos (antes → agora):

| Instância | Fecho | m = 256 | m = 2.048 | m = 16.384 (agora) |
|---|---|---|---|---|
| bangalore | 0,11 → 0,10 | 0,63 → 0,27 | 3,91 → 0,42 | 0,89 |
| random_mixed_square | 0,10 → 0,08 | 1,07 → 0,31 | 10,17 → 0,47 | 1,31 |
| vienna | 0,22 → 0,19 | 4,06 → 0,71 | 89,22 → 1,64 | 4,37 |
| istanbul | 0,40 → 0,33 | 2,48 → 0,92 | 63,94 → 1,43 | 2,93 |
| warsaw | 0,57 → 0,49 | 3,97 → 1,46 | 68,53 → 2,35 | 4,74 |
| tokyo | 1,14 → 0,90 | 7,88 → 3,11 | 46,31 → 4,39 | 6,80 |

- **Escala.** De m = 256 a 16.384 (64× mais vértices, até 983 mil no total), o
  tempo cresce de 2 a 6×. Em m = 2.048 o B&B ficou de 9 a 54× mais rápido que
  antes. O "antes" com m = 16.384 não foi medido.
- **Mesma árvore.** O número de chamadas e o de nós ficam a até 1,2% dos de
  antes (no pior caso, `random_mixed_square` com m = 256, 6.512 → 6.586
  chamadas); só a construção de cada chamada difere, por arredondamento.
- **O que pesava, e o que mudou:**
  - polimento com uma barreira por face: agora é feito sobre uma aproximação
    interna, refinada quando o Newton para;
  - recorte O(m²) de regiões sobrepostas: interseção O(m log m), guardada por
    par de regiões;
  - proposta de inserção e checagens de visita O(m): buscas binárias com
    certificado ou com a mesma conta da varredura;
  - cópia das regiões a cada chamada: empréstimo por troca.

  Os detalhes estão em
  [convex-cycle.md](../../docs/algorithms/convex-cycle.md#regions-with-many-vertices-2026-10-21)
  e [tspn.md](../../docs/algorithms/tspn.md#large-convex-regions-2026-10-21).
- **Corpus.** Comparamos o solver da `main` com a branch em 43 casos dos 558: os
  3 com fecho de 32 vértices ou mais e 40 sorteados (semente 20261021). Os
  argumentos foram os mesmos, com limite de 60 s. A árvore é idêntica em todos
  os 43, e a branch é 1,16× mais rápida (média geométrica dos casos acima de
  0,05 s).

**Limitações.**
- Uma execução por instância, num Mac só.
- Instâncias exploratórias, não realistas.
- O Fekete não rodou nestas instâncias. O SOCP dele cresce com m a cada
  chamada.
- A comparação com a `main` no corpus é uma amostra. O benchmark do artigo
  precisa ser refeito antes de um merge.
- Gerador e dados brutos: local, em `benchmarks/workspace/experiments/`
  (`tspn-smoothed` e `corpus-check`).

## oracle-vertex-scaling-2026-10-21

**Pergunta.** Como o custo de uma chamada ao oráculo do ciclo cresce com o
número de vértices por região, comparado ao SOCP do Gurobi, e até onde ele
pode cair? **Formulação.** Ciclo de ordem fixa por k = 3, 5 ou 10 regiões
convexas disjuntas, cada uma o fecho convexo inteiro de m pontos (m = 8 a
32.768) num disco ou numa elipse (ângulos aleatórios), com coordenadas inteiras
divididas por 2^13. Com m = 32.768, os fechos ficam com cerca de 16 mil vértices
por região. Duas amostras por caso, 3 repetições, com a ordem dos solvers
rodando entre elas. Mac M4 Pro, uma thread, processo único. Programa:
`main-vertex_scaling.cpp`, nesta branch.
- **Nosso oráculo:** o pedido do B&B, ou seja, estágio binary64 com
  `max_gap` = 2,5e-4 × o comprimento do ciclo pelos centros e certificado
  intervalar. Quatro variantes:
  - frio, sem cache;
  - quente, com o `ConvexCycleWorkspace` reaproveitado;
  - com identificadores (`region_ids`), como o B&B poderia chamar;
  - apertado, frio com gap de 1e-6, para casar com a tolerância padrão do
    Gurobi.
- **Gurobi:** parâmetros padrão, uma thread, modelo montado a cada chamada, como
  no oráculo SOCP do Fekete.

**Resultado.**
- **Primeiro, um recorte quadrático.** O teste de ciclo zero recortava uma região
  por todas as arestas da outra, em O(m²), mesmo com regiões disjuntas. Com 16
  mil vértices por região, o nosso ficava só 10–20× mais rápido que o Gurobi.
- **Depois, O(log m) por chamada com cache.** O estágio deixou de ler as
  regiões inteiras a cada chamada:
  - a prova de pertinência acha um triângulo do leque e o prova com três sinais
    exatos;
  - o suporte do limite dual sai de um vértice extremo certificado por
    intervalos como mínimo exato, com uma janela de dois vértices para os
    empates das arestas ótimas;
  - a atualização por coordenada da construção localiza o alvo no mapa de
    último passo de Dror et al. por busca binária, como o solver de origem fixa
    por busca binária, e mantém os testes de aceitação do laço;
  - um teste de separação em O(log m) evita copiar uma região para o recorte;
  - com identificadores, cada região é normalizada, copiada e tem a média
    calculada uma vez por workspace, em vez de ser lida e copiada a cada chamada.

  Cada caminho volta ao laço anterior quando não consegue certificar, e dá o
  mesmo resultado que ele.
- **Tempo por chamada com identificadores e razão Gurobi / nosso** (disco):

  | m (vértices por região) | 8 | 512 | 2.048 | 8.192 | 32.768 (~16 mil) |
  |---|---|---|---|---|---|
  | k = 3 | 1,5 µs (179×) | 2,4 µs (5.030×) | 3,5 µs (10.316×) | 7,3 µs (38.315×) | 9,6 µs (95.087×) |
  | k = 10 | 4,6 µs (140×) | 12,5 µs (2.506×) | 22,5 µs (5.480×) | 32,8 µs (23.681×) | 43,1 µs (72.287×) |

  Nas elipses dá o mesmo: 85.508× com k = 3 e 50.531× com k = 10, no maior
  tamanho. Contra a `main` (quente, antes deste trabalho), a chamada com k = 3 e
  ~16 mil vértices fica 9.844× mais rápida (95 ms → 9,6 µs). Sem identificadores
  (quente), o custo continua O(m) na leitura das regiões e fica em 0,26 ms. O
  frio e o apertado continuam O(m), porque normalizam a região a cada chamada:
  com gap de 1e-6, a razão contra o Gurobi é de cerca de 900× no maior tamanho.
- **Mesmo resultado.** Nas 252 medições, os limites e contatos das variantes
  frio, quente e apertado são idênticos aos da `main`, e os da variante com
  identificadores, aos da quente. O nosso sempre fecha o gap pedido
  (`GapClosed`), o Gurobi sempre termina ótimo, e o limite superior com gap de
  1e-6 nunca passa do objetivo do Gurobi em mais de 8,4e-7 relativo.
- **Testes.** `logarithmic_queries()`, em `main-cycle_tests.cpp`, compara cada
  caminho com o seu laço em fechos de reticulado de até 9.000 vértices:
  - 3.720 pertinências, incluindo vértices, pontos médios e pontos a um ulp da
    borda;
  - 1.033 suportes, sempre abaixo do mínimo exato racional e iguais aos do laço,
    233 deles com empate exato numa aresta;
  - 1.503 atualizações por coordenada;
  - 57 separações confirmadas por interseção exata;
  - o estágio inteiro com e sem identificadores, com os limites cercando o
    ótimo exato.

  As suítes do ciclo, do certificado, das interseções, do TSPN e da ordem livre
  passam.

**Limitações.** Chamadas isoladas, sem as sequências e as heranças do B&B;
regiões disjuntas e bem separadas; um Mac pessoal. O que falta para o B&B
ganhar o mesmo:
- passar identificadores e não copiar as regiões em `selected`;
- tornar sublinear o par mais próximo das sequências de duas regiões, hoje
  O(m²);
- escrever as provas novas em `convex-cycle.md`.

O gerador e os dados brutos ficam locais em `experiments/vertex-scaling`.

## path-vs-cycle-mac-2026-10-21

**Pergunta.** O TSPN (ciclo livre) costuma fechar mais rápido que o TPP com
extremos fixos nos mesmos polígonos. Isso vem da árvore de busca ou do
oráculo? **Formulação.** Os 558 casos de `fekete-comparison/instances.bin`,
cada um como caminho, com o início e o fim do caso, e como ciclo nos mesmos
polígonos. Mesmo binário (código de `2467373`), gap `UB ≤ 1,001·LB`,
factibilidade 1e-8, uma thread, 60 s. O ciclo usa
`cache,features,root,interval`; o caminho, os padrões dele. As duas formulações
rodam em seguida no mesmo worker (a ordem alterna por caso), com 6 workers no
Mac M4 Pro e uma repetição.

**Resultado.**
- **Fechadas em 60 s:** 556 como caminho e 557 como ciclo. Só o caminho fecha
  o caso 189; só o ciclo fecha os casos 130 e 542.
- **Decomposição.** Razões ciclo/caminho em média geométrica, nas instâncias
  fechadas nas duas formas:

  | Instâncias | n | Tempo | Nós | Chamadas | Tempo de oráculo por chamada |
  |---|---|---|---|---|---|
  | todas | 544 | 0,58× | 0,43× | 0,52× | 0,69× |
  | caminho ≥ 0,1 s | 89 | 0,28× | 0,35× | 0,38× | 0,65× |
  | caminho ≥ 1 s | 25 | 0,14× | 0,18× | 0,20× | 0,65× |

  Nas instâncias difíceis, a maior parte vem da árvore: o ciclo precisa de 3 a
  5× menos nós (a correlação entre a razão de nós e a de tempo, em log, é 0,97).
  O oráculo também pesa: cada chamada do ciclo custa 0,65× a do caminho
  (mediana de 7,8 µs contra 12,0 µs).
- **A diferença por chamada não vem do tamanho das sequências.** Nas 89
  instâncias, o comprimento médio das sequências nos nós é quase o mesmo (13,4
  regiões no caminho, 14,2 no ciclo), e o tempo por chamada e por região fica
  em 0,64×.
- **Por família**, por chamada: tesselações 0,48× (43,8 contra 20,9 µs),
  aleatórias 0,50× e OSM 0,77×. O oráculo ocupa a mesma fração do tempo nas
  duas formas (63%).
- **O ciclo é mais rápido em 415 de 544 instâncias**, mesmo começando com gap
  inicial de 100% (a raiz tem limite zero), contra 67% do caminho, em mediana.

**Leitura.** Com extremos fixos, a ordem é menos determinada: há mais ordens
quase equivalentes entre o início e o fim, e o limite parcial sobe mais devagar.
O caso 189, com zigue-zague entre fileiras, mostra o contrário: ali os extremos
fixos desfazem a ambiguidade. Essa explicação é uma hipótese; a medição só
separa árvore e oráculo.

**Limitações.** Uma repetição num Mac pessoal, com 6 execuções simultâneas e
limite de 60 s. As otimizações do oráculo diferem entre os modos: ciclo e
caminho têm reaproveitamentos próprios, e a medição não separa o efeito de
cada um. Dados locais: `experiments/path-vs-cycle-20261021`.

## fekete-reference-dantzig-2026-10-20

**Pergunta.** A comparação de
[`fekete-reference-armijo-2026-10-17`](#fekete-reference-armijo-2026-10-17) se
mantém noutra máquina, de outro fabricante? **Formulação.** A mesma da armijo,
só com as configurações A, B e C: 300 s e 3 repetições, e depois 900 s nas
abertas; dantzig (Intel Core i9-12900K, 8 núcleos de performance com SMT mais 8
de eficiência, 24 CPUs, Linux; [`machines.md`](../machines.md)), em 20 e 21/10.
- **Threads:** a A usa as 16 CPUs dos núcleos de performance; a B e a C, uma
  execução por núcleo de performance, presa a ele.
- **Fekete:** o mesmo upstream `5cd8450` com o mesmo lockfile (Gurobi 12.0.0).
- **Nosso:** o mesmo código da armijo (os commits depois de `2467373` só mudam
  um teste), compilado na dantzig para x86-64 genérico, como lá.
- **Carga alheia:** a partir das 11h (horário local), outro usuário ocupou de 4 a
  6 núcleos durante quase toda a A e toda a B e a C.
  - Na A, as execuções sob carga ficaram a 1% da mediana da própria instância
    (nas que levam ≥ 1 s), provavelmente porque a carga ficou nos núcleos de
    eficiência.
  - Nos núcleos da B, houve carga em 1 de 3348 execuções; nos da C, em 55 de
    1674, 9 delas com mais de 10 ms. As medianas de três repetições não mudam.
  - **Nas extensões de 900 s (21/10) a carga mudou de lugar.** Ficaram 4
    processos alheios nas threads irmãs dos núcleos de performance 0 a 3. A A
    não mudou: fechou as mesmas instâncias que na armijo, em 0,96× o tempo
    dela. Já 25 das 54 execuções da B dividiram o núcleo físico com eles e
    ficaram cerca de 1,6× mais lentas que as limpas (razão de tempo
    dantzig/armijo de 1,04 contra 0,65). Por isso, a B com 900 s na dantzig
    não entra nas comparações; no CSV, são as linhas da B com `time_limit` 900.
    O caso 189 com 900 s também dividiu o núcleo.

**Resultado.**

| Configuração | Resolvidas em 300 s (3 repetições), dantzig | armijo | Tours válidos a 1e-7 |
|---|---|---|---|
| Fekete A (CHR, 16 threads) | 545; 549 com 900 s | 542; 549 com 900 s | 151 |
| Fekete B, CHR (1 thread) | 528 | 523 | 151 |
| Fekete B, LEFP (1 thread) | 534 | 529 | 157 |
| Nosso, C (1 thread) | 557 | 557 | 558 |

- **O que a armijo resolve, a dantzig também resolve.** A dantzig fecha
  algumas a mais: 3 na A e 5 em cada raiz da B. A A resolve as 542 do artigo e
  mais 3; os tempos publicados por eles são 1,26× os dela (instâncias ≥ 1 s).
  Com 900 s, a A fecha 549 (98,39%): as mesmas instâncias que na armijo e no
  artigo, com as mesmas 9 abertas.
- **As máquinas favorecem solvers diferentes.** Razão de tempo dantzig/armijo,
  em média geométrica, nas instâncias que a armijo leva ≥ 1 s:
  - Fekete A: 0,98;
  - Fekete B: 0,78 com a CHR e 0,77 com a LEFP;
  - nosso C: 1,20.

  No conjunto todo, 0,94, 0,79 e 1,19. Nas instâncias mais lentas, a diferença
  entre repetições não passa de 4%. Os binários não são compilados para a CPU,
  então a diferença vem do hardware. O artigo deles foi medido em AMD (Ryzen 9
  7900).
- **Por isso o speedup depende da máquina.** Com 300 s nas duas, sem as
  extensões (Fekete / nosso, média geométrica nas instâncias que os dois fecham):

  | Comparação | dantzig | armijo | Onde o Fekete leva ≥ 1 s: dantzig / armijo |
  |---|---|---|---|
  | A (16 threads) / C (1 thread) | 16,0× (545) | 20,1× (542) | 61× / 75× |
  | B, CHR / C (1 thread cada) | 45,1× (528) | 66,7× (523) | 180× / 256× |
  | B, LEFP / C (1 thread cada) | 44,5× (534) | 65,8× (529) | 159× / 239× |

  Contando a extensão de 900 s da A, a primeira linha fica em 16,5× na dantzig
  e 21,0× na armijo (549 instâncias); onde o Fekete leva ≥ 1 s, 67× e 84×. Nas
  duas máquinas o nosso é mais rápido em todas as instâncias que os dois
  fecham.
- **Caso 189 com 900 s:** o nosso fica com LB 429,45 e UB 437,44 (gap 1,83%, no
  núcleo dividido), ainda melhores dos dois lados que os melhores do Fekete na
  dantzig (LB 420,19 na A, UB 441,52 na B com LEFP).

**Limitações.**
- Uma rodada por máquina. A B com 900 s ficou contaminada pela carga alheia e
  foi descartada.
- Outro usuário ocupou núcleos de eficiência durante a rodada. O efeito medido
  é desprezível, mas a máquina não estava exclusiva.
- Dois processadores, um de cada fabricante. Não dá para separar o que vem da
  microarquitetura e o que vem de frequência e limite de potência (o Ryzen 7
  9700X tem TDP padrão de 65 W; não conferimos a configuração da armijo).

**Dados.** `per-case.csv`, no formato de
[`fekete-reference-armijo-2026-10-17`](#fekete-reference-armijo-2026-10-17). Os
registros completos ficam locais em `experiments/fekete-reference@dantzig`.

## fekete-reference-armijo-2026-10-17

**Pergunta.** Com o código original de Fekete et al., sem modificações e na
configuração do artigo, quanto o nosso solver ganha nas 558 instâncias do
artigo, na mesma máquina? **Formulação.** TSPN de ciclo livre (ordem livre, sem
ponto fixo), regiões poligonais de `instances_socg_simplified.zip` (sha256
`21084118…`); eps 0,001; 300 s por execução, 3 repetições em ordem embaralhada
(semente 20261017), e depois uma execução de 900 s (`--extend`) nas instâncias
que alguma repetição deixou aberta (só A, B e C). armijo (Ryzen 7 9700X, 8
núcleos e 16 CPUs, Linux; [`machines.md`](../machines.md)), sem outros jobs,
uma configuração por vez, de 17 a 19/10. O caso 189, o único aberto no nosso,
teve ainda uma execução de até 6 h com uma thread. Comandos `tpp.py fekete-paper`,
`fekete-single-thread`, `ours-single-thread`, `fekete-paper-lefp` e
`ours-threads`
([protocolo](../README.md#medições-de-referência-o-fekete-original-e-o-nosso-configurações-a-a-e)).
- **Fekete:** upstream `5cd8450` sem modificações (Gurobi 12.0.0, Boost 1.83.0,
  CGAL 6.0.1, g++ 13.3), chamado como em
  `evaluation/rq4_scalability/01_run_solver.py` (`FarthestPoly`, `DfsBfs`,
  tolerâncias originais) e cronometrado como lá (`time.time()` em volta de
  `solve_annotated_instance`; Gurobi aquecido fora da medição). "Resolvida" é o
  `is_optimal` dele, `(UB−LB) ≤ eps·UB + eps`.
  - **A**, a configuração do artigo: raiz CHR (`OrderRoot`), uma execução por
    vez com 16 threads presas às 16 CPUs. O artigo usou 24 threads num Ryzen 9
    7900 (12 núcleos).
  - **B**: uma thread, raízes CHR e LEFP, oito execuções simultâneas, cada uma
    presa a um núcleo físico.
  - **D**: como a A, com a raiz LEFP.
- **Nosso:** `tpp-unordered --cycle` com `cache,features,root,interval`,
  factibilidade 1e-8 e aritmética gmp, o mesmo binário (sha256 `73c1e229…`,
  código de `2467373`) em todas as rodadas. "Resolvida" é `UB ≤ 1,001·LB` com
  limites provados.
  - **C**: uma thread, no mesmo regime da B.
  - **E**: no regime da A, com `--subtree-threads 16 --max-frontier-mib 1`
    (subárvores em paralelo desde a raiz).
- Os tours dos dois são validados a 1e-7 fora da medição (`validate_cycle`).

**Resultado.**

| Configuração | Resolvidas em 300 s (nas 3 repetições) | Após 900 s nas abertas | Tours válidos a 1e-7 |
|---|---|---|---|
| Fekete A (CHR, 16 threads) | 542 (97,13%) | 549 (98,39%) | 151 |
| Fekete D (LEFP, 16 threads) | 539 | — | 159 |
| Fekete B, CHR (1 thread) | 523 | 541 | 151 |
| Fekete B, LEFP (1 thread) | 529 | 538 | 159 |
| Nosso, C (1 thread) | 557 | 557 (558 com o caso 189 em 1 h 29 min) | 558 |
| Nosso, E (16 threads) | 557 | — | 558 |

- **A A reproduz o artigo.** Ela resolve 97,13% em 300 s e 98,39% em 900 s, com
  9 abertas e gap máximo de 5,00%. O artigo reporta 97,13%, 98,39% e 5,53%.
- **E é mais rápida que as execuções do próprio artigo.** O repositório deles
  publica os resultados por instância
  (`evaluation/rq4_scalability/results_optimality_gaps_300` e `long_runs`,
  máquinas `algra01` a `algra06`, 24 threads).
  - Com a CHR, as resolvidas em 300 s são as mesmas 542 e, em 900 s, as mesmas
    7 das 16 estendidas.
  - Nas 119 que levam pelo menos 1 s, os tempos publicados são 1,23× os da A em
    média geométrica (1,11× a 1,32× entre os percentis 10 e 90).
  - Com a LEFP, o artigo resolve 538 e a D, 539, e os tempos publicados são
    1,31× os da D.

  Logo, as 16 threads da armijo não desfavorecem o Fekete.
- **A CHR é a melhor raiz dele, como no artigo.** Com 16 threads, a LEFP (D)
  resolve 539 contra 542 da CHR (6 só na A, 3 só na D), com tempos parecidos:
  razão geométrica D/A de 1,02× nas 536 comuns.
- **Tempo.** O tempo de uma instância é a mediana das 3 repetições quando as 3
  fecham; senão, o da execução de 900 s (D e E não têm essa extensão). O nosso
  é mais rápido em todas as instâncias que os dois fecham, em todos os pares
  (razão mínima: 1,73×, A contra C).

  | Fekete / nosso | Fechadas pelos dois | Geométrica | Mediana | Onde o Fekete leva ≥ 1 s | 41–60 regiões |
  |---|---|---|---|---|---|
  | A / E (16 threads cada) | 549 | 23,2× | 16,8× | 114,0× (n = 120) | 78,3× (n = 136) |
  | D / E (16 threads cada) | 539 | 21,7× | 17,6× | 76,2× (n = 104) | 56,8× (n = 127) |
  | A (16 threads) / C (1 thread) | 549 | 21,0× | 16,0× | 84,1× (n = 120) | 60,0× (n = 136) |
  | B, CHR / C (1 thread cada) | 541 | 72,9× | 72,7× | 291,5× (n = 185) | 303,8× (n = 128) |
  | B, LEFP / C (1 thread cada) | 538 | 68,6× | 67,0× | 253,5× (n = 181) | 261,2× (n = 126) |

  Somados, os tempos das 549 instâncias comuns dão 7282 s na A, 42 s na C e
  21 s na E. As 8 que só o nosso fecha levam de 2,2 s a 52 s na C.
- **De onde vem o ganho (B contra C, 1 thread cada).** A comparação usa a
  mesma definição de nó nos dois: nós retirados da fila, inclusive os podados
  na retirada (`num_iterations` dele, `nodes` nosso). Nas 166 instâncias em que
  o Fekete leva ≥ 1 s com a CHR e os dois fecham nas 3 repetições (melhor
  repetição de cada), o speedup de 257× se decompõe assim:
  - a árvore dele tem 1,08× o tamanho da nossa (0,97× nas 111 com 41 a 60
    regiões);
  - cada nó nosso custa 1/238 do dele: 8,9 ms contra 33 µs, em mediana.

  Com a LEFP, a árvore dele tem 1,37× a nossa, e o custo por nó é 175× maior. O
  ganho vem do custo de cada nó: com a raiz CHR, as duas buscas têm o mesmo
  tamanho. Os filhos de um nó do Fekete são resolvidos ao ramificar, e os nossos
  podem ser avaliados só ao sair da fila. Por isso a comparação é de nós, não
  de chamadas ao oráculo.
- **Paralelismo do nosso (E contra C).** Nas 19 instâncias em que a C leva
  ≥ 1 s, as 16 threads dão 6,6× em média geométrica (de 0,97× a 11,4×; a mais
  lenta, o caso 542, cai de 52 s para 7,1 s). No conjunto todo, 1,14×, porque a
  maioria leva milissegundos; a soma cai de 136 s para 32,5 s.
- **Abertas.** Nenhuma instância fecha só no Fekete. A única aberta no nosso
  em 900 s, o caso 189 (`kuala_lumpur_n060_seed3530`), também fica aberta em
  todas as variantes dele; mesmo assim, os limites da C em 900 s (LB 431,26, UB
  437,44, gap 1,41%) já são melhores dos dois lados que os melhores dele (LB
  421,15 na A, UB 441,52 na B com LEFP; gap de 4,71% na A). Com até 6 h, uma
  thread e sem teto de chamadas, a C fecha o caso 189 em 5363 s (LB 436,556,
  UB 436,992; 27 milhões de nós, 173 milhões de chamadas, 4,3 GB). O último
  tour melhor aparece aos 1122 s; o resto é prova. Assim o nosso fecha as 558
  com uma thread. Na E, as três repetições pararam no teto de 10⁸ chamadas
  aos 265 s, com LB 408,47 (abaixo dos 428,28 da C em 300 s) e UB 437,17.
- **Como as chamadas ao oráculo terminam (C, repetição 1).** Cada chamada do
  ciclo fecha de um único jeito: prova intervalar de um candidato da construção
  geométrica, polimento por pontos interiores ou caminho racional exato, o
  único que tem a busca ancorada na borda
  ([`tspn.md`](../../docs/algorithms/tspn.md),
  [`convex-cycle.md`](../../docs/algorithms/convex-cycle.md#binary64-stage-with-a-requested-gap-2026-10-08)).
  Nas 557 instâncias sem o caso 189 (5,80 milhões de chamadas), 97,37% fecham
  pela prova intervalar e 2,63% pelo polimento; toda tentativa de polimento
  fechou, e nenhuma chamada foi ao caminho racional. O oráculo ocupa 46% do
  tempo do solver, e o polimento, 3,2%. Por família, o polimento fecha 0,98%
  das chamadas nas OSM, 5,3% nas aleatórias e 25,7% nas tesselações (19% do
  tempo delas), onde as regiões se tocam. No caso 189, a execução que fecha
  (173 milhões de chamadas) tem 94,2% e 5,8%, também sem caminho racional; o
  caminho racional rodou uma única vez, na execução de 300 s desse caso.
- **Tours.** A maior distância de um tour do Fekete a um polígono tem mediana
  2,7e-6 e máximo 9,6e-4, nas unidades da instância. Em 22 instâncias o UB
  dele fica abaixo do LB provado do nosso, por no máximo 5,2e-7 relativo. É a
  tolerância de factibilidade dele (1e-3), não um erro do solver. Nos tours do
  nosso, a maior distância é 2,9e-11.

**Limitações.**
- Uma máquina só. A A usa 16 threads, contra 24 no artigo, mas resolve as
  mesmas instâncias que as execuções publicadas, cerca de 1,2× mais rápido.
- As execuções de 900 s e a de 6 h têm uma repetição; usaram o mesmo build do
  Fekete e o mesmo binário nosso das rodadas de 300 s. O Fekete não teve
  execução de 6 h no caso 189.
- Os critérios diferem: o dele mede o gap relativo ao UB e soma eps absoluto.
  Pelo critério do nosso, a A fecha 526 em 300 s. Os limites do Fekete são
  numéricos (SOCP); os do nosso são provados.
- As rodadas de 17 e 18/10 do nosso tinham um teto de 10⁸ chamadas ao
  oráculo. Ele só foi atingido no caso 189: na E e numa primeira execução de
  6 h, que parou aos 2677 s com gap de 0,59% e foi repetida sem teto em 19/10.
  Nas demais, quem encerrou foi o fechamento do gap ou o limite de tempo.
- O resultado do nosso vale para esse binário (`2467373`). Uma nova versão só se
  compara a estas linhas se rodar na armijo no mesmo protocolo; noutra máquina,
  A e B têm de ser repetidas (cerca de 10 h).

**Dados.** `per-case.csv` tem uma linha por caso, configuração, raiz e limite de
tempo (`time_limit` 300, 900 ou 21600):
- `solved`, `solved_strict` (`UB ≤ 1,001·LB`) e `tours_valid` contam
  repetições;
- `seconds_*` resumem os tempos das repetições;
- `lower_bound` e `upper_bound` são os melhores entre as repetições;
- `tour_max_distance` é a maior distância de um tour a um polígono.

Os registros completos (tours, contadores, CPUs e carga de cada execução) ficam
locais em `experiments/fekete-reference@armijo`.

## tspn-case189-zigzag-2026-10-16

**Pergunta.** O caso 189 (`kuala_lumpur_n060_seed3530`) fecha com extremos
fixos em 12 s, mas fica aberto no TSPN em 300 s. Onde está a dificuldade, e
alguma regra de ramificação a resolve? **Formulação.** TSPN de ciclo livre, gap
`UB ≤ 1,001·LB`, `cache,features,root,interval`, com as correções de
[`cycle-coincident-contacts-2026-10-16`](#cycle-coincident-contacts-2026-10-16);
Mac M4 Pro, uma thread salvo indicação; variantes de uma mesma tabela rodaram
juntas, com a mesma carga.

**Resultado.**
- **Dificuldade caminho × ciclo.** Nas 557 instâncias fechadas nas duas
  formulações, a correlação dos log-tempos é 0,94 e o ciclo leva, em mediana,
  0,41× o tempo do caminho
  ([`float-recovery-dantzig-2026-10-08`](#float-recovery-dantzig-2026-10-08)
  contra a comparação TSPN de 16/10). Os desvios são grandes nos dois sentidos:
  o caso 189 vai de 12 s a aberto; o 130 vai de 978 s a 3,9 s, e o 493 de 84 s
  a 0,003 s.
- **Ótimo do caso 189.** A ILS (`--primal-ils`) acha em ~30 s um tour de
  436,992. Com `lazy --lazy-oracles --dive-interval 0`, fronteira de 512 MiB e
  8 threads de subárvore, o B&B certifica esse tour dentro de 0,1% (LB 436,556)
  em 675 s e 154 M nós (600 s, checkpoint e retomada). O tour do Fekete de 16/10
  (441,52) fica 1,04% acima.
- **Não são os toques.** Encolhendo cada casa 0,1% (ou 1%) em direção à média
  dos vértices, nenhum par se toca, e o gap em 120 s fica em 2,58% (2,70%),
  contra 3,23% no original. A dificuldade é o zigue-zague entre fileiras, com
  muitas ordens de comprimento quase igual.
- **Ramificação forte.** `branch` compara a menor inserção das 3 regiões
  ausentes mais distantes. Comparar todas as ausentes (experimento, sem
  commit) leva o caso 189 a 0,89% em 120 s e 0,40% em 600 s (com ILS e
  `lazy`), contra 1,65% com 3. No protocolo, nas 558 com 60 s, a mesma regra
  deixa o resto 7–12× mais lento (média geométrica) e perde 5 instâncias.
  Restringir as candidatas às que estão a ≥ α da maior distância (α = 0,3 a
  0,9), ou só trocar a mais distante quando a inserção dela fecha menos de
  θ = 0,1–0,3 do gap do nó, não evita a troca: com α = 0,7 o caso 189 cai para
  0,23–0,5% em 60 s, mas `vienna_n060_seed3619` (30 s) deixa de fechar e
  `warsaw_n060_seed4239` e `dubai_n060_seed3351` ficam 1,4–1,9× mais lentas,
  também com o tour da ILS desde o início.

**Leitura.** Nenhuma regra fixa de ramificação testada melhora o caso 189 sem
piorar outras instâncias; nada mudou no solver. O caso fecha com mais tempo e
threads. **Limitações.** Uma máquina e uma repetição; as regras foram medidas
em 7 instâncias escolhidas (as que pioraram com todas as candidatas) e no
corpus só na versão com todas. Dados locais e patches dos experimentos:
`experiments/zigzag-case189`.

## cycle-coincident-contacts-2026-10-16

**Pergunta.** Na [comparação TSPN × Fekete de 16/10](#tspn-fekete-558-2026-10-16)
(558 originais, 300 s), a
única instância aberta, o caso 189 (`kuala_lumpur_n060_seed3530`, 60 casas
geminadas do OSM), mandou 77% das chamadas do oráculo de ciclo ao polimento por
pontos interiores. Por que a construção geométrica (busca em coordenadas) não
fecha essas chamadas? **Formulação.** TSPN de ciclo livre, gap
`UB ≤ 1,001·LB` (absoluto 0), `cache,features,root,interval`, uma thread;
Mac M4 Pro. Contrato e prova do estágio binary64 inalterados: as duas mudanças
só escolhem propostas ([`convex-cycle.md`](../../docs/algorithms/convex-cycle.md)).

**Diagnóstico.** 8.880 chamadas capturadas do caso 189 (uma a cada 200) foram
repetidas no estágio binary64 de três formas: só a construção, a construção
reiniciada a partir do próprio resultado até parar de melhorar, e com o
polimento. A construção fechava 25,2%. Em 55% ela parava num ciclo
comprovadamente acima do ótimo (mediana 6,9%, máximo 31,6%), e reiniciá-la
fechava só 0,3% dessas. Em 14% o ciclo estava dentro do gap, mas faltava o
certificado (sempre com elos nulos). Duas causas:
- **Movimento em bloco em binary64.** Duas casas com a parede comum dada pelos
  mesmos vértices binary64 têm interseção recortada reduzida ao canto distante.
  O par de contatos coincidentes, que estava no canto bom, era movido para lá e
  fixado. Correção: em binary64 o bloco só se move se o movimento não alongar
  seus dois elos.
- **Duais dos elos curtos.** As quatro propostas aplicam uma regra só a todos
  os elos curtos, e cada bloco de contatos coincidentes costuma precisar de um
  vetor diferente. Correção: uma quinta proposta escolhe por trecho de elos
  curtos, por programação dinâmica, já que `D(u)` se separa por trecho.

**Resultado.**
- **Mesmas 8.880 chamadas.** A construção passa a fechar 78,5% (25,2% antes);
  tempo total do estágio 0,69 → 0,31 s.
- **Caso 189, B&B de 300 s lado a lado com a `main` (`2cb14df`).** Chamadas
  3,73 M → 12,47 M, nós 0,83 M → 1,98 M, polimento 77% → 12% das chamadas, LB
  426,37 → 428,58, gap 2,55% → 2,05%. Continua aberto.
- **558 originais, 60 s, variantes intercaladas por caso, um processo por
  vez.** 557/558 fechadas nas duas (o caso 189 fica aberto), 558/558 tours
  válidos (`validate_cycle`, 1e-7), nenhum par de limites incompatível. Razão
  de tempo `main`/nova (média geométrica): 1,001× nos casos ≥ 10 ms (n = 150),
  1,012× nos ≥ 0,1 s (n = 57), 1,017× nos ≥ 1 s (n = 12); a mais lenta, 0,91×
  num caso de 0,2 s. Polimento fora do caso 189: 3,48% → 2,59% das chamadas.

**Limitações.** Uma máquina e uma repetição; fora do caso 189 o efeito no tempo
é neutro. No caso 189 sobram ~20% de chamadas em que a construção para acima
do ótimo (mediana 1,6%), a maioria sem elos curtos; a causa não foi investigada. O oráculo
de caminho usa a mesma regra única para os elos curtos (`interval_dual_lower`)
e não recebeu a escolha por trecho. Dados locais:
`experiments/cycle-coincident-contacts`.

## tspn-fekete-558-2026-10-16

**Pergunta.** No TSPN, como o nosso solver se compara ao de Fekete et al. nas
558 instâncias originais do artigo deles, lado a lado e na mesma máquina?
**Formulação.** TSPN de ciclo livre (ordem livre, sem ponto fixo), regiões
poligonais fechadas de `instances_socg_simplified.zip`; gap relativo de 0,1% nos
dois; 300 s por execução (teto do processo 360 s), uma thread cada, quatro
execuções simultâneas, uma repetição; Mac M4 Pro. `tspn-compare
--socp-defaults`: o Fekete roda com as tolerâncias originais dele (Gurobi
padrão, `FEASIBILITY_TOLERANCE` 1e-3, `SPANNING_TOLERANCE` 9e-4); o nosso
(`2cb14df`) com factibilidade 1e-8 e `cache,features,root,interval`. Tours
validados de forma independente a 1e-7 (`validate_cycle`). "Fechou" é gap
numérico fechado: os limites do nosso são provados, os do Fekete são numéricos.

**Resultado.**
- **Nosso:** 557/558 com `UB ≤ 1,001·LB` (soma 385 s, a mais longa em 29,8 s);
  558/558 tours válidos. A aberta é o caso 189 (ver
  [`tspn-case189-zigzag-2026-10-16`](#tspn-case189-zigzag-2026-10-16)).
- **Fekete:** 536/558 pelo critério dele, `(UB−LB)/UB ≤ 0,1%` (526 por
  `UB ≤ 1,001·LB`); as outras 22 pararam no tempo. Só 155 dos 558 tours dele
  validam a 1e-7, o esperado com a tolerância de factibilidade de 1e-3.
- **Tempo, nas 536 fechadas pelos dois:** o nosso é mais rápido em 535 (a
  exceção, `buenos_aires_n010_seed60`, leva 1,1 ms contra 0,7 ms). Razão
  Fekete/nosso: média aritmética 91,1×, geométrica 41,4×, mediana 41,5×; nas de
  41–60 regiões, 173× geométrica (n = 124). Restrita às 155 com tour do Fekete
  válido, que são as menores, a média geométrica é 13,8×.

**Limitações.** Uma repetição, num Mac pessoal com quatro execuções
simultâneas. As tolerâncias de factibilidade diferem (as originais do Fekete
contra 1e-8 no nosso). As 22 abertas do Fekete ficam fora das razões, o que
subestima a vantagem. Dados locais: `campaigns/tspn-fekete-558-original`.

## tspn-paula-long-armijo-2026-10-14

**Pergunta.** Terminando subárvores em vez de descartar
([`tspn-frontier-completion-2026-10-13`](#tspn-frontier-completion-2026-10-13)),
quantas das 13 abertas o B&B certifica em 24 h? **Formulação.** A de
[`tspn-paula-long-armijo-2026-10-11`](#tspn-paula-long-armijo-2026-10-11), com
`lazy` sem mergulhos, fronteira de 128 MiB e até 24 h por instância, a partir do
melhor tour; solver de `a00ab58`, na armijo, 8 processos de uma thread, um por
núcleo físico, em ordem crescente de gap (`--repeat-bnb`, segunda tentativa).

**Resultado.**
- **Certificadas (7):** 100prb1173-10x10 em 2,0 h (279 M nós), 75lin105 em
  3,5 h (572 M), 100vm1084 em 4,8 h (625 M), 100rat783-10x10 em 5,4 h (668 M),
  81vm1084-9x9 em 5,5 h (433 M), 100C10k.0 em 15,2 h (1,79 bi) e 100nrw1379 em
  17,8 h (2,11 bi). A pilha não passou de 188 nós.
- **Coleção inteira:** 229 de 235 certificadas (eram 222). Abertas: 80rd400
  (4,59%), 100rat783 (7,01%), 89pcb442 (8,10%), 100pcb1173 (8,18%), 100pr1002
  (8,42%) e 99d493 (14,63%); esses gaps são os da rodada com descarte, o maior
  LB de todas. As tentativas de 24 h dessas seis terminaram com LB pior
  (13,7–23,7%), como esperado de uma busca em profundidade que não fecha, e
  processaram 1,7–2,8 bi nós cada.
- **Tours:** nenhum B&B encurtou o tour inicial; os 235 melhores continuam
  empatados com o melhor da Paula, todos válidos. Nas 229 certificadas, o melhor
  dela é ótimo.
- **Vazão:** 1,2–1,95× os nós por segundo da rodada com descarte (com 8
  processos em vez de 4).
- **Previsão:** as que fecharam precisaram de 3,5–35× os nós expandidos até o
  congelamento da rodada anterior, ~2× por 0,77 ponto de gap no congelamento.

**Limitações.** Sem checkpoint, as 24 h das seis abertas não são
reaproveitadas. Uma tentativa por instância, 8 processos simultâneos.

## tspn-bounds-and-threads-2026-10-14

**Pergunta.** Depois da rodada de completar subárvores
([`tspn-paula-long-armijo-2026-10-13`](#tspn-paula-long-armijo-2026-10-13) e o
resultado de 14/10: 229/235), o que os dados ensinam sobre o custo de
certificar, e o que reduz esse custo? **Formulação.** TSPN de ciclo livre, gap
relativo 1e-6, `cache,features,root,interval`, `lazy` sem mergulhos, 128 MiB,
completando subárvores, a partir do melhor tour, até o certificado (limite de
1.800 s). Mac M4 Pro, 6 processos simultâneos, com outros programas abertos:
razões entre variantes da mesma instância, não tempos absolutos.

**Resultado.**
- **Regra de custo.** Nas 15 instâncias certificadas na armijo, os nós para
  certificar crescem ~2,31× por ponto do gap provado numa rodada de 600 s
  (r = 0,98 em log; desvio dos resíduos 0,13 década). Tudo com gap ≤ 7,8% aos
  600 s fechou e tudo com ≥ 8,8% está aberto. Pela mesma regra, e pela do gap
  no congelamento (2× por 0,77 ponto), as abertas pedem, numa thread da
  armijo: 80rd400 ~3–4 dias, 100rat783 ~10–13 dias, 100pcb1173, 100pr1002 e
  89pcb442 1–3 meses, e 99d493 décadas.
- **Para onde vai o trabalho** (56a280 e 45ts225): cada expansão gera 17–18
  filhos, e 93% caem na triagem por inserção (~1,1 entra na fila); dos nós
  retirados, 12–16% são podados pelo oráculo e 84–88% são expandidos.
- **Opções de limite**, nós e tempo até o certificado relativos à base:
  `--multi-insertion-bound` 0,43–0,76 dos nós e 3,0–6,9× o tempo (56a280,
  45ts225, 60pr299, 84fl417, 50gil262); em 100i2500-708 (100 regiões) não fecha
  em 1.800 s, contra 440 s. Ele poda 57–58% dos nós em que é calculado e ocupa
  89% do tempo. `one-tree`: os mesmos nós. Ramificação forte: 0,95–1,84× os nós e
  1,5–3,2× o tempo. `--insertion-lookahead 8`: 0,78–0,90 dos nós e 0,96–1,25×
  o tempo. `--path-certificate-dual`: não fecha em 1.800 s em 2 de 3.
- **Held–Karp** (1-árvore com penalidades) sobre distâncias entre regiões,
  que vale como limite: 81,0% do tour em 99d493 (nosso LB: 85,4%) e 61,5% em
  80rd400 (95,4%), com o subgradiente convergido.
- **Subárvores repetidas** (a mesma sequência por caminhos diferentes): 5 em
  28.472, 15 em 19.825 e 15 em 38.270 nós expandidos, as mesmas sequências de
  polígonos com peças diferentes.
- **Lado primal:** a partir do tour da ILS de 100i1000-410 (0,19% acima),
  completar subárvores achou o ótimo aos 417 s e o certificou aos 622 s; a busca
  de melhor limite da armijo (11/10) o achou a 4 s do fim.
- **Threads de subárvore e checkpoints**
  ([contrato](../../docs/algorithms/unordered-tpp.md#orçamento-de-memória-da-fronteira-2026-10-09)):
  50gil262 com 2, 4 e 8 threads fecha com os mesmos 2.425.412 nós em 35,7,
  19,3 e 10,8 s; interrompida aos 20 s e retomada de um checkpoint, soma
  exatamente esses nós (402.454 + 2.022.958).

**Leitura.** O custo de certificar é exponencial no gap que a busca de melhor
limite alcança, e a régua permite comparar ideias: 8 threads valem ~2,3
pontos de gap; o `--multi-insertion-bound` de graça valeria ~0,3–1 ponto, e
com o custo atual não paga. Nenhuma das opções nem o Held–Karp aproxima o
que falta em 99d493 (~10 pontos). Paralelizar e retomar fecham as instâncias
a dias de distância; as outras pedem um limite novo.

**Limitações.** Uma repetição, Mac carregado. As projeções são extrapolações
de 15 pontos, sem garantia fora deles.

## tspn-frontier-completion-2026-10-13

**Pergunta.** Na armijo, as 13 instâncias abertas congelaram no piso dos nós
descartados ([abaixo](#tspn-paula-long-armijo-2026-10-13)). Uma busca que
passa do orçamento pode continuar completa, sem perder nós, e quanto o nó ainda
encolhe sem mudar a busca? **Formulação.** TSPN de ciclo livre, gap relativo
1e-6, `cache,features,root,interval`, `lazy` sem mergulhos a partir do melhor
tour (da campanha da armijo, ou da rodada de 600 s de 2026-10-09 para 50gil262
e 45ts225). `main` (`95b81cb`) × branch `frontier-completion`. macOS arm64
(M4 Pro), várias execuções simultâneas, com outros programas abertos.
Contratos em
[`unordered-tpp.md`](../../docs/algorithms/unordered-tpp.md#orçamento-de-memória-da-fronteira-2026-10-09).

**Resultado.**
- **Nó de 96 B** (antes 168): buscas idênticas (chamadas, nós, limites e
  contadores de visita) em 6/6 instâncias de 100 regiões com `lazy` e 300 mil
  chamadas, em 4/4 na política padrão com 30 mil, na regressão SoCG (48/48,
  2,75 contra 2,76 s, todos válidos) e no corpus de caminhos (40/40,
  `--stride 14`, gap 1e-3, todos exatos). Bytes por nó no pico, com
  `lazy`: caminhos ancorados 195–197 → 123–125, ciclos 251–297 → 179–225. Com
  1,5 M chamadas, o RSS de 75lin105 cai de 1,74 para 1,08 GB, e o de
  100prb1173-10x10 de 2,23 para 1,69 GB, no mesmo tempo; com
  `--sequence-storage deltas`, para 0,93 e 1,57 GB. Na política padrão, um
  caminho resolvido ganha ~24 B.
- **Completar** com 8 MiB: 50gil262 fecha com os mesmos 2.425.412 nós da busca
  sem limite, em 53,7 s contra 63,3 s (RSS 14 contra 145 MiB), e 45ts225 com
  3.889.590 nós contra 3.889.597, em 46,1 s contra 58,8 s (RSS 16 contra 788
  MiB). A pilha nunca passou de 55 nós. Descartando, com os mesmos 8 MiB, as
  duas param com `memory_limit` em 5,7% e 7,8%. Em 100i2500-708 (três processos,
  sem paginar), os mesmos 22.587.951 nós levam 620,9 s sem limite (pico de
  9,88 M nós, RSS 4,77 GB), 515,3 s com 1 GiB (RSS 1,23 GB) e 446,1 s com 128 MiB
  (RSS 170 MB, pilha de até 66 nós).
- **LB sem fechar** (120 s): 75lin105 fica com gap 5,89% sem limite, 9,05%
  com 512 MiB e 11,57% com 128 MiB; 100prb1173-10x10, 5,30%, 8,00% e 10,17%.
  A pilha chegou a 63–119 nós.
- **Testes:** `tpp-tspn-tests` (12 instâncias com orçamento de 1 byte: ao
  completar, padrão e `lazy`, fecham no ótimo sem orçamento; ao descartar, os
  limites cercam o ótimo) e `tpp-unordered-tests`, também sob ASan/UBSan, e o
  solver instrumentado com 4 MiB em 50gil262 por 240 s (1,09 M nós, `packed` e
  `deltas`).

**Leitura.** Com o incumbente ótimo, as duas ordens abrem os mesmos nós, e
terminar subárvores é mais rápido por nó: a pilha fica no cache, e a fila
grande não. A memória deixa de ser o limite de uma certificação; o tempo
passa a ser. Uma execução que não fecha tem LB pior que uma fila sem limite
no mesmo tempo, porque o LB sobe uma subárvore por vez; as campanhas reportam
o maior LB de todas as execuções.

**Limitações.** Um Mac com outros programas, uma repetição. Uma primeira
rodada de 100i2500-708 e 100i2000-604 com 6 processos paginou e foi
descartada; as contagens de nós em 120 s (75lin105: 6,65 M com 512 MiB contra
5,41 M sem limite) também vêm de 6 processos e são só indicativas. Um incumbente subótimo faria a
busca em profundidade abrir nós que a de melhor limite poderia evitar.

## tspn-paula-long-armijo-2026-10-13

**Pergunta.** Com a fronteira compacta e a parada no piso
([`tspn-frontier-compact-2026-10-12`](#tspn-frontier-compact-2026-10-12)),
quanto mais o B&B certifica nas 15 instâncias da Paula ainda abertas?
**Formulação.** A mesma de
[`tspn-paula-long-armijo-2026-10-11`](#tspn-paula-long-armijo-2026-10-11):
TSPN de ciclo livre, gap relativo 1e-6, `cache,features,root,interval`,
`lazy` sem mergulhos a partir do melhor tour válido, até 4 h, tours validados
(`validate_cycle`, 1e-7). Solver de `95b81cb` (`frontier-compact`), na armijo,
um processo de uma thread por núcleo físico. Duas rodadas de
`paula-campaign --phases bnb`: 88pr439 e 100i3000-803 (`--repeat-bnb`, segunda
tentativa) com 2 processos e 50 GiB de fronteira cada; depois as outras 13
(primeira tentativa) com 4 processos e 22 GiB cada. Dados locais:
`campaigns/paula-long-v1`.

**Resultado.**
- **Certificadas:** 88pr439 em 10.753 s (241 M nós, pico de 101 M na fila e
  20,2 GiB) e 100i3000-803 em 7.536 s (144 M nós, 62,7 M e 34,5 GiB). Na
  tentativa anterior, com 22 GiB e o nó antigo, tinham parado em 0,83% e 0,70%.
- **As outras 13** pararam com `memory_limit`, com o LB congelado no piso dos
  descartados, entre 2.844 e 7.955 s; sem a parada no piso teriam seguido até
  as 4 h (a rodada levou 4,5 h em vez de ~16 h). Gap provado (antes → agora):
  81vm1084-9x9 5,98 → 1,45%, 100prb1173-10x10 5,59 → 1,70%, 75lin105 5,89 →
  1,72%, 100rat783-10x10 5,27 → 1,92%, 100vm1084 6,22 → 2,41%, 100C10k.0 7,66
  → 3,71%, 100nrw1379 7,80 → 3,89%, 80rd400 8,77 → 4,59%, 100rat783 10,79 →
  7,01%, 89pcb442 12,27 → 8,10%, 100pcb1173 12,23 → 8,18%, 100pr1002 12,67 →
  8,42% e 99d493 18,51 → 14,63%.
- **Coleção inteira**, juntando todas as rodadas: 222 de 235 certificadas
  (eram 220); nas 13 abertas, gap de 1,45% a 14,63% (mediana 3,89%; era 7,7%).
- **Tours:** nenhum B&B encurtou o tour inicial; os 235 melhores continuam
  empatados com o melhor da Paula, todos válidos.
- **Descarte:** as 13 descartaram 82 M (100prb1173-10x10) a 1,69 bilhão
  (99d493) de nós, contra 60–127 M expandidos. Antes de congelar, o gap caía
  ~0,7–0,9 ponto por duplicação do tempo (75lin105: 4,07% aos 600 s, 3,16% aos
  1.200 s, 2,28% aos 2.400 s).

**Leitura.** A memória continuava sendo o limite, e os nós descartados, que
teriam de ser resolvidos para certificar, eram a maior parte do trabalho
restante. Isso motivou
[`tspn-frontier-completion-2026-10-13`](#tspn-frontier-completion-2026-10-13).

**Limitações.** Uma tentativa por instância nesta rodada, 2 ou 4 processos
simultâneos. Os LBs reportados são de execuções com descarte; uma execução que
termina subárvores tem LB pior no mesmo tempo, e o relatório usa o maior.

## tspn-frontier-compact-2026-10-12

**Pergunta.** As rodadas longas da armijo
([abaixo](#tspn-paula-long-armijo-2026-10-11)) mostraram que a memória da
fronteira limitava o LB. Quanto se ganha em memória e tempo por nó sem mudar a
busca? **Formulação.** TSPN de ciclo livre, gap relativo 1e-6,
`cache,features,root,interval`. `lazy` sem mergulhos a partir do melhor tour
da campanha da armijo, com 300 mil chamadas (orçamento fixo: as buscas são
comparáveis nó a nó), um processo por vez, mediana de 3 repetições com a ordem
alternada; também a regressão SoCG (48 casos, padrão, 60 s) e o corpus de
caminhos (40 casos, `--stride 14`, gap 1e-3). `main` (`0722d66`) × branch
`frontier-compact`. macOS arm64 (M4 Pro). Contratos em
[`unordered-tpp.md`](../../docs/algorithms/unordered-tpp.md#orçamento-de-memória-da-fronteira-2026-10-09).

**Resultado.**
- **Perfil** (`main`, 100i2500-708, 60 s): verificação de visitas 38%,
  oráculo 37% (construção 24%), manutenção da busca 25%. Por tempo próprio:
  consultas de contato 18,5%, limites das regiões 10,4%, alocação ~14%,
  ordenação de todas as regiões a cada nó ~7%, `best_contact` ~6%.
- **Memória** da fila de `lazy` (bytes por nó no pico): ciclos de 100 regiões
  870–950 → 286 (100i2500-708) e 238 (100pr1002); caminhos ancorados num
  ponto 253 → 197 (88pr439) e 252 → 196 (64lin318). Na política padrão, +4%.
- **Tempo** (300 mil chamadas): 100i2500-708 7,73 → 5,68 s (1,36×), 100pr1002
  8,07 → 6,05 s (1,33×), 100i3000-803 7,59 → 5,58 s (1,36×), 88pr439 5,30 →
  4,50 s (1,18×), 64lin318 6,08 → 5,36 s (1,13×). SoCG 1,04× (48/48), corpus de
  caminhos 1,06× (40/40, todos válidos).
- **Identidade:** mesmas chamadas, nós, limites e contadores de visita em todas
  as comparações. Testes nativos também sob ASan/UBSan, e o solver
  instrumentado com fronteira de 16 MiB (despir e descartar a cada poucos nós).
- **Descartados:** parar os limites de visita contra um piso (0,82–0,94×) e um
  teste de caixa na distância ponto-caminho (0,75–0,8×), que quebram o laço
  vetorizado; blocos maiores no Or-opt da ILS (6 sementes × 120 s nas 5
  instâncias em que nem toda semente chega à Paula: 11 sementes chegam com o
  bloco 3 do padrão, 12 com 8, 8 com 16).

**Leitura.** Com o mesmo orçamento, uma fronteira de ciclo guarda ~3× mais nós
e uma de caminho ~1,3×, e cada nó custa 13–36% menos tempo. A busca agora para
com `memory_limit` quando o LB congela no piso dos descartados.

**Limitações.** Um Mac, execuções curtas (≤ 8 s); em rodadas de horas o ganho
de memória vale enquanto a fronteira não está despida.

## tspn-paula-long-armijo-2026-10-11

**Pergunta.** Nas 21 instâncias da Paula que nenhuma rodada tinha certificado
até 2026-10-09, a ILS com 10 sementes alcança o melhor tour dela, e um B&B
longo com muita memória certifica alguma? **Formulação.** TSPN de ciclo livre,
gap relativo 1e-6, `cache,features,root,interval`, tours validados
(`validate_cycle`, 1e-7). `tpp.py paula-campaign` com o solver de `0722d66`, na
armijo (AMD Ryzen 7 9700X, 8 núcleos, 128 GB, Linux), um processo de uma thread
fixado por núcleo físico. ILS: 10 sementes × 1.200 s por instância
(`--primal-ils 0.99`), 8 simultâneas. B&B: 4 h a partir do melhor tour
(`lazy` sem mergulhos) nas 8 instâncias com gap ≤ 5%, 4 simultâneas com
fronteira de 22 GiB. Referência: melhor, média e pior de 10 execuções do
ILS-BCD na tabela do apêndice do rascunho (comprimentos com 2 casas, outra
máquina). Dados locais: `campaigns/paula-long-v1`.

**Resultado.**
- **B&B:** 6 das 8 fecharam: 100i2500-708 (22 min), 100i2000-604 (27 min),
  100i1000-410 (29 min), 72vm1084-8x9 (44 min), 64lin318 (46 min) e 50eil101
  (78 min), com picos de fronteira de 3,9 a 14,7 GiB (~1 KB por nó nos ciclos
  de 100 regiões). 88pr439 e 100i3000-803 pararam por memória aos 22 GiB, com
  gaps de 0,83% e 0,70%. Em 100i1000-410, o B&B achou aos 1.757 s um tour
  0,19% mais curto que o da ILS, igual ao melhor da Paula, e provou que ele é
  ótimo.
- **Coleção inteira**, juntando todas as rodadas: 220 de 235 certificadas
  (eram 214); nas 15 abertas, gap provado de 0,70% a 18,5% (mediana 7,7%).
- **Tours:** o nosso melhor empata com o melhor dela nas 21 e nunca fica acima
  (nas certificadas, o dela é ótimo). Execução média contra execução média: 2
  melhores (100i3000-803, 100pcb1173), 16 iguais, 3 piores (100i1000-410,
  100i2000-604, 100rat783-10x10). Sementes que alcançam o melhor dela: 10/10
  em 16 instâncias; 100rat783-10x10 9/10, 100pcb1173 7/10, 100i2000-604 4/10,
  100C10k.0 2/10 e 100i1000-410 0/10 (todas a +0,19%). Tempo mediano até o
  melhor dela, entre as sementes que chegam: até 27 s em 18 instâncias, 63 s em
  100vm1084 e 215 s em 100i3000-803.

**Leitura.** Nas que fecharam, a memória era o limite: as rodadas de 600 s
com 1,5 GiB tinham parado em gaps de 2,4% a 4,3%. O LB congela quando o mínimo
da fronteira alcança o piso dos nós descartados (88pr439 aos 7.200 s, e a
execução seguiu por mais uma hora), o que motivou
[`tspn-frontier-compact-2026-10-12`](#tspn-frontier-compact-2026-10-12). Em
100i1000-410, o ótimo difere do tour da ILS sobretudo por um trecho de ~16
regiões em outra posição, fora do alcance do Or-opt e das janelas; só o B&B o
encontrou.

**Limitações.** Uma execução de B&B por instância, com 4 a 8 processos
simultâneos (tempos conservadores). A comparação com a Paula usa a tabela
dela: outra máquina, tempo de CPU por execução e comprimentos com 2 casas.

## tspn-search-retune-2026-10-09

**Pergunta.** Com o oráculo ~50× mais barato e a triagem em binary64, as
escolhas de busca de [`tspn-paula-lower-bound-2026-10-07`](#tspn-paula-lower-bound-2026-10-07)
ainda valem? **Formulação.** Mesmo protocolo daquela seção: TSPN de ciclo
livre, gap 1e-6, uma thread, `cache,features,root,interval`. Nas triagens,
as variantes partem do mesmo incumbente (`--initial-path`, o tour de 120 s de
`--primal-ils 0.95`), e a métrica é a fração do gap da base que a variante
fecha. As 13 de desenvolvimento e as 25 de validação são as mesmas de
2026-10-07. Binários `e2d1898` (triagem) e `7c29102` (validação, 209
fechadas, pipeline), com a mesma busca abaixo do orçamento de memória.
macOS arm64, 6–8 processos simultâneos com `--max-frontier-mib 1536`. Nenhuma
rodada relatada terminou por memória. Dados locais:
`campaigns/cycle-insertion-binary64-20261009/retune`.

**Resultado.**
- **Memória.** A fronteira passou a crescer na mesma proporção que a vazão.
  Em 100pr1002, 60 s chegam a 4,2 M nós e 3,0–3,8 GB. A primeira triagem
  (8 processos de 120 s) fez a máquina de 24 GB usar swap, e rodadas
  terminaram até 46 s depois do prazo. Ela foi descartada. Daí o orçamento de
  memória da fronteira (contrato em
  [`unordered-tpp.md`](../../docs/algorithms/unordered-tpp.md#orçamento-de-memória-da-fronteira-2026-10-09)).
  Na versão que só descartava nós, `lazy` parou aberto em 45ts225 (LB/UB
  0,989 aos 44 s); despindo primeiro, fecha em 67 s. Com fila em `deque`, o
  pico de RSS de 100pr1002 (30 s, sem orçamento) cai de 2,39 para 1,58 GB.
- **Triagem (13 de desenvolvimento, 120 s).** `ln` (`lazy` sem mergulhos):
  +29% do gap da base (6/1). Só sem mergulhos: +15%. Inserções múltiplas:
  −69% sozinhas, −12,5% com `ln` e −17% sem mergulhos; elas também deixam
  as que fecham 3–5× mais lentas (60pr299: 18 → 85 s). Strong branching
  (`branch`) piora também (−4 a −8%, na primeira triagem).
- **Validação (25, 120 s).** `ln`: +39,5% do gap da base (mediana 37%,
  27–70%), melhor em 17/17 das abertas. LB/UB médio 0,935 → 0,958. Fecha as
  mesmas 8, 5–30% mais devagar. Só sem mergulhos: +23% (17/17). `mln`: +11,5%
  (13/4).
- **Sem tour inicial (209 fechadas, 60 s).** A base fecha as 209. `ln`
  também, a 0,79× em média geométrica (n = 27 com a base ≥ 0,1 s). Só sem
  mergulhos fica a 0,20× e perde 6. Sem um incumbente bom, os mergulhos
  continuam necessários.
- **Pipeline completo (38 abertas aos 60 s, 600 s, `--primal-ils 0.5`).**
  Mesmo protocolo de 2026-10-07, 8 processos.

  | | 2026-10-07 (`mln`, binário antigo) | `ln` (`7c29102`) | mesmo binário, `mln` |
  |---|---:|---:|---:|
  | Gap certificado: mediana / média / máx. | 5,7% / 6,2% / 20,5% | **2,7% / 3,9% / 18,5%** | 3,9% / 4,4% / 19,2% |
  | Mediana em ≤ 60 / 61–99 / 100 regiões | 0,9% / 8,7% / 8,0% | 0% / 6,0% / 5,6% | — |
  | Gap fechado (1e-6) | 4 | **17** | 14 |
  | Tour ≤ melhor da Paula + 0,005 | 34 | 33 | 32 |

  Contra 2026-10-07, `ln` reduz o gap nas 34 que estavam abertas (fecha 59%
  do gap restante em média, mediana 49%, mínimo 10%). Os tours são os mesmos,
  salvo o ruído do ILS (um fica 0,08% mais longo). Todos validaram.

**Leitura.** Com o oráculo barato, o que limita o LB é o número de nós
expandidos, e qualquer custo por nó pesa mais. As inserções múltiplas, que
custam `O(m·k)` por nó, deixam de compensar, e a configuração recomendada
para o LB com incumbente do ILS passa a ser `--primal-ils 0.5
--cycle-optimization lazy --lazy-oracles --dive-interval 0`, sem
`--multi-insertion-bound`. Os padrões não mudam: sem tour inicial, os
mergulhos e a avaliação imediata dos filhos continuam melhores.

**Limitações.** Uma repetição por variante, 6–8 processos simultâneos (a
mesma carga para as variantes de uma mesma rodada); a comparação com
2026-10-07 junta binário, máquina e momento diferentes, e o ILS tem ruído. O
`run_lb.py` desta campanha não guardava os contadores do orçamento; nas
rodadas de 600 s, o RSS por processo ficou em 2,3–2,7 GB com 1,5 GiB de
orçamento, antes da fila em `deque`, sem swap.

## tspn-cycle-insertion-binary64-2026-10-09

**Pergunta.** Com o oráculo de ciclo em binary64, a triagem de inserções do
ciclo, ainda racional, passou a dominar o tempo. Ela pode ser provada em
binary64? E os limites de inserção do caminho e de inserções múltiplas, que
subtraíam uma margem fixa, podem virar limites provados sem custo?
**Formulação.** O mesmo TSPN de ciclo livre e o mesmo protocolo de
[`tspn-cycle-float-2026-10-08`](#tspn-cycle-float-2026-10-08): gap relativo
1e-6, 60 s, uma thread, um processo por vez, `cache,features,root,interval`,
tours validados (`validate_cycle`, 1e-7). Variantes intercaladas por caso:
`main` (binário `7cb5c29`, a `main` de antes) × `6ab9504` (triagem binary64,
limites de caminho e de inserções múltiplas provados, cache das regiões
normalizadas, poda das consultas de visita), e no SoCG também
`--no-cycle-binary-insertion` do mesmo binário (só a triagem racional) e
`7326d8d` (provas do oráculo com limite de erro *a priori*). Contratos e
provas em
[`unordered-tpp.md`](../../docs/algorithms/unordered-tpp.md#certificado-convexo-e-interseções),
[`tspn.md`](../../docs/algorithms/tspn.md#cyclic-insertion-bounds) e
[`convex-cycle.md`](../../docs/algorithms/convex-cycle.md#correctness-of-the-bounds).
"Fechou" é gap numérico fechado com limites provados, não ótimo algébrico.
Dados locais: `campaigns/cycle-insertion-binary64-20261009`.

**Resultado.**
- **SoCG (48 casos).** As três variantes fazem a mesma busca (mesmas chamadas
  nos 48), e todas fecham e validam. Tempo total: 26,6 s (`main`), 26,0 s (só
  a triagem racional) e 3,5 s (`6ab9504`). A triagem cai de 22,4 para 0,15 s.
  `main` → `6ab9504`: 5,9× em média geométrica nos casos ≥ 0,1 s (n = 11,
  2,5–16×) e 9,2× nos ≥ 1 s (n = 5). O caso mais longo
  (`random_41-60`, 60 regiões) cai de 9,8 para 1,2 s. Com `7326d8d`, as provas
  intervalares do oráculo caem de 1,08 para 0,37 s e o total, de 3,5 para
  2,7 s (1,29× nos casos ≥ 0,1 s, n = 8). Os limites mudam no último bit, e 6
  dos 48 casos fazem buscas um pouco diferentes (134.300 → 134.174 chamadas).
- **Paula (235 instâncias, `main` × `6ab9504`).** Fecham 209 contra 207:
  56a280 (14,9 s) e 50lin318 (11,8 s) passam a fechar, e nenhuma deixa de
  fechar. As 470 execuções validam. Nas 169 de ciclo: 11,5× nos 23 casos
  fechados pelas duas com a referência ≥ 0,1 s (4,7–18,9×), soma 71,8 → 5,9 s.
  Nas 14 abertas nas duas, o novo faz 4,4× mais chamadas (3,9–5,1×), e o gap
  relativo mediano cai de 0,152 para 0,113 (todas menores; LB 3,8% maior na
  mediana, 3,3–6,4%). As 66 com ponto, resolvidas como caminho, não mudam
  (1,01×), como esperado. Chamadas racionais: 7 de 99,1 M (3 que o polimento
  deixa abertas em 80rd400 e 4 interrompidas pelo prazo).
- **Caminho (40 casos do corpus, `--stride 14`, gap 1e-3).** Com os limites
  provados, as buscas ficam idênticas às da `main` (mesmas chamadas e nós em
  40/40). O tempo total cai de 9,25 para 8,61 s com `7326d8d`, pelo filtro
  estático de orientação; sem ele, sobe 1,5%.
- **Inserções múltiplas (opção).** Em 4.000 nós aleatórios, o limite provado
  coincide com o antigo, com margem, até 1e-9 relativo. Com limite de 20.000
  chamadas em cinco instâncias da Paula, as buscas ficam praticamente iguais,
  a ~15% de tempo a mais.

**Leitura.** A triagem racional era o último gargalo do ciclo. Provada em
binary64, ela custa menos que as consultas de visita, e o oráculo volta a
dominar (1,8 dos 2,7 s no SoCG). Os limites de poda do B&B deixam de depender
de margens: todos são provados (racionais, intervalos ou limite de erro *a
priori*). Nas abertas, o ganho de vazão vira LB, mas não fecha o gap: o
gargalo continua combinatório.

**Limitações.** Uma repetição num Mac pessoal; o SoCG e a Paula rodaram com
um processo por vez, sem outro trabalho pesado. Nos casos de milissegundos,
as razões são ruído. A Paula foi medida com `6ab9504`, sem o ganho do
oráculo de `7326d8d`. Com `dual` (dual herdado racional), a triagem continua
racional, assim como `dual-screen`.

## tspn-cycle-float-2026-10-08

**Pergunta.** O oráculo de ciclo do TSPN pode fechar o gap pedido em binary64,
com limites provados, como o oráculo de caminho já faz? E as regiões que são
pontos ou segmentos? **Formulação.** TSPN de ciclo livre, gap relativo 1e-6
(`UB ≤ (1+1e-6)·LB`, absoluto 0) e, à parte, 1e-3; 60 s por execução, uma
thread por instância, um processo por vez, `cache,features,root,interval`.
Mesmo binário (`7cb5c29`, branch `cycle-float-oracle`; macOS arm64, AppleClang,
C++23, GMP), duas variantes intercaladas por caso (a ordem alterna):
referência `--no-cycle-float --no-float-degenerate` (o fluxo exato de antes) ×
padrão (estágio binary64 do ciclo e regiões degeneradas no oráculo de caminho;
contrato e prova em
[`convex-cycle.md`](../../docs/algorithms/convex-cycle.md#binary64-stage-with-a-requested-gap-2026-10-08)).
Tours validados de forma independente (`validate_cycle`, 1e-7). Conjuntos: a
regressão SoCG de [`tspn-paula-cycle-2026-10-06`](#tspn-paula-cycle-2026-10-06)
(48 casos, 4 por estrato, seed 20260930) e a coleção da Paula inteira (235
instâncias; local, sem autorização de redistribuição). "Fechou" é gap numérico
fechado, com limites provados; não é ótimo algébrico. Dados locais:
`campaigns/cycle-float-oracle-20261008`.

**Resultado.**
- **SoCG, 1e-6.** 48/48 fecharam e validaram nas duas variantes. No padrão,
  nenhuma das 134.300 chamadas ao oráculo usou aritmética racional: 120.734
  fecharam pela prova intervalar de um candidato da construção e 13.566 pelo
  polimento (toda tentativa de polimento fechou, com 28 iterações de Newton em
  média). Na referência, as 137.781 chamadas usaram o certificado exato e
  104.505 delas, recuperação racional. Tempo total 131,1 → 23,5 s; oráculo
  109,9 → 2,8 s (referência: construção 18,8 s, certificação exata 58,5 s,
  recuperação 32,6 s; padrão: construção 1,2 s, provas intervalares 1,0 s,
  polimento 0,6 s). Média geométrica das razões de tempo: 6,7× nos casos
  ≥ 0,1 s (n = 19, mín. 2,3×) e 7,3× nos ≥ 1 s (n = 12). O caso mais longo
  (`random_41-60` de 60 regiões) leva 59,0 s na referência (numa rodada
  anterior, parou nos 60 s) e 8,8 s no padrão.
- **SoCG, 1e-3.** 48/48 nas duas variantes; 0 chamadas racionais em 124.978
  (11.331 pelo polimento, 17 iterações em média); 6,8× nos casos ≥ 0,1 s
  (n = 19, mín. 2,3×); oráculo 99,6 → 2,3 s.
- **Por chamada.** Replay das 11.411 chamadas capturadas dos casos 02, 11 e 19
  (`cycle-replay --float on|off`): nenhum par de intervalos incompatível,
  todos os contatos aprovados pelo verificador exato independente, todas as
  chamadas fechadas em binary64. Mediana por chamada: 12–20 µs contra
  560–910 µs.
- **Paula, 235 instâncias.** Fecharam 207 no padrão e 197 na referência: 10
  só no padrão (53gil262, 60pr299, 84fl417, 42rat99-6x7, 49gil262-7x7,
  49lin318-7x7, 50a280, 50kroB100, 50lin105 e 50pcb442), nenhuma só na
  referência; as 470 execuções validaram. Nos 65 casos fechados pelas duas com
  a referência ≥ 0,1 s: 16,2× (1,5–101×), soma 335,4 → 43,2 s. As 66
  instâncias com ponto são resolvidas como caminho pela âncora, e antes toda
  chamada delas ia ao fluxo exato: 69,5× (n = 32, 41–101×). As 169 de ciclo:
  4,0× (n = 33, 1,5–87×), com a triagem de inserções dominando o tempo que
  sobra. Nas 28 abertas nas duas, o padrão faz de 3 a 130× mais chamadas no mesmo
  tempo e reduz o gap relativo em todas (mediana 0,214 → 0,117; LB 7% maior
  na mediana, 4–18%). Chamadas: 64,9 M no padrão, das quais 6 racionais (3
  interrompidas pelo prazo de 60 s e 3 que o polimento deixou abertas, em
  80rd400); 3,35 M na referência, 3,30 M delas racionais. Tempo médio por
  chamada: 735 → 13 µs.

**Leitura.** O estágio binary64 fecha praticamente toda chamada, inclusive com
as tolerâncias de 2,5e-7 relativo dos refinamentos, e o oráculo deixa de ser
o gargalo. No ciclo, o tempo restante vai para a triagem de inserções, que
ainda é racional: no caso 39 da SoCG, `insertion_lower_bounds` leva ~60% das
amostras e o oráculo, ~13%. Próximas direções em
[`tspn.md`](../../docs/algorithms/tspn.md#next-directions-2026-10-09).

**Limitações.** Uma repetição, num Mac pessoal sem isolamento de carga; nos
casos de milissegundos as razões são ruído. Razões com a referência perto dos
60 s são limites inferiores do ganho. Só a configuração do protocolo
(`cache,features,root,interval`) foi medida; `memo`, `share-bounds` e o
portfólio continuam verificando reúsos com o certificado exato. Com um gap
muito mais apertado que o do B&B, o polimento pode ficar aberto (ver o
contrato), e a chamada usa o fluxo exato.

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
- **Repetição em outra máquina, com o orçamento da Paula** (armijo: AMD Ryzen 7
  9700X, Ubuntu 24.04, GCC, commit `73fd995`; 8 processos fixados nos 8
  núcleos físicos; 1.200 s por execução, o limite de CPU de cada execução
  dela; uma repetição). O GMP não foi encontrado, então a aritmética exata usou
  o `cpp_rational` do Boost: os casos fechados levaram em mediana 1,44× o
  tempo do Mac. Os tempos absolutos desta máquina não valem para comparação,
  mas as configurações rodaram juntas e nas mesmas condições.

  | | Antes | Agora |
  |---|---:|---:|
  | Pipeline (`--primal-ils 0.5`): gap mediano / máximo | 10,4% / 31,1% | **5,3% / 20,0%** |
  | Mediana em ≤ 60 / 61–99 / 100 polígonos | 3,3% / 17,1% / 13,2% | 0,1% / 7,8% / 7,6% |
  | Pipeline: gap fechado (1e-6) | 5 | **7** (+50kroB100, 50lin318) |
  | Só LB, com o melhor tour fornecido: gap mediano / máximo | 9,0% / 29,7% | **4,5% / 19,1%** |
  | Só LB: gap fechado | 8 | **10** (+84fl417, 50kroB100) |

  Nenhuma instância piora. O tour do pipeline é igual nas duas configurações:
  em 35 das 38 iguala o melhor da Paula, e 100C10k.0, 100i1000-410 e
  100i2000-604 ficam 0,07–0,27% acima. Nas fechadas (60 s, sem tour), as duas
  fecham 196 (45tsp225 não fecha em nenhuma), os ótimos coincidem e `mln` é
  1,21× mais rápido em média geométrica. Somando as rodadas, 10 das 38
  abertas ficam com gap fechado, todas no tour da Paula: 42rat99-6x7,
  49gil262-7x7, 49lin318-7x7, 50kroB100, 50lin105, 50lin318, 50pcb442,
  53gil262, 56a280 e 84fl417. Com isso, em 08/10, eram 207 das 235 com
  certificado (depois, 229: ver
  [`tspn-paula-long-armijo-2026-10-14`](#tspn-paula-long-armijo-2026-10-14)).
- **Limitações.** Duas máquinas, com 8 a 10 processos simultâneos e uma
  repetição nas rodadas longas. A triagem
  orientou a escolha. O limite novo usa binary64 com margem, não aritmética
  racional (ver o contrato). O LB continua sendo o gargalo: 34 das 38 seguem
  abertas aos 600 s, com até 20% de gap nas instâncias de 61–100 regiões.

## verified-fallback-2026-10-08

**Mudança.** O último recurso do oráculo de caminho (solver racional completo)
devolvia como LB e UB o comprimento do próprio caminho, sem certificado. Agora
os contatos exatos passam pelo certificado KKT exato de ciclos; se ele falha, o
oráculo devolve UB = comprimento e LB = dual das direções do caminho (contador
`oracle_unverified_fallbacks`). **Verificação.** (1) Suíte direcional com 3.000
caixas inteiras e 200 polígonos convexos aleatórios, em que o modo de
diagnóstico `shadow_rational` executa o último recurso em toda chamada:
484.502 checagens, 0 falhas, 0 verificações falhas. (2) Mesmo Mac, binário
anterior (`739aacb`) × novo, 24 casos de `fekete-comparison/instances.bin`
(0,05–1,5 s na dantzig, todos com chamadas racionais na variante
`--no-float-recovery`), gap `UB ≤ 1,001·LB`, nas duas variantes: 48 pares com
nós, chamadas, limites e todos os contadores idênticos, 96/96 execuções
fechadas e validadas; 11.477 chamadas racionais na variante sem polimento,
nenhuma no último recurso. **Leitura.** Sem efeito de desempenho, porque o
último recurso não foi acionado em nenhuma medição (0 nas 1.116 execuções de
`float-recovery-dantzig-2026-10-08`); o ganho é de correção: nenhum limite
devolvido depende da construção dos mapas. **Limitações.** Uma repetição e
poucos casos, suficientes para mostrar identidade, não para medir tempo. Dados
em `experiments/verified-fallback-check` do workspace local.

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

**Comparação com o Fekete.** Os tempos da variante padrão, com os do Fekete de
[`free-order-dantzig-2026-10-06`](#free-order-dantzig-2026-10-06) (campanha de
02/10, não lado a lado), estão em
[`float-recovery-dantzig-2026-10-08/per-case.csv`](float-recovery-dantzig-2026-10-08/per-case.csv),
com as mesmas colunas da tabela de 06/10 (casos numerados a partir de 1; acima,
`caso 129` usa a numeração a partir de 0 e é o caso 130). Nos 553 casos fechados
pelos dois, a razão Fekete/nosso tem média aritmética **122,6×**, mediana
**57,6×** e média geométrica 65,4× (06/10: 111,6×, 41,3× e 49,8×); mínimo 8,7×
(caso 452, antes 3,4×) e máximo 2.302× (caso 174). Por fonte (média
geométrica): OSM 81,7×, aleatórias 66,8×, tessellation 25,6×. Por número de
polígonos: 4–10 29,1×; 11–20 49,9×; 21–40 77,1×; 41–60 149,8×. Nos três casos
que o Fekete fechou depois (131, 420, 558) o nosso levou 22,6 s, 44,9 s e
46,8 s. Nos 5 abertos do Fekete, o nosso levou de 77 s a 978 s (caso 130, com
carga maior que na variante de referência, que levou 707 s com a mesma busca).
Os tempos absolutos dessa rodada têm a carga da máquina compartilhada e tendem a
subestimar a razão.

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
