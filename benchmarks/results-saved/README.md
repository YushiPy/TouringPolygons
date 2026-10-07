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
| Casos com gap fechado | **558/558** | **550/558** |
| Soma dos tempos nos 550 casos fechados por ambos | 345 s | 172.667 s (maior: caso 231, 49.896 s ≈ 13,9 h) |
| Soma dos tempos nos 558 casos | 1.899 s (maior: caso 130, 722 s) | — |

Nos 550 casos fechados por ambos, o nosso solver foi mais rápido em **550/550**
(menor razão 3,4×, caso 452). Razão Fekete/nosso: média geométrica **49,0×**,
mediana 40,8×. Por fonte: OSM (313) 71,2×; aleatórias (159) 39,2×; tessellation
(78) 17,3×. Por número de polígonos: 5–10 (128) 20,9×; 11–20 (136) 38,2×; 21–40
(149) 63,0×; 41–60 (137) 106,0×. Todas as 558 trajetórias nossas validaram.
Nosso solver no `470a0c9` × a campanha anterior dele na mesma pasta: 26.116 s →
1.899 s no total (5,6× geométrico).

**Casos em aberto no Fekete (8):** 65, 66, 130, 131, 420, 493, 542, 558
(numeração a partir de 1). Foram interrompidos manualmente depois de dias sem
fechar o gap (o tempo por caso não foi gravado nessa versão). Os limites
parciais do Fekete são compatíveis com o ótimo do nosso solver (LB do Fekete ≤
nosso UB e UB do Fekete ≥ nosso LB). Na antiga comparação (`fekete-comparison`,
teto de 6 h) os abertos eram 65, 66, 130, 131, 231, 493, 542 e 558: o 231 fechou
aqui em 13,9 h e o 420 ficou aberto. O caso 131 chegou a fechar em 42.500 s em
2026-10-06, mas com `FEASIBILITY_TOLERANCE` 1e-8 e `SPANNING_TOLERANCE` 0,0009
(nunca ajustada junto, apesar de a biblioteca recomendá-la logo abaixo da
primeira); não é a configuração original e ficou fora da comparação.

**Limitações — leia antes de citar a razão.**
- A campanha do Fekete é de 2 de outubro, com 8 casos em paralelo na máquina; o
  nosso solver rodou 1 caso por vez (2 nos casos 65 e 66). Com uma thread por
  caso em 24 núcleos o efeito esperado é pequeno, mas não foi medido.
- Os caminhos do Fekete dessa campanha não têm validação independente
  registrada (`valid` ausente), só `endpoint_valid`; com a tolerância de
  cobertura de 1e-3 do Fekete, a validação a 1e-7 pode não passar em todos.
- Os 8 casos abertos do Fekete são censurados: as razões os excluem e, portanto,
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
  com pontos ou de 100 regiões), onde o Fekete também não fecha; os resultados
  do ILS da Paula não estão no rascunho (nem no PDF) e não foram comparados.
- TPP com depósito no centro da bbox (`convert-paula`, mesmo binário, 60 s,
  gap 1e-6): 196 fechadas contra 197 do TSPN, 2× mais rápido onde ambos fecham,
  mesma fronteira (0 de 22 acima de 60 polígonos).
- Qualidade do UB nas abertas, sem certificado (2026-10-07, ε = 0,001). Como
  os resultados do ILS da Paula não estão disponíveis, o tour do GTSP serve de
  substituto. O B&B quase não melhora o UB entre 60 e 600 s. Com 600 s (parcial,
  21 de 38 instâncias), as 4 de 45–60 polígonos ficam em 0,993–0,998 do tour
  do GTSP; as 17 de 64–100 ficam em 0,990–1,039, 12 acima de 1; a LNS por
  janelas tira até 2,5 pontos percentuais nas de 64–99 e nada nas de 100. Gap do B&B aos 600 s:
  2–35%. A **busca local iterada** opcional (`--primal-ils 0.5`, 60 s,
  5 instâncias) levou 80rd400 de 1,035 a 0,998, 100pr1002 de 1,010 a 0,996 e
  64lin318 a 0,999; manteve 50kroA100 (0,999) e 100i1000-410 (1,006, polígonos
  muito sobrepostos). No 56a280, o B&B sozinho em 60 s achou 0,993 ou 1,006
  conforme a carga da máquina; com ILS, 0,993 (UB/LB 1,021 contra o LB de
  600 s). Opção desligada por padrão; a avaliação de 600 s nas 38 abertas ainda
  está em andamento.

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
