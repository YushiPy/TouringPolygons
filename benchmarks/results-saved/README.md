# Benchmarks preservados

Esta pasta guarda resumos compactos dos resultados, não cópias de campanhas.
Execuções completas, entradas repetidas, trajetórias, logs, builds e patches de
experimento ficam nos diretórios locais ignorados (`benchmarks/workspace/`).

Pastas com dados:

| Pasta | Conteúdo |
|---|---|
| [cetsp-table2-mac-2026-10-27](cetsp-table2-mac-2026-10-27/per-instance.csv) | As 62 instâncias 2D da Tabela 2 de Coutinho et al. (2016): nossos limites com 1 e 8 threads (30 min), os deles (4 h), o UB de Mennell e o tour de Carrabs et al. (2020), uma linha por instância; e, nas 30 abertas para os dois, o tempo até superar estritamente os limites de 4 h deles ([`time-to-beat.csv`](cetsp-table2-mac-2026-10-27/time-to-beat.csv)); resumo [abaixo](#cetsp-table2-mac-2026-10-27). |
| [fekete-smoothed-dantzig-2026-10-22](fekete-smoothed-dantzig-2026-10-22/per-run.csv) | As instâncias suavizadas na dantzig, no protocolo da referência (C ×3; B e A com 300 s e extensão de 900 s), uma linha por instância, configuração, raiz e limite; resumo [abaixo](#fekete-smoothed-dantzig-2026-10-22). |
| [fekete-smoothed-mac-2026-10-21](fekete-smoothed-mac-2026-10-21/per-run.csv) | Fekete (1 thread, CHR e LEFP) × nosso nas 24 instâncias suavizadas, 300 s, uma linha por execução com limites, gap e validação do tour; resumo [abaixo](#fekete-smoothed-mac-2026-10-21). |
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

## cetsp-table2-mac-2026-10-27

**Pergunta.** Nas 62 instâncias 2D de Mennell da Tabela 2 de Coutinho et al.
(2016), fechamos todas as que eles fecham? E o que acontece nas outras?

**Formulação.**
- **Instâncias.** Os arquivos de Mennell como estão no repositório de Cerrone
  (github.com/CerroneCarmine/CETSP), e as 21 TSPLIB em 2D reconstruídas dos
  arquivos 3D de Mennell; todas em
  [`benchmarks/suites/cetsp/`](../suites/cetsp/README.md). As 27 que também
  vieram de Walton Coutinho são idênticas.
- **Binários.** Para as 42 que não são geométricas, `tpp-convex-cetsp` do
  commit 7e5a169. O oráculo e o B&B são os de 541d862; a heurística ganhou o
  filtro do Or-opt e "ruína e reconstrução" em sequências quando o tour vira
  em poucos discos. As 20 geométricas vêm da
  [segunda campanha](#cetsp-campaign2-mac-2026-10-26) (541d862) e não rodaram
  de novo.
- **Protocolo.** O da segunda campanha, no Mac M4 Pro só com a campanha:
  - 1 thread, 1.800 s, 6 ao mesmo tempo, teto de 2,5 GB;
  - as que ficaram abertas, com 8 threads, uma por vez, 1.800 s, teto de
    12 GB.

  A campanha foi interrompida uma vez no passo 2; a instância em curso
  (team5_499) rodou de novo desde o começo.
- **Gap e verificação.** Gap relativo de 1e-6, com limites provados na
  instância em binary64. Todos os tours foram verificados em racionais exatos
  e cobrem também os discos decimais.
- **Referências.**
  - Coutinho et al.: 1 thread, i7 de 3,4 GHz (2012), 4 h. O UB da tabela
    deles é o de Mennell (2009).
  - Os tours de Carrabs et al. (2020), do mesmo repositório, servem de segunda
    referência de UB.
- **Dados.** Uma linha por instância em
  [`per-instance.csv`](cetsp-table2-mac-2026-10-27/per-instance.csv).

**Resultado.**
- **Fechadas: 32 de 62, contra 23 deles.**
  - **As 23 deles.** Os mesmos ótimos (diferença de até 0,001), todos mais
    rápidos com 1 thread, de 17× a 18.000× (mediana de 100×). Exemplos:
    - lin318-0.10: 57,5 s contra 8.541 s;
    - team1_100rdmRad: 24 ms contra 269 s;
    - team3_300rdmRad: 37 ms contra 682 s.
  - **9 abertas para eles:**

| Instância | Ótimo (1e-6) | Tempo | Threads | LB deles, 4 h | Acima do ótimo: Mennell | Carrabs |
|---|---|---|---|---|---|---|
| team5_499rdmRad | 446,191 | 0,5 s | 1 | 442,637 | +1,82% | +0,07% |
| concentricCircles2 | 153,132 | 72 s | 1 | 149,868 | 0,00% | +0,67% |
| kroD100rdmRad | 141,829 | 98 s | 1 | 136,620 | 0,00% | +0,36% |
| d493rdmRad | 134,226 | 116 s | 1 | 125,312 | +4,39% | +0,38% |
| dsj1000-0.10 | 373,759 | 126 s | 1 | 373,730 | +0,63% | +0,14% |
| chaoSingleDep | 1.039,611 | 271 s | 1 | 1.000,151 | 0,00% | 0,00% |
| team3_300 | 461,889 | 376 s | 1 | 447,534 | +0,94% | +3,01% |
| lin318rdmRad | 2.047,107 | 501 s | 1 | 1.807,681 | +1,63% | +0,42% |
| rotatingDiamonds3 | 380,883 | 718 s | 8 | 348,608 | 0,00% | 0,00% |

  Os 0,00% são diferenças abaixo da terceira casa, em que a tabela de
  Coutinho et al. arredonda.
- **LB nas 30 abertas para os dois.** O nosso (o melhor entre 1 e 8 threads)
  é maior que o de 4 h deles em todas:
  - de +2,8% (pcb442-0.10) a +20,0% (team2_200rdmRad), com mediana de +13,6%;
  - o gap também é menor em todas.
- **Tours** (o melhor entre 1 e 8 threads), nas 39 abertas para eles:
  - **contra o de Mennell:** melhor em 34 e igual em 1. Pior em 4:
    - rotatingDiamonds3 e concentricCircles2, por menos de 0,001 (são ótimas;
      a diferença é o arredondamento da tabela);
    - rotatingDiamonds4 e 5, por 0,04% e 0,16% (heurística anterior);
  - **contra o de Carrabs et al.:** melhor em 35 e igual em 1. Pior em 3:
    - bonus1000, por 0,53%;
    - dsj1000rdmRad, por 0,87%;
    - team2_200rdmRad, por menos de 0,01%.
- **Memória.**
  - Picos de 1,6 GB com 1 thread e de 8,5 GB com 8 threads.
  - O descarte entrou em todas as abertas, com pisos de 0,03% a 2,0% acima do
    LB final.
  - kroD100-0.02, com 8 threads, esvaziou a fila aos 1.714 s e terminou no
    piso, com gap de 0,26%.

**Tempo até superar os limites de 4 h deles (as 30 abertas para os dois).**
- **Protocolo.**
  - Binário do commit 841c144 (`--target-lower`, `--target-gap`), o mesmo B&B
    e a mesma heurística de 7e5a169. Configuração da campanha: orçamento de
    1.800 s (heurística de até 90 s), 1 thread, teto de 2,5 GB. Duas
    instâncias ao mesmo tempo, com `nice`, no Mac em uso leve.
  - Cada execução para quando o LB provado é **estritamente maior** que o
    deles e o gap estritamente menor. A tabela deles arredonda o LB em 3
    casas e o gap em 3 casas do percentual; os alvos somam meia unidade da
    última casa (LB + 0,0005; gap − 0,0005 ponto percentual).
  - O tempo vem da trajetória dos limites, gravada no momento em que cada alvo
    é alcançado. É o tempo total, com a leitura, o B&B preliminar de 1 s (só
    com o tour dos centros) e a heurística, durante a qual o LB não sobe.
  - O gap deles usa o tour de Mennell; o nosso, o nosso tour.
- **Resultado.** Os dois limites deles foram superados nas 30, e uma linha
  por instância está em
  [`time-to-beat.csv`](cetsp-table2-mac-2026-10-27/time-to-beat.csv).
  - **LB:** de 0,12 s a 111 s, com mediana de 50 s. Contra os 14.400 s
    deles, de 129× a 118.000×, com mediana de 324×.
  - **Gap:** razão de 151× a 9.973×, com mediana de 350×.
  - **Onde o tempo vai.** Em 6, todas de raio aleatório, o B&B preliminar
    supera o LB deles em até 0,86 s; dsj1000rdmRad, em 0,12 s. Nas outras
    24, o LB deles é superado pelo B&B depois da heurística, em mediana de
    3,0 s (máximo de 34 s, pcb442-0.10); o resto é a heurística, que usa os
    90 s inteiros nas maiores.

**Limitações.**
- Máquinas de épocas diferentes (Mac de 2024 contra i7 de 2012), 30 min
  contra 4 h. Os fatores de tempo não descontam o hardware.
- "Fechada" é gap relativo de até 1e-6, não ótimo algébrico.
- Carrabs et al. (2020) também publicam limites inferiores, ainda não
  comparados. Até lá, "LB maior" vale contra Coutinho et al., não contra o
  melhor conhecido.
- As 20 geométricas usam a heurística anterior; os UBs delas podem melhorar.
- As instâncias 3D (Tabela 3) não rodaram.
- No tempo até superar os limites, eles publicam só os limites ao fim de 4 h.
  A razão supõe que precisaram das 4 h inteiras; se o LB deles parou de subir
  antes, ela exagera. A frase segura é "superamos em X s o LB que eles têm ao
  fim de 4 h". Essa medição rodou com outro processo ao lado.

## cetsp-campaign2-mac-2026-10-26

**Pergunta.** Com a memória da busca reorganizada (famílias de filhos, teto
respeitado) e o B&B rodando antes da heurística, o que muda em relação à
[primeira campanha](#cetsp-campaign-mac-2026-10-25)?

**Formulação.**
- **Binário.** `tpp-convex-cetsp` do commit 541d862, com o mesmo oráculo e o
  mesmo B&B preguiçoso, mais:
  - filhos guardados por família, a 6 bytes cada, e a fila guardando famílias;
  - teto de memória respeitado, com compactação e, se preciso, descarte dos
    filhos de limite mais alto; o limiar do descarte vira um piso para o
    limite inferior final;
  - um segundo de B&B com o tour dos centros antes da heurística.

  Detalhes em
  [convex-cycle.md](../../docs/algorithms/convex-cycle.md#disks-2026-10-25).
- **Protocolo.** O mesmo da primeira campanha: script único, Mac M4 Pro
  reiniciado e só a campanha rodando.
  - Mennell com 1 thread: 1.800 s, 6 ao mesmo tempo, teto de 2,5 GB.
  - As abertas com 8 threads: uma por vez, 1.800 s, teto de 12 GB.
  - Behdani e Smith: 60 s cada, 1 thread.
  - A campanha foi interrompida uma vez no passo 1 e retomada desde o começo
    dele; nenhum resultado parcial entrou.
- **Gap e verificação.**
  - Gap relativo de 1e-6, com limites provados na instância em binary64.
  - Os 752 tours foram verificados em racionais exatos.
- **Referência.** Coutinho et al. (2016): 1 thread, i7 de 3,4 GHz (2012), 4 h.
  O B&B deles informa o ótimo ou o seu LB; o UB das tabelas é o de Mennell
  (2009), o melhor conhecido, contra o qual eles medem o gap.

**Resultado em Mennell (1 thread).**

| Instância | Nosso [LB; UB] | Tempo | Coutinho et al. [LB; UB], 4 h | Tempo |
|---|---|---|---|---|
| bubbles1 | **349,135** | 3,7 ms | 349,135 | 0,10 s |
| bubbles2 | **428,279** | 8,9 ms | 428,279 | 0,22 s |
| bubbles3 | **529,955** | 7,9 s | 529,955 | 193,1 s |
| bubbles4 | [729,869; 804,989] | 30 min | [690,578; 829,888] | 4 h |
| bubbles5 | [918,237; 1.038,299] | 30 min | [851,822; 1.062,335] | 4 h |
| bubbles6 | [1.084,003; 1.291,209] | 30 min | [993,981; 1.383,139] | 4 h |
| bubbles7 | [1.227,591; 1.653,713] | 30 min | [1.123,522; 1.720,214] | 4 h |
| bubbles8 | [1.373,777; 1.991,033] | 30 min | [1.252,715; 2.101,373] | 4 h |
| bubbles9 | [1.518,665; 2.329,164] | 30 min | [1.374,407; 2.426,274] | 4 h |
| chaoSingleDep | **1.039,611** | 270,8 s | [1.000,151; 1.039,610] | 4 h |
| concentricCircles1 | **53,158** | 19 ms | 53,158 | 5,2 s |
| concentricCircles2 | **153,132** | 71,9 s | [149,868; 153,132] | 4 h |
| concentricCircles3 | [262,250; 270,615] | 30 min | [247,624; 271,076] | 4 h |
| concentricCircles4 | [402,488; 453,696] | 30 min | [358,887; 454,457] | 4 h |
| concentricCircles5 | [513,899; 646,866] | 30 min | [459,411; 645,381] | 4 h |
| rotatingDiamonds1 | **32,389** | 2,6 ms | 32,389 | 0,09 s |
| rotatingDiamonds2 | **140,477** | 0,84 s | 140,477 | 730,4 s |
| rotatingDiamonds3 | [378,867; 380,883] | 30 min | [348,608; 380,882] | 4 h |
| rotatingDiamonds4 | [663,718; 772,472] | 30 min | [593,350; 770,660] | 4 h |
| rotatingDiamonds5 | [1.244,150; 1.514,624] | 30 min | [1.106,577; 1.510,752] | 4 h |

**As 12 abertas, com 8 threads.**

| Instância | Nosso [LB; UB] | Gap | Tempo |
|---|---|---|---|
| bubbles4 | [739,533; 804,084] | 8,0% | 30 min |
| bubbles5 | [931,039; 1.035,311] | 10,1% | 30 min |
| bubbles6 | [1.103,101; 1.234,811] | 10,7% | 30 min |
| bubbles7 | [1.250,979; 1.636,317] | 23,6% | 30 min |
| bubbles8 | [1.399,922; 1.925,830] | 27,3% | 30 min |
| bubbles9 | [1.550,456; 2.301,693] | 32,6% | 30 min |
| concentricCircles3 | [265,518; 270,223] | 1,7% | 30 min |
| concentricCircles4 | [408,090; 452,345] | 9,8% | 30 min |
| concentricCircles5 | [524,582; 633,486] | 17,2% | 30 min |
| rotatingDiamonds3 | **380,883** | 1e-6 | 718 s |
| rotatingDiamonds4 | [678,494; 770,975] | 12,0% | 30 min |
| rotatingDiamonds5 | [1.272,815; 1.513,146] | 15,9% | 30 min |

- **Fechadas: as mesmas 9 de 20** da primeira campanha, com concentricCircles2,
  chaoSingleDep e rotatingDiamonds3 abertas para Coutinho et al.
- **Velocidade.** Agora somos mais rápidos em todas as que os dois fecham:
  - rotatingDiamonds2: 0,84 s contra 730 s (870×);
  - concentricCircles1: 19 ms contra 5,2 s (270×);
  - bubbles3: 7,9 s contra 193 s (24×);
  - bubbles1, bubbles2 e rotatingDiamonds1: de 25× a 35×.

  A causa é o segundo de B&B antes da heurística, que fecha as pequenas
  sozinho.
- **Todas as abertas rodaram até o prazo.** Na primeira campanha, a memória
  encerrava a maioria entre 3 e 13 min.
- **LB.** Nas 11 que seguem abertas, o nosso é maior que o de 4 h deles:
  - de 5,7% (bubbles4) a 12,4% (rotatingDiamonds5) com 1 thread;
  - de 7,1% a 15,0% com 8 threads.

  O LB também subiu em relação à primeira campanha em todas, menos
  concentricCircles3 e rotatingDiamonds3, que já iam até o prazo. Por
  exemplo, rotatingDiamonds5 com 8 threads foi de 1.231,9 para 1.272,8.
- **UB (com 8 threads)**, contra o de Mennell que consta na tabela deles:
  - Melhor em 9: bubbles4 a 9 (de −2,5% a −10,7%) e concentricCircles3 a 5
    (−0,3%, −0,5% e −1,8%).
  - Igual em rotatingDiamonds3.
  - Pior em rotatingDiamonds4 e 5: +0,04% e +0,16%.
- **O descarte por memória esteve ativo em todas as abertas.** Os pisos ficaram
  de 0,08% a 0,75% acima do LB final: dentro do teto, mais tempo levaria o LB
  até o piso e não além.

**Resultado em Behdani e Smith (720 instâncias).**
- **Fechadas e conferidas.** As 720 fecham, os 720 ótimos coincidem com os do
  apêndice de Coutinho et al. até a terceira casa, e os 720 tours passam na
  verificação exata.
- **Tempo.** Todas fecham no segundo inicial de B&B, sem a heurística: média
  de 3,4 ms, máximo de 13,9 ms.
- **Contra eles:** 0,6 a 6,5 ms por grupo, contra 17 ms a 1,66 s, de 19× a
  255× mais rápido em cada grupo. As árvores são de 2 a 9 vezes menores.

| r | Grupo | Nós (média) | Tempo médio | Coutinho et al.: nós | tempo |
|---|---|---|---|---|---|
| 0,25 | CETSP-06 | 7 | 0,6 ms | 26 | 29 ms |
| 0,25 | CETSP-12 | 19 | 2,9 ms | 139 | 189 ms |
| 0,25 | CETSP-20 | 110 | 6,5 ms | 1.016 | 1.656 ms |
| 0,5 | CETSP-20 | 54 | 6,3 ms | 437 | 719 ms |
| 1,0 | CETSP-06 | 6 | 0,7 ms | 13 | 17 ms |
| 1,0 | CETSP-20 | 28 | 6,0 ms | 163 | 261 ms |

(Grupos selecionados; os 24 estão em `summary-behdani.md` da campanha.)

**Limitações.**
- Uma execução por instância. O i7 de Coutinho et al. é de 2012; os tempos são
  indicativos.
- Exatidão: gap de 1e-6 com limites provados, não ótimo algébrico. Os limites
  valem na instância em binary64, que difere da decimal em ~1e-14 por
  coordenada.
- **Memória.** Ela continua sendo o limite das grandes, agora sob forma de
  piso. A fatia da busca é 40% do teto, por segurança na compactação, e pode
  subir.
- Dados: local, em
  `benchmarks/workspace/experiments/cetsp-native/campaign-20261025-2308/`.

## cetsp-campaign-mac-2026-10-25

**Substituído** por [cetsp-campaign2](#cetsp-campaign2-mac-2026-10-26) (mesmas 9
fechadas; limites, tempos e memória melhores).

**Pergunta.** Com o B&B preguiçoso (limites de inserção certificados), as
threads e a ILS nova, quanto do CETSP de Coutinho et al. (2016) se resolve
num Mac: as 20 instâncias 2D geométricas de Mennell e as 720 de Behdani e
Smith?

**Formulação.**
- **Binário.** `tpp-convex-cetsp` do commit f31c123, com o oráculo
  geométrico de discos, filhos preguiçosos e a ILS de ordem completa.
  Detalhes em
  [convex-cycle.md](../../docs/algorithms/convex-cycle.md#disks-2026-10-25).
- **Gap e certificação.**
  - Gap relativo de 1e-6, com limites provados na instância lida em binary64.
  - Os 752 tours foram verificados à parte em racionais exatos: cobertura e
    comprimento com raízes arredondadas para cima.
  - Os 32 tours de Mennell cobrem também os discos decimais.
- **Execução.**
  - Script único (`run-all.sh`), com o Mac reiniciado e só a campanha
    rodando: Mac M4 Pro, 12 núcleos, 24 GB.
  - Teto de memória por processo; ao atingi-lo, o B&B para com o limite que
    tem.
  - **Mennell, 1 thread:** 1.800 s, 6 ao mesmo tempo, teto de 2,5 GB.
  - **As abertas, 8 threads:** uma por vez, 1.800 s, teto de 12 GB.
  - **Behdani e Smith:** os 240 conjuntos de pontos com raios 0,25, 0,5 e
    1,0, 60 s cada, 1 thread, 6 ao mesmo tempo.
- **Referência.**
  - Coutinho et al. (2016): CPLEX 12.4, 1 thread, i7 de 3,4 GHz (2012), 4 h
    por instância.
  - Tabela 2 para Mennell; apêndice (Tabelas 4 a 9) para os 720 ótimos de
    Behdani e Smith.

**Resultado em Mennell (1 thread).**

| Instância | n | Nosso [LB; UB] | Tempo (parada) | Coutinho et al. [LB; UB], 4 h | Tempo |
|---|---|---|---|---|---|
| bubbles1 | 36 | **349,135** | 2,3 s | 349,135 | 0,1 s |
| bubbles2 | 76 | **428,279** | 3,0 s | 428,279 | 0,2 s |
| bubbles3 | 126 | **529,955** | 6,6 s | 529,955 | 193,1 s |
| bubbles4 | 184 | [724,169; 804,989] | 759 s (memória) | [690,578; 829,888] | 4 h |
| bubbles5 | 250 | [907,188; 1.038,299] | 554 s (memória) | [851,822; 1.062,335] | 4 h |
| bubbles6 | 324 | [1.054,857; 1.290,783] | 250 s (memória) | [993,981; 1.383,139] | 4 h |
| bubbles7 | 406 | [1.191,883; 1.647,400] | 228 s (memória) | [1.123,522; 1.720,214] | 4 h |
| bubbles8 | 496 | [1.335,053; 1.987,761] | 245 s (memória) | [1.252,715; 2.101,373] | 4 h |
| bubbles9 | 594 | [1.472,616; 2.313,268] | 248 s (memória) | [1.374,407; 2.426,274] | 4 h |
| chaoSingleDep | 200 | **1.039,611** | 290,5 s | [1.000,151; 1.039,610] | 4 h |
| concentricCircles1 | 16 | **53,158** | 0,7 s | 53,158 | 5,2 s |
| concentricCircles2 | 36 | **153,132** | 69,2 s | [149,868; 153,132] | 4 h |
| concentricCircles3 | 60 | [262,337; 270,615] | 1.749 s (memória) | [247,624; 271,076] | 4 h |
| concentricCircles4 | 104 | [392,007; 453,696] | 251 s (memória) | [358,887; 454,457] | 4 h |
| concentricCircles5 | 148 | [494,493; 646,866] | 194 s (memória) | [459,411; 645,381] | 4 h |
| rotatingDiamonds1 | 20 | **32,389** | 0,6 s | 32,389 | 0,1 s |
| rotatingDiamonds2 | 60 | **140,477** | 6,6 s | 140,477 | 730,4 s |
| rotatingDiamonds3 | 180 | [378,951; 380,883] | 1.800 s (tempo) | [348,608; 380,882] | 4 h |
| rotatingDiamonds4 | 320 | [640,749; 772,472] | 232 s (memória) | [593,350; 770,660] | 4 h |
| rotatingDiamonds5 | 680 | [1.201,664; 1.514,288] | 244 s (memória) | [1.106,577; 1.510,752] | 4 h |

**As 12 abertas, com 8 threads.**

| Instância | Nosso [LB; UB] | Gap | Tempo (parada) |
|---|---|---|---|
| bubbles4 | [733,569; 804,084] | 8,8% | 681 s (memória) |
| bubbles5 | [920,265; 1.035,311] | 11,1% | 498 s (memória) |
| bubbles6 | [1.090,123; 1.234,811] | 11,7% | 630 s (memória) |
| bubbles7 | [1.222,084; 1.636,317] | 25,3% | 305 s (memória) |
| bubbles8 | [1.379,449; 1.925,830] | 28,4% | 624 s (memória) |
| bubbles9 | [1.505,233; 2.301,693] | 34,6% | 241 s (memória) |
| concentricCircles3 | [265,556; 270,223] | 1,7% | 1.800 s (tempo) |
| concentricCircles4 | [402,518; 452,345] | 11,0% | 447 s (memória) |
| concentricCircles5 | [509,638; 633,486] | 19,6% | 283 s (memória) |
| rotatingDiamonds3 | **380,883** | 1e-6 | 713 s |
| rotatingDiamonds4 | [654,751; 770,975] | 15,1% | 211 s (memória) |
| rotatingDiamonds5 | [1.231,918; 1.513,125] | 18,6% | 242 s (memória) |

- **Fechadas: 9 de 20.**
  - Com 1 thread: as 6 que Coutinho et al. resolvem, concentricCircles2 (69
    s) e chaoSingleDep (291 s).
  - Com 8 threads: rotatingDiamonds3 (713 s).
  - As três estavam abertas para eles em 4 h. Os UBs deles eram ótimos até a
    terceira casa: 153,132, 1.039,610 e 380,882 (o nosso, 380,8823 a
    380,8827).
- **Velocidade nas que os dois fecham:** rotatingDiamonds2, 6,6 s contra
  730 s (110×); bubbles3, 6,6 s contra 193 s (29×); concentricCircles1, 0,7
  s contra 5,2 s.
  - Em bubbles1, bubbles2 e rotatingDiamonds1 os deles são menores (0,1–0,2
    s): o nosso tempo é quase todo da heurística.
- **LB nas 11 que seguem abertas:** com 1 thread, em 3 a 29 min, de 4,9%
  (bubbles4) a 9,2% (concentricCircles4) acima do de 4 h deles; com 8
  threads, de 6,2% a 12,2%.
- **UB:**
  - Melhor que o deles em 9: bubbles4 a 9 (de 2,3% a 6,7% com 1 thread; até
    10,7% em bubbles6 com 8 threads), concentricCircles3 e 4, e
    concentricCircles5 com 8 threads (−1,8%).
  - Pior em rotatingDiamonds4 e 5: +0,04% e +0,16% com 8 threads.
- **A memória, não o tempo, encerrou 11 das 12 abertas com 1 thread** (de
  194 a 1.749 s). Ao parar, a árvore preguiçosa tem de 67 a 134 milhões de
  nós, a ~16 bytes cada. Com 8 threads, os 12 GB acabam em 3,5 a 11 min.

**Resultado em Behdani e Smith (720 instâncias).**
- **Fechadas e conferidas.** As 720 fecham no gap de 1e-6, os 720 ótimos
  coincidem com os do apêndice de Coutinho et al. até a terceira casa, e os
  720 tours passam na verificação exata.
- **Árvores** de 2 a 9 vezes menores que as deles. No grupo mais difícil
  (CETSP-20, r = 0,25): 110 nós de média, contra 1.016.
- **Tempo.**
  - O B&B leva em média 0,3 ms por instância (no máximo 1,5 ms de média num
    grupo), contra 17 ms a 1,66 s deles.
  - O tempo total é dominado pela ILS inicial: média de 0,52 s, máximo de
    2,2 s. Ela roda pelo menos 200 iterações sem melhora antes de parar, o
    que é excessivo aqui.
  - Por isso o total é maior que o deles em 18 dos 24 grupos (por exemplo,
    CETSP-20 com r = 1,0: 1,06 s contra 0,26 s). É menor em 6: CETSP-06,
    que não roda a ILS, e CETSP-16 a 20 com r = 0,25 (CETSP-20: 0,85 s
    contra 1,66 s).

| r | Grupo | Fechadas | Nós (média) | B&B (média) | Total (média) | Coutinho et al.: nós | tempo |
|---|---|---|---|---|---|---|---|
| 0,25 | CETSP-06 | 30/30 | 6 | 0,0 ms | 0,002 s | 26 | 0,029 s |
| 0,25 | CETSP-10 | 30/30 | 14 | 0,1 ms | 0,346 s | 86 | 0,107 s |
| 0,25 | CETSP-16 | 30/30 | 53 | 0,7 ms | 0,602 s | 443 | 0,673 s |
| 0,25 | CETSP-20 | 30/30 | 110 | 1,5 ms | 0,852 s | 1.016 | 1,656 s |
| 0,5 | CETSP-20 | 30/30 | 53 | 0,7 ms | 1,029 s | 437 | 0,719 s |
| 1,0 | CETSP-20 | 30/30 | 28 | 0,4 ms | 1,055 s | 163 | 0,261 s |

(Grupos selecionados; os 24 grupos estão em `summary-behdani.md` da campanha.)

**Limitações.**
- Uma execução por instância. O i7 de Coutinho et al. é de 2012; os tempos
  são indicativos, não uma comparação de máquina igual.
- Exatidão: gap de 1e-6 com limites provados, não ótimo algébrico (com
  discos não há caminho exato). Os limites valem na instância em binary64,
  que difere da decimal em ~1e-14 por coordenada.
- A memória da árvore limita as instâncias grandes. Nós menores, ou
  reaproveitar os nós podados, deixariam rodar até o tempo.
- O piso de 200 iterações da ILS pesa nas instâncias pequenas; rodar o B&B
  primeiro, por um instante, as fecharia em milissegundos.
- Dados: local, em
  `benchmarks/workspace/experiments/cetsp-native/campaign-20261025-1752/`
  (JSONL, tours, `summary-*.md` e `log.txt`).

## cetsp-geometric-mac-2026-10-25

**Substituído** por [cetsp-campaign](#cetsp-campaign-mac-2026-10-25), com o
B&B preguiçoso, as threads e a ILS nova; os números abaixo são da versão
anterior (filhos resolvidos ao criar).

**Pergunta.** Com uma construção geométrica no oráculo de discos (Alhazen,
Newton no tipo, blocos para discos sobrepostos) e um B&B com tour inicial por
busca local, quanto do CETSP 2D de Mennell se resolve, comparado a Coutinho
et al. (2016)?

**Formulação.**
- **Oráculo:** `tpp_convex_solve_cycle_disks`, como em
  [cetsp-native](#cetsp-native-mac-2026-10-25), agora com a proposta
  geométrica.
  - Atualizações de Alhazen fixam um tipo, e o Newton nos ângulos o fecha.
  - Contatos coincidentes andam em bloco: ponto da lente, um círculo ativo ou
    separação.
  - O canto só fica se o seu dual se divide.
  - A barreira é o recurso de reserva.
  - Detalhes em
    [convex-cycle.md](../../docs/algorithms/convex-cycle.md#disks-2026-10-25).
- **B&B:** `tpp-convex-cetsp`.
  - Best-first, ramificando no disco descoberto mais distante.
  - Nós de 8 bytes, re-resolvidos ao expandir.
  - Tour inicial: tour dos centros, inserção e busca local em sequências,
    depois ILS de ordem completa; 5% do tempo, parando ao estagnar.
  - Gap relativo de 1e-6; o oráculo recebe 0,25 × gap × UB.
- **Execução.**
  - 1.800 s por instância (Coutinho et al.: 4 h), 1 thread, 7 ao mesmo tempo
    num Mac M4 Pro (12 núcleos, 24 GB).
  - As fáceis rodaram ao lado de outras 6.
- **Certificação e verificação.**
  - Os limites são provados na instância lida em binary64.
  - Cada tour foi verificado à parte em racionais exatos: cobre todos os
    discos binary64, e o comprimento, com raízes arredondadas para cima,
    não passa do UB.
  - Na instância decimal, um contato exatamente no círculo pode errar o disco
    por ~1e-14 (o arredondamento do centro). Os tours gravados a partir do
    commit 9b2502d recuam os contatos 2⁻⁴⁰ do raio e cobrem também a decimal.
- **Referência.** Tabela 2 de Coutinho et al. (CPLEX 12.4, 1 thread, i7 de
  3,4 GHz, 4 h).

**Resultado (30 min por instância).**

| Instância | n | Nosso [LB; UB] | Tempo | Coutinho et al. [LB; UB] | Tempo |
|---|---|---|---|---|---|
| bubbles1 | 36 | **349,135** | 0,3 s | 349,135 | 0,1 s |
| bubbles2 | 76 | **428,279** | 0,9 s | 428,279 | 0,2 s |
| bubbles3 | 126 | **529,955** | 13,0 s | 529,955 | 193,1 s |
| bubbles4 | 184 | [715,694; 804,989] | 30 min | [690,578; 829,888] | 4 h |
| bubbles5 | 250 | [894,334; 1.080,069] | 30 min | [851,822; 1.062,335] | 4 h |
| bubbles6 | 324 | [1.048,670; 1.401,378] | 30 min | [993,981; 1.383,139] | 4 h |
| bubbles7 | 406 | [1.185,752; 1.675,336] | 30 min | [1.123,522; 1.720,214] | 4 h |
| bubbles8 | 496 | [1.325,330; 2.030,876] | 30 min | [1.252,715; 2.101,373] | 4 h |
| bubbles9 | 594 | [1.459,890; 2.435,313] | 30 min | [1.374,407; 2.426,274] | 4 h |
| chaoSingleDep | 200 | [1.039,604; 1.039,611] | 30 min | [1.000,151; 1.039,610] | 4 h |
| concentricCircles1 | 16 | **53,158** | 0,2 s | 53,158 | 5,2 s |
| concentricCircles2 | 36 | **153,132** | 461,4 s | [149,868; 153,132] | 4 h |
| concentricCircles3 | 60 | [256,601; 270,034] | 30 min | [247,624; 271,076] | 4 h |
| concentricCircles4 | 104 | [389,618; 454,702] | 30 min | [358,887; 454,457] | 4 h |
| concentricCircles5 | 148 | [493,793; 633,342] | 30 min | [459,411; 645,381] | 4 h |
| rotatingDiamonds1 | 20 | **32,389** | 0,2 s | 32,389 | 0,1 s |
| rotatingDiamonds2 | 60 | **140,477** | 6,4 s | 140,477 | 730,4 s |
| rotatingDiamonds3 | 180 | [368,817; 380,886] | 30 min | [348,608; 380,882] | 4 h |
| rotatingDiamonds4 | 320 | [637,314; 770,661] | 30 min | [593,350; 770,660] | 4 h |
| rotatingDiamonds5 | 680 | [1.189,862; 1.510,861] | 30 min | [1.106,577; 1.510,752] | 4 h |

- **Fechadas: 7 de 20 em 30 min** (8 com a de 4 h, abaixo). São as 6 que
  Coutinho et al. resolvem e **concentricCircles2**, que eles deixaram
  aberta em 4 h.
  - O ótimo de concentricCircles2 é 153,132: o UB deles era ótimo.
  - Mais rápido em rotatingDiamonds2 (114×), bubbles3 (15×) e
    concentricCircles1 (26×).
  - Nas três menores o tempo é quase só a heurística (0,2–0,9 s).
- **Abertas.**
  - Em 30 min, o LB passa o de 4 h deles nas 13, de 3,6% (bubbles4,
    concentricCircles3) a 8,6% (concentricCircles4) acima.
  - chaoSingleDep fica a 6,9e-6 de fechar; fecha em 1.794 s na rodada de
    4 h, com só 3 processos.
  - UB melhor em 5 (bubbles4, 7 e 8; concentricCircles3 e 5), praticamente
    igual em 4 (chao, rotatingDiamonds3 a 5) e pior em 4 (bubbles5, 6 e 9;
    concentricCircles4).
- **Custo do oráculo no B&B.**
  - 20–30 µs por chamada nas abertas.
  - A construção fecha 99,9–100% das chamadas; de 85% a 98% nas menores, de
    bubbles e rotatingDiamonds1.

**Resultado (4 h, o orçamento de Coutinho et al.)** nas três abertas mais
próximas de fechar, 3 ao mesmo tempo (binário 9b2502d):

| Instância | Nosso [LB; UB] | Tempo | Coutinho et al. [LB; UB], 4 h |
|---|---|---|---|
| chaoSingleDep | **1.039,611** | 1.794 s | [1.000,151; 1.039,610] |
| rotatingDiamonds3 | [377,271; 380,886], gap 0,95% | 4 h | [348,608; 380,882] |
| concentricCircles3 | [261,526; 270,034], gap 3,2% | 4 h | [247,624; 271,076] |

- **chaoSingleDep fechada em 30 min:** o ótimo é 1.039,611, e o UB de 4 h
  deles era ótimo. Com isso, **8 de 20 fechadas, duas delas abertas para
  Coutinho et al.**
- O tour de chaoSingleDep visita 35 dos 200 discos. Ele foi verificado à
  parte e cobre também os discos decimais, com margem r² − d² ≥ 4,5e-11.
  O comprimento provado é 1.039,61089118.

**Oráculo de ordem fixa × Gurobi.**
- **Formulação.** O ciclo pelas 20 instâncias em ordem angular é um SOCP. O
  Gurobi 12.0.3 roda com 1 thread, tolerâncias padrão e a forma cônica que
  ele resolveu com mais frequência. O tempo inclui a montagem do modelo.
- **Medida.** Melhor de 5 nos dois lados, com a máquina ociosa.
- **Certificação.** O nosso: certificado em todas.
  - A construção fecha 19, com gap de ~1e-12 relativo.
  - O rotatingDiamonds4 a frio cai na barreira e fecha no gap pedido (1e-6
    absoluto).
- **Gurobi.**
  - "Numerical trouble" em 8 das 20: bubbles1, 2, 3 e 7,
    concentricCircles2 e 3, rotatingDiamonds1 e 4.
  - Das 12 "ótimas", 8 têm objetivo abaixo do nosso LB certificado, de 3e-9
    a 4e-7 relativo: pontos levemente inviáveis.
  - Outras formas do mesmo modelo (bola direta, lado direito constante,
    homogêneo) falharam em instâncias diferentes, ou deram status subótimo
    com valor abaixo do ótimo.
- **Tempo.** O nosso é de 11× a 82× mais rápido, com mediana de 17×. A
  exceção é o rotatingDiamonds4 a frio, com 2×.

**Limitações.**
- Uma execução por instância, num Mac compartilhado entre 7 processos; o
  i7 de Coutinho et al. é de 2012.
- Exatidão: gap relativo de 1e-6 com limites provados, não ótimo algébrico,
  porque discos não têm caminho exato (Alhazen). Os limites são da instância
  em binary64; a decimal difere em ~1e-14 por coordenada, o que move o ótimo
  em no máximo 2·Σ|Δ| ≈ 1e-11.
- B&B de exploração (`main-cetsp.cpp`), não integrado ao B&B do TSPN.
- Dados e scripts: local, em `benchmarks/workspace/experiments/cetsp-native/`
  (`table-v6.jsonl`, `tours-table/`, `long-v7.jsonl`, `tours-long/`,
  `gurobi-cycle.jsonl`, `verify_tour.py`).

## cetsp-native-mac-2026-10-25

**Substituído** por [cetsp-geometric](#cetsp-geometric-mac-2026-10-25), com a
construção geométrica e a heurística nova.

**Pergunta.** Um oráculo nativo para discos, com o mesmo "propor e
certificar", faz o CETSP andar melhor que o sanduíche de polígonos e que
Coutinho et al. (2016)?

**Formulação.**
- **Oráculo:** `tpp_convex_solve_cycle_disks`.
  - Proposta por Newton com barreira de disco.
  - `L` pela dualidade fraca com o suporte do disco em forma fechada; `U` por
    contatos provados dentro dos discos.
  - Detalhes em
    [convex-cycle.md](../../docs/algorithms/convex-cycle.md#disks-2026-10-25).
- **B&B:** a ferramenta de exploração `tpp-convex-cetsp`.
  - Best-first, com sequências cíclicas a partir do depósito.
  - Ramifica no disco descoberto mais distante, em todas as posições, com a
    simetria de sentido eliminada.
  - Os filhos são avaliados ao serem criados, com corte pelo UB, e o oráculo
    recebe um gap de 0,25 × gap × UB.
  - UB inicial por uma heurística de inserção simples.
- **Execução.** Gap relativo de 1e-6, 1 thread por instância, 600 s. Mac M4
  Pro **em uso**, com 6 instâncias ao mesmo tempo nas maiores.
- **Referência.** Tabela 2 de Coutinho et al. (CPLEX 12.4, 1 thread, i7 de
  3,4 GHz, 4 h).

**Resultado.**

| Instância | Nosso | Coutinho et al. |
|---|---|---|
| rotatingDiamonds1 | 32,3890, 0,02 s | 32,389, 0,09 s |
| bubbles1 | 349,1349, 0,05 s | 349,135, 0,10 s |
| bubbles2 | 428,2793, 0,04 s | 428,279, 0,22 s |
| concentricCircles1 | 53,1580, 2,3 s | 53,158, 5,2 s |
| rotatingDiamonds2 | 140,4774, **22,9 s** | 140,477, 730 s |
| bubbles3 | 529,9548, 124 s | 529,955, 193 s |
| rotatingDiamonds3 | [348,86; 463,08], 600 s | [348,61; 380,88], 4 h |
| chaoSingleDep | [999,54; 1.093,66], 600 s | [1.000,15; 1.039,61], 4 h |
| bubbles4 | [685,52; 805,12], 600 s | [690,58; 829,89], 4 h |
| concentricCircles2 | [147,59; 159,73], 600 s | [149,87; 153,13], 4 h |

- **Fechadas.** Nas seis que os dois fecham, os ótimos coincidem até a
  terceira casa, e o nosso é mais rápido em todas: de 1,6× (bubbles3) a 32×
  (rotatingDiamonds2), sem descontar a diferença de máquina.
- **Abertas.**
  - Em 10 min, o nosso LB chega perto do de 4 h deles e passa em
    rotatingDiamonds3.
  - O UB é o ponto fraco: a heurística de inserção dá tours de 5 a 22% acima
    dos melhores conhecidos (Mennell) e domina o gap.
- **Custo do oráculo.** De 0,08 a 1,5 ms por chamada e de 190 a 290
  iterações de Newton, pelo gap apertado. Em bubbles3, o sanduíche com m = 128
  fechou em 3 s, contra 124 s aqui: o oráculo de discos ainda é mais lento por
  chamada que o de polígonos nessa instância.

**Limitações.**
- Uma execução por instância, com o Mac carregado.
- B&B de exploração, sem a triagem de inserções nem o ILS do TSPN.
- O oráculo não tem o equivalente da construção (conjunto ativo e Newton na
  fronteira), o próximo passo natural.
- Dados: local, em `benchmarks/workspace/experiments/cetsp-native/`.

## cetsp-sandwich-mac-2026-10-25

**Pergunta.** Dá para resolver o CETSP (vizinhanças em disco) com o nosso TSPN,
de forma certificada, sem código novo?

**Formulação.**
- **Sanduíche de polígonos.** Cada disco é trocado por dois m-ágonos
  regulares:
  - um dentro dele: o TSPN nesses polígonos dá um tour viável, ou seja, um UB
    do CETSP;
  - outro que o contém: o ótimo do TSPN neles é um LB.

  As duas contenções são conferidas em aritmética exata depois do
  arredondamento. O depósito é um ponto.
- **Instâncias e referência.** As instâncias 2D de Mennell (na época, uma
  cópia local; os dados são os de `benchmarks/suites/cetsp/cerrone/Dataset/`,
  só o cabeçalho de comentários difere), comparadas com a Tabela 2 de
  Coutinho et al. (INFORMS J. Comput. 2016): B&B com SOCP no CPLEX 12.4, 1
  thread, i7 de 3,4 GHz, limite de 4 h.
- **Execução.** TSPN cíclico, gap relativo de 1e-6, `cache`, `features`,
  `root`, `interval`, fronteira de 1 GiB, 1 thread por lado. Mac M4 Pro **em
  uso** (outro programa aberto).

**Resultado.**
- **Correção.** O LB externo nunca passou do ótimo de Coutinho et al., e o UB
  interno nunca ficou abaixo dele, em todas as instâncias resolvidas por eles
  (bubbles1 a 3, concentricCircles1, rotatingDiamonds1 e 2).
- **O gap do sanduíche segue (π/m)².** Ele cai 4× a cada vez que m dobra: de
  2,2e-3 em m = 32 a 9e-6 em m = 512 (concentricCircles1).
- **Árvores menores:** de 2 a 37 vezes menos nós que as de Coutinho et al.
  - bubbles1: 28 contra 66;
  - concentricCircles1: 554 contra 2.943;
  - bubbles3: 592 a 645 contra 31.995;
  - rotatingDiamonds2: ~52 mil contra 275 mil.
- **Tempos com m = 128** (gap do sanduíche de ~1e-4):

  | Instância | Nosso (LB…UB) | Tempo por lado | Coutinho et al. |
  |---|---|---|---|
  | bubbles3 | 529,946…529,979 | 2–3 s | ótimo 529,955, 193 s |
  | rotatingDiamonds2 | 140,470…140,479 | 362–380 s | ótimo 140,477, 730 s |
  | concentricCircles1 | 53,153…53,161 | 5 s | ótimo 53,158, 5,2 s |
  | chaoSingleDep | 1.001,0…1.039,8 | 1.800 s, aberta | 1.000,2…1.039,6 em 4 h |
  | concentricCircles2 | 146,6…153,1 | 1.800 s, aberta | 149,9…153,1 em 4 h |

- **O gargalo é o custo por nó.** De 6 a 370 ms, contra ~33 µs no corpus do
  Fekete. Com discos muito sobrepostos, dois terços das chamadas vão ao
  polimento (contatos coincidentes), e polir polígonos de centenas de vértices
  é caro. Em m = 256, o lado interno de bubbles3 não fechou em 120 s.

**Limitações.**
- Uma execução, com o Mac em uso; os tempos são indicativos.
- Só 2D.
- O sanduíche custa duas execuções e tem gap mínimo de (π/m)².
- Um oráculo nativo para discos tiraria esse gap e o custo do polimento por
  face: o suporte tem forma fechada e o limite dual vale do mesmo jeito.
- Dados e scripts: local, em `benchmarks/workspace/experiments/cetsp-sandwich/`.

## sao-paulo-official-mac-2026-10-23

**Pergunta** (Ernesto, 23/10). O tour da instância de São Paulo é ótimo como
TSPN, sem ponto inicial?

**Formulação.**
- **Instância.** O arquivo `96sp4966.dat`, enviado pela Paula em 23/10 (local,
  em `third_party/paula-tspn/sao-paulo/`, não redistribuído).
  - 96 polígonos não convexos, de 51 a 57 vértices (4.966 no total), em
    coordenadas UTM (metros).
  - Foram transladados por inteiros (314.000; 7.345.000), o que é exato em
    binary64 e não muda as distâncias.
  - **Dois pares se sobrepõem**, embora o rascunho `ber2025` diga que os
    polígonos são disjuntos. Numerando a partir de 1, na ordem do arquivo:
    - 83 e 85: 45.839 m², ou 0,54% e 0,75% das áreas deles;
    - 90 e 91: 5.400 m², 0,10% de cada.

    A área total dos 96 polígonos é de 943 km².
- **Origem, segundo Birgin e Martínez (2022, Seção 6.4).** Mapa do GeoSampa,
  com 156.852 vértices, simplificado para 51 a 57 vértices por polígono e
  encolhido 20% em direção ao centro do retângulo envolvente.
- **Tour inicial.** O tour publicado (199.362,73), levado para as coordenadas
  oficiais a partir da reconstrução de
  [`sao-paulo-figure-mac-2026-10-23`](#sao-paulo-figure-mac-2026-10-23).
  - As duas instâncias diferem por uma translação pura, com resíduo de até
    4,2 cm por vértice.
  - Os pontos foram projetados nos polígonos.
- **Execução.** A mesma de lá: TSPN cíclico, gap relativo de 1e-6, `lazy` sem
  mergulhos, fronteira de 512 MiB, 6 threads de subárvore. Em paralelo, o ILS
  a partir do zero (1 thread, 1.200 s). Mac M4 Pro, solver da branch.

**Resultado.**
- **O tour publicado é ótimo,** com gap de 1e-6: LB 199.362,529 e UB
  199.362,729, em 2.932 s e 170 milhões de nós, sem chamadas racionais. Em
  distância, o gap é de 0,2 m em 199 km.
- **O ILS acha o mesmo tour sozinho:** 199.362,72880, igual ao do B&B até
  1e-9.
  - As duas poligonais coincidem em 56 dos 57 pontos de dobra (a menos de
    0,1 mm).
  - A ordem de visita listada pelo ILS difere da publicada em 6 ligações,
    sem mudar o comprimento.

**Limitações.**
- Uma execução.
- Gap de 1e-6, não ótimo algébrico.
- Dados e scripts: local, em `benchmarks/workspace/experiments/sao-paulo/` (no
  checkout principal).

## sao-paulo-figure-mac-2026-10-23

**Pergunta** (Ernesto, 23/10). O tour da instância de São Paulo é ótimo como
TSPN, sem ponto inicial, ou existe outro melhor?

**Formulação.**
- **Instância.** A de Birgin e Martínez (2022): 96 bairros disjuntos e não
  convexos, de 51 a 57 vértices (4.966 no total). No rascunho `ber2025` da
  Paula, o melhor tour do ILS-BCD mede 199.362,73.
- **Reconstrução.** O arquivo oficial ainda não chegou. A instância foi
  reconstruída da figura do rascunho: coordenadas da figura em cm, com 5 casas,
  multiplicadas pela escala que dá ao tour desenhado o comprimento publicado
  (6.034,45 por cm).
  - O arredondamento move cada coordenada até cerca de 0,03 unidade.
  - Sete pontos de visita do tour publicado caem até 0,037 fora do seu
    polígono, e dois pares de polígonos passam a se tocar.
  - Os pontos foram projetados nos polígonos. O tour inicial resultante mede
    199.362,7497.
- **Execução.** Mac M4 Pro, solver da branch `oracle-vertex-scaling` em
  `d1bac40`. TSPN cíclico com gap relativo de 1e-6, `cache`, `features`,
  `root`, `interval`, `lazy` sem mergulhos, fronteira de 512 MiB, 6 threads
  de subárvore, a partir do tour publicado. À parte, o ILS próprio do solver
  (`--primal-ils 0.99`) a partir do zero, com 1 thread por 1.200 s.

**Resultado.**
- **O B&B fecha em 840 s** (170 milhões de nós): LB 199.362,550 e UB
  199.362,750, que é o próprio tour publicado. Nenhum tour da reconstrução é
  mais curto que 199.362,55.
- **O ILS acha outro tour,** de 199.362,737, que difere do publicado em 10 das
  96 ligações. Os dois ficam a 6e-8 um do outro, abaixo do erro da
  reconstrução.

**Limitações.**
- Não é a instância oficial. O erro de arredondamento pode mudar o ótimo em até
  cerca de 4e-5 relativo (limite folgado: duas ligações por região, 0,04 cada).
  A resposta definitiva, com o arquivo da Paula, está em
  [`sao-paulo-official-mac-2026-10-23`](#sao-paulo-official-mac-2026-10-23).
- Uma execução de cada.
- Dados e scripts: local, em `benchmarks/workspace/experiments/sao-paulo/` (no
  checkout principal), com instruções para o arquivo oficial.

## no-polish-mac-2026-10-23

**Pergunta** (Ernesto, 23/10). Dá para eliminar os pontos interiores e fechar
as chamadas só com a geometria em precisão dupla?

**Formulação.** Os 60 casos de
[`ablation-mac-2026-10-21`](#ablation-mac-2026-10-21), configuração C, uma
thread, 40 s, 6 processos com variantes intercaladas, Mac M4 Pro. O polimento
foi desligado por um build local com `ConvexCycleDoubleOptions::float_polish`
forçado a falso; o patch não foi arquivado. Branch em `667d572`, depois com o
passo de [`anchor-step-mac-2026-10-23`](#anchor-step-mac-2026-10-23).

**Resultado.**

| Variante | Fecham | Chamadas racionais | Tempo, média geom. nos 58 que os dois fecham |
|---|---|---|---|
| C | 59 | 0 | 1 |
| Sem polimento | 58 | 60.461 | 6,51× (mediana 9,1×, até 143×) |
| Sem polimento, com o passo | 58 | 60.258 | 6,56× |

- **O que muda.** Os nós são os mesmos; só o custo por chamada sobe (oráculo
  8,5×). As chamadas que o polimento fechava vão ao caminho exato racional.
- **Por classe** (fração fechada pelo polimento em C; tempo sem ele):
  - OSM disjuntas (12): 9,7%; 1,37×;
  - OSM só com toques (22): 0,3%; 3,1×;
  - tesselações (6): 24,1%; 17,7×;
  - aleatórias com sobreposição (17): 6,4%; 33,5×.
- **O passo não ajuda aqui,** porque o custo não está na bissecção. Para
  eliminar os pontos interiores, faltaria uma construção que feche os casos com
  contatos coincidentes e sobreposições.

**Limitações.**
- Uma execução por caso.
- A variante com o passo rodou depois, sem C intercalada, com a mesma carga.
- Dados: local, em `benchmarks/workspace/experiments/no-polish/`.

## anchor-step-mac-2026-10-23

**Pergunta** (Ernesto, 23/10). A bissecção na aresta não gasta chamadas
demais ao TPP? Por que não Newton?

**Formulação.**
- **Chamadas.** 320 chamadas reais do B&B, 8 por caso, espaçadas em cada uma
  das capturas de 40 casos de
  [`oracle-gurobi-corpus-mac-2026-10-21`](#oracle-gurobi-corpus-mac-2026-10-21).
- **Execução.** Cada chamada foi resolvida até gap zero pelo caminho exato
  racional (`tpp_convex_solve_cycle`), com limite de 30 s, de dois jeitos:
  como ele roda (construção primeiro) e só com a busca por âncoras
  (`refine_contacts=false`).
- **Contadores novos** em `ConvexCycleResult`: TPPs ancorados, sondas de
  bissecção e passos em forma fechada. Ferramenta: `tpp-convex-anchor-probes`.
- **O passo em forma fechada** (novo): o tipo combinatório de cada ciclo
  ancorado, resolvido por desdobramento com a âncora refletindo na aresta,
  propõe a próxima sonda (o papel do Newton, aqui exato). Detalhes em
  [convex-cycle.md](../../docs/algorithms/convex-cycle.md#general-boundary-reduction-and-correctness).
- Mac M4 Pro, 4 processos em segundo plano.

**Resultado.**

| Caminho exato | Sondas (máx. numa chamada) | TPPs ancorados, máx. | Pior chamada | Total |
|---|---|---|---|---|
| Construção primeiro | 0 | 2 | 0,20 s | 5,1 s |
| Só âncoras, só bissecção | 1.797 (469) | 482 | 30 s, interrompida | 120,5 s |
| Só âncoras, com o passo | 2 (1) | 22 | 0,41 s | 26,9 s |

- **Com a construção primeiro,** 297 das 320 chamadas fecham sem TPP
  ancorado, e as outras 23 com no máximo 2, em vértices.
- **Só com âncoras,** 99% fecham nos vértices da menor região (mediana de 4
  TPPs). Quatro precisaram de bissecção:
  - de 424 a 469 sondas cada, de 16 a 30 s, uma interrompida;
  - com o passo, no máximo 1 sonda e 0,41 s.
- **O ótimo é o mesmo** nas 639 comparações em que as duas versões fecham.
- **Conta da bissecção.** Ela pede cerca de 14 sondas para 10⁻⁴, não 10 mil.
  Em racionais exatos, porém, o alvo é um racional de denominador enorme, e
  O(log Q) chega a centenas de sondas.
- **Regressão.** No caso assimétrico dos quatro quadrados, a busca só por
  âncoras fecha com 7 TPPs ancorados e nenhuma sonda (teste
  `rotated_squares`).
- **Corpus.** Nos 558 casos, chamadas, nós e limites são idênticos bit a bit
  aos da branch antes do passo; o B&B não chega ao caminho exato.

**Limitações.**
- 320 chamadas; as difíceis vêm de três casos (49, 138 e 391).
- Tempos sem a máquina dedicada.
- Dados: local, em `benchmarks/workspace/experiments/anchor-probes/`.

## oracle-million-mac-2026-10-22

**Pergunta.** Uma chamada ao oráculo do ciclo chega a ser um milhão de vezes
mais rápida que o SOCP do Gurobi, com a mesma tolerância?

**Formulação.**
- **Instâncias.** Três discos disjuntos (k = 3), no leiaute de
  [`oracle-vertex-scaling-2026-10-21`](#oracle-vertex-scaling-2026-10-21),
  com m = 32.768 a 262.144 vértices por região e duas amostras por m (semente
  20261022). O reticulado inteiro cresce com m (raio R ≥ m²/4), para que os m
  pontos de cada círculo fiquem todos no fecho. As coordenadas são divididas
  por uma potência de 2, para regiões de raio ~1024.
- **Nosso oráculo.** Estágio binary64 com certificado intervalar e regiões
  preparadas uma vez (`region_ids` e workspace, como no B&B). Duas
  tolerâncias: gap de 1e-6 × o comprimento (a tolerância padrão do Gurobi) e o
  gap do B&B (2,5e-4). Dois tempos:
  - a primeira chamada logo depois do Gurobi;
  - a média de um bloco de chamadas seguidas por pelo menos 0,2 s (~30 mil
    chamadas), como o B&B chama o oráculo.

  O workspace guarda só a geometria das regiões, nunca soluções: cada chamada
  resolve do zero.
- **Gurobi** 13.0.3: parâmetros padrão, uma thread, modelo montado a cada
  chamada (como no oráculo SOCP do Fekete), normais unitárias.
- **Execução.** Mac M4 Pro, processo único, 2 repetições (1 em m = 262.144),
  com o Mac em uso por outros programas.

**Resultado** (medianas; razão = tempo total do Gurobi / tempo do nosso, ambos
com gap de 1e-6):

| m por região | Gurobi | Nosso, média do bloco | Nosso, 1ª chamada | Razão (bloco) | Razão (1ª chamada) |
|---|---|---|---|---|---|
| 32.768 | 1,36 s | 5,9 µs | 18,6 µs | 230.970× | 73.194× |
| 65.536 | 6,74 s | 6,7 µs | 49,1 µs | 1.000.816× | 137.189× |
| 131.072 | 15,6 s | 6,6 µs | 18,2 µs | 2.352.609× | 858.870× |
| 262.144 | 57,9 s | 7,1 µs | 22,5 µs | 8.097.751× | 2.568.988× |

- **Mesmo problema.** Nas 14 execuções, o objetivo do Gurobi fica dentro do
  intervalo certificado [L, U] do nosso, cuja largura relativa é de no
  máximo 5,6e-7.
- **Um milhão.** Com m = 262.144 (786 mil vértices), a razão passa de um
  milhão em todas as formas medidas:
  - 7,9 milhões× na pior execução, pela média do bloco;
  - de 1,2 a 2,6 milhões× na primeira chamada depois do Gurobi (nas duas
    varreduras; com cache frio, essa chamada oscila mais);
  - 19 milhões× com o gap do B&B (3,0 µs).

  Só a otimização do Gurobi, sem montar o modelo, também dá 8,0 milhões×.
- **Escala.** Com m 8× maior, o nosso vai de 5,9 a 7,1 µs. O Gurobi vai de
  1,4 a 58 s, cerca de m^1,8.
- **Como fecha.** Todas as chamadas fecham pela prova intervalar sobre a
  construção, sem iterações de Newton: os discos são disjuntos e bem
  separados.

**Limitações.**
- É uma chamada isolada com k = 3 em discos sintéticos, não o B&B nem uma
  instância inteira. No TSPN, a maior razão medida por nó contra o código do
  Fekete é de 168 mil× (dantzig, m = 16.384,
  [`fekete-smoothed-dantzig-2026-10-22`](#fekete-smoothed-dantzig-2026-10-22)).
- A razão não tem teto. A nossa chamada cresce com log m, e um SOCP pelo
  menos com m, então um milhão é uma escolha de m. A chamada fria, que
  normaliza as regiões, continua O(m): cerca de 900× na seção de 21/10.
- **Ruído.** Num teste anterior à varredura, com o Mac em uso, o mesmo
  Gurobi levou 8,1 s em vez de 1,4 s (m = 32.768) e 90 s em vez de 15,6 s
  (m = 131.072).
- **Repetição com o Mac ocioso (23/10).** Mesmo binário e mesmas instâncias:
  - Gurobi em 1,37, 6,81, 15,63 e 57,95 s (antes, 1,36, 6,74, 15,60 e
    57,86 s);
  - o nosso em 6,0 a 7,2 µs, ou 8,0 milhões× com m = 262.144 (pior
    execução: 7,9 milhões×);
  - a primeira chamada variou de 16,8 a 65,1 µs.
- O Gurobi usa uma thread, como no oráculo do Fekete. Com mais threads, ele
  seria mais rápido.
- Programa: `main-vertex_scaling.cpp`, variante `ids_tight`. Gerador e dados
  brutos: local, em `experiments/vertex-scaling/` (`generate_large.py`,
  `large/`).

## tspn-smoothed-m65536-mac-2026-10-22

**Pergunta.** Por que o nosso TSPN não fechava as instâncias suavizadas com
65.536 vértices por região, se com 16.384 fecha em segundos?

**Formulação.**
- **Instâncias.** As seis da
  [`tspn-smoothed-vertex-scaling-2026-10-21`](#tspn-smoothed-vertex-scaling-2026-10-21),
  com m = 65.536 (de 2,4 a 3,9 milhões de vértices por instância), geradas
  por `tpp.py smoothed-instances --variants 65536 --no-annotations`.
- **Execução.** Argumentos da configuração C, uma thread, Mac M4 Pro, uma
  execução por instância e binário. Limite de 600 s para o binário corrigido
  e 300 s para o anterior (b983a59).

**Diagnóstico.** Algumas chamadas ao oráculo de `vienna` e `warsaw` caíam no
caminho racional, cuja normalização em GMP, `O(m)` por região, consumia o
orçamento. Num replay de duas chamadas capturadas de `vienna` (`k = 30`, 1,97
milhão de vértices), o polimento parava nas 600 iterações de Newton:
- a banda `w-4..w+4` da aproximação interna ficava sempre um pouco atrás do
  vértice extremo, nos arcos de milhares de arestas curtas, e as 8 rodadas de
  refinamento de cada nível só acrescentavam faces;
- com μ pequeno, o Armijo aceitava passos minúsculos que não mudavam o
  objetivo (inclinação −9×10⁻¹⁴), repetidos até o teto de iterações.

**Correção.** A banda passou a ser graduada: fina em `w` e com espaçamentos
dobrando até o do esqueleto. O refinamento agora ignora a região cujo
extremo já está no subconjunto, e o Newton para quando um passo aceito não
muda o objetivo. Os detalhes estão em
[convex-cycle.md](../../docs/algorithms/convex-cycle.md#polish-on-inner-approximations).

**Resultado** (gap relativo de 0,1%, numérico; nenhuma chamada racional com a
correção):

| Instância | Vértices | Antes (s) | Agora (s) | Chamadas (agora) |
|---|---|---|---|---|
| bangalore | 3.923.307 | 2,07 | 2,03 | 9.514 |
| random_mixed_square | 2.420.203 | 1,95 | 2,27 | 6.599 |
| vienna | 3.276.804 | não parou no limite | 13,19 | 8.366 |
| istanbul | 3.866.633 | 5,62 | 6,13 | 25.822 |
| warsaw | 3.775.363 | caminho racional | 11,11 | 39.200 |
| tokyo | 3.895.636 | 12,06 | 11,18 | 89.933 |

- **Limites.** Nas quatro que os dois binários fecham, LB e UB são idênticos.
- **Antes.** A execução de `vienna` foi morta 30 s depois do limite de 300 s,
  ainda no caminho racional. O B&B só confere o relógio entre chamadas, e uma
  chamada racional com milhões de vértices não é interrompida a tempo. A de
  `warsaw` não rodou com o binário anterior. Com as três mudanças desligadas
  por variável de ambiente (o mesmo código do binário anterior), ela também
  ficou no caminho racional e não parou no limite de 60 s.
- **Cada mudança.** Com o mesmo esquema de variáveis, `vienna` fecha só com
  a parada do Newton. Já `warsaw` precisa das três: banda graduada (na
  inicialização e no refinamento), refinamento só com extremo novo e parada
  do Newton. Nenhuma combinação parcial testada fechou. Os tempos dessa
  ablação não são reportados, porque o Mac estava em uso.
- **Custo em m = 16.384.** Medianas de quatro execuções intercaladas por
  binário: a correção fica 1,055× mais lenta na média geométrica (de 1,00×
  em `bangalore` a 1,13× em `warsaw`), com o número de chamadas a até 0,4%
  do anterior. Não isolamos qual das três mudanças custa esse tempo.
- **Corpus.** Nos 558 casos (configuração C, uma thread, teto de 300 mil
  chamadas), chamadas, nós e limites são idênticos bit a bit aos da branch em
  d110ce2, inclusive nos 4 interrompidos pelo teto.

**Limitações.**
- Uma execução por instância em m = 65.536, num Mac só.
- O Fekete não rodou em m = 65.536: já em m = 16.384 ele fica sem tour na
  maioria das instâncias em 900 s.
- O tempo da dantzig em
  [`fekete-smoothed-dantzig-2026-10-22`](#fekete-smoothed-dantzig-2026-10-22)
  é do binário anterior. Com a correção, m = 16.384 deve ficar até ~13% mais
  lento, sem mudar o que fecha.
- Dados brutos: local, em `benchmarks/workspace/experiments/large-limit/`.

## fekete-smoothed-dantzig-2026-10-22

**Pergunta.** A comparação nas instâncias suavizadas
([`fekete-smoothed-mac-2026-10-21`](#fekete-smoothed-mac-2026-10-21)) se mantém
na máquina de referência, no protocolo do artigo?

**Formulação.**
- **Instâncias.** As mesmas 24 instâncias suavizadas: seis do corpus, como
  fecho e com 256, 2.048 e 16.384 vértices por região. Foram geradas na
  dantzig por `tpp.py smoothed-instances` (sha256 `e22aa168…`); os polígonos e
  as anotações conferem, instância por instância, com os do Mac. Os dois
  solvers leem o mesmo arquivo, com as coordenadas como o modelo do Fekete as
  escreve.
- **Configurações.** As da referência, via `--instances`:
  - C: o nosso, 1 thread, 300 s, 3 repetições, branch em 6ea2f9b;
  - B: Fekete com 1 thread, raízes CHR e LEFP, uma execução por núcleo de
    performance;
  - A: Fekete com 16 threads, raiz CHR, uma execução por vez.
- **Limites.** B e A com 300 s e uma repetição, mais a extensão de 900 s no que
  ficou aberto.
- **Execução.** dantzig, 22/10, sem carga alheia nos núcleos das execuções
  (no máximo 0,16, numa da C).
- **Dados.** Uma linha por instância, configuração, raiz e limite em
  [per-run.csv](fekete-smoothed-dantzig-2026-10-22/per-run.csv).

**Resultado** (Fekete com até 900 s; "aberta" é o gap final, `(UB−LB)/UB`):

| Variante | Nosso (C, 1 thread) | Fekete B (1 thread, melhor raiz) | Fekete A (16 threads) |
|---|---|---|---|
| Fecho | fecha 6/6 em 0,2–2,1 s | fecha 4/6 em 34–190 s; tokyo e warsaw ficam com 0,1% e 0,2% | fecha 6/6 em 6–245 s |
| m = 256 | fecha 6/6 em 0,6–6,5 s | fecha 0/6; gap de 0,4% a 23% | fecha 2/6 (random_mixed em 119 s, bangalore em 256 s); as outras com 0,6% a 3,4% |
| m = 2.048 | fecha 6/6 em 0,8–8,8 s | fecha 0/6; gap de 8,6% a 25% | fecha 0/6; gap de 0,3% a 9,1% |
| m = 16.384 | fecha 6/6 em 1,6–12,1 s | sem tour nas 6 | sem tour em 5 das 6 (random_mixed com 22%) |

- **Onde o Fekete fecha, ele é bem mais lento.**
  - Com 16 threads (A): de 30 a 202× o nosso com 1 thread nos fechos, e 194×
    e 461× nas duas de m = 256.
  - Com 1 thread (B, melhor raiz): de 185 a 353× nos fechos.
- **Custo por nó** (mediana de segundos por nó, `num_iterations` dele contra
  `nodes` nosso; rodadas de 300 s):

  | Variante | Nosso | B, CHR | A, CHR (16 threads, tempo de parede) |
  |---|---|---|---|
  | Fecho | 29 µs | 14 ms (495×) | 2,6 ms (92×) |
  | m = 256 | 93 µs | 3,8 s (40.848×) | 73 ms (789×) |
  | m = 2.048 | 139 µs | 10 s (71.840×) | 0,73 s (5.271×) |
  | m = 16.384 | 259 µs | 44 s (168.112×) | 21 s (79.173×) |

- **Reproduz o Mac.** As mesmas 4 instâncias fecham com a B. A razão por nó da
  B vai de 437 a 149 mil× no Mac e de 495 a 168 mil× aqui.
- **As 16 threads da A** fecham os 6 fechos e 2 instâncias de m = 256, mas
  nenhuma a partir de m = 2.048.
- **O nosso na dantzig é de 1,7 a 2,3× mais lento que no Mac** (mediana
  2,0×), no mesmo sentido da referência, em que o nosso fica mais lento na
  Intel.
- **Execuções interrompidas.** 4 execuções da B com LEFP em m = 16.384 foram
  mortas a 450 s na rodada de 300 s: um nó dele não pode ser interrompido. A
  extensão de 900 s as repetiu, e de novo sem tour.
- **Tours dele.** Os tours do Fekete ficam a até 5e-5 dos polígonos, como no
  corpus.

**Limitações.**
- Uma repetição do Fekete e seis instâncias, exploratórias e não realistas.
- O nosso tem 3 repetições.
- Não há variante de 65.536 vértices: com ela, o nosso ainda cai no caminho
  racional do oráculo (ver o docstring de `smoothed_instances.py`).
- Dados brutos ficam local (`fekete-reference-smoothed@dantzig`).

## ablation-mac-2026-10-21

**Pergunta.** Quanto cada componente da configuração C pesa no nosso solver
(experimento opcional 3 da pauta)?

**Formulação.**
- **Casos.** 60 casos do corpus, sorteados (semente 20261022) entre os 135 com
  pelo menos mil chamadas ao oráculo.
- **Solver.** O binário da `main`, ou seja, o solver do artigo, com uma thread
  e limite de 40 s.
- **Variantes.** C, e C sem um componente de cada vez:
  - sem o estágio binary64 do oráculo (`--no-cycle-float`, tudo pelo caminho
    exato);
  - sem o certificado intervalar (`interval`);
  - sem as features de partida (`features`);
  - sem cache e memo (`cache`);
  - sem a raiz separada (`root`);
  - sem a triagem de inserção em binary64 (`--no-cycle-binary-insertion`,
    triagem racional).
- **Execução.** Mac M4 Pro, seis execuções ao mesmo tempo, uma repetição.
- **Razões.** Médias geométricas contra C, nos casos que os dois fecham.

**Resultado.**

| Variante | Fecham (de 60) | Tempo × C | Nós × C | Oráculo × C |
|---|---|---|---|---|
| C | 59 | 1 | 1 | 1 |
| sem estágio binary64 | 54 | 41 | 1,00 | 63 |
| sem triagem binary64 | 58 | 10,6 | 1,00 | 1,05 |
| sem cache e memo | 58 | 1,26 | 1,00 | 1,41 |
| sem raiz separada | 59 | 1,20 | 1,22 | 1,25 |
| sem certificado intervalar | 59 | 0,99 | 1,00 | 0,99 |
| sem features | 59 | 0,98 | 1,00 | 0,97 |

- **O estágio binary64 é o componente decisivo.** Sem ele, as chamadas vão ao
  caminho racional (1,2 milhão de chamadas racionais), o oráculo fica 63× mais
  lento e 5 casos a mais deixam de fechar.
- **A triagem de inserção em binary64 vem em seguida.** A versão racional não
  muda a árvore, mas custa 10×.
- **A raiz separada é a única variante que muda a árvore**: sem ela, 22% mais
  nós.
- **O certificado intervalar e as features não mudam nada aqui.** Com o
  estágio binary64 ligado, quase nenhuma chamada chega ao certificado exato,
  que é onde o intervalo atua.

**Limitações.**
- Uma repetição, uma amostra de 60 casos e um limite de 40 s, que corta 6 casos
  da variante sem estágio binary64 (as razões dela usam só os casos fechados).
- Dados brutos ficam local, em `benchmarks/workspace/experiments/ablation/`.

## oracle-gurobi-corpus-mac-2026-10-21

**Pergunta.** Nas relaxações reais do B&B do corpus, quanto custa uma chamada do
nosso oráculo do ciclo contra o SOCP do Gurobi (o tipo de oráculo do Fekete)?
É o experimento 1 da pauta do artigo.

**Formulação.**
- **Captura.** O nosso B&B na configuração C (uma thread) rodou em 40 casos
  sorteados (semente 20261021) entre os 135 do corpus com pelo menos mil
  chamadas, gravando uma chamada a cada 50 (`--oracle-capture-every 50`).
  Foram 55.898 chamadas.
- **Repetição.** Repetimos até 400 por caso, igualmente espaçadas e na ordem
  original, num total de 6.977 chamadas. Cada chamada tem k de 5 a 41 regiões
  (mediana 15) e 79 vértices (mediana).
- **Nosso.** O mesmo pedido do B&B: gap, cutoff, partida a quente, features e
  um workspace por caso, como um trabalhador do B&B.
- **Gurobi.** O SOCP do mesmo ciclo, com parâmetros padrão, uma thread e o
  modelo montado a cada chamada, como no oráculo do Fekete. As restrições de
  semiplano usam normais unitárias, para que a tolerância de factibilidade
  dele seja uma distância.
- **Execução.** `main-vertex_scaling.cpp --replay`, binário da branch. Mac M4
  Pro, quatro processos ao mesmo tempo, uma repetição.

**Resultado.**
- **Tempo por chamada** (mediana): 9,0 µs o nosso e 0,95 ms o Gurobi (0,85 ms
  só na otimização). A razão é de 93× na média geométrica (p10 57×, p90 131×,
  e 84× contando só a otimização dele), quase constante em k.
- **Como o nosso fecha.** 95% das chamadas fecham pela construção e pela prova
  intervalar, e 5% pelo polimento; nenhuma usa o caminho racional. O intervalo
  provado tem mediana de 6e-15 relativo. Os maiores são os das chamadas que
  param no cutoff (`CertifiedBound`).
- **A resposta do Gurobi não é um limite.** Em 3.557 chamadas o objetivo dele
  fica abaixo do nosso limite inferior provado, por até 2,2e-5 relativo.
  - Não é erro nosso: nessas chamadas o nosso intervalo fecha em cerca de
    1e-14.
  - É a tolerância de factibilidade dele, 1e-6 por restrição: com cerca de 15
    contatos, cada um pode ficar até 1e-6 fora da sua região, o que encurta o
    ciclo.
  - Com as linhas sem normalizar (o primeiro modelo), essa tolerância vale
    1e-6/|aresta| e a diferença chegava a 1,1e-4.
- **Comparação com a decomposição de 03d4c2c.** Lá, cada nó nosso custava
  1/238 do nó do Fekete. Aqui, uma chamada isolada custa 1/93 de um SOCP do
  Gurobi; o restante do custo por nó vem do que cada nó dele faz além de um
  SOCP.

**Limitações.**
- O nosso é o binário da branch. No corpus, os caminhos do oráculo são os da
  `main`, salvo o atalho de 52a3f70.
- O SOCP é o nosso modelo do oráculo do Fekete, não o código dele.
- Amostra: 40 casos, até 400 chamadas por caso.
- Capturas e saídas ficam local, em
  `benchmarks/workspace/experiments/oracle-gurobi-corpus/`.

## fekete-smoothed-mac-2026-10-21

**Pergunta.** Nas mesmas instâncias suavizadas
([abaixo](#tspn-smoothed-vertex-scaling-2026-10-21)), como se sai o solver do
Fekete et al., cujo SOCP por nó cresce com o número de vértices?

**Formulação.**
- **Instâncias.** As 24 da seção abaixo: seis do corpus, cada uma com o fecho
  convexo e com m = 256, 2.048 e 16.384.
- **Anotações.** Para o Fekete, cada instância virou um `AnnotatedInstance`
  anotado pelo próprio `add_order_annotations` dele (ordem no fecho e
  interseções), sem simplificar a geometria, de modo que os dois solvers
  recebem os mesmos polígonos. O `meta` é o da instância original, mais uma
  nota da suavização.
- **Fekete.** Código original (5cd8450) no build do Mac, chamado como na
  referência (`fekete_reference_child.py`, a chamada do script do artigo), com
  1 thread e as raízes CHR e LEFP, como na configuração B.
- **Nosso.** Configuração C (uma thread, gap 0,1%, `cache`, `features`, `root`
  e `interval`), binário d110ce2.
- **Execução.** Limite de 300 s e eps 0,001. Mac M4 Pro, quatro execuções ao
  mesmo tempo (das 8 P-cores), uma repetição. Os tours são validados a 1e-7 com
  distâncias do Shapely (nos fechos, também com o `validate_cycle` exato).
- **Dados.** Uma linha por execução em
  [per-run.csv](fekete-smoothed-mac-2026-10-21/per-run.csv), e figura em
  [escala-vertices](../../docs/reports/escala-vertices/README.md).

**Resultado.**
- **Nosso:** fecha as 24 instâncias, a mais demorada em 7,5 s, todas com tour
  válido.
- **Fekete:** fecha 4 das 24 com cada raiz, todas fechos: bangalore,
  istanbul, random_mixed e vienna. Com a melhor raiz levou de 19 a 109 s,
  contra 0,1 a 0,4 s nosso, ou seja, de 170 a 362× mais. Nos fechos de tokyo
  e warsaw termina com gap de 0,76% e 0,27%.
- **Nenhuma variante suavizada fecha no Fekete.** Ao fim dos 300 s, com a
  melhor raiz:
  - em m = 256, gap de 6 a 10%;
  - em m = 2.048, gap de 12 a 19%;
  - tokyo não tem tour nenhum a partir de m = 256;
  - em m = 16.384, nenhuma instância tem tour. Ele explora de 9 a 18 nós e
    termina com o limite inferior de 14 a 27% abaixo do ótimo (14 a 23% com a melhor raiz). Um nó não pode
    ser interrompido no meio, então essas execuções passam do limite e duram
    até 458 s.
- **Custo por nó** (mediana de segundos por nó explorado, a contagem da
  decomposição de 03d4c2c):

  | Variante | Fekete | Nosso | Razão |
  |---|---|---|---|
  | Fecho | 6,4 ms | 15 µs | 437× |
  | m = 256 | 1,3 s | 50 µs | 26 mil× |
  | m = 2.048 | 4,0 s | 77 µs | 52 mil× |
  | m = 16.384 | 25 s | 169 µs | 149 mil× |

- **Tours do Fekete.** Nos fechos, os tours dele ficam a 6e-6 até 5e-5 dos
  polígonos (absoluto, com coordenadas de centenas de metros), como no corpus:
  é tolerância, não erro.

- **Fekete com 8 threads** (configuração A: raiz CHR, uma execução por vez,
  m = 256; linhas em [per-run-threads8.csv](fekete-smoothed-mac-2026-10-21/per-run-threads8.csv)).
  - Ele limita o Gurobi a 1 thread (`soc.cpp`), então o paralelismo é só a
    avaliação dos filhos.
  - Fecha 1 das 6, random_mixed em 162 s (479× o nosso). As outras param com
    gap de 0,51% a 8,30%.
  - Explora de 7 a 60× mais nós que com 1 thread. No ritmo por iteração, as 8
    threads são de 4 a 6× mais rápidas; o resto vem de o custo por nó variar
    ao longo da busca.
- **Controle da concorrência.** O warsaw com m = 256 e 1 thread, sozinho na
  máquina, chega ao mesmo gap (12,23%), com 92 nós em vez de 90 e um ritmo 5%
  melhor. As execuções simultâneas não distorcem a comparação com 1 thread.

**Limitações.**
- Uma repetição.
- Quatro execuções simultâneas num Mac só, sem fixar os núcleos (o macOS não
  permite); o controle acima mede o efeito em 5%.
- Instâncias exploratórias, não realistas.
- Com 8 threads, o Fekete só foi medido em m = 256 (acima).
- Entradas, saídas e logs brutos ficam local, em
  `benchmarks/workspace/experiments/fekete-smoothed/`.

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
- **Corpus.** Comparamos o solver da `main` (binário de 16/10, mesmo código de
  solver da `main` atual) com a branch (d110ce2) nos 558 casos, com os
  argumentos da configuração C, uma thread e um teto de 300 mil chamadas ao
  oráculo, para que as execuções interrompidas parem no mesmo ponto.
  - Em 557 casos, chamadas, nós e limites são idênticos bit a bit, inclusive
    nos 4 interrompidos pelo teto.
  - No caso 538, os nós (5.408) e os limites são idênticos, mas a branch faz
    55 chamadas a mais (9.328 contra 9.273). A bisseção leva ao primeiro
    commit da branch (52a3f70), o atalho de separação antes do recorte: com
    regiões quase encostadas, o recorte em binary64 guardava uma lasca de
    arredondamento, que o atalho descarta. É só uma diferença de construção,
    que os certificados não usam. Os fontes da `main` atual dão as mesmas
    9.273 chamadas do binário de 16/10.
  - Os tempos nos núcleos de eficiência dão a branch 1,11× mais rápida (média
    geométrica dos casos idênticos acima de 0,05 s).
- **Corpus na dantzig, protocolo do artigo** (22/10; configuração C, 300 s, 3
  repetições, uma execução por núcleo de performance; branch em 613fa46,
  `main` em e9c491f; os números de comparação da `main` estão na seção da
  dantzig).
  - As 1.671 execuções resolvidas têm nós, chamadas e limites idênticos aos da
    `main`, bit a bit, inclusive o caso 538. A diferença dele aparece só no
    build arm64 do Mac; a dantzig usa x86-64 genérico.
  - A branch fecha as mesmas 557 instâncias e é mais rápida em todas: 1,044×
    na média geométrica (de 1,011× a 1,075×), contra a rodada da `main` com a
    máquina ociosa, nos 118 casos que levam ≥ 0,05 s.
  - Houve carga alheia nos núcleos de 13 execuções com ≥ 0,05 s (6% mais lentas
    em média); as medianas de três repetições a absorvem.

**Limitações.**
- Uma execução por instância, num Mac só.
- Instâncias exploratórias, não realistas.
- O Fekete não rodou nestas instâncias. O SOCP dele cresce com m a cada
  chamada.
- Antes de um merge, o benchmark do artigo precisa ser refeito nas máquinas de
  referência. O tamanho da árvore não muda, mas os tempos sim.
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
  mesmo resultado que ele. A exceção é o atalho do recorte. Com regiões quase
  encostadas, o recorte em binary64 pode guardar uma lasca de arredondamento,
  e o atalho a descarta. Só a construção muda: no corpus inteiro, isso explica
  o único caso diferente, o 538
  ([abaixo](#tspn-smoothed-vertex-scaling-2026-10-21), bisseção até 52a3f70).
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
- **Modelo do Gurobi com normais unitárias** (refeito em 21/10, depois da
  repetição no corpus [acima](#oracle-gurobi-corpus-mac-2026-10-21)): os tempos
  do Gurobi quase não mudam. No maior tamanho, as razões ficam em 92.668× e
  76.572× (disco, k = 3 e 10) e em 73.703× e 51.046× (elipse), a mesma ordem
  de antes. O limite superior apertado nunca passa do objetivo do Gurobi em
  mais de 8,7e-7.
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
