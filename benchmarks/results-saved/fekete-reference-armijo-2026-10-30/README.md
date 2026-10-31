# Armijo: configuração F com o critério de parada de Fekete

TSPN de ciclo livre, nas 558 instâncias originais de Fekete et al. (4–60
polígonos). F foi medida em 30/10/2026 na armijo (Ryzen 7 9700X, Linux,
uma thread por execução, oito processos), em três repetições de 300 s e uma
extensão de 900 s nas abertas. O mesmo binário da C foi reutilizado e
conferido por SHA-256:
`73c1e229630e9ed9e788094c938a64b339f97dbfeeb9c5f462c2b32f29c68b23`.
O driver é `c3b3a50`; o binário é de `2467373`.

A única mudança nos argumentos foi de `--absolute-gap 0 --relative-gap
0.0009990009990009992` (C: U ≤ 1,001L) para `--absolute-gap 0.001
--relative-gap 0.001` (F: U−L ≤ 0,001U + 0,001), o critério do código
original de Fekete. Os limites de F são provados; os do SOCP de Fekete são
numéricos. Os tours foram validados fora da medição a 1e-7.

## Comparação com C e com o código original

| Medida | C (17/10) | F (30/10) |
|---|---:|---:|
| Resolvidas nas três repetições de 300 s | 557 | 557 |
| Resolvidas após a extensão de 900 s | 557 | 557 |
| Instâncias comuns com B, CHR, uma thread | 541 | 541 |
| Speedup mediano sobre B | 72,6568× | 69,0107× |
| Média aritmética do speedup sobre B | 196,0051× | 183,2106× |
| Menor speedup sobre B | 1,8865× | 1,8365× |
| Mediana com cerca de 5 polígonos | 12,0218× | 11,0028× |
| Mediana com cerca de 60 polígonos | 429,7119× | 394,8284× |
| Soma das medianas dos tempos, 557 resolvidas (s) | 136,0802 | 147,1986 |

O tempo por caso é a mediana das três repetições se todas fecham; senão, o
da execução de 900 s que fecha. F é mais rápida que B em todas as 541
instâncias comuns. A média geométrica das razões de tempo F/C é 1,06545:
nesta medição F ficou 6,5% mais lenta. Não se atribui essa diferença apenas
ao critério, pois as rodadas ocorreram em datas distintas.

Nas 166 instâncias que B fecha nas três repetições e em que sua melhor
repetição leva pelo menos 1 s, usando a melhor repetição de cada solver:
o speedup geométrico B/F é 244,7918×, a razão de nós é 1,082712× e a razão
de custo por nó é 226,0914×. Nós são os retirados da fila: `num_iterations`
de B e `nodes` de F. Assim, 245 ≈ 1,08 × 226, com arredondamento.

Na repetição 1 das 557 resolvidas, a prova intervalar fecha 74,2677% das
chamadas em tesselações e 99,0240% nas OSM. A maior distância dos tours de F
a um polígono nas três repetições é 2,910383e-11. Em 22 instâncias, o melhor
UB de B fica abaixo do melhor LB provado de F; os desvios de factibilidade
de B permanecem dentro da tolerância 1e-3 adotada no SOCP.

## Estado e limites da comparação

A configuração A (parâmetros do artigo, adaptados a 16 threads na armijo) e a B (CHR, uma thread) são as
medições originais de 17/10, na mesma máquina: 542/549 e 523/541 resolvidas
em 300/900 s. Não foram repetidas em 30/10.
Os scripts originais do artigo usam 24 threads. O resumo e o pôster identificam
diretamente a nossa medição com 16 threads, sem chamá-la de reprodução literal
da configuração de paralelismo do artigo.

## Agregação das distâncias conferida nos registros brutos

A comparação de precisão do resumo e do pôster usa **B/CHR contra F**,
uma thread por execução, nas 558 instâncias, inclusive as que ficam abertas.
Para cada instância, toma-se a maior distância a qualquer polígono entre seus
três tours das rodadas de 300 s. A mediana, o percentil e o máximo são então
calculados sobre esses 558 valores, sem arredondamento intermediário.

| Medida | B/CHR | F |
|---|---:|---:|
| Registros de execução | 1674 | 1674 |
| Mediana dos máximos por instância | 2,648977721572538e-6 | 1,9860273225978185e-15 |
| P95 (posto teto de 0,95 × 558) | 1,7083242320488533e-4 | 1,1368683772161603e-13 |
| Maior distância | 8,229625996190154e-4 | 2,9103830456733704e-11 |
| Instâncias acima de 1e-7 | 407 | 0 |

Para publicação, a mediana de B/CHR é **2,65e-6** e seu máximo é **8,23e-4**
(três algarismos significativos, arredondados diretamente dos dados brutos).
O máximo de **9,568426805603701e-4** pertence à raiz **LEFP**, no caso
`kuala_lumpur_n040_seed3462`; ele não pertence ao recorte CHR da comparação
principal. O antigo CSV compacto já arredondava a mediana CHR para 2,65e-6;
a estatística publicada agora não passa por esse arredondamento intermediário.
Todos esses desvios do SOCP estão dentro da tolerância adotada de 1e-3.
O valor conservador 3e-11 para os nossos tours permanece válido.

O caso 189 (`kuala_lumpur_n060_seed3530`) fica aberto em F após 900 s.
A extensão F de 21600 s tem apenas plano e manifesto na cópia local,
sem execução concluída; não há tempo de fechamento de F para publicar.
Os 5362,91 s (89 min) pertencem à configuração C, não a F.

`per-case.csv` guarda A, B/CHR e F, no formato da campanha de 17/10:
uma linha por caso, configuração, raiz e orçamento; contagens por repetição,
mediana/mínimo/máximo de tempo, melhor LB/UB e maior distância do tour.
`analysis.json` contém os agregados sem arredondamento e hashes das fontes.
Os registros completos e a análise executável ficam apenas no workspace
local, em `experiments/fekete-reference@armijo/`.
