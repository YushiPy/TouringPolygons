# TSPN — triagem de portfólio em casos grandes (2026-09-29)

Seis casos com 30–60 regiões, selecionados antes da medição. A triagem usou
5 s nativos e teto por processo de 12 s; OSM39 recebeu três repetições com
10 s / 15 s. Formulação fechada, ordem livre e sem ponto fixo; gap alvo `1e-6`,
factibilidade `1e-8`, validação `1e-7`.

Todas as trajetórias passaram pela validação. Nos seis casos principais, cada
modo nativo fechou 3/6 gaps; Fekete fechou 2/6. Os únicos dois casos com ambos
fechados foram random30 e tessellation30; o nativo foi mais rápido nos dois em
todos os quatro modos. Os outros casos são limites de orçamento, não speedups
resolvidos. No OSM39, o nativo fechou 3/3 repetições; medianas: default `4,998 s`,
DFS/BFS `1,501 s`, corrida `1,654 s`, cooperativo `1,705 s`. Fekete permaneceu
com gap aberto nas três repetições, portanto não há speedup pareado nesse caso.

A seleção excluiu 18 casos anteriores e Bangalore; três reservas não foram
medidas. Resultados são uma triagem estratificada pequena, sem alegação de
melhoria universal. Dados por repetição e polígonos duplicados foram removidos.
