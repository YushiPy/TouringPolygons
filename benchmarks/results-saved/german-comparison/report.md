# Análise comparativa — German comparison

Análise das 558 instâncias, unidas por `case_index` e `sha256`. Os histogramas de tempo incluem todos os tempos registrados; speedup e comprimento usam apenas as 550 instâncias concluídas pelo Fekete.

**Complemento estratificado:** a comparação por origem (OSM, aleatórias e tessellations/Voronoi), faixa de tamanho e sobreposição está em [`analysis/source-stratified-report.md`](analysis/source-stratified-report.md). Os rótulos e medidas por caso estão em [`analysis/instance-classification.csv`](analysis/instance-classification.csv).

## Resultado principal

| Métrica | Nosso solver | Fekete et al. |
| --- | --- | --- |
| Instâncias certificadas/concluídas | 558/558 | 550/558 |
| Tempo médio registrado | 80.6222 s | 495.218 s |
| Tempo mediano registrado | 0.0782821 s | 0.24411 s |
| Tempo total registrado | 12.5 h | 76.76 h |

## Tempos e speedup

No conjunto comum concluído, a mediana do speedup Fekete/nosso é **5.10667×** e a média geométrica é **4.80243×**. Nosso solver foi mais rápido em 492 de 550 instâncias; o Fekete foi mais rápido em 58. Nesse mesmo conjunto, as medianas de tempo são 0.0694589 s contra 0.235709 s.

| Faixa de regiões | Casos | Mediana nosso (s) | Fekete concluídos | Mediana Fekete (s) | Timeouts Fekete |
| --- | --- | --- | --- | --- | --- |
| 4–10 | 128 | 0.000514292 | 128 | 0.00640817 | 0 |
| 11–20 | 136 | 0.0124725 | 136 | 0.0612365 | 0 |
| 21–30 | 75 | 0.14298 | 75 | 0.473447 | 0 |
| 31–40 | 74 | 0.704311 | 74 | 2.29073 | 0 |
| 41–50 | 75 | 5.82811 | 75 | 15.0854 | 0 |
| 51–60 | 70 | 35.6049 | 62 | 72.0492 | 8 |

## Comprimento

Nas 550 instâncias concluídas por ambos, a razão comprimento(Fekete)/comprimento(nosso) tem mediana **1** e média **1.00006**. O Fekete retornou comprimento maior em 445 casos e menor em 105; diferenças abaixo de 1 podem ser efeito numérico/feasibility tolerance, não evidência de um ótimo melhor que o certificado.

## Precisão geométrica

A distância geométrica é a mínima entre a polilinha da trajetória e cada alvo; zero significa contato ou cruzamento. A contagem anterior de 16.303 contra 15.883 era o número de pares rota–alvo medidos nas trajetórias disponíveis, não o número de contatos. Ela misturava 558 trajetórias nossas e 551 de Fekete; sete trajetórias ausentes de Fekete, cada uma em um caso com 60 alvos, explicam a diferença líquida de 420 observações. Para comparar os métodos sobre exatamente a mesma amostra, usamos os 550 casos concluídos por ambos: 15.823 pares instância–alvo.

Para cada instância, $B$ é o maior lado da caixa envolvente de $s$, $t$ e todos os vértices. A métrica é o pior afastamento relativo $d/B$ entre seus alvos; reportamos o P95 sobre os 550 casos pareados.

| Métrica | Nosso solver | Fekete bruta | Fekete snapped |
| --- | --- | --- | --- |
| P95 do pior $d/B$ por caso | $2.51\times10^{-17}$ (≈ 0) | $1.356\times10^{-6}$ | $1.356\times10^{-6}$ |

As pequenas folgas são compatíveis com tolerâncias numéricas no subproblema SOCP de Fekete et al., mas a comparação geométrica não isola sua causa.

A comparação de comprimento usa o comprimento recalculado das trajetórias do Fekete e o comprimento final certificado do nosso solver. Como ambos produzem uma solução ótima, pequenas diferenças abaixo de 1 na razão podem ser efeito de arredondamento e tolerância numérica.
