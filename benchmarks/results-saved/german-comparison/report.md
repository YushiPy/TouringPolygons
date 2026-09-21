# Análise comparativa — German comparison

Análise das 558 instâncias, unidas por `case_index` e `sha256`. Os histogramas de tempo incluem todos os tempos registrados; speedup e comprimento usam apenas as 550 instâncias concluídas pelo Fekete.

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

A métrica usada é a distância Euclidiana mínima entre a polilinha completa da trajetória e cada polígono; zero significa que a trajetória toca ou cruza o polígono. A tabela mostra o maior erro por instância e também a distribuição por ponto de visita.

| Métrica | Nosso solver | Fekete bruta | Fekete snapped |
| --- | --- | --- | --- |
| Instâncias com trajetória | 558 | 551 | 551 |
| Mediana do maior erro por instância | 0 | 3.40404e-06 | 3.40404e-06 |
| P95 do maior erro por instância | 1.13687e-13 | 9.60514e-05 | 9.60514e-05 |
| Instâncias ≤ 1e−7 | 558 | 130 | 130 |
| Instâncias > 1e−7 | 0 | 421 | 421 |

Por ponto de visita, a mesma métrica tem a seguinte distribuição:

| Métrica | Nosso solver | Fekete bruta | Fekete snapped |
| --- | --- | --- | --- |
| Pontos de visita | 16303 | 15883 | 15883 |
| Mediana da distância | 0 | 0 | 0 |
| P95 da distância | 0 | 9.31276e-06 | 9.31276e-06 |
| Pontos ≤ 1e−7 | 16303 | 11715 | 11715 |

A comparação de comprimento usa o comprimento recalculado das trajetórias do Fekete e o comprimento final certificado do nosso solver. Como ambos produzem uma solução ótima, pequenas diferenças abaixo de 1 na razão podem ser efeito de arredondamento e tolerância numérica.
