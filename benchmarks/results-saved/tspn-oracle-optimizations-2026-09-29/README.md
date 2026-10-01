# TSPN — otimizações do oráculo (2026-09-29/30)

**Formulação e contrato.** B&B para tour fechado de polígonos simples, ordem
livre, sem ponto fixo. Telas principais usaram 2 s de solver e teto de processo
de 8 s; gap alvo `1e-6`, factibilidade `1e-8`, validação independente `1e-7`.
Gap numérico fechado não certifica ótimo racional do TSPN completo.

**Resultado.** A shortlist `cache + features + root` (CFR) retornou 12/12
trajetórias válidas e fechou 9/12 gaps no conjunto pequeno/grande. Nos sete
casos com ambos os solvers válidos e gap fechado, venceu 7/7, speedup mediano
`3,686×`. Candidate E fechou OSM39 em 3/3 repetições, mediana `0,736 s`.
Uma relaxação capturada de 19 regiões passou de mais de 8 s para cerca de
110 ms; é um caso diagnóstico, não uma garantia para outras entradas.

**Limites.** OSM50, random59 e tessellation60 permaneceram com gap aberto na
triagem. `lazy` ainda deixou OSM39 com gap aberto de aproximadamente 8,5% no
limite de 2 s. A pequena diferença de tempo entre versões não foi atribuída
causalmente ao algoritmo. Binário Candidate E: SHA-256
`277452400f5c472919e3acbace5ac03a07abb6d5ec252ee5fd04f8c7d70fb3f2`.
Builds, matrizes de variantes, patches e execuções completas foram removidos.

O fixture de regressão da relaxação de 19 regiões está em
`packages/convex-tpp/cpp/tests/cycle_active_contacts.json`; esta nota mantém o
resumo e a identificação do experimento que originou a regressão.
