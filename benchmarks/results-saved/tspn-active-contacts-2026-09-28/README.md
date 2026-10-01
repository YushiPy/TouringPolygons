# TSPN — contatos ativos e custo serial (2026-09-28)

**Formulação.** Tour fechado por regiões poligonais simples, ordem cíclica
livre, sem ponto fixo; sobreposição permitida. 33 casos, cinco repetições na
comparação final. Gap alvo `1e-6`; factibilidade `1e-8`; validação independente
`1e-7`. O campo `exact` significa gap solicitado fechado, não ótimo racional do
TSPN completo. Bounds Fekete são numéricos.

**Resultado.** Os dois solvers retornaram trajetórias válidas em todas as 165
execuções. O B&B nativo fechou 33/33 gaps e Fekete 25/33. Entre os 25 casos em
que ambos fecharam o gap, o nativo venceu 17/25, com speedup geométrico mediano
`1,70×`; não é uma estimativa universal. `fekete_3_n10` caiu de cerca de
19,0 s para 15,7 ms após a correção de contatos ativos. O caso `fekete_6_n15`
passou de execução com gap aberto para objetivo fechado em cerca de 108 ms.

A otimização reutiliza contatos ativos, mantendo certificado independente e
fallback completo. O custo serial OpenMP também foi removido antes do relógio
interno. O B&B segue exponencial; algumas instâncias ainda perdem para Fekete.
Foram removidos os raws por repetição e as cópias das mesmas instâncias.
