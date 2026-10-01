# TSPN — intervalos, dual-screen e bounds compartilhados (2026-09-30)

Tour fechado, ordem livre, sem ponto fixo; gap alvo `1e-6`, factibilidade
`1e-8`, validação independente `1e-7`. Os 96 resultados nativos foram válidos;
60 fecharam o gap e 36 pararam no limite de solver, sem timeouts de processo.
As bounds Fekete são referências numéricas; gap fechado no B&B não implica
ótimo racional do TSPN completo.

**Certificado intervalar.** Na comparação repetida large6 venceu as nove
comparações pareadas com gap fechado, speedup mediano `1,263×`; também reduziu
gaps nos casos grandes abertos. No OSM39 com portfólio e três repetições,
mediana foi `0,754 s` no controle e `0,507 s` com intervalos, 3/3 vitórias
pareadas (`1,481×`).

**Outras opções.** `dual-screen` não melhorou o tempo dos casos fechados.
`share-bounds` teve hits e podas certificados, mas não reduziu o tempo de parede
no screen. Mantenha todas as opções experimentais desligadas por padrão até
validação mais ampla; o ganho medido do filtro intervalar não é universal.
