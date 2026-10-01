# TSPN — triagem estratificada inicial (2026-09-28)

Triagem de 12 casos SOCG escolhidos lexicograficamente, uma repetição, 2 s de
solver e 12 s por processo. A seleção colocou os quatro casos OSM em Bangalore,
portanto não representa o corpus. Formulação: tour fechado, ordem livre, sem
ponto fixo; gap `1e-6`, factibilidade `1e-8`, validação `1e-7`.

A etapa GMP + cutoff + contatos herdados fechou 8/12 gaps nativos e 5/12 no
Fekete; nos cinco casos com ambos fechados, o nativo venceu 3/5, mediana `1,68×`.
Houve 11/12 tours nativos válidos. Essas medidas são históricas e anteriores às
correções de contatos ativos e à seleção seeded; a versão final está resumida
em [`tspn-active-contacts-2026-09-28`](../tspn-active-contacts-2026-09-28/README.md).
Uma alteração de recuperação posterior ainda não tinha sido validada naquele
momento. Tabelas de repetição, dados brutos e patches foram removidos.
