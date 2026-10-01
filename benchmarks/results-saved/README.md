# Benchmarks preservados

Esta pasta guarda um resumo compacto dos resultados, não cópias de campanhas.
Execuções completas, entradas repetidas, trajetórias, logs, builds e patches de
experimento ficam nos diretórios locais ignorados (`benchmarks/campaigns/` e
`benchmarks/results/`).

| Pasta | Conteúdo preservado |
|---|---|
| [fekete-comparison](fekete-comparison/README.md) | Comparação completa usada pelo material SIICUSP; CSVs e corpus são dependências do exportador. |
| [convex-cycle-gurobi-reference-2026-09-25](convex-cycle-gurobi-reference-2026-09-25/README.md) | Resumo e `instances.json`, fixture pequeno exigido pelos testes e benchmarks. |
| [convex-cycle-final-2026-09-27](convex-cycle-final-2026-09-27/README.md) | Resultado final do ciclo convexo sintético; campanhas intermediárias estão consolidadas em uma linha cada. |
| [tspn-active-contacts-2026-09-28](tspn-active-contacts-2026-09-28/README.md) | Comparação TSPN após correções de contatos ativos. |
| [tspn-oracle-optimizations-2026-09-29](tspn-oracle-optimizations-2026-09-29/README.md) | Triagem de otimizações do oráculo e resultado de Candidate E. |
| [tspn-portfolio-holdout-2026-09-29](tspn-portfolio-holdout-2026-09-29/README.md) | Holdout de seis casos, três repetições por modo. |
| [tspn-portfolio-large-2026-09-29](tspn-portfolio-large-2026-09-29/README.md) | Triagem de seis casos grandes e repetição OSM39. |
| [tspn-cycle-reuse-cutoff-2026-09-30](tspn-cycle-reuse-cutoff-2026-09-30/README.md) | Reuso de relaxações e propagação de cutoff. |
| [tspn-dual-interval-sharing-2026-09-30](tspn-dual-interval-sharing-2026-09-30/README.md) | Certificado intervalar, dual-screen e bounds compartilhados. |
| [tspn-held-karp-learning-2026-09-30](tspn-held-karp-learning-2026-09-30/README.md) | Triagem do bound one-tree e branching aprendido. |
| [tspn-socg-stratified-2026-09-28](tspn-socg-stratified-2026-09-28/README.md) | Triagem inicial, amostra enviesada e resultados depois substituídos. |
| [tspn-fekete-2026-09-27](tspn-fekete-2026-09-27/README.md) | Comparação inicial do solver TSPN com o baseline SOCP. |
| [tspn-fekete-final-2026-09-27](tspn-fekete-final-2026-09-27/README.md) | Recaptura final da comparação inicial; sem dados brutos. |
| [convex-cycle-complete-2026-09-27](convex-cycle-complete-2026-09-27/README.md) | Medição preliminar, substituída pela campanha final. |
| [convex-cycle-optimized-2026-09-27](convex-cycle-optimized-2026-09-27/README.md) | Medição intermediária, consolidada na campanha final. |
| [convex-cycle-performance-2026-09-27](convex-cycle-performance-2026-09-27/README.md) | Medição intermediária, consolidada na campanha final. |

Cada resumo informa formulação, orçamento e tolerâncias relevantes, status de
gap/exatidão e limitações. Um gap numérico fechado no B&B TSPN não significa
ótimo racional do problema completo; bounds SOCP de Fekete são numéricos.
