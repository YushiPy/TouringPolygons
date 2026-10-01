# TSPN — holdout de portfólio (2026-09-29)

Seis casos pré-selecionados, três repetições por modo, limite nativo de 1 s e
teto por processo de 8 s. Formulação: tour fechado, ordem livre, sem ponto fixo.
Gap alvo `1e-6`; factibilidade `1e-8`; validação independente `1e-7`.
Comparações de tempo contam apenas casos em que todas as repetições de ambos os
solvers validaram e fecharam o gap.

| Modo nativo | Casos pareados | Vitórias nativas | Speedup mediano Fekete/nativo |
|---|---:|---:|---:|
| Default | 5 | 4/5 | 2,26× |
| DFS/BFS | 5 | 4/5 | 3,16× |
| Corrida independente | 5 | 4/5 | 2,00× |
| Portfólio cooperativo | 5 | 4/5 | 2,19× |

O nativo retornou tours válidos e fechou 18/18 execuções. Fekete retornou
tours válidos em 18/18 e fechou 15/18. O caso de tesselação com 20 regiões
favoreceu Fekete; não houve timeouts de processo. O holdout é pequeno e não
mostra vantagem consistente do portfólio cooperativo sobre DFS/BFS ou corrida.
A seleção usou hash de nome, excluiu Bangalore e foi feita antes das medições.
Os bounds externos permanecem numéricos.
