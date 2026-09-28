# Resultado final: ciclo convexo exato, interseções e desempenho

Trabalho concluído no worktree `/private/tmp/tpp-convex-cycle`, branch
`codex/convex-cycle-disjoint`, sem commit. As alterações locais e as referências
Gurobi anteriores foram preservadas. Este relatório substitui o estado parcial
em `../convex-cycle-performance-2026-09-27/handoff.md`.

## Algoritmo e exatidão

As APIs gerais aceitam polígonos convexos fechados com toque, sobreposição e
contenção. A ordem é fixa, o ciclo é fechado e pode se auto-intersectar.
As duas versões compartilham propostas por contatos ativos/reflexões e a busca
por âncoras nas fronteiras. Elas reutilizam as recorrências e mapas direcionais
C++ existentes. Não há SOCP de produção, discretização nem epsilon de aceitação.

O certificado independente trata completamente os links zero por propagação de
conjuntos duais no disco unitário. Condições de suporte necessárias e suficientes
provam optimalidade; a busca trata extremos de âncora não diferenciáveis e usa
reconstrução racional. O racional devolve contatos exatos e o objetivo como soma
de raízes de quadrados racionais, sem erro de otimização.

Double usa as mesmas construções, com limites próprios de ponto flutuante.
A configuração medida permite recuperação racional explícita: seis instâncias
usaram reconstrução de features e uma usou recuperação do ciclo zero. Nenhuma
precisou recuperar uma âncora nesta campanha. Os contadores constam dos dados.
É possível desativar recuperações; double puro ainda pode falhar por arredondamento,
como registrado nos testes das referências. `FloatingPointLimit` não significa
ótimo exato. O exemplo cujo único ponto comum é `(1/3,1/3)` tem comprimento zero
no racional e intervalo double `[0, 1.5700924586837752e-16]`.

## Complexidade

Com N vértices e k regiões, o certificado custa O(N+k³) operações racionais,
ou O(N) sem links zero. Uma varredura da proposta sem blocos zero custa O(N);
ela é limitada estruturalmente a k+1 varreduras antes da busca completa.

Na busca geral, A=O(kN) segmentos de âncora, H=1+log Q para reconstrução do
parâmetro racional, C=O(N²+k²N log(kN)) para o mapa de fonte fixa e V=O(N+k³)
para o certificado dão O(N²+A H(C+V+H)) sem aceleração. A proposta opcional
adiciona R=O(k(N²+V)) a cada invocação. Espaço O(kN+H), em valores aritméticos;
os custos de bits dos racionais não são constantes. A prova e os contratos
estão em [convex-cycle.md](../../../docs/algorithms/convex-cycle.md) e
[convex-cycle-certificate.md](../../../docs/algorithms/convex-cycle-certificate.md).

## Medição final

20 instâncias, 15 repetições cada, uma execução de aquecimento por backend,
ordem de execução rotativa, Gurobi com uma thread e ambiente já inicializado.
Tempos C++ incluem validação e certificação. Gurobi inclui construção, solução,
extração e destruição do modelo; seu tempo de otimização também está separado.
A campanha final foi medida após terminar os demais builds/testes.

| Quatro referências originais, k=2–5 | Medianas em ms |
|---|---:|
| Racional | 0.017–0.288 |
| Double, recuperação habilitada | 0.013–0.361 |
| Gurobi, chamada completa | 0.299–0.445 |

Ambas as versões ficaram à frente de Gurobi nas 20 medianas: fatores de
1.29–21.33 no racional e 1.11–22.38 em double.
Todos os resultados racionais são certificados ótimos; todos os resultados
double são factíveis. Todos os intervalos independentes do racional/Gurobi
se sobrepõem. Comparamos objetivos e limites, nunca igualdade de contatos.
A maior diferença entre limites superiores double/racional foi
1.5700924586837752e-16; o maior gap double
foi 9.3132257461547852e-10, nas instâncias de escala grande.

As duas degenerescências novas eram lentas na campanha
`../convex-cycle-complete-2026-09-27/`. Liberar conjuntamente vértices coincidentes
para uma proposta de reflexão, e antecipar a recuperação do ponto comum,
removeram as buscas redundantes. Ambas as propostas continuam certificadas.

Esta suíte sintética não estabelece dominância universal. A comparação é da
formulação SOCP de ciclo com ordem fixa, não do algoritmo completo de Fekete
para escolher ordens. Gurobi usa tolerâncias numéricas registradas em config.json;
seus limites numéricos não são tratados como certificados exatos. Os contatos
Gurobi recebem correção racional mínima de factibilidade fora da medição antes
da verificação independente. Tempos absolutos e diferenças pequenas variam
com a máquina e a carga.

## Validação e pendência externa

- Ciclos: testes finais passaram; 66 comparações disjuntas, 96 casos aleatórios
  com interseções, buscas acelerada e sem aceleração, variantes double puro,
  contatos não diádicos, escalas racionais 2^±1100 e regressão de extremos não
  diferenciáveis. As quatro referências Gurobi originais foram comparadas.
- Certificados: 1.040 pares de raios analíticos independentes, incluindo variantes
  com blocos maiores, passaram; também passaram entradas inválidas e bounds.
- Mapas direcionais: 1.604 verificações, zero falhas ou casos não resolvidos.
- `./scripts/sanity_check.sh --no-install`: terminou com código zero, incluindo
  as oito suítes convexas e o smoke benchmark não convexo. Foi executado antes
  das últimas otimizações restritas ao ciclo; os testes de ciclo/certificado
  foram repetidos depois delas.
- WASM: build e `node wasm/test-intersections.mjs` passaram, incluindo Wrong1.
- `RUN_BROWSER=0 npm run test:all`: bloqueado na obtenção de `contourpy` por
  falha de DNS em files.pythonhosted.org; não se afirma sucesso dessa suíte.
- Revisão do diff e `git diff --check` passaram; nenhuma remoção de arquivo
  existente. Hashes da campanha final conferidos com o código medido.

Entradas, dados brutos, parâmetros, logs, hashes e análises estão juntos nesta
campanha. Não há benchmark nem processo de teste deixado em execução.
