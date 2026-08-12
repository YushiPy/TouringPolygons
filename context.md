
## O Problema — TPP (Touring Polygons Problem)

### Definição

Dado um ponto inicial $s$, um ponto final $t$ e uma sequência de polígonos $P_1, \dots, P_k$ no plano, encontrar o caminho euclidiano de menor comprimento partindo de $s$, visitando cada $P_i$ (tocando a borda ou atravessando), e chegando em $t$. A **ordem de visita é fixada** no TPP clássico.

### Variantes e hierarquia de dificuldade

1. **TPP convexo, ordem fixa** — polinomial, $O(nk\log(n/k))$ e $O(nk^2\log(n/k))$ [Dror 2003], $O(nk)$ e $O(nk^2)$ [Tan & Jiang 2017] para polígonos disjuntos e com interseção, respectivamente.
2. **TPP não convexo, ordem fixa** — NP-difícil [Dror 2003]
3. **TPP não convexo, sem ordem fixa** — NP-difícil (generaliza TSP e TPP não convexo); equivalente ao VRP com um único veículo sem capacidade
4. **CVRPN (Capacitated VRP with Neighborhoods)** — múltiplos veículos com capacidade; generaliza o TPP sem ordem fixa (caso $s = t = \text{depósito}$, um veículo, sem capacidade)

### Relação com outros problemas

- **TSP:** o TSP é um problema clássico e o TPP generaliza ele, pois podemos fazer `(s = t)` e obter o TSP.

- **TSPN (TSP with Neighborhoods):** tour com ponto inicial e final iguais, mas não especificados, visitando regiões em ordem livre — o TPP sem ordem é um TSPN com $s$ e $t$ fixos e possivelmente distintos, assim o TPP não é uma generalização do TSPN, ao mesmo tempo o TSPN não é uma generalização do TPP, pois não é possível recriar o ponto incial e final distintos. Soluções exatas via Mixed-Integer Nonlinear Programming (MINLP): Gentilini et al. [2013] e Fekete et al. [2026]

- **CETSP (Close-Enough TSP):** VRP onde as regiões são discos — caso especial. O caminho começa e termina no depósito (um ponto em $\mathbb{R^2}$), visita cada região (disco) e retorna ao depósito, sem ordem fixa. Soluções exatas via Branch and Bound: Coutinho et al. [2016], Zhang et al. [2023]

- **CVRPN:** VRP onde clientes são polígonos — o artigo do doutorando Rogério S. Matos (orientado pelo Ernesto) resolve isso com heurística (GVNS + block coordinate descent)

---

## Estado do Projeto

### O que já foi feito

**TPP convexo, ordem fixa (COMPLETO):**

- Implementação do algoritmo de Dror et al. [2003] em C++23
	- Estruturas: região de primeiro contato $T_i$ e mapa de último passo $S_i$
	- 3 implementações: busca linear $O(n^2)$, busca binária $O(nk\log(n/k))$, memoizada (melhor na prática)
	- A implementação memoizada usa estratégia própria de cache integrada à busca binária, mais eficiente que a programação dinâmica original
- Implementação do algoritmo de Tan & Jiang [2017] — $O(nk)$ mas sem memoização; pior na prática
- Ainda não implementamos o caso com interseção de polígonos.
- Baseline SOCP via Gurobi para verificação de corretude
- Ferramenta de visualização interativa (instâncias + soluções + last-step maps para o caso convexo)
- TODO: Implementar caso com interseção de polígonos (Dror et al. [2003] e Tan & Jiang [2017] descrevem extensão, mas não implementam)

**TPP não convexo, ordem fixa (EM DESENVOLVIMENTO):**

- Decomposição convexa via algoritmo de Greene, implementado com CGAL
- Branch and Bound: cada nó atribui peça convexa a polígono; lower bound = resolver TPP convexo com convex hulls nos polígonos não atribuídos
- Solução inicial heurística: colocar pontos com distância de no máximo $\epsilon$ entre si no perímetro de cada polígono não convexo. Depois resolver usando BFS.
- Performance: reduz espaço de busca em várias ordens de grandeza vs. enumeração completa
- **Ainda a fazer:** melhorar bounds, incorporar geometria no B&B (last-step maps revelam quais partes da borda são alcançáveis), explorar Chazelle-Dobkin e convex covers
- Baseline MIQCP via Gurobi para verificação (reformulação como MIQCP: variáveis binárias $w_{i\ell}$ para escolha de peça convexa, restrições SOC para distâncias)

### Próximos passos (em ordem cronológica)

1. Concluir implementação e análise de Tan & Jiang
2. Desenvolver e aprimorar B&B para TPP não convexo com ordem fixa
3. **[BEPE]** TPP não convexo sem ordem fixa — B&B sobre permutações, usando TPP solver como subrotina
4. **[Pós-BEPE]** CVRPN exato — continuação natural; múltiplos veículos com capacidade
5. Consolidação de resultados e experimentos finais

---

## Detalhes Técnicos da Implementação

### Stack

- **Linguagem:** C++23
- **Geometria:** CGAL (exclusivamente para decomposição convexa de Greene)
	- CGAL é GPL para uso comercial → repositório sob GPL v3
	- Remoção futura possível via Hertel-Mehlhorn (já implementado) ou Greene/Chazelle autorais
- **Solver:** Gurobi (somente para baselines/verificação — não é central ao algoritmo)
- **Decomposição convexa:** Greene's algorithm [1983] via CGAL (~10 peças por polígono nas instâncias de teste)

### Vantagem sobre trabalhos relacionados

Fekete et al. [SoCG 2026] e Coutinho et al. [IJOC 2016] resolvem o subproblema de ordem fixa via SOCP com Gurobi — não é possível evitar Gurobi nessa abordagem (~US$12k-15k/ano para uso comercial). Nossa abordagem reduz o subproblema a uma instância do TPP convexo, resolvida com algoritmo autoral sem dependências comerciais obrigatórias.

---

## Notas Diversas

- **Repositório:** será público sob GPL v3 (compatível com CGAL)
- **Processo número FAPESP:** 2025/13861-1
- **SIICUSP:** Gabriel participou em 2025 e pretende participar em 2026 (outubro) — motivo adicional para não sair em outubro-dezembro
- **Data de saída para BEPE:** entre 6-10 de janeiro de 2026 (evitar 1 de janeiro — feriado, universidade fechada)
- **Relatório parcial:** ainda a fazer; cobre apenas o TPP convexo (item 1 do projeto)
- **LaTeX:** usar `\(\)` para matemática inline (não `$$`); texto em linha única (LaTeX faz wrapping), usar tabs para indentação, evitar espaços em branco desnecessários.
- **Texto:** evitar usar travessões (—) em LaTeX, pois passam a impressão de texto gerado por IA, prefira usar vírgulas, parênteses ou ponto e vírgula para separar ideias.

---

## Referências Bibliográficas (arquivo referencias.bib)

Aqui está a seção de referências reformatada. Vou substituir no context.md:

---

## Referências Bibliográficas

A seguir, as referências bibliográficas estão organizadas por categoria: (1) trabalhos diretamente relacionados ao TPP, (2) contexto clássico de TSP e TSPN, (3) trabalhos exatos relacionados, (4) geometria computacional relevante para a decomposição convexa, (5) motivação aplicada para last-mile delivery e AMR, e (6) referências do orientador no exterior para demonstrar expertise em métodos exatos para VRP. Cada referência é descrita brevemente para destacar sua relevância para o projeto.

O nome de cada referência é o mesmo usado no arquivo `referencias.bib` para facilitar a consulta, quando for citar a referência no texto, use `\cite{nome-da-referencia}`. As referências completas estão no arquivo `referencias.bib` em formato BibTeX, não é necessário reescrevê-las aqui, apenas as descrições e a organização por categoria.

### Problema Base

**tpp-dror-2003** — *Touring a sequence of polygons*, Dror, Efrat, Lubiw, Mitchell. STOC 2003.
Introduz o TPP. Prova que o caso de polígonos convexos disjuntos com ordem fixa é solúvel em $O(nk\log(n/k))$ via programação dinâmica com last-step maps, também mostra uma solução em tempo $O(nk^2\log(n/k))$ para o caso de polígonos não disjuntos. O paper também descreve uma solução que considera "cercas", as quais o caminho deve permancer dentro, mas esse resultado é pouco relevante para o nosso projeto. Prova que o caso não convexo é NP-difícil. É a referência central do projeto inteiro.

**tpp-tan-2017** — *Efficient algorithms for touring a sequence of convex polygons and related problems*, Tan, Jiang. TAMC 2017.
Melhora a complexidade do TPP convexo disjunto de $O(nk\log(n/k))$ para $O(nk)$ e $O(nk^2\log(n/k))$ para $O(nk^2)$ quando temos interseções. A abordagem é puramente iterativa e não admite a estratégia de memoização que desenvolvemos, sendo pior na prática apesar da complexidade teórica melhor. Relevante por mostrar familiaridade com a literatura e porque estamos implementando para comparação.

---

### Contexto Clássico

**tsp-cook-2011** — *The Traveling Salesman Problem: A Computational Study*, Cook, Applegate, Bixby, Chvátal. Princeton University Press, 2011.
Referência canônica para o TSP. Citado para contextualizar o problema clássico do qual o TSPN e o TPP sem ordem fixa são generalizações.

**tspn-arkin-1994** — *Approximation algorithms for the geometric covering salesman problem*, Arkin, Hassin. Discrete Applied Mathematics, 1994.
Introduz formalmente o TSPN (TSP with Neighborhoods). Citado para contextualizar a família de problemas em que o nosso se insere.

---

### Trabalhos Exatos Relacionados

**cetsp-coutinho-2016** — *A branch-and-bound algorithm for the Close-Enough Traveling Salesman Problem*, Coutinho, do Nascimento, Pessoa, Subramanian. INFORMS Journal on Computing, 2016.
Primeiro algoritmo exato para o CETSP (TSPN com vizinhanças circulares). Usa B&B onde o subproblema de ordem fixa é formulado como SOCP e resolvido com Gurobi. Resolve instâncias com até mil nós. Relevante como trabalho relacionado — abordagem similar à nossa em estrutura (B&B + subproblema exato), mas depende de Gurobi para o subproblema central, o que não pode ser contornado.

**cetsp-zhang-2023** — *Results for the close-enough traveling salesman problem with a branch-and-bound algorithm*, Zhang, Sauppe, Jacobson. Computational Optimization and Applications, 2023.
Extensão do trabalho de Coutinho com estratégias de bound melhoradas para o CETSP. Também usa SOCP + Gurobi. Relevante como estado da arte para o caso circular.

**tspn-fekete-2026** — *A Branch-And-Bound Algorithm for the Traveling Salesman Problem with Difficult Neighborhoods*, Fekete, Kniep, Krupke, Perk. SoCG 2026.
Primeiro algoritmo exato de B&B para o TSPN com vizinhanças poligonais arbitrárias incluindo não convexas. Usa SOCP + Gurobi para resolver o subproblema de ordem fixa. É o trabalho mais próximo do nosso: diferenças são (1) eles resolvem um tour (`s=t`, mas não são definidos, sendo parte das variáveis a serem otimizadas) enquanto nosso problema tem `s` e `t` possivelmente distintos, mas fixos, e (2) eles dependem de Gurobi para o subproblema central, enquanto nossa abordagem usa o solver autoral do TPP convexo, sem dependências comerciais obrigatórias.

**tspn-gentilini-2013** — *The Travelling Salesman Problem with neighbourhoods: MINLP solution*, Gentilini, Margot, Shimada. Optimization Methods and Software, 2013.
Formula o TSPN como MINLP e resolve com solver comercial. Experimentos limitados a instâncias pequenas ou convexas. Relevante como exemplo de abordagem via programação matemática para o TSPN, que é alternativa ao B&B.

---

### Geometria Computacional

**cpd-greene-1983** — *The decomposition of polygons into convex parts*, Greene. In: Computational Geometry (ed. Preparata), JAI Press, 1983.
Descreve o algoritmo de Greene para decomposição convexa de polígonos simples. É o algoritmo usado na nossa implementação via CGAL. Produz em média ~10 peças por polígono nas instâncias de teste. É ótimo se você considerar apenas arestas entre vértices, mas pode ser melhorado com o algoritmo de Chazelle-Dobkin que considera *Steiner Points*. No entanto, entre os dois é o único que está de fato implementado (na biblioteca CGAL), assim usamos no momento.

**ocd-chazelle-1985** — *Optimal convex decompositions*, Chazelle, Dobkin. In: Computational Geometry (ed. Toussaint), North-Holland, 1985.
Descreve um algoritmo para decompor polígonos no número mínimo possível de peças convexas, usando Steiner points. Sempre produz menos ou igual peças que Greene. Menos peças → menor search space no B&B → potencialmente muito mais rápido. Não está implementado ainda; implementar seria uma melhoria significativa, mas também seria uma tarefa considerável. Por isso, é um candidato a melhoria futura, mas não é a abordagem atual.

**hcd-hertel-1985** — *Fast triangulation of the plane with respect to simple polygons*, Hertel, Mehlhorn. Information and Control, 1985.
Descreve o algoritmo de Hertel-Mehlhorn para decomposição convexa aproximada. Mais simples que Greene e Chazelle, não é ótimo mas é uma 4-aproximação. Já está implementado autoralmente (sem CGAL), o que o torna relevante para eliminar a dependência do CGAL caso necessário para uso comercial.

---

### Motivação Aplicada

**crowdsourced-alnaggar-2021** — *Crowdsourced delivery: A review of platforms and academic literature*, Alnaggar, Gzara, Bookbinder. Omega, 2021.
Survey sobre entrega crowdsourced (last-mile delivery). Relevante para motivar o problema: cada courier opera em uma região não convexa (bairro, zona urbana) e a ordem de visita não é fixada.

**crowdshipping-le-2019** — *Supply, demand, operations, and management of crowd-shipping services*, Le, Stathopoulos, Van Woensel, Ukkusuri. Transportation Research Part C, 2019.
Revisão sobre crowd-shipping. Complementa Alnaggar na motivação de last-mile delivery com vizinhanças poligonais.

**amr-dong-2007** — *Heuristic approaches for a TSP variant: The automatic meter reading shortest tour problem*, Dong, Yang, Chen. In: Extending the Horizons, Springer, 2007.
Aborda leitura automática de medidores (AMR) como variante do TSP. Relevante para motivar o CETSP e por extensão o TPP: o veículo não precisa ir ao local exato do medidor, apenas passar dentro do alcance do sinal.

**amr-shuttleworth-2008** — *Advances in meter reading: Heuristic solution of the CETSP over a street network*, Shuttleworth, Golden, Smith, Wasil. In: The Vehicle Routing Problem, Springer, 2008.
Aplica o CETSP ao contexto de AMR em rede viária. Complementa Dong na motivação de AMR.

**rwp-haslett-2008** — *Essentials of Radio Wave Propagation*, Haslett. Cambridge University Press, 2008.
Livro sobre propagação de ondas de rádio. Justifica por que a zona de cobertura de um medidor é não convexa na prática: obstáculos físicos (paredes, prédios) criam zonas de sombra que tornam a região de cobertura irregular e não convexa, motivando o uso de polígonos não convexos em vez de discos.

---

### Referências do Orientador no Exterior

**coelho2013** — *The exact solution of several classes of inventory-routing problems*, Coelho, Laporte. Computers & Operations Research, 2013.
Trabalho do Leandro Coelho com Laporte em métodos exatos para problemas de roteamento com estoque. Citado na proposta BEPE para demonstrar o histórico do Leandro em métodos exatos para VRP.

**desaulniers2016** — *A branch-price-and-cut algorithm for the inventory-routing problem*, Desaulniers, Rakke, Coelho. Transportation Science, 2016.
Algoritmo branch-price-and-cut do grupo do Leandro. Citado para demonstrar expertise em métodos exatos avançados para VRP — diretamente relevante para o que vamos fazer na BEPE.

---

### Ferramentas

**cgal** — *CGAL 6.0*, The CGAL Project, 2025. https://www.cgal.org
Biblioteca de geometria computacional usada exclusivamente para a decomposição convexa de Greene. Licença GPL para uso comercial — força o repositório a ser GPL v3. Pode ser removida futuramente usando Hertel-Mehlhorn autoral, reimplementando Greene sem dependências, ou sendo o primeiro a implementar Chazelle-Dobkin para obter menos peças convexas.

**gurobi** — *Gurobi Optimizer 12.0*, Gurobi Optimization LLC, 2025. https://www.gurobi.com
Solver comercial usado apenas para baselines de verificação (SOCP para TPP convexo, MIQCP para TPP não convexo). Não é central ao algoritmo — pode ser removido sem perda de funcionalidade, apenas de verificação.

**matos2024** — *Solving the capacitated vehicle routing problem with polygonal neighborhoods using a metaheuristic enhanced with block coordinate descent*, Matos, Birgin. Manuscrito, 2024.
Paper do doutorando Rogério S. Matos orientado pelo Ernesto. Resolve o CVRPN com heurística (GVNS + block coordinate descent). É a principal referência para o CVRPN — o nosso trabalho futuro visa um método exato para o mesmo problema. Ainda não publicado; citar como manuscrito.
